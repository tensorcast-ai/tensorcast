// Copyright (c) 2025-2026, TensorCast Team.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/logging_init.h"
#include "core/common/memory/cuda_memory.h"
#include "core/common/memory/pinned_buffer_pool.h"
#include "core/common/memory/streaming_pinned_buffer.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/error_handling.h"
#include "core/store/materialization/dataplane/contracts/buffer_pool.h"
#include "core/store/materialization/dataplane/contracts/sink.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "core/store/materialization/dataplane/metadata/index_reader.h"
#include "core/store/materialization/dataplane/metadata/safetensors_util.h"
#include "core/store/materialization/dataplane/runtime/pump.h"
#include "core/store/materialization/dataplane/runtime/streaming_buffer_adapter.h"
#include "core/store/materialization/dataplane/sinks/gpu_memory_sink.h"
#include "core/store/materialization/dataplane/sources/file_partition_source.h"
#include "core/store/materialization/dataplane/sources/multi_safetensors_source.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

#include <nccl.h>

namespace tensorcast::store::loader {
namespace {

#define TC_RETURN_IF_ERROR(expr) \
  do {                           \
    absl::Status _st = (expr);   \
    if (!_st.ok()) {             \
      return _st;                \
    }                            \
  } while (0)

#define TC_ASSIGN_OR_RETURN(lhs, expr) \
  do {                                 \
    auto _or = (expr);                 \
    if (!_or.ok()) {                   \
      return _or.status();             \
    }                                  \
    (lhs) = std::move(_or).value();    \
  } while (0)

common::AsyncRuntime& pump_benchmark_runtime() {
  static common::AsyncRuntime runtime(
      common::AsyncRuntime::Options{
          .thread_name_prefix = "tensorcast-pump-bench",
      });
  return runtime;
}

absl::StatusOr<size_t> pread_fully(int fd, uint64_t off, void* dst, size_t bytes) {
  size_t total = 0;
  char* ptr = static_cast<char*>(dst);
  while (total < bytes) {
    const ssize_t got = ::pread(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::ErrnoToStatus(errno, "pread failed");
    }
    if (got == 0) {
      break;
    }
    total += static_cast<size_t>(got);
  }
  return total;
}

enum class BenchMode : uint8_t {
  kLoader = 0,
  kDiskBaseline = 1,
  kDiskFragmentation = 2,
  kGpuPeerBaseline = 3,
  kSafetensorsDiskBaseline = 4,
  kSafetensorsHostBaseline = 5,
  kH2dBaseline = 6,
  kH2d2dBaseline = 16,
  kSafetensorsODirectHostBaseline = 7,
  kSafetensorsODirectDiskBaseline = 8,
  kSafetensorsHotHostBaseline = 11,
  kSafetensorsHotDiskBaseline = 12,
  kNcclBaseline = 9,
  kNcclLaunchTax = 10,
  kMaterializeD = 13,
  kMaterializedDiskBaseline = 14,
  kSafetensorsDramMirrorHostBaseline = 15,
};
enum class StrategyKind : uint8_t { kA_Eager = 0, kB_LazyCommit = 1, kC_BatchedOptimal = 2, kC_HostPack = 3 };
enum class NcclOp : uint8_t { kBroadcast = 0, kSendrecv = 1 };

constexpr uint64_t kCorrectnessSampleMaxBytes = 4ull * 1024ull * 1024ull;
constexpr int kGpuPeerDefaultSrcDevice = 0;
constexpr int kGpuPeerDefaultDstDevice = 1;
constexpr int kDefaultNcclWarmupIters = 10;
constexpr int kDefaultNcclIters = 100;
constexpr uint64_t kMaxPlanSegments = 10ull * 1000ull * 1000ull;

struct TensorMeta {
  std::string name;
  uint64_t offset = 0;
  uint64_t size = 0;
  std::vector<int64_t> shape;
  std::vector<int64_t> stride;
  std::string dtype;
  size_t elem_size = 0;
};

struct SegmentCopy {
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;
  size_t bytes = 0;
};

struct TensorSlicePlan {
  std::string name;
  uint64_t dst_offset = 0;
  uint64_t bytes = 0;
  int64_t rows = 0;
  int64_t cols = 0;
  size_t elem_size = 0;
};

struct PhaseTimesSec {
  double open_meta = 0.0;
  double open_copy = 0.0;
  double get_calls_total = 0.0;
  double pack = 0.0;
  double net = 0.0;
  double commit = 0.0;
  double total_ready = 0.0;
};

struct BytesCounters {
  uint64_t disk_read_bytes = 0;
  uint64_t h2d_bytes = 0;
  uint64_t d2d_bytes = 0;
  uint64_t d2h_bytes = 0;
  uint64_t nccl_tx_bytes = 0;
  uint64_t nccl_rx_bytes = 0;
};

struct ResourceSnapshot {
  uint64_t pinned_host_bytes = 0;
  uint64_t gpu_alloc_bytes = 0;
  uint64_t vmm_reserved_bytes = 0;
  uint64_t vmm_mapped_bytes = 0;
  size_t vmm_granularity_bytes = 0;
};

class NullPositionedSink final : public PositionedSink {
 public:
  explicit NullPositionedSink(uint64_t total_size) : total_size_(total_size) {}

  absl::Status write_at(uint64_t offset, const void* /*src*/, size_t bytes) override {
    if (bytes == 0) {
      return absl::OkStatus();
    }
    if (total_size_ > 0 && offset + bytes > total_size_) {
      return absl::OutOfRangeError("NullPositionedSink: write exceeds total_size");
    }
    bytes_written_ += static_cast<uint64_t>(bytes);
    return absl::OkStatus();
  }

  [[nodiscard]] uint64_t bytes_written() const {
    return bytes_written_;
  }

 private:
  uint64_t total_size_ = 0;
  uint64_t bytes_written_ = 0;
};

class HostBufferSink final : public PositionedSink {
 public:
  HostBufferSink(gsl::not_null<uint8_t*> base, uint64_t total_size) : base_(base), total_size_(total_size) {}

  absl::Status write_at(uint64_t offset, const void* src, size_t bytes) override {
    if (bytes == 0) {
      return absl::OkStatus();
    }
    if (total_size_ > 0 && offset + static_cast<uint64_t>(bytes) > total_size_) {
      return absl::OutOfRangeError("HostBufferSink: write exceeds total_size");
    }
    std::memcpy(base_.get() + offset, src, bytes);
    bytes_written_ += static_cast<uint64_t>(bytes);
    return absl::OkStatus();
  }

  [[nodiscard]] uint64_t bytes_written() const {
    return bytes_written_;
  }

 private:
  gsl::not_null<uint8_t*> base_;
  uint64_t total_size_ = 0;
  uint64_t bytes_written_ = 0;
};

struct RunResult {
  StrategyKind strategy{};
  PhaseTimesSec t{};
  BytesCounters bytes{};
  ResourceSnapshot res{};
  uint64_t selected_tensors = 0;
  uint64_t selected_copies = 0;
  uint64_t planned_segments = 0;
  uint64_t planner_segments_pre_merge = 0;
  uint64_t planner_segments_merged = 0;
  uint64_t planner_src_runs = 0;
  double planner_src_run_avg_bytes = 0.0;
  uint64_t planner_src_run_max_bytes = 0;
  uint64_t planned_ranges = 0;
  uint64_t output_bytes = 0;
  uint64_t n_collectives = 0;
  uint64_t gpu_sched_waits = 0;
  double gpu_sched_wait_sec = 0.0;

  // Strategy C extra stats (only meaningful when strategy==kC_BatchedOptimal).
  uint64_t c_staged_tensors = 0;
  uint64_t c_staged_reads = 0;
  uint64_t c_direct_primary_reads = 0;
  uint64_t c_direct_dedup_copies = 0;
  uint64_t c_fallback_copies = 0;
};

struct LoaderConfig {
  BenchMode mode = BenchMode::kLoader;
  StrategyKind strategy = StrategyKind::kA_Eager;
  bool run_both_strategies = false;
  int tp_world_size = 1;
  int tp_rank = 0;
  std::string tp_devices;

  int device_id = 0;
  int io_threads = 4;
  int buffer_chunks = 8;
  int64_t bbuf_size_kb = 262144; // 256 MiB total (approx)
  bool use_pinned_host_buffer = true;
  // NUMA placement for pinned host bounce buffers:
  // -1: default OS policy; -2: best-effort auto (based on CUDA device NUMA node via sysfs).
  int pinned_numa_node = -1;
  bool pinned_numa_prefault = false;
  bool gpu_sched_enabled = true;
  uint64_t gpu_sched_limit_bytes = DEFAULT_GPU_SCHED_LIMIT_BYTES;
  uint64_t gpu_sched_limit_copies = DEFAULT_GPU_SCHED_LIMIT_COPIES;
  uint64_t strategy_c_staging_bytes = 1024ull * 1024ull * 1024ull;

  bool enable_collectives = false;
  std::string master_addr = "127.0.0.1";
  int master_port = 29500;
  std::string rendezvous_id = "tensorcast_safetensors_benchmark";
  int rendezvous_timeout_sec = 300;
  int nccl_timeout_sec = 600;
  bool nccl_blocking_wait = false;
  std::string nccl_op = "broadcast";
  uint64_t nccl_min_bytes = 1024;
  uint64_t nccl_max_bytes = 1024ull * 1024ull * 1024ull;
  std::string nccl_sizes;
  int nccl_iters = kDefaultNcclIters;
  int nccl_warmup = kDefaultNcclWarmupIters;
  int nccl_overhead_iters = 10000;

  uint64_t check_correctness_samples = 0;

  std::filesystem::path safetensors_dir;
  std::filesystem::path load_plan_json_path;
  std::filesystem::path materialized_dir;
  std::filesystem::path materialized_meta_path;

  // Disk microbench params
  std::filesystem::path disk_bench_path;
  uint64_t disk_bench_bytes = 8ull * 1024ull * 1024ull * 1024ull; // 8 GiB
  DirectIoMode disk_io_mode = DirectIoMode::kAuto;
  uint64_t disk_frag_segment_bytes = 256ull * 1024ull; // 256 KiB
  uint64_t disk_frag_segments = 32768; // ~8 GiB total
  uint64_t disk_frag_stride_bytes = 4ull * 1024ull * 1024ull; // 4 MiB stride

  // GPU peer microbench params
  uint64_t gpu_peer_bytes = 1024ull * 1024ull * 1024ull; // 1 GiB

  // H2D microbench params (pinned host -> GPU)
  uint64_t h2d_bench_bytes = 8ull * 1024ull * 1024ull * 1024ull; // 8 GiB
  bool h2d_per_gpu_pinned_pool = false;

  // H2D 2D microbench params (pinned host strided source -> packed GPU dst; typical for axis=1 slices).
  // Defaults approximate Qwen2.5-32B MLP gate/up weight ([rows=5120, cols=27648], fp16; TP=4 slice width=6912).
  uint64_t h2d_2d_width_bytes = 13824; // 6912 * 2
  uint64_t h2d_2d_height = 5120;
  uint64_t h2d_2d_src_pitch_bytes = 55296; // 27648 * 2
  uint64_t h2d_2d_dst_pitch_bytes = 13824; // packed
};

// Forward declarations (used by Strategy D helpers).
absl::Status run_disk_baseline(const LoaderConfig& cfg);
absl::Status log_run_result(const LoaderConfig& cfg, const RunResult& r);

std::string join_device_ids(const std::vector<int>& device_ids) {
  std::string out;
  for (size_t i = 0; i < device_ids.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&out, ",");
    }
    absl::StrAppend(&out, device_ids[i]);
  }
  return out;
}

absl::Status nccl_result_to_status(ncclResult_t rc, std::string_view what) {
  if (rc == ncclSuccess) {
    return absl::OkStatus();
  }
  return absl::InternalError(absl::StrCat(what, ": ", ncclGetErrorString(rc)));
}

class NcclClique {
 public:
  struct Options {
    std::vector<int> device_ids; // rank -> cuda device id
    int nccl_timeout_sec = 600;
  };

  NcclClique() = default;

  ~NcclClique() {
    release();
  }

  NcclClique(const NcclClique&) = delete;
  NcclClique& operator=(const NcclClique&) = delete;

  NcclClique(NcclClique&& other) noexcept {
    *this = std::move(other);
  }

  NcclClique& operator=(NcclClique&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release();
    options_ = std::move(other.options_);
    device_ids_ = std::move(other.device_ids_);
    comms_ = std::move(other.comms_);
    streams_ = std::move(other.streams_);
    done_events_ = std::move(other.done_events_);
    other.options_ = Options{};
    other.device_ids_.clear();
    other.comms_.clear();
    other.streams_.clear();
    other.done_events_.clear();
    return *this;
  }

  static absl::StatusOr<NcclClique> create(const Options& options) {
    if (tensorcast::cuda::is_fake()) {
      return absl::FailedPreconditionError("NCCL collectives require real CUDA; FakeCuda does not support NCCL");
    }
    if (!tensorcast::cuda::is_available()) {
      return absl::FailedPreconditionError("NCCL collectives require CUDA");
    }
    if (options.device_ids.empty()) {
      return absl::InvalidArgumentError("NcclClique::create: device_ids is empty");
    }
    if (options.nccl_timeout_sec <= 0) {
      return absl::InvalidArgumentError("NcclClique::create: nccl_timeout_sec must be > 0");
    }

    int device_count = 0;
    TC_RETURN_IF_ERROR(tensorcast::cuda::get_device_count(&device_count));
    if (device_count <= 0) {
      return absl::FailedPreconditionError("CUDA reports zero devices");
    }

    absl::flat_hash_set<int> seen;
    seen.reserve(options.device_ids.size());
    for (const int device_id : options.device_ids) {
      if (device_id < 0 || device_id >= device_count) {
        return absl::InvalidArgumentError(
            absl::StrCat("NcclClique::create: invalid device id: ", device_id, " (device_count=", device_count, ")"));
      }
      if (!seen.insert(device_id).second) {
        return absl::InvalidArgumentError(absl::StrCat("NcclClique::create: duplicate device id: ", device_id));
      }
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
      TC_RETURN_IF_ERROR(tensorcast::cuda::device_synchronize());
    }

    NcclClique clique;
    clique.options_ = options;
    clique.device_ids_ = options.device_ids;

    const int world_size = static_cast<int>(options.device_ids.size());
    clique.comms_.assign(static_cast<size_t>(world_size), nullptr);
    {
      const ncclResult_t rc = ncclCommInitAll(clique.comms_.data(), world_size, options.device_ids.data());
      if (rc != ncclSuccess) {
        for (auto& comm : clique.comms_) {
          if (comm != nullptr) {
            (void)ncclCommAbort(comm);
            (void)ncclCommDestroy(comm);
            comm = nullptr;
          }
        }
        return nccl_result_to_status(rc, "ncclCommInitAll");
      }
    }

    clique.streams_.assign(static_cast<size_t>(world_size), nullptr);
    clique.done_events_.assign(static_cast<size_t>(world_size), nullptr);
    for (int rank = 0; rank < world_size; ++rank) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(options.device_ids[rank]));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&clique.streams_[rank], cudaStreamNonBlocking));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&clique.done_events_[rank], cudaEventDisableTiming));
    }

    return clique;
  }

  [[nodiscard]] int world_size() const {
    return static_cast<int>(device_ids_.size());
  }

  [[nodiscard]] const std::vector<int>& device_ids() const {
    return device_ids_;
  }

  absl::Status barrier() {
    for (int rank = 0; rank < world_size(); ++rank) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[rank]));
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(streams_[rank]));
    }
    return absl::OkStatus();
  }

  static absl::Status group_start() {
    return nccl_result_to_status(ncclGroupStart(), "ncclGroupStart");
  }

  absl::Status broadcast_u8(int rank, const void* send, void* recv, size_t bytes, int root_rank) {
    if (rank < 0 || rank >= world_size()) {
      return absl::InvalidArgumentError("NcclClique::broadcast_u8: rank out of range");
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[rank]));
    return nccl_result_to_status(
        ncclBroadcast(send, recv, bytes, ncclUint8, root_rank, comms_[rank], streams_[rank]), "ncclBroadcast");
  }

  absl::Status send_u8(int src_rank, const void* src_ptr, size_t bytes, int dst_rank) {
    if (src_rank < 0 || src_rank >= world_size() || dst_rank < 0 || dst_rank >= world_size()) {
      return absl::InvalidArgumentError("NcclClique::send_u8: rank out of range");
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[src_rank]));
    return nccl_result_to_status(
        ncclSend(src_ptr, bytes, ncclUint8, dst_rank, comms_[src_rank], streams_[src_rank]), "ncclSend");
  }

  absl::Status recv_u8(int dst_rank, void* dst_ptr, size_t bytes, int src_rank) {
    if (src_rank < 0 || src_rank >= world_size() || dst_rank < 0 || dst_rank >= world_size()) {
      return absl::InvalidArgumentError("NcclClique::recv_u8: rank out of range");
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[dst_rank]));
    return nccl_result_to_status(
        ncclRecv(dst_ptr, bytes, ncclUint8, src_rank, comms_[dst_rank], streams_[dst_rank]), "ncclRecv");
  }

  absl::Status group_end_and_wait(std::string_view what) {
    TC_RETURN_IF_ERROR(nccl_result_to_status(ncclGroupEnd(), "ncclGroupEnd"));
    for (int rank = 0; rank < world_size(); ++rank) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[rank]));
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(done_events_[rank], streams_[rank]));
    }
    return wait_all_with_timeout(absl::Seconds(options_.nccl_timeout_sec), what);
  }

 private:
  void abort_all(std::string_view reason) {
    LOG(ERROR) << absl::StrCat("Aborting NCCL communicators: ", reason);
    for (auto& comm : comms_) {
      if (comm == nullptr) {
        continue;
      }
      (void)ncclCommAbort(comm);
    }
  }

  absl::Status wait_all_with_timeout(absl::Duration timeout, std::string_view what) {
    const absl::Time start = absl::Now();
    const absl::Time deadline = start + timeout;
    std::vector<bool> ready(static_cast<size_t>(world_size()), false);

    while (true) {
      bool all_ready = true;
      for (int rank = 0; rank < world_size(); ++rank) {
        if (ready[static_cast<size_t>(rank)]) {
          continue;
        }
        TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids_[rank]));
        bool is_ready = false;
        const absl::Status st = tensorcast::cuda::event_query(done_events_[rank], &is_ready);
        if (!st.ok()) {
          abort_all(absl::StrCat("cudaEventQuery failed for rank=", rank, " what=", what, " status=", st.ToString()));
          return st;
        }
        if (is_ready) {
          ready[static_cast<size_t>(rank)] = true;
        } else {
          all_ready = false;
        }
      }
      if (all_ready) {
        return absl::OkStatus();
      }
      if (absl::Now() >= deadline) {
        abort_all(absl::StrCat("timeout what=", what));
        const absl::Duration elapsed = absl::Now() - start;
        return absl::DeadlineExceededError(
            absl::StrCat(
                "NCCL operation timed out (what=",
                what,
                " elapsed=",
                absl::FormatDuration(elapsed),
                " timeout=",
                absl::FormatDuration(timeout),
                " world_size=",
                world_size(),
                " device_ids=[",
                join_device_ids(device_ids_),
                "])"));
      }
      absl::SleepFor(absl::Milliseconds(1));
    }
  }

  void release() {
    if (device_ids_.empty()) {
      return;
    }

    for (size_t rank = 0; rank < device_ids_.size(); ++rank) {
      if (streams_.empty() || done_events_.empty()) {
        break;
      }
      if (const absl::Status st = tensorcast::cuda::set_device(device_ids_[rank]); !st.ok()) {
        LOG(WARNING) << "Failed to set CUDA device during NcclClique teardown: " << st;
        continue;
      }
      if (rank < done_events_.size() && done_events_[rank] != nullptr) {
        const absl::Status st = tensorcast::cuda::event_destroy(done_events_[rank]);
        if (!st.ok()) {
          LOG(WARNING) << "Failed to destroy CUDA event during NcclClique teardown: " << st;
        }
        done_events_[rank] = nullptr;
      }
      if (rank < streams_.size() && streams_[rank] != nullptr) {
        const absl::Status st = tensorcast::cuda::stream_destroy(streams_[rank]);
        if (!st.ok()) {
          LOG(WARNING) << "Failed to destroy CUDA stream during NcclClique teardown: " << st;
        }
        streams_[rank] = nullptr;
      }
    }

    for (auto& comm : comms_) {
      if (comm == nullptr) {
        continue;
      }
      (void)ncclCommDestroy(comm);
      comm = nullptr;
    }

    comms_.clear();
    streams_.clear();
    done_events_.clear();
    device_ids_.clear();
  }

  Options options_;
  std::vector<int> device_ids_;
  std::vector<ncclComm_t> comms_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> done_events_;
};

struct FileSegment {
  std::filesystem::path path;
  uint64_t base_offset = 0;
  uint64_t data_size = 0;
  int owner_rank = 0;
};

[[maybe_unused]] absl::StatusOr<std::vector<FileSegment>> compute_file_segments(
    const std::vector<std::filesystem::path>& shards,
    int tp_world_size) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("compute_file_segments: shards is empty");
  }
  if (tp_world_size <= 0) {
    return absl::InvalidArgumentError("compute_file_segments: tp_world_size must be > 0");
  }

  std::vector<FileSegment> out;
  out.reserve(shards.size());
  uint64_t base = 0;
  for (size_t i = 0; i < shards.size(); ++i) {
    const auto& p = shards[i];
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open ", p.string()));
    }
    auto info_or = ParseSafetensorsHeader(fd);
    ::close(fd);
    if (!info_or.ok()) {
      return info_or.status();
    }

    FileSegment seg;
    seg.path = p;
    seg.base_offset = base;
    seg.data_size = info_or->data_size;
    seg.owner_rank = static_cast<int>(i % static_cast<size_t>(tp_world_size));
    out.push_back(std::move(seg));
    base += seg.data_size;
  }
  return out;
}

[[maybe_unused]] absl::StatusOr<size_t> find_file_index_for_offset(
    const std::vector<FileSegment>& segments,
    uint64_t offset) {
  for (size_t i = 0; i < segments.size(); ++i) {
    const auto& s = segments[i];
    const uint64_t end = s.base_offset + s.data_size;
    if (offset >= s.base_offset && offset < end) {
      return i;
    }
  }
  return absl::OutOfRangeError("Offset not mapped to any safetensors file segment");
}

struct TpContiguousSlice {
  uint64_t offset_in_tensor = 0;
  uint64_t bytes = 0;
};

[[maybe_unused]] absl::StatusOr<uint64_t> compute_tp_slice_bytes(
    const TensorMeta& t,
    const LoaderConfig& cfg,
    int tp_rank) {
  if (cfg.tp_world_size <= 0 || tp_rank < 0 || tp_rank >= cfg.tp_world_size) {
    return absl::InvalidArgumentError("Invalid tp_world_size/tp_rank");
  }
  if (t.size == 0) {
    return 0ull;
  }
  if (cfg.tp_world_size == 1) {
    return t.size;
  }

  if (t.shape.size() >= 2 && t.elem_size > 0) {
    const int64_t rows = t.shape[0];
    const int64_t cols = t.shape[1];
    if (rows <= 0 || cols <= 0) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid 2D shape for tensor ", t.name));
    }
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * static_cast<uint64_t>(t.elem_size);
    const int64_t row_start = (rows * tp_rank) / cfg.tp_world_size;
    const int64_t row_end = (rows * (tp_rank + 1)) / cfg.tp_world_size;
    if (row_end <= row_start) {
      return 0ull;
    }
    return static_cast<uint64_t>(row_end - row_start) * row_bytes;
  }

  const uint64_t start = (t.size * static_cast<uint64_t>(tp_rank)) / static_cast<uint64_t>(cfg.tp_world_size);
  const uint64_t end = (t.size * static_cast<uint64_t>(tp_rank + 1)) / static_cast<uint64_t>(cfg.tp_world_size);
  if (end <= start) {
    return 0ull;
  }
  return end - start;
}

[[maybe_unused]] absl::StatusOr<TpContiguousSlice> compute_tp_contiguous_slice(
    const TensorMeta& t,
    const LoaderConfig& cfg,
    int tp_rank) {
  if (cfg.tp_world_size <= 0 || tp_rank < 0 || tp_rank >= cfg.tp_world_size) {
    return absl::InvalidArgumentError("Invalid tp_world_size/tp_rank");
  }
  if (t.size == 0) {
    return TpContiguousSlice{.offset_in_tensor = 0, .bytes = 0};
  }
  if (cfg.tp_world_size == 1) {
    return TpContiguousSlice{.offset_in_tensor = 0, .bytes = t.size};
  }

  if (t.shape.size() >= 2 && t.elem_size > 0) {
    const int64_t rows = t.shape[0];
    const int64_t cols = t.shape[1];
    if (rows <= 0 || cols <= 0) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid 2D shape for tensor ", t.name));
    }
    const uint64_t row_bytes = static_cast<uint64_t>(cols) * static_cast<uint64_t>(t.elem_size);
    const int64_t row_start = (rows * tp_rank) / cfg.tp_world_size;
    const int64_t row_end = (rows * (tp_rank + 1)) / cfg.tp_world_size;
    if (row_end <= row_start) {
      return TpContiguousSlice{.offset_in_tensor = 0, .bytes = 0};
    }
    return TpContiguousSlice{
        .offset_in_tensor = static_cast<uint64_t>(row_start) * row_bytes,
        .bytes = static_cast<uint64_t>(row_end - row_start) * row_bytes,
    };
  }

  const uint64_t start = (t.size * static_cast<uint64_t>(tp_rank)) / static_cast<uint64_t>(cfg.tp_world_size);
  const uint64_t end = (t.size * static_cast<uint64_t>(tp_rank + 1)) / static_cast<uint64_t>(cfg.tp_world_size);
  if (end <= start) {
    return TpContiguousSlice{.offset_in_tensor = 0, .bytes = 0};
  }
  return TpContiguousSlice{.offset_in_tensor = start, .bytes = end - start};
}

[[maybe_unused]] absl::StatusOr<uint64_t> copy_tp_slice_to_contiguous(
    const TensorMeta& t,
    const LoaderConfig& cfg,
    int tp_rank,
    const uint8_t* tensor_ptr,
    uint8_t* dst_ptr) {
  if (cfg.tp_world_size <= 0 || tp_rank < 0 || tp_rank >= cfg.tp_world_size) {
    return absl::InvalidArgumentError("Invalid tp_world_size/tp_rank");
  }
  if (t.size == 0) {
    return 0ull;
  }

  if (cfg.tp_world_size == 1) {
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy(dst_ptr, tensor_ptr, static_cast<size_t>(t.size), cudaMemcpyDeviceToDevice));
    return t.size;
  }

  const auto slice_or = compute_tp_contiguous_slice(t, cfg, tp_rank);
  if (!slice_or.ok()) {
    return slice_or.status();
  }
  const auto& slice = *slice_or;
  if (slice.bytes == 0) {
    return 0ull;
  }
  TC_RETURN_IF_ERROR(
      tensorcast::cuda::memcpy(
          dst_ptr, tensor_ptr + slice.offset_in_tensor, static_cast<size_t>(slice.bytes), cudaMemcpyDeviceToDevice));
  return slice.bytes;
}

absl::StatusOr<BenchMode> parse_mode(std::string_view s) {
  if (s == "loader") {
    return BenchMode::kLoader;
  }
  if (s == "disk_baseline") {
    return BenchMode::kDiskBaseline;
  }
  if (s == "disk_fragmentation") {
    return BenchMode::kDiskFragmentation;
  }
  if (s == "gpu_peer_baseline") {
    return BenchMode::kGpuPeerBaseline;
  }
  if (s == "safetensors_disk_baseline") {
    return BenchMode::kSafetensorsDiskBaseline;
  }
  if (s == "safetensors_host_baseline") {
    return BenchMode::kSafetensorsHostBaseline;
  }
  if (s == "h2d_baseline") {
    return BenchMode::kH2dBaseline;
  }
  if (s == "h2d_2d_baseline") {
    return BenchMode::kH2d2dBaseline;
  }
  if (s == "safetensors_o_direct_host_baseline") {
    return BenchMode::kSafetensorsODirectHostBaseline;
  }
  if (s == "safetensors_o_direct_disk_baseline") {
    return BenchMode::kSafetensorsODirectDiskBaseline;
  }
  if (s == "safetensors_hot_host_baseline") {
    return BenchMode::kSafetensorsHotHostBaseline;
  }
  if (s == "safetensors_hot_disk_baseline") {
    return BenchMode::kSafetensorsHotDiskBaseline;
  }
  if (s == "materialize_d") {
    return BenchMode::kMaterializeD;
  }
  if (s == "materialized_disk_baseline") {
    return BenchMode::kMaterializedDiskBaseline;
  }
  if (s == "safetensors_dram_mirror_host_baseline") {
    return BenchMode::kSafetensorsDramMirrorHostBaseline;
  }
  if (s == "nccl_baseline") {
    return BenchMode::kNcclBaseline;
  }
  if (s == "nccl_launch_tax") {
    return BenchMode::kNcclLaunchTax;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown --mode: ", s));
}

absl::StatusOr<DirectIoMode> parse_direct_io_mode(std::string_view s) {
  if (s == "auto") {
    return DirectIoMode::kAuto;
  }
  if (s == "direct") {
    return DirectIoMode::kDirect;
  }
  if (s == "buffered") {
    return DirectIoMode::kBuffered;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown --disk_io_mode: ", s, " (expected: auto|direct|buffered)"));
}

absl::StatusOr<NcclOp> parse_nccl_op(std::string_view s) {
  if (s == "broadcast") {
    return NcclOp::kBroadcast;
  }
  if (s == "sendrecv") {
    return NcclOp::kSendrecv;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown --nccl_op: ", s, " (expected: broadcast|sendrecv)"));
}

absl::StatusOr<std::vector<uint64_t>> parse_u64_csv(std::string_view csv, std::string_view flag_name) {
  std::vector<uint64_t> out;
  if (csv.empty()) {
    return out;
  }
  for (const auto& token : absl::StrSplit(csv, ',', absl::SkipEmpty())) {
    uint64_t v = 0;
    char* end = nullptr;
    errno = 0;
    const std::string s(token);
    const uint64_t parsed = std::strtoull(s.c_str(), &end, 10);
    if (end == s.c_str() || (end != nullptr && *end != '\0') || errno != 0) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid uint64 value in ", flag_name, ": ", std::string(token)));
    }
    v = static_cast<uint64_t>(parsed);
    if (v == 0) {
      continue;
    }
    out.push_back(v);
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

absl::StatusOr<std::vector<int>> parse_i32_csv(std::string_view csv, std::string_view flag_name) {
  std::vector<int> out;
  if (csv.empty()) {
    return out;
  }
  for (const auto& token : absl::StrSplit(csv, ',', absl::SkipEmpty())) {
    char* end = nullptr;
    errno = 0;
    const std::string s(token);
    const int64_t parsed = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || (end != nullptr && *end != '\0') || errno != 0) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid int value in ", flag_name, ": ", std::string(token)));
    }
    if (parsed < 0 || parsed > std::numeric_limits<int>::max()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out-of-range int value in ", flag_name, ": ", std::string(token)));
    }
    out.push_back(static_cast<int>(parsed));
  }
  return out;
}

absl::StatusOr<std::vector<int>> build_tp_device_ids(const LoaderConfig& cfg) {
  std::vector<int> device_ids;
  if (!cfg.tp_devices.empty()) {
    TC_ASSIGN_OR_RETURN(device_ids, parse_i32_csv(cfg.tp_devices, "--tp_devices"));
  } else {
    device_ids.reserve(static_cast<size_t>(cfg.tp_world_size));
    for (int i = 0; i < cfg.tp_world_size; ++i) {
      device_ids.push_back(i);
    }
  }
  if (device_ids.size() != static_cast<size_t>(cfg.tp_world_size)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "--tp_devices must have exactly tp_world_size entries (tp_world_size=",
            cfg.tp_world_size,
            " tp_devices_count=",
            device_ids.size(),
            ")"));
  }

  absl::flat_hash_set<int> seen;
  seen.reserve(device_ids.size());
  for (const int device_id : device_ids) {
    if (!seen.insert(device_id).second) {
      return absl::InvalidArgumentError(absl::StrCat("Duplicate device id in --tp_devices: ", device_id));
    }
  }

  int device_count = 0;
  TC_RETURN_IF_ERROR(tensorcast::cuda::get_device_count(&device_count));
  if (device_count <= 0) {
    return absl::FailedPreconditionError("CUDA reports zero devices");
  }
  for (const int device_id : device_ids) {
    if (device_id < 0 || device_id >= device_count) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid device id in --tp_devices: ", device_id, " (device_count=", device_count, ")"));
    }
  }
  return device_ids;
}

absl::StatusOr<int> read_sysfs_pci_numa_node_for_cuda_device(int device_id) {
  constexpr int kBusIdLen = 32;
  char bus_id[kBusIdLen];
  if (const cudaError_t rc = cudaDeviceGetPCIBusId(bus_id, kBusIdLen, device_id); rc != cudaSuccess) {
    return absl::InternalError(
        absl::StrCat("cudaDeviceGetPCIBusId failed: ", cudaGetErrorString(rc), " device_id=", device_id));
  }

  std::filesystem::path p = "/sys/bus/pci/devices";
  p /= absl::AsciiStrToLower(std::string(bus_id));
  p /= "numa_node";

  std::ifstream f(p);
  if (!f.is_open()) {
    return absl::NotFoundError(absl::StrCat("Failed to open sysfs NUMA node file: ", p.string()));
  }
  int node = -1;
  f >> node;
  if (!f.good() && !f.eof()) {
    return absl::InternalError(absl::StrCat("Failed to parse sysfs NUMA node from: ", p.string()));
  }
  return node;
}

absl::StatusOr<int> resolve_pinned_numa_node_for_device(const LoaderConfig& cfg, int device_id) {
  if (cfg.pinned_numa_node >= 0) {
    return cfg.pinned_numa_node;
  }
  if (cfg.pinned_numa_node != -2) {
    return -1;
  }
  int node = -1;
  TC_ASSIGN_OR_RETURN(node, read_sysfs_pci_numa_node_for_cuda_device(device_id));
  if (node < 0) {
    return -1;
  }
  return node;
}

absl::StatusOr<std::vector<uint64_t>> build_power2_sweep(uint64_t min_bytes, uint64_t max_bytes) {
  if (min_bytes == 0 || max_bytes == 0) {
    return absl::InvalidArgumentError("nccl_min_bytes/nccl_max_bytes must be > 0");
  }
  if (min_bytes > max_bytes) {
    return absl::InvalidArgumentError("nccl_min_bytes must be <= nccl_max_bytes");
  }
  std::vector<uint64_t> out;
  for (uint64_t cur = min_bytes; cur <= max_bytes;) {
    out.push_back(cur);
    if (cur > max_bytes / 2) {
      break;
    }
    cur *= 2;
  }
  return out;
}

absl::StatusOr<StrategyKind> parse_strategy(std::string_view s) {
  if (s == "a" || s == "A" || s == "eager") {
    return StrategyKind::kA_Eager;
  }
  if (s == "b" || s == "B" || s == "lazy_commit") {
    return StrategyKind::kB_LazyCommit;
  }
  if (s == "c" || s == "C" || s == "batched_optimal" || s == "optimal") {
    return StrategyKind::kC_BatchedOptimal;
  }
  if (s == "hp" || s == "host_pack" || s == "c_host_pack") {
    return StrategyKind::kC_HostPack;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown --strategy: ", s));
}

double seconds_since(absl::Time start) {
  return absl::ToDoubleSeconds(absl::Now() - start);
}

size_t dtype_elem_size_bytes(std::string_view torch_dtype) {
  if (torch_dtype == "torch.uint8" || torch_dtype == "torch.int8") {
    return 1;
  }
  if (torch_dtype == "torch.int16") {
    return 2;
  }
  if (torch_dtype == "torch.int32" || torch_dtype == "torch.float32") {
    return 4;
  }
  if (torch_dtype == "torch.int64" || torch_dtype == "torch.float64") {
    return 8;
  }
  if (torch_dtype == "torch.float16" || torch_dtype == "torch.bfloat16") {
    return 2;
  }
  return 0;
}

struct AxisSlice {
  int axis = 0;
  int64_t start = 0;
  int64_t size = 0;
};

struct PlannedCopyTask {
  const TensorMeta* meta = nullptr;
  std::string ckpt_name;
  std::string dst_param;
  std::vector<AxisSlice> slices;
  uint64_t src_sort_key = 0;
};

struct PlannedCopyInstance {
  const TensorMeta* meta = nullptr;
  std::string ckpt_name;
  std::string dst_param;
  std::vector<AxisSlice> slices;
  uint64_t src_sort_key = 0;
  uint64_t contiguous_src_offset = 0;
  uint64_t dst_offset = 0;
  uint64_t bytes = 0;
  bool contiguous = false;
};

struct RankLoadPlan {
  uint64_t unique_tensors = 0;
  uint64_t copies = 0;
  uint64_t output_bytes = 0;
  std::vector<SegmentCopy> segments; // src_offset/dst_offset/bytes
};

absl::StatusOr<nlohmann::json> load_json_file(const std::filesystem::path& path);
absl::StatusOr<std::vector<AxisSlice>> parse_axis_slices(const nlohmann::json& slices);
absl::Status validate_axis_slices(const TensorMeta& t, const std::vector<AxisSlice>& slices);
absl::StatusOr<std::vector<int64_t>> build_contiguous_stride_elems(const std::vector<int64_t>& shape);

absl::StatusOr<PlannedCopyInstance> analyze_copy_instance(
    const TensorMeta& t,
    std::string ckpt_name,
    std::string dst_param,
    std::vector<AxisSlice> slices) {
  PlannedCopyInstance out;
  out.meta = &t;
  out.ckpt_name = std::move(ckpt_name);
  out.dst_param = std::move(dst_param);
  out.slices = std::move(slices);

  if (t.size == 0) {
    out.bytes = 0;
    out.contiguous = true;
    out.contiguous_src_offset = t.offset;
    out.src_sort_key = t.offset;
    return out;
  }

  const int ndim = static_cast<int>(t.shape.size());
  if (ndim == 0) {
    if (!out.slices.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("Scalar tensor does not support axis slicing: ", t.name));
    }
    out.bytes = t.size;
    out.contiguous = true;
    out.contiguous_src_offset = t.offset;
    out.src_sort_key = t.offset;
    return out;
  }

  TC_RETURN_IF_ERROR(validate_axis_slices(t, out.slices));

  std::vector<int64_t> starts(t.shape.size(), 0);
  std::vector<int64_t> sizes = t.shape;
  for (const auto& sl : out.slices) {
    starts[static_cast<size_t>(sl.axis)] = sl.start;
    sizes[static_cast<size_t>(sl.axis)] = sl.size;
  }

  __int128 numel = 1;
  for (const int64_t d : sizes) {
    numel *= static_cast<__int128>(d);
  }
  if (numel <= 0) {
    out.bytes = 0;
    out.contiguous = true;
    out.contiguous_src_offset = t.offset;
    out.src_sort_key = t.offset;
    return out;
  }
  if (numel > static_cast<__int128>(std::numeric_limits<uint64_t>::max() / std::max<size_t>(1, t.elem_size))) {
    return absl::InvalidArgumentError(absl::StrCat("Slice byte size overflow for tensor ", t.name));
  }
  const uint64_t bytes_total = static_cast<uint64_t>(numel) * static_cast<uint64_t>(t.elem_size);
  out.bytes = bytes_total;

  std::vector<int64_t> stride_elems;
  if (t.stride.size() == t.shape.size()) {
    stride_elems = t.stride;
  } else {
    TC_ASSIGN_OR_RETURN(stride_elems, build_contiguous_stride_elems(t.shape));
  }

  __int128 elem_off = 0;
  for (size_t i = 0; i < t.shape.size(); ++i) {
    elem_off += static_cast<__int128>(starts[i]) * static_cast<__int128>(stride_elems[i]);
  }
  if (elem_off < 0 ||
      elem_off > static_cast<__int128>(std::numeric_limits<uint64_t>::max() / std::max<size_t>(1, t.elem_size))) {
    return absl::InvalidArgumentError(absl::StrCat("Slice offset overflow for tensor ", t.name));
  }
  const uint64_t src = t.offset + static_cast<uint64_t>(elem_off) * static_cast<uint64_t>(t.elem_size);
  out.src_sort_key = src;
  out.contiguous_src_offset = src;

  const size_t last = t.shape.size() - 1;
  size_t leading_single = 0;
  while (leading_single < t.shape.size() && sizes[leading_single] == 1) {
    ++leading_single;
  }
  const size_t pivot = (leading_single < t.shape.size()) ? leading_single : last;
  bool trailing_full = true;
  for (size_t i = pivot + 1; i < t.shape.size(); ++i) {
    if (!(starts[i] == 0 && sizes[i] == t.shape[i])) {
      trailing_full = false;
      break;
    }
  }
  out.contiguous = trailing_full;
  return out;
}

absl::StatusOr<std::vector<PlannedCopyInstance>> parse_plan_copy_instances_for_rank(
    const LoaderConfig& cfg,
    const std::vector<TensorMeta>& metas,
    uint64_t* out_unique_tensors) {
  if (out_unique_tensors != nullptr) {
    *out_unique_tensors = 0;
  }
  nlohmann::json root;
  TC_ASSIGN_OR_RETURN(root, load_json_file(cfg.load_plan_json_path));
  if (!root.is_object()) {
    return absl::InvalidArgumentError("Load plan JSON must be an object");
  }
  if (!root.contains("ranks") || !root.at("ranks").is_array()) {
    return absl::InvalidArgumentError("Load plan JSON must contain ranks[]");
  }

  absl::flat_hash_map<std::string, const TensorMeta*> by_name;
  by_name.reserve(metas.size());
  for (const auto& m : metas) {
    by_name.emplace(m.name, &m);
  }

  const nlohmann::json& ranks = root.at("ranks");
  const nlohmann::json* rank_obj = nullptr;
  for (const auto& r : ranks) {
    if (!r.is_object()) {
      return absl::InvalidArgumentError("ranks[] must be an object");
    }
    if (!r.contains("tp_rank") || !r.contains("tp_world_size")) {
      continue;
    }
    const int tp_rank = r.at("tp_rank").get<int>();
    const int tp_world_size = r.at("tp_world_size").get<int>();
    if (tp_rank == cfg.tp_rank && tp_world_size == cfg.tp_world_size) {
      rank_obj = &r;
      break;
    }
  }
  if (rank_obj == nullptr) {
    return absl::NotFoundError(
        std::format("No matching rank entry in plan for tp_rank={} tp_world_size={}", cfg.tp_rank, cfg.tp_world_size));
  }
  if (!rank_obj->contains("tensors") || !rank_obj->at("tensors").is_object()) {
    return absl::InvalidArgumentError("rank.tensors must be an object");
  }

  std::vector<PlannedCopyInstance> out;
  absl::flat_hash_set<std::string> unique;
  unique.reserve(rank_obj->at("tensors").size());

  for (auto it = rank_obj->at("tensors").begin(); it != rank_obj->at("tensors").end(); ++it) {
    const std::string ckpt_name = it.key();
    const nlohmann::json& t = it.value();
    if (!t.is_object()) {
      return absl::InvalidArgumentError(std::format("rank.tensors[{}] must be an object", ckpt_name));
    }
    const auto m_it = by_name.find(ckpt_name);
    if (m_it == by_name.end()) {
      return absl::NotFoundError(std::format("Plan references tensor not found in checkpoint: {}", ckpt_name));
    }
    const TensorMeta& meta = *m_it->second;

    if (t.contains("shape")) {
      const auto shape = t.at("shape").get<std::vector<int64_t>>();
      if (shape != meta.shape) {
        return absl::InvalidArgumentError(std::format("Shape mismatch for {} (plan vs ckpt)", ckpt_name));
      }
    }
    if (t.contains("dtype")) {
      const std::string dtype = t.at("dtype").get<std::string>();
      if (dtype != meta.dtype) {
        return absl::InvalidArgumentError(std::format("Dtype mismatch for {} (plan vs ckpt)", ckpt_name));
      }
    }
    if (!t.contains("copies") || !t.at("copies").is_array()) {
      return absl::InvalidArgumentError(std::format("rank.tensors[{}].copies must be an array", ckpt_name));
    }
    for (const auto& c : t.at("copies")) {
      if (!c.is_object()) {
        return absl::InvalidArgumentError(std::format("rank.tensors[{}].copies[] must be an object", ckpt_name));
      }
      if (!c.contains("dst_param") || !c.at("dst_param").is_string()) {
        return absl::InvalidArgumentError(
            std::format("rank.tensors[{}].copies[].dst_param must be a string", ckpt_name));
      }
      std::vector<AxisSlice> slices;
      if (c.contains("slices")) {
        TC_ASSIGN_OR_RETURN(slices, parse_axis_slices(c.at("slices")));
      }
      PlannedCopyInstance inst;
      TC_ASSIGN_OR_RETURN(
          inst,
          analyze_copy_instance(
              meta, /*ckpt_name=*/ckpt_name, /*dst_param=*/c.at("dst_param").get<std::string>(), std::move(slices)));
      out.push_back(std::move(inst));
      unique.insert(ckpt_name);
    }
  }

  std::sort(out.begin(), out.end(), [](const PlannedCopyInstance& a, const PlannedCopyInstance& b) {
    if (a.src_sort_key != b.src_sort_key) {
      return a.src_sort_key < b.src_sort_key;
    }
    if (a.ckpt_name != b.ckpt_name) {
      return a.ckpt_name < b.ckpt_name;
    }
    return a.dst_param < b.dst_param;
  });

  if (out_unique_tensors != nullptr) {
    *out_unique_tensors = unique.size();
  }
  return out;
}

absl::StatusOr<nlohmann::json> load_json_file(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open JSON file: ", path.string()));
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse JSON file ", path.string(), ": ", e.what()));
  }
  return j;
}

absl::StatusOr<std::vector<AxisSlice>> parse_axis_slices(const nlohmann::json& slices) {
  if (!slices.is_array()) {
    return absl::InvalidArgumentError("copies[].slices must be an array");
  }
  std::vector<AxisSlice> out;
  out.reserve(slices.size());
  for (const auto& s : slices) {
    if (!s.is_object()) {
      return absl::InvalidArgumentError("copies[].slices[] must be an object");
    }
    AxisSlice sl;
    if (!s.contains("axis") || !s.contains("start") || !s.contains("size")) {
      return absl::InvalidArgumentError("copies[].slices[] must contain axis/start/size");
    }
    sl.axis = s.at("axis").get<int>();
    sl.start = s.at("start").get<int64_t>();
    sl.size = s.at("size").get<int64_t>();
    out.push_back(sl);
  }
  return out;
}

absl::Status validate_axis_slices(const TensorMeta& t, const std::vector<AxisSlice>& slices) {
  const int ndim = static_cast<int>(t.shape.size());
  if (ndim == 0) {
    if (!slices.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("Scalar tensor does not support axis slicing: ", t.name));
    }
    return absl::OkStatus();
  }
  absl::flat_hash_set<int> seen;
  seen.reserve(slices.size());
  for (const auto& sl : slices) {
    if (sl.axis < 0 || sl.axis >= ndim) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid slice axis for ", t.name, ": ", sl.axis));
    }
    if (!seen.insert(sl.axis).second) {
      return absl::InvalidArgumentError(absl::StrCat("Duplicate slice axis for ", t.name, ": ", sl.axis));
    }
    if (sl.start < 0 || sl.size < 0) {
      return absl::InvalidArgumentError(absl::StrCat("Negative slice start/size for ", t.name));
    }
    const int64_t dim = t.shape[static_cast<size_t>(sl.axis)];
    if (sl.start > dim || sl.start + sl.size > dim) {
      return absl::InvalidArgumentError(
          absl::StrCat("Slice out of bounds for ", t.name, ": axis=", sl.axis, " start=", sl.start, " size=", sl.size));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<int64_t>> build_contiguous_stride_elems(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return std::vector<int64_t>{};
  }
  std::vector<int64_t> stride(shape.size(), 1);
  __int128 cur = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    if (cur > std::numeric_limits<int64_t>::max()) {
      return absl::InvalidArgumentError("Stride overflow");
    }
    stride[i] = static_cast<int64_t>(cur);
    cur *= static_cast<__int128>(shape[i]);
  }
  return stride;
}

absl::StatusOr<std::vector<SegmentCopy>> build_segments_for_slices(
    const TensorMeta& t,
    const std::vector<AxisSlice>& slices,
    uint64_t dst_base,
    uint64_t* out_copy_bytes,
    uint64_t* out_src_sort_key) {
  if (out_copy_bytes != nullptr) {
    *out_copy_bytes = 0;
  }
  if (out_src_sort_key != nullptr) {
    *out_src_sort_key = t.offset;
  }

  if (t.size == 0) {
    return std::vector<SegmentCopy>{};
  }

  const int ndim = static_cast<int>(t.shape.size());
  if (ndim == 0) {
    if (!slices.empty()) {
      return absl::InvalidArgumentError(absl::StrCat("Scalar tensor does not support axis slicing: ", t.name));
    }
    if (out_copy_bytes != nullptr) {
      *out_copy_bytes = t.size;
    }
    if (out_src_sort_key != nullptr) {
      *out_src_sort_key = t.offset;
    }
    return std::vector<SegmentCopy>{
        SegmentCopy{.src_offset = t.offset, .dst_offset = dst_base, .bytes = static_cast<size_t>(t.size)}};
  }

  TC_RETURN_IF_ERROR(validate_axis_slices(t, slices));

  std::vector<int64_t> starts = std::vector<int64_t>(t.shape.size(), 0);
  std::vector<int64_t> sizes = t.shape;
  for (const auto& sl : slices) {
    starts[static_cast<size_t>(sl.axis)] = sl.start;
    sizes[static_cast<size_t>(sl.axis)] = sl.size;
  }

  __int128 numel = 1;
  for (const int64_t d : sizes) {
    numel *= static_cast<__int128>(d);
  }
  if (numel <= 0) {
    return std::vector<SegmentCopy>{};
  }
  if (numel > static_cast<__int128>(std::numeric_limits<uint64_t>::max() / std::max<size_t>(1, t.elem_size))) {
    return absl::InvalidArgumentError(absl::StrCat("Slice byte size overflow for tensor ", t.name));
  }
  const uint64_t bytes_total = static_cast<uint64_t>(numel) * static_cast<uint64_t>(t.elem_size);
  if (out_copy_bytes != nullptr) {
    *out_copy_bytes = bytes_total;
  }

  std::vector<int64_t> stride_elems;
  if (t.stride.size() == t.shape.size()) {
    stride_elems = t.stride;
  } else {
    TC_ASSIGN_OR_RETURN(stride_elems, build_contiguous_stride_elems(t.shape));
  }

  const size_t last = t.shape.size() - 1;
  size_t leading_single = 0;
  while (leading_single < t.shape.size() && sizes[leading_single] == 1) {
    ++leading_single;
  }
  const size_t pivot = (leading_single < t.shape.size()) ? leading_single : last;
  bool trailing_full = true;
  for (size_t i = pivot + 1; i < t.shape.size(); ++i) {
    if (!(starts[i] == 0 && sizes[i] == t.shape[i])) {
      trailing_full = false;
      break;
    }
  }

  // Fast path: contiguous source block slice.
  if (trailing_full) {
    __int128 elem_off = 0;
    for (size_t i = 0; i < t.shape.size(); ++i) {
      const int64_t coord = starts[i];
      elem_off += static_cast<__int128>(coord) * static_cast<__int128>(stride_elems[i]);
    }
    if (elem_off < 0 ||
        elem_off > static_cast<__int128>(std::numeric_limits<uint64_t>::max() / std::max<size_t>(1, t.elem_size))) {
      return absl::InvalidArgumentError(absl::StrCat("Slice offset overflow for tensor ", t.name));
    }
    const uint64_t src = t.offset + static_cast<uint64_t>(elem_off) * static_cast<uint64_t>(t.elem_size);
    if (out_src_sort_key != nullptr) {
      *out_src_sort_key = src;
    }
    return std::vector<SegmentCopy>{SegmentCopy{
        .src_offset = src,
        .dst_offset = dst_base,
        .bytes = static_cast<size_t>(bytes_total),
    }};
  }

  __int128 outer = 1;
  for (size_t i = 0; i < last; ++i) {
    outer *= static_cast<__int128>(sizes[i]);
  }
  if (outer <= 0) {
    return std::vector<SegmentCopy>{};
  }
  if (outer > static_cast<__int128>(kMaxPlanSegments)) {
    const uint64_t segments = static_cast<uint64_t>(kMaxPlanSegments + 1);
    return absl::FailedPreconditionError(
        absl::StrCat("Plan slice expands into too many segments for tensor ", t.name, ": segments>", segments));
  }

  const uint64_t outer_u64 = static_cast<uint64_t>(outer);
  const uint64_t inner_elems = static_cast<uint64_t>(sizes[last]);
  const size_t inner_bytes = static_cast<size_t>(inner_elems * static_cast<uint64_t>(t.elem_size));
  std::vector<SegmentCopy> segs;
  segs.reserve(static_cast<size_t>(outer_u64));

  std::vector<int64_t> idx(last, 0);
  uint64_t dst = dst_base;
  for (uint64_t seg_i = 0; seg_i < outer_u64; ++seg_i) {
    __int128 elem_off = 0;
    for (size_t ax = 0; ax < last; ++ax) {
      const int64_t coord = starts[ax] + idx[ax];
      elem_off += static_cast<__int128>(coord) * static_cast<__int128>(stride_elems[ax]);
    }
    elem_off += static_cast<__int128>(starts[last]) * static_cast<__int128>(stride_elems[last]);
    if (elem_off < 0) {
      return absl::InvalidArgumentError(absl::StrCat("Negative slice offset for tensor ", t.name));
    }
    const uint64_t src = t.offset + static_cast<uint64_t>(elem_off) * static_cast<uint64_t>(t.elem_size);
    if (out_src_sort_key != nullptr && seg_i == 0) {
      *out_src_sort_key = src;
    }
    segs.push_back(
        SegmentCopy{
            .src_offset = src,
            .dst_offset = dst,
            .bytes = inner_bytes,
        });
    dst += inner_bytes;

    // Odometer increment.
    for (size_t ax = last; ax-- > 0;) {
      idx[ax] += 1;
      if (idx[ax] < sizes[ax]) {
        break;
      }
      idx[ax] = 0;
    }
  }
  return segs;
}

absl::StatusOr<RankLoadPlan> build_rank_load_plan(const LoaderConfig& cfg, const std::vector<TensorMeta>& metas) {
  if (cfg.load_plan_json_path.empty()) {
    return absl::InvalidArgumentError("build_rank_load_plan: load_plan_json_path is empty");
  }
  nlohmann::json root;
  TC_ASSIGN_OR_RETURN(root, load_json_file(cfg.load_plan_json_path));
  if (!root.is_object()) {
    return absl::InvalidArgumentError("Load plan JSON must be an object");
  }
  if (!root.contains("ranks") || !root.at("ranks").is_array()) {
    return absl::InvalidArgumentError("Load plan JSON must contain ranks[]");
  }

  absl::flat_hash_map<std::string, const TensorMeta*> by_name;
  by_name.reserve(metas.size());
  for (const auto& m : metas) {
    by_name.emplace(m.name, &m);
  }

  const nlohmann::json& ranks = root.at("ranks");
  const nlohmann::json* rank_obj = nullptr;
  for (const auto& r : ranks) {
    if (!r.is_object()) {
      return absl::InvalidArgumentError("ranks[] must be an object");
    }
    if (!r.contains("tp_rank") || !r.contains("tp_world_size")) {
      continue;
    }
    const int tp_rank = r.at("tp_rank").get<int>();
    const int tp_world_size = r.at("tp_world_size").get<int>();
    if (tp_rank == cfg.tp_rank && tp_world_size == cfg.tp_world_size) {
      rank_obj = &r;
      break;
    }
  }
  if (rank_obj == nullptr) {
    return absl::NotFoundError(
        std::format("No matching rank entry in plan for tp_rank={} tp_world_size={}", cfg.tp_rank, cfg.tp_world_size));
  }
  if (!rank_obj->contains("tensors") || !rank_obj->at("tensors").is_object()) {
    return absl::InvalidArgumentError("rank.tensors must be an object");
  }

  std::vector<PlannedCopyTask> tasks;
  absl::flat_hash_set<std::string> unique;
  unique.reserve(rank_obj->at("tensors").size());

  for (auto it = rank_obj->at("tensors").begin(); it != rank_obj->at("tensors").end(); ++it) {
    const std::string ckpt_name = it.key();
    const nlohmann::json& t = it.value();
    if (!t.is_object()) {
      return absl::InvalidArgumentError(std::format("rank.tensors[{}] must be an object", ckpt_name));
    }
    const auto m_it = by_name.find(ckpt_name);
    if (m_it == by_name.end()) {
      return absl::NotFoundError(std::format("Plan references tensor not found in checkpoint: {}", ckpt_name));
    }
    const TensorMeta& meta = *m_it->second;

    if (t.contains("shape")) {
      const auto shape = t.at("shape").get<std::vector<int64_t>>();
      if (shape != meta.shape) {
        return absl::InvalidArgumentError(std::format("Shape mismatch for {} (plan vs ckpt)", ckpt_name));
      }
    }
    if (t.contains("dtype")) {
      const std::string dtype = t.at("dtype").get<std::string>();
      if (dtype != meta.dtype) {
        return absl::InvalidArgumentError(std::format("Dtype mismatch for {} (plan vs ckpt)", ckpt_name));
      }
    }
    if (!t.contains("copies") || !t.at("copies").is_array()) {
      return absl::InvalidArgumentError(std::format("rank.tensors[{}].copies must be an array", ckpt_name));
    }
    for (const auto& c : t.at("copies")) {
      if (!c.is_object()) {
        return absl::InvalidArgumentError(std::format("rank.tensors[{}].copies[] must be an object", ckpt_name));
      }
      if (!c.contains("dst_param") || !c.at("dst_param").is_string()) {
        return absl::InvalidArgumentError(
            std::format("rank.tensors[{}].copies[].dst_param must be a string", ckpt_name));
      }
      PlannedCopyTask task;
      task.meta = &meta;
      task.ckpt_name = ckpt_name;
      task.dst_param = c.at("dst_param").get<std::string>();
      if (c.contains("slices")) {
        TC_ASSIGN_OR_RETURN(task.slices, parse_axis_slices(c.at("slices")));
      }
      uint64_t unused_bytes = 0;
      uint64_t src_key = meta.offset;
      auto segs_or = build_segments_for_slices(meta, task.slices, /*dst_base=*/0, &unused_bytes, &src_key);
      if (!segs_or.ok()) {
        return segs_or.status();
      }
      task.src_sort_key = src_key;
      tasks.push_back(std::move(task));
      unique.insert(ckpt_name);
    }
  }

  std::sort(tasks.begin(), tasks.end(), [](const PlannedCopyTask& a, const PlannedCopyTask& b) {
    if (a.src_sort_key != b.src_sort_key) {
      return a.src_sort_key < b.src_sort_key;
    }
    if (a.ckpt_name != b.ckpt_name) {
      return a.ckpt_name < b.ckpt_name;
    }
    return a.dst_param < b.dst_param;
  });

  RankLoadPlan plan;
  plan.unique_tensors = unique.size();
  plan.copies = tasks.size();

  uint64_t dst_cursor = 0;
  for (const auto& task : tasks) {
    uint64_t copy_bytes = 0;
    uint64_t src_key = 0;
    std::vector<SegmentCopy> segs;
    TC_ASSIGN_OR_RETURN(segs, build_segments_for_slices(*task.meta, task.slices, dst_cursor, &copy_bytes, &src_key));
    dst_cursor += copy_bytes;
    plan.output_bytes = dst_cursor;
    plan.segments.insert(plan.segments.end(), segs.begin(), segs.end());
  }
  return plan;
}

absl::StatusOr<std::vector<std::filesystem::path>> collect_safetensors_inputs(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> out;
  if (!dir.empty()) {
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
      if (ec) {
        return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to list ", dir.string()));
      }
      if (!entry.is_regular_file()) {
        continue;
      }
      const auto& p = entry.path();
      if (absl::EndsWith(p.filename().string(), ".safetensors")) {
        out.push_back(p);
      }
    }
  }
  if (out.empty()) {
    return absl::NotFoundError("No .safetensors inputs found (use --safetensors_dir).");
  }
  std::ranges::sort(out, [](const std::filesystem::path& a, const std::filesystem::path& b) {
    const std::string af = a.filename().string();
    const std::string bf = b.filename().string();
    if (af != bf) {
      return af < bf;
    }
    return a.string() < b.string();
  });
  return out;
}

absl::StatusOr<std::vector<TensorMeta>> parse_canonical_index(std::string_view canonical_index_json) {
  using nlohmann::json;
  std::vector<TensorMeta> metas;
  json idx;
  try {
    idx = json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", e.what()));
  }
  if (!idx.is_object()) {
    return absl::InvalidArgumentError("Canonical index JSON must be an object");
  }
  metas.reserve(idx.size());
  for (auto it = idx.begin(); it != idx.end(); ++it) {
    const std::string& name = it.key();
    const json& arr = it.value();
    if (!arr.is_array() || arr.size() < 6) {
      return absl::InvalidArgumentError(absl::StrCat("Invalid tensor entry for key: ", name));
    }
    TensorMeta m;
    m.name = name;
    m.offset = arr[0].get<uint64_t>();
    m.size = arr[1].get<uint64_t>();
    m.shape = arr[2].get<std::vector<int64_t>>();
    m.stride = arr[3].get<std::vector<int64_t>>();
    m.dtype = arr[4].get<std::string>();
    m.elem_size = dtype_elem_size_bytes(m.dtype);
    __int128 numel = 1;
    for (int64_t d : m.shape) {
      if (d < 0) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid negative dimension for tensor ", name));
      }
      numel *= static_cast<__int128>(d);
    }

    if (m.elem_size == 0) {
      if (numel > 0) {
        const auto denom = static_cast<uint64_t>(numel);
        if (denom != 0 && m.size % denom == 0) {
          m.elem_size = static_cast<size_t>(m.size / denom);
        }
      }
    }
    if (m.elem_size == 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Unable to infer element size for tensor ", name, " dtype=", m.dtype));
    }

    const __int128 expect_bytes128 = numel * static_cast<__int128>(m.elem_size);
    if (expect_bytes128 < 0 || expect_bytes128 > std::numeric_limits<uint64_t>::max()) {
      return absl::InvalidArgumentError(absl::StrCat("Tensor byte size overflow for tensor ", name));
    }
    const uint64_t expect_bytes = static_cast<uint64_t>(expect_bytes128);
    if (expect_bytes != m.size) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "Tensor size mismatch for ", name, ": expected=", expect_bytes, " bytes but got=", m.size, " bytes"));
    }
    metas.push_back(std::move(m));
  }
  std::sort(metas.begin(), metas.end(), [](const TensorMeta& a, const TensorMeta& b) { return a.name < b.name; });
  return metas;
}

absl::StatusOr<std::vector<const TensorMeta*>> select_tensors_all(const std::vector<TensorMeta>& metas) {
  std::vector<const TensorMeta*> selected;
  selected.reserve(metas.size());
  for (const auto& m : metas) {
    selected.push_back(&m);
  }
  return selected;
}

std::vector<std::pair<uint64_t, size_t>> split_even_ranges(uint64_t base, uint64_t bytes, int parts) {
  std::vector<std::pair<uint64_t, size_t>> out;
  if (bytes == 0 || parts <= 0) {
    return out;
  }
  out.reserve(static_cast<size_t>(parts));
  for (int i = 0; i < parts; ++i) {
    const uint64_t start = base + (bytes * static_cast<uint64_t>(i)) / static_cast<uint64_t>(parts);
    const uint64_t end = base + (bytes * static_cast<uint64_t>(i + 1)) / static_cast<uint64_t>(parts);
    if (end > start) {
      out.push_back({start, static_cast<size_t>(end - start)});
    }
  }
  return out;
}

std::vector<std::pair<uint64_t, size_t>> merge_adjacent_ranges(std::vector<std::pair<uint64_t, size_t>> ranges) {
  if (ranges.empty()) {
    return ranges;
  }
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  std::vector<std::pair<uint64_t, size_t>> merged;
  merged.reserve(ranges.size());
  uint64_t cur_off = ranges[0].first;
  uint64_t cur_end = cur_off + ranges[0].second;
  for (size_t i = 1; i < ranges.size(); ++i) {
    const auto& [off, len] = ranges[i];
    const uint64_t end = off + len;
    if (off <= cur_end) {
      cur_end = std::max(cur_end, end);
      continue;
    }
    merged.push_back({cur_off, static_cast<size_t>(cur_end - cur_off)});
    cur_off = off;
    cur_end = end;
  }
  merged.push_back({cur_off, static_cast<size_t>(cur_end - cur_off)});
  return merged;
}

struct SrcRunStats {
  uint64_t runs = 0;
  double avg_bytes = 0.0;
  uint64_t max_bytes = 0;
};

SrcRunStats compute_src_run_stats(const std::vector<SegmentCopy>& segments_sorted_by_src) {
  SrcRunStats st;
  uint64_t total_bytes = 0;
  uint64_t run_bytes = 0;
  uint64_t expected_next_src = 0;
  bool has_run = false;
  for (const auto& s : segments_sorted_by_src) {
    if (s.bytes == 0) {
      continue;
    }
    const uint64_t bytes = static_cast<uint64_t>(s.bytes);
    total_bytes += bytes;
    if (!has_run) {
      has_run = true;
      st.runs = 1;
      run_bytes = bytes;
      expected_next_src = s.src_offset + bytes;
      continue;
    }
    if (s.src_offset == expected_next_src) {
      run_bytes += bytes;
      expected_next_src += bytes;
      continue;
    }
    st.max_bytes = std::max(st.max_bytes, run_bytes);
    st.runs += 1;
    run_bytes = bytes;
    expected_next_src = s.src_offset + bytes;
  }
  if (has_run) {
    st.max_bytes = std::max(st.max_bytes, run_bytes);
  }
  if (st.runs > 0) {
    st.avg_bytes = static_cast<double>(total_bytes) / static_cast<double>(st.runs);
  }
  return st;
}

struct SegmentPlannerStats {
  uint64_t segments_pre_merge = 0;
  uint64_t segments_post_merge = 0;
  uint64_t src_runs_pre_merge = 0;
  uint64_t src_runs_post_merge = 0;
  double src_run_avg_bytes = 0.0;
  uint64_t src_run_max_bytes = 0;
};

std::pair<std::vector<SegmentCopy>, SegmentPlannerStats> merge_adjacent_segments_by_src(
    std::vector<SegmentCopy> segments) {
  SegmentPlannerStats st;
  segments.erase(
      std::remove_if(segments.begin(), segments.end(), [](const SegmentCopy& s) { return s.bytes == 0; }),
      segments.end());

  std::sort(segments.begin(), segments.end(), [](const SegmentCopy& a, const SegmentCopy& b) {
    if (a.src_offset != b.src_offset) {
      return a.src_offset < b.src_offset;
    }
    if (a.dst_offset != b.dst_offset) {
      return a.dst_offset < b.dst_offset;
    }
    return a.bytes < b.bytes;
  });

  st.segments_pre_merge = segments.size();
  const SrcRunStats pre = compute_src_run_stats(segments);
  st.src_runs_pre_merge = pre.runs;

  std::vector<SegmentCopy> merged;
  merged.reserve(segments.size());
  for (const auto& seg : segments) {
    if (merged.empty()) {
      merged.push_back(seg);
      continue;
    }
    auto& prev = merged.back();
    const uint64_t prev_src_end = prev.src_offset + static_cast<uint64_t>(prev.bytes);
    const uint64_t prev_dst_end = prev.dst_offset + static_cast<uint64_t>(prev.bytes);
    if (prev_src_end == seg.src_offset && prev_dst_end == seg.dst_offset) {
      prev.bytes += seg.bytes;
      continue;
    }
    merged.push_back(seg);
  }

  st.segments_post_merge = merged.size();
  const SrcRunStats post = compute_src_run_stats(merged);
  st.src_runs_post_merge = post.runs;
  st.src_run_avg_bytes = post.avg_bytes;
  st.src_run_max_bytes = post.max_bytes;
  return {std::move(merged), st};
}

absl::StatusOr<std::vector<SegmentCopy>> plan_tensor_segments(
    const TensorMeta& t,
    const LoaderConfig& cfg,
    uint64_t dst_base) {
  if (cfg.tp_world_size <= 0 || cfg.tp_rank < 0 || cfg.tp_rank >= cfg.tp_world_size) {
    return absl::InvalidArgumentError("Invalid --tp_world_size/--tp_rank");
  }
  if (t.size == 0) {
    return std::vector<SegmentCopy>{};
  }
  if (cfg.tp_world_size == 1) {
    return std::vector<SegmentCopy>{
        SegmentCopy{.src_offset = t.offset, .dst_offset = dst_base, .bytes = static_cast<size_t>(t.size)}};
  }
  const auto slice_or = compute_tp_contiguous_slice(t, cfg, cfg.tp_rank);
  if (!slice_or.ok()) {
    return slice_or.status();
  }
  const auto& slice = *slice_or;
  if (slice.bytes == 0) {
    return std::vector<SegmentCopy>{};
  }
  return std::vector<SegmentCopy>{SegmentCopy{
      .src_offset = t.offset + slice.offset_in_tensor,
      .dst_offset = dst_base,
      .bytes = static_cast<size_t>(slice.bytes)}};
}

class RemappedSource final : public SeekableSource {
 public:
  struct Segment {
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t end_offset;
  };

  RemappedSource(gsl::not_null<SeekableSource*> backing, std::vector<Segment> segments)
      : backing_(backing), segments_(std::move(segments)) {
    std::sort(segments_.begin(), segments_.end(), [](const Segment& a, const Segment& b) {
      return a.dst_offset < b.dst_offset;
    });
    if (!segments_.empty()) {
      total_bytes_ = segments_.back().end_offset;
    }
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    return total_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    absl::MutexLock lk(&offset_mu_);
    auto got_or = read_at(current_offset_, dst, max_bytes);
    if (got_or.ok()) {
      current_offset_ += static_cast<uint64_t>(*got_or);
    }
    return got_or;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (bytes == 0) {
      return 0;
    }
    if (offset >= total_bytes_) {
      return static_cast<size_t>(0);
    }
    const Segment* seg = find_segment_(offset);
    if (seg == nullptr) {
      return absl::OutOfRangeError("RemappedSource: offset not mapped");
    }
    const uint64_t avail = seg->end_offset - offset;
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(bytes, avail));
    const uint64_t delta = seg->src_offset - seg->dst_offset;
    return backing_->read_at(offset + delta, dst, to_read);
  }

 private:
  const Segment* find_segment_(uint64_t offset) const {
    if (segments_.empty()) {
      return nullptr;
    }
    size_t lo = 0;
    size_t hi = segments_.size();
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      const Segment& s = segments_[mid];
      if (offset < s.dst_offset) {
        hi = mid;
      } else if (offset >= s.end_offset) {
        lo = mid + 1;
      } else {
        return &s;
      }
    }
    return nullptr;
  }

  gsl::not_null<SeekableSource*> backing_;
  std::vector<Segment> segments_;
  uint64_t total_bytes_ = 0;
  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_) = 0;
};

class HeapBufferPool final : public BufferPool {
 public:
  HeapBufferPool(size_t chunk_size, int capacity) : chunk_size_(chunk_size), capacity_(capacity) {
    capacity_ = std::max(capacity_, 1);
    buffers_.reserve(static_cast<size_t>(capacity_));
    slot_state_.assign(static_cast<size_t>(capacity_), SlotState::kFree);
    for (int i = 0; i < capacity_; ++i) {
      buffers_.push_back(std::make_unique<std::byte[]>(chunk_size_));
      free_.push(i);
    }
  }

  size_t chunk_size() const override {
    return chunk_size_;
  }

  int capacity() const override {
    return capacity_;
  }

  absl::StatusOr<int> get_free_chunk() override {
    absl::MutexLock lk(&mu_);
    while (free_.empty() && !shutdown_) {
      free_cv_.Wait(&mu_);
    }
    if (shutdown_) {
      return absl::CancelledError("HeapBufferPool shutdown");
    }
    int slot = free_.front();
    free_.pop();
    slot_state_[static_cast<size_t>(slot)] = SlotState::kProducerOwned;
    return slot;
  }

  void return_chunk(int slot_id) override {
    absl::MutexLock lk(&mu_);
    if (slot_id < 0 || slot_id >= capacity_) {
      return;
    }
    slot_state_[static_cast<size_t>(slot_id)] = SlotState::kFree;
    free_.push(slot_id);
    free_cv_.Signal();
  }

  absl::Status mark_chunk_ready(int slot_id, uint64_t global_chunk_idx, size_t valid_bytes) override {
    absl::MutexLock lk(&mu_);
    if (slot_id < 0 || slot_id >= capacity_) {
      return absl::InvalidArgumentError("Invalid slot_id");
    }
    if (shutdown_) {
      return absl::CancelledError("HeapBufferPool shutdown");
    }
    slot_state_[static_cast<size_t>(slot_id)] = SlotState::kReady;
    ready_.push(
        ReadyChunk{
            .slot_id = slot_id,
            .global_chunk_id = global_chunk_idx,
            .bytes_in_chunk = valid_bytes,
            .data_ptr = buffers_[static_cast<size_t>(slot_id)].get(),
        });
    ready_cv_.Signal();
    return absl::OkStatus();
  }

  absl::StatusOr<ReadyChunk> get_ready_chunk() override {
    absl::MutexLock lk(&mu_);
    while (ready_.empty() && !shutdown_ && !production_complete_) {
      ready_cv_.Wait(&mu_);
    }
    if (shutdown_) {
      return absl::CancelledError("HeapBufferPool shutdown");
    }
    if (ready_.empty() && production_complete_) {
      return absl::OutOfRangeError("production complete");
    }
    ReadyChunk c = ready_.front();
    ready_.pop();
    slot_state_[static_cast<size_t>(c.slot_id)] = SlotState::kConsumerOwned;
    return c;
  }

  void signal_production_complete() override {
    absl::MutexLock lk(&mu_);
    production_complete_ = true;
    ready_cv_.SignalAll();
  }

  void shutdown() override {
    absl::MutexLock lk(&mu_);
    shutdown_ = true;
    free_cv_.SignalAll();
    ready_cv_.SignalAll();
  }

  void* get_chunk_data_ptr(int slot_id) override {
    if (slot_id < 0 || slot_id >= capacity_) {
      return nullptr;
    }
    return buffers_[static_cast<size_t>(slot_id)].get();
  }

 private:
  enum class SlotState : uint8_t { kFree = 0, kProducerOwned, kReady, kConsumerOwned };

  size_t chunk_size_;
  int capacity_;
  std::vector<std::unique_ptr<std::byte[]>> buffers_;

  absl::Mutex mu_;
  absl::CondVar free_cv_;
  absl::CondVar ready_cv_;
  std::queue<int> free_ ABSL_GUARDED_BY(mu_);
  std::queue<ReadyChunk> ready_ ABSL_GUARDED_BY(mu_);
  std::vector<SlotState> slot_state_ ABSL_GUARDED_BY(mu_);
  bool production_complete_ ABSL_GUARDED_BY(mu_) = false;
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
};

struct BounceBufferPlan {
  int chunks = 0;
  size_t chunk_bytes = 0;
  uint64_t requested_total_bytes = 0;
  uint64_t total_bytes = 0;
};

size_t round_up_to(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t rem = value % alignment;
  if (rem == 0) {
    return value;
  }
  const size_t add = alignment - rem;
  if (value > std::numeric_limits<size_t>::max() - add) {
    return 0;
  }
  return value + add;
}

absl::StatusOr<BounceBufferPlan> plan_bounce_buffer(const LoaderConfig& cfg) {
  const int chunks = std::max(1, cfg.buffer_chunks);
  const uint64_t requested_total_bytes = static_cast<uint64_t>(std::max<int64_t>(1, cfg.bbuf_size_kb)) * 1024ull;
  const uint64_t chunk_bytes_u64 =
      (requested_total_bytes + static_cast<uint64_t>(chunks) - 1) / static_cast<uint64_t>(chunks);
  if (chunk_bytes_u64 == 0 || chunk_bytes_u64 > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("Invalid bounce buffer chunk size");
  }
  const size_t alignment = common::memory::PinnedBufferPool::kDirectIOAlignment;
  const size_t chunk_bytes = round_up_to(static_cast<size_t>(chunk_bytes_u64), alignment);
  if (chunk_bytes == 0) {
    return absl::InvalidArgumentError("Bounce buffer chunk size overflow after alignment");
  }
  const uint64_t total_bytes = static_cast<uint64_t>(chunks) * static_cast<uint64_t>(chunk_bytes);
  if (total_bytes == 0 || total_bytes > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError("Bounce buffer total bytes overflow");
  }
  return BounceBufferPlan{
      .chunks = chunks,
      .chunk_bytes = chunk_bytes,
      .requested_total_bytes = requested_total_bytes,
      .total_bytes = total_bytes,
  };
}

absl::StatusOr<std::unique_ptr<BufferPool>> make_bounce_buffer_pool(
    const LoaderConfig& cfg,
    const BounceBufferPlan& plan,
    ResourceSnapshot* out_res) {
  if (cfg.use_pinned_host_buffer) {
    common::memory::PinnedBufferPool::Options pool_opts;
    pool_opts.name = "benchmark_bounce";
    pool_opts.prefault = cfg.pinned_numa_prefault;
    TC_ASSIGN_OR_RETURN(pool_opts.numa_node, resolve_pinned_numa_node_for_device(cfg, cfg.device_id));

    auto pinned_pool = std::make_shared<common::memory::PinnedBufferPool>(
        static_cast<size_t>(plan.total_bytes), plan.chunk_bytes, std::move(pool_opts));
    const size_t pool_chunk_size = pinned_pool->slice_bytes();
    auto spb = std::make_shared<common::memory::StreamingPinnedBuffer>(
        static_cast<size_t>(plan.chunks), pool_chunk_size, pinned_pool);
    TC_RETURN_IF_ERROR(spb->initialize());
    if (out_res != nullptr) {
      out_res->pinned_host_bytes = plan.total_bytes;
    }
    return std::make_unique<StreamingBufferAdapter>(
        gsl::not_null<std::shared_ptr<common::memory::StreamingPinnedBuffer>>{spb});
  }

  if (out_res != nullptr) {
    out_res->pinned_host_bytes = 0;
  }
  return std::make_unique<HeapBufferPool>(plan.chunk_bytes, plan.chunks);
}

class VmmAllocation final {
 public:
  VmmAllocation() = default;

  VmmAllocation(VmmAllocation&& other) noexcept {
    *this = std::move(other);
  }

  VmmAllocation& operator=(VmmAllocation&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release();
    device_id_ = other.device_id_;
    base_ = other.base_;
    size_rounded_ = other.size_rounded_;
    handle_ = other.handle_;
    mapped_ = other.mapped_;
    granularity_ = other.granularity_;

    other.device_id_ = -1;
    other.base_ = 0;
    other.size_rounded_ = 0;
    other.handle_ = 0;
    other.mapped_ = false;
    other.granularity_ = 0;
    return *this;
  }

  ~VmmAllocation() {
    release();
  }

  VmmAllocation(const VmmAllocation&) = delete;
  VmmAllocation& operator=(const VmmAllocation&) = delete;

  absl::Status reserve_and_map(uint64_t requested_bytes, int device_id) {
    release();
    device_id_ = device_id;
    if (requested_bytes == 0) {
      return absl::InvalidArgumentError("VmmAllocation: requested_bytes is 0");
    }
    TC_RETURN_IF_ERROR(tensorcast::cuda::cu_init(0));
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

    size_t granularity = 0;
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::cu_mem_get_allocation_granularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if (granularity == 0) {
      return absl::InternalError("VmmAllocation: granularity is 0");
    }
    granularity_ = granularity;
    const uint64_t rounded = ((requested_bytes + granularity - 1) / granularity) * granularity;
    size_rounded_ = rounded;

    TC_RETURN_IF_ERROR(tensorcast::cuda::cu_mem_address_reserve(&base_, rounded, granularity, 0, 0));
    TC_RETURN_IF_ERROR(tensorcast::cuda::cu_mem_create(&handle_, rounded, &prop, 0));
    TC_RETURN_IF_ERROR(tensorcast::cuda::cu_mem_map(base_, rounded, 0, handle_, 0));
    CUmemAccessDesc access{};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    TC_RETURN_IF_ERROR(tensorcast::cuda::cu_mem_set_access(base_, rounded, &access, 1));
    mapped_ = true;
    return absl::OkStatus();
  }

  void* base_ptr() const {
    return reinterpret_cast<void*>(base_);
  }

  uint64_t reserved_bytes() const {
    return size_rounded_;
  }

  uint64_t mapped_bytes() const {
    return mapped_ ? size_rounded_ : 0;
  }

  size_t granularity_bytes() const {
    return granularity_;
  }

 private:
  void release() {
    if (base_ == 0 || size_rounded_ == 0) {
      base_ = 0;
      size_rounded_ = 0;
      handle_ = 0;
      mapped_ = false;
      granularity_ = 0;
      device_id_ = -1;
      return;
    }
    if (mapped_) {
      (void)tensorcast::cuda::cu_mem_unmap(base_, size_rounded_);
      mapped_ = false;
    }
    if (handle_ != 0) {
      (void)tensorcast::cuda::cu_mem_release(handle_);
      handle_ = 0;
    }
    (void)tensorcast::cuda::cu_mem_address_free(base_, size_rounded_);
    base_ = 0;
    size_rounded_ = 0;
    granularity_ = 0;
    device_id_ = -1;
  }

  int device_id_ = -1;
  CUdeviceptr base_ = 0;
  uint64_t size_rounded_ = 0;
  CUmemGenericAllocationHandle handle_ = 0;
  bool mapped_ = false;
  size_t granularity_ = 0;
};

absl::StatusOr<std::pair<std::vector<TensorMeta>, uint64_t>> load_metas_from_safetensors(
    const std::vector<std::filesystem::path>& shards,
    double* out_open_meta_sec) {
  const absl::Time t0 = absl::Now();
  auto idx_or = loader::build_from_safetensors(shards, std::nullopt);
  if (!idx_or.ok()) {
    return idx_or.status();
  }
  auto metas_or = parse_canonical_index(idx_or->canonical_index_json);
  if (!metas_or.ok()) {
    return metas_or.status();
  }
  if (out_open_meta_sec != nullptr) {
    *out_open_meta_sec = seconds_since(t0);
  }
  return std::make_pair(std::move(metas_or).value(), idx_or->total_size_bytes);
}

absl::StatusOr<std::vector<std::pair<uint64_t, size_t>>> build_pump_ranges_for_copy(
    const std::vector<SegmentCopy>& segments,
    int io_threads,
    uint64_t* out_total_bytes,
    uint64_t* out_ranges_count) {
  std::vector<std::pair<uint64_t, size_t>> dst_ranges;
  dst_ranges.reserve(segments.size());
  *out_total_bytes = 0;
  for (const auto& s : segments) {
    if (s.bytes == 0) {
      continue;
    }
    dst_ranges.push_back({s.dst_offset, s.bytes});
    *out_total_bytes += s.bytes;
  }
  // Merge adjacent destination ranges to reduce dispatch overhead.
  dst_ranges = merge_adjacent_ranges(std::move(dst_ranges));
  // Ensure we have enough ranges to feed producer threads for the common case
  // where the workload is a single large contiguous range.
  if (dst_ranges.size() == 1 && io_threads > 1) {
    const auto [off, len] = dst_ranges[0];
    dst_ranges = split_even_ranges(off, len, io_threads);
  }
  *out_ranges_count = dst_ranges.size();
  return dst_ranges;
}

absl::StatusOr<RunResult> run_strategy_a_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  RunResult r;
  r.strategy = StrategyKind::kA_Eager;
  const absl::Time t_total = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);

  // Meta
  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  const auto& metas = meta_or->first;
  const uint64_t total_payload_bytes = meta_or->second;
  r.t.open_meta = open_meta;

  std::vector<SegmentCopy> planned_copy_segments;
  uint64_t planned_output_bytes = 0;
  if (!cfg.load_plan_json_path.empty()) {
    const absl::Time t_get = absl::Now();
    RankLoadPlan plan;
    TC_ASSIGN_OR_RETURN(plan, build_rank_load_plan(cfg, metas));
    planned_copy_segments = std::move(plan.segments);
    planned_output_bytes = plan.output_bytes;
    r.selected_tensors = plan.unique_tensors;
    r.selected_copies = plan.copies;
    r.output_bytes = planned_output_bytes;
    r.t.get_calls_total = seconds_since(t_get);
    r.planned_segments = planned_copy_segments.size();
  } else {
    auto selected_or = select_tensors_all(metas);
    if (!selected_or.ok()) {
      return selected_or.status();
    }
    const auto& selected = *selected_or;
    r.selected_tensors = selected.size();
    r.selected_copies = selected.size();
    const absl::Time t_get = absl::Now();
    uint64_t planned_bytes = 0;
    uint64_t planned_segments = 0;
    uint64_t dst_cursor = 0;
    for (const TensorMeta* tm : selected) {
      auto segs_or = plan_tensor_segments(*tm, cfg, dst_cursor);
      if (!segs_or.ok()) {
        return segs_or.status();
      }
      for (const auto& seg : *segs_or) {
        planned_bytes += seg.bytes;
        dst_cursor += seg.bytes;
        ++planned_segments;
      }
    }
    r.t.get_calls_total = seconds_since(t_get);
    r.planned_segments = planned_segments;
    r.output_bytes = planned_bytes;
  }

  // Copy all payload bytes to GPU as a "file buffer" baseline.
  const absl::Time t_copy = absl::Now();
  MultiSafetensorsSource src(shards);

  common::memory::GpuDeviceMemory allocation;
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  TC_RETURN_IF_ERROR(allocation.allocate(static_cast<size_t>(total_payload_bytes), cfg.device_id));

  loader::GpuMemorySink::Options sink_opts{
      .gpu_base_ptr = gsl::not_null<void*>{allocation.get()},
      .total_size = total_payload_bytes,
      .chunk_size = 128 * 1024 * 1024,
      .device_id = cfg.device_id,
      .allocation = nullptr,
      .gpu_sched_enabled = cfg.gpu_sched_enabled,
      .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
      .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
  };
  loader::GpuMemorySink sink(sink_opts);

  const int io_threads = std::max(1, cfg.io_threads);
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));

  auto ranges = split_even_ranges(/*base=*/0, total_payload_bytes, io_threads);
  r.planned_ranges = ranges.size();
  r.bytes.disk_read_bytes = total_payload_bytes;
  r.bytes.h2d_bytes = total_payload_bytes;
  r.res.gpu_alloc_bytes = total_payload_bytes;

  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  r.t.open_copy = seconds_since(t_copy);
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  if (sched_after.waits >= sched_before.waits) {
    r.gpu_sched_waits = sched_after.waits - sched_before.waits;
  }
  if (sched_after.wait_sec >= sched_before.wait_sec) {
    r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
  }

  if (!planned_copy_segments.empty()) {
    common::memory::GpuDeviceMemory output;
    const absl::Status out_st = output.allocate(static_cast<size_t>(planned_output_bytes), cfg.device_id);
    if (!out_st.ok()) {
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      (void)tensorcast::cuda::get_memory_info(&free_bytes, &total_bytes, cfg.device_id);
      LOG(INFO) << std::format(
          "strategy_a: skipping optional GPU pack (output allocation failed) output_bytes={} free_bytes={} total_bytes={} status={}",
          static_cast<uint64_t>(planned_output_bytes),
          static_cast<uint64_t>(free_bytes),
          static_cast<uint64_t>(total_bytes),
          out_st.ToString());
    } else {
      r.res.gpu_alloc_bytes = total_payload_bytes + planned_output_bytes;
      r.bytes.d2d_bytes = planned_output_bytes;

      cudaStream_t pack_stream = nullptr;
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));
      const absl::Time t_pack = absl::Now();
      const auto* src_base = static_cast<const uint8_t*>(allocation.get());
      auto* dst_base = static_cast<uint8_t*>(output.get());
      // Optimize D2D packing for strided slices (e.g., axis=1 column slices expanded into per-row segments):
      // coalesce a run of equal-width segments into a single cudaMemcpy2DAsync.
      const size_t n = planned_copy_segments.size();
      for (size_t i = 0; i < n;) {
        const SegmentCopy& s0 = planned_copy_segments[i];
        if (s0.bytes == 0) {
          ++i;
          continue;
        }

        if (i + 1 < n) {
          const SegmentCopy& s1 = planned_copy_segments[i + 1];
          if (s1.bytes == s0.bytes && s1.src_offset > s0.src_offset && s1.dst_offset > s0.dst_offset) {
            const uint64_t src_pitch_u64 = s1.src_offset - s0.src_offset;
            const uint64_t dst_pitch_u64 = s1.dst_offset - s0.dst_offset;
            const uint64_t width_u64 = static_cast<uint64_t>(s0.bytes);

            if (src_pitch_u64 >= width_u64 && dst_pitch_u64 >= width_u64 && src_pitch_u64 > width_u64) {
              size_t height = 2;
              size_t j = i + 1;
              while (j + 1 < n) {
                const SegmentCopy& cur = planned_copy_segments[j];
                const SegmentCopy& nxt = planned_copy_segments[j + 1];
                if (cur.bytes != s0.bytes || nxt.bytes != s0.bytes) {
                  break;
                }
                if (nxt.src_offset <= cur.src_offset || nxt.dst_offset <= cur.dst_offset) {
                  break;
                }
                if ((nxt.src_offset - cur.src_offset) != src_pitch_u64 ||
                    (nxt.dst_offset - cur.dst_offset) != dst_pitch_u64) {
                  break;
                }
                ++height;
                ++j;
              }

              const uint64_t src_end = s0.src_offset + (static_cast<uint64_t>(height - 1) * src_pitch_u64) + width_u64;
              const uint64_t dst_end = s0.dst_offset + (static_cast<uint64_t>(height - 1) * dst_pitch_u64) + width_u64;
              if (src_end <= total_payload_bytes && dst_end <= planned_output_bytes) {
                SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
                    dst_base + s0.dst_offset,
                    static_cast<size_t>(dst_pitch_u64),
                    src_base + s0.src_offset,
                    static_cast<size_t>(src_pitch_u64),
                    static_cast<size_t>(width_u64),
                    height,
                    cudaMemcpyDeviceToDevice,
                    pack_stream));
                i += height;
                continue;
              }
            }
          }
        }

        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                dst_base + s0.dst_offset, src_base + s0.src_offset, s0.bytes, cudaMemcpyDeviceToDevice, pack_stream));
        ++i;
      }
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
      r.t.pack = seconds_since(t_pack);
      (void)tensorcast::cuda::stream_destroy(pack_stream);
    }
  }

  r.t.total_ready = seconds_since(t_total);
  return r;
}

struct StrategyAState {
  std::vector<TensorMeta> metas;
  std::vector<const TensorMeta*> selected;
  uint64_t total_payload_bytes = 0;
  std::unique_ptr<common::memory::GpuDeviceMemory> gpu_payload;
  std::unique_ptr<common::memory::GpuDeviceMemory> output;
  uint64_t output_bytes = 0;
  std::vector<TensorSlicePlan> tensor_plans;
};

absl::StatusOr<std::pair<RunResult, StrategyAState>> run_strategy_a_baseline_with_state(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  StrategyAState s;
  RunResult r;
  r.strategy = StrategyKind::kA_Eager;
  const absl::Time t_total = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);

  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  s.metas = std::move(meta_or->first);
  s.total_payload_bytes = meta_or->second;
  r.t.open_meta = open_meta;

  auto selected_or = select_tensors_all(s.metas);
  if (!selected_or.ok()) {
    return selected_or.status();
  }
  s.selected = *selected_or;
  r.selected_tensors = s.selected.size();

  const absl::Time t_copy = absl::Now();
  MultiSafetensorsSource src(shards);

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  s.gpu_payload = std::make_unique<common::memory::GpuDeviceMemory>();
  TC_RETURN_IF_ERROR(s.gpu_payload->allocate(static_cast<size_t>(s.total_payload_bytes), cfg.device_id));

  loader::GpuMemorySink::Options sink_opts{
      .gpu_base_ptr = gsl::not_null<void*>{s.gpu_payload->get()},
      .total_size = s.total_payload_bytes,
      .chunk_size = 128 * 1024 * 1024,
      .device_id = cfg.device_id,
      .allocation = nullptr,
      .gpu_sched_enabled = cfg.gpu_sched_enabled,
      .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
      .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
  };
  loader::GpuMemorySink sink(sink_opts);

  const int io_threads = std::max(1, cfg.io_threads);
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));

  auto ranges = split_even_ranges(/*base=*/0, s.total_payload_bytes, io_threads);
  r.planned_ranges = ranges.size();
  r.bytes.disk_read_bytes = s.total_payload_bytes;
  r.bytes.h2d_bytes = s.total_payload_bytes;
  r.res.gpu_alloc_bytes = s.total_payload_bytes;

  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  r.t.open_copy = seconds_since(t_copy);
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  if (sched_after.waits >= sched_before.waits) {
    r.gpu_sched_waits = sched_after.waits - sched_before.waits;
  }
  if (sched_after.wait_sec >= sched_before.wait_sec) {
    r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
  }

  const absl::Time t_get = absl::Now();
  uint64_t dst_cursor = 0;
  uint64_t planned_segments = 0;
  for (const TensorMeta* tm : s.selected) {
    auto segs_or = plan_tensor_segments(*tm, cfg, dst_cursor);
    if (!segs_or.ok()) {
      return segs_or.status();
    }
    for (const auto& seg : *segs_or) {
      dst_cursor += seg.bytes;
      ++planned_segments;
    }
  }
  r.t.get_calls_total = seconds_since(t_get);
  r.planned_segments = planned_segments;

  r.t.total_ready = seconds_since(t_total);
  return std::make_pair(r, std::move(s));
}

struct MultiRankStrategyACollectivesResult {
  std::vector<int> device_ids;
  std::vector<RunResult> results; // per-rank
};

absl::StatusOr<MultiRankStrategyACollectivesResult> run_strategy_a_collectives_single_process(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (cfg.tp_world_size <= 1) {
    return absl::InvalidArgumentError("run_strategy_a_collectives_single_process: tp_world_size must be > 1");
  }
  if (!cfg.enable_collectives) {
    return absl::InvalidArgumentError("run_strategy_a_collectives_single_process: enable_collectives must be true");
  }
  if (cfg.nccl_blocking_wait) {
    // NOLINTNEXTLINE
    (void)::setenv("NCCL_BLOCKING_WAIT", "1", /*overwrite=*/1);
  }

  MultiRankStrategyACollectivesResult out;
  TC_ASSIGN_OR_RETURN(out.device_ids, build_tp_device_ids(cfg));

  NcclClique clique;
  TC_ASSIGN_OR_RETURN(
      clique,
      NcclClique::create(
          NcclClique::Options{
              .device_ids = out.device_ids,
              .nccl_timeout_sec = cfg.nccl_timeout_sec,
          }));

  const int world_size = clique.world_size();
  out.results.assign(static_cast<size_t>(world_size), RunResult{});
  for (int rank = 0; rank < world_size; ++rank) {
    out.results[static_cast<size_t>(rank)].strategy = StrategyKind::kA_Eager;
  }

  const absl::Time t_total = absl::Now();

  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  std::vector<TensorMeta> metas = std::move(meta_or->first);

  auto selected_or = select_tensors_all(metas);
  if (!selected_or.ok()) {
    return selected_or.status();
  }
  const std::vector<const TensorMeta*>& selected = *selected_or;

  std::vector<FileSegment> segments;
  TC_ASSIGN_OR_RETURN(segments, compute_file_segments(shards, world_size));

  struct RankCtx {
    std::unique_ptr<common::memory::GpuDeviceMemory> owned_payload;
    std::unique_ptr<common::memory::GpuDeviceMemory> output;
    uint64_t owned_total_bytes = 0;
    uint64_t output_bytes = 0;
    std::vector<uint64_t> owned_file_base;
    std::vector<bool> owned_file;
    std::vector<TensorSlicePlan> tensor_plans;
  };

  std::vector<RankCtx> ranks(static_cast<size_t>(world_size));
  for (int rank = 0; rank < world_size; ++rank) {
    RankCtx& ctx = ranks[static_cast<size_t>(rank)];
    ctx.owned_file_base.assign(segments.size(), 0);
    ctx.owned_file.assign(segments.size(), false);
    for (size_t file_idx = 0; file_idx < segments.size(); ++file_idx) {
      if (segments[file_idx].owner_rank != rank) {
        continue;
      }
      ctx.owned_file[file_idx] = true;
      ctx.owned_file_base[file_idx] = ctx.owned_total_bytes;
      ctx.owned_total_bytes += segments[file_idx].data_size;
    }
  }

  for (int rank = 0; rank < world_size; ++rank) {
    RunResult& r = out.results[static_cast<size_t>(rank)];
    r.t.open_meta = open_meta;
    r.selected_tensors = selected.size();

    const int device_id = out.device_ids[rank];
    const auto sched_before = loader::get_gpu_scheduler_stats(device_id);

    const absl::Time t_copy = absl::Now();
    RankCtx& ctx = ranks[static_cast<size_t>(rank)];
    if (ctx.owned_total_bytes > 0) {
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
      ctx.owned_payload = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(ctx.owned_payload->allocate(static_cast<size_t>(ctx.owned_total_bytes), device_id));

      MultiSafetensorsSource backing_src(shards);
      std::vector<RemappedSource::Segment> remap;
      remap.reserve(segments.size());
      uint64_t dst_cursor = 0;
      for (size_t file_idx = 0; file_idx < segments.size(); ++file_idx) {
        if (!ctx.owned_file[file_idx]) {
          continue;
        }
        remap.push_back(
            RemappedSource::Segment{
                .dst_offset = dst_cursor,
                .src_offset = segments[file_idx].base_offset,
                .end_offset = dst_cursor + segments[file_idx].data_size,
            });
        dst_cursor += segments[file_idx].data_size;
      }
      RemappedSource src(gsl::not_null<SeekableSource*>{&backing_src}, std::move(remap));

      loader::GpuMemorySink sink(
          loader::GpuMemorySink::Options{
              .gpu_base_ptr = gsl::not_null<void*>{ctx.owned_payload->get()},
              .total_size = ctx.owned_total_bytes,
              .chunk_size = 128 * 1024 * 1024,
              .device_id = device_id,
              .allocation = nullptr,
              .gpu_sched_enabled = cfg.gpu_sched_enabled,
              .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
              .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
          });

      const int io_threads = std::max(1, cfg.io_threads);
      BounceBufferPlan bb;
      TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
      std::unique_ptr<BufferPool> pool_ptr;
      TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));

      auto ranges_out = split_even_ranges(/*base=*/0, ctx.owned_total_bytes, io_threads);
      r.planned_ranges = ranges_out.size();
      r.bytes.disk_read_bytes = ctx.owned_total_bytes;
      r.bytes.h2d_bytes = ctx.owned_total_bytes;

      TC_RETURN_IF_ERROR(
          loader::pump_ranges(
              src, sink, *pool_ptr, ranges_out, io_threads, pump_benchmark_runtime().blocking_executor()));
      TC_RETURN_IF_ERROR(sink.close());
    } else {
      r.planned_ranges = 0;
      r.bytes.disk_read_bytes = 0;
      r.bytes.h2d_bytes = 0;
      r.res.pinned_host_bytes = 0;
    }
    r.t.open_copy = seconds_since(t_copy);
    const auto sched_after = loader::get_gpu_scheduler_stats(device_id);
    if (sched_after.waits >= sched_before.waits) {
      r.gpu_sched_waits = sched_after.waits - sched_before.waits;
    }
    if (sched_after.wait_sec >= sched_before.wait_sec) {
      r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
    }
  }

  for (int rank = 0; rank < world_size; ++rank) {
    RunResult& r = out.results[static_cast<size_t>(rank)];
    RankCtx& ctx = ranks[static_cast<size_t>(rank)];
    LoaderConfig rank_cfg = cfg;
    rank_cfg.tp_rank = rank;

    const absl::Time t_get = absl::Now();
    uint64_t dst_cursor = 0;
    uint64_t planned_segments = 0;
    ctx.tensor_plans.clear();
    ctx.tensor_plans.reserve(selected.size());
    for (const TensorMeta* tm : selected) {
      auto segs_or = plan_tensor_segments(*tm, rank_cfg, dst_cursor);
      if (!segs_or.ok()) {
        return segs_or.status();
      }
      const auto& segs = *segs_or;
      uint64_t tensor_bytes = 0;
      for (const auto& seg : segs) {
        tensor_bytes += seg.bytes;
        dst_cursor += seg.bytes;
        ++planned_segments;
      }
      TensorSlicePlan tp;
      tp.name = tm->name;
      tp.dst_offset = dst_cursor - tensor_bytes;
      tp.bytes = tensor_bytes;
      if (tm->shape.size() >= 2) {
        tp.rows = tm->shape[0];
        tp.cols = tm->shape[1];
      }
      tp.elem_size = tm->elem_size;
      ctx.tensor_plans.push_back(std::move(tp));
    }
    ctx.output_bytes = dst_cursor;
    r.t.get_calls_total = seconds_since(t_get);
    r.planned_segments = planned_segments;

    if (ctx.output_bytes > 0) {
      const int device_id = out.device_ids[rank];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
      ctx.output = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(ctx.output->allocate(static_cast<size_t>(ctx.output_bytes), device_id));
    }
  }

  std::vector<const uint8_t*> owned_base(static_cast<size_t>(world_size), nullptr);
  std::vector<uint8_t*> out_base(static_cast<size_t>(world_size), nullptr);
  for (int rank = 0; rank < world_size; ++rank) {
    RankCtx& ctx = ranks[static_cast<size_t>(rank)];
    owned_base[static_cast<size_t>(rank)] =
        ctx.owned_payload ? static_cast<const uint8_t*>(ctx.owned_payload->get()) : nullptr;
    out_base[static_cast<size_t>(rank)] = ctx.output ? static_cast<uint8_t*>(ctx.output->get()) : nullptr;
  }

  std::vector<double> pack_sec(static_cast<size_t>(world_size), 0.0);
  std::vector<double> net_sec(static_cast<size_t>(world_size), 0.0);
  std::vector<uint64_t> nccl_tx_bytes(static_cast<size_t>(world_size), 0);
  std::vector<uint64_t> nccl_rx_bytes(static_cast<size_t>(world_size), 0);
  std::vector<uint64_t> n_collectives(static_cast<size_t>(world_size), 0);

  TC_RETURN_IF_ERROR(clique.barrier());

  for (size_t ti = 0; ti < selected.size(); ++ti) {
    const TensorMeta& tm = *selected[ti];

    size_t file_idx = 0;
    TC_ASSIGN_OR_RETURN(file_idx, find_file_index_for_offset(segments, tm.offset));
    const FileSegment& fs = segments[file_idx];
    if (tm.offset + tm.size > fs.base_offset + fs.data_size) {
      return absl::InvalidArgumentError(absl::StrCat("Tensor crosses file boundary: ", tm.name));
    }
    const int owner_rank = fs.owner_rank;

    RankCtx& owner_ctx = ranks[static_cast<size_t>(owner_rank)];
    if (!owner_ctx.owned_file[file_idx] || owned_base[static_cast<size_t>(owner_rank)] == nullptr) {
      return absl::InternalError(absl::StrCat("Owner rank missing file payload for ", fs.path.string()));
    }
    const uint64_t tensor_file_off = tm.offset - fs.base_offset;
    const uint8_t* tensor_ptr =
        owned_base[static_cast<size_t>(owner_rank)] + owner_ctx.owned_file_base[file_idx] + tensor_file_off;

    const absl::Time t_a4 = absl::Now();
    double pack_inc = 0.0;

    const TensorSlicePlan& owner_plan = owner_ctx.tensor_plans[ti];
    if (owner_plan.bytes > 0) {
      uint8_t* dst = out_base[static_cast<size_t>(owner_rank)];
      if (dst == nullptr) {
        return absl::InternalError("Output buffer not allocated for owner rank");
      }
      const int device_id = out.device_ids[owner_rank];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_id));
      const absl::Time t_pack = absl::Now();
      auto copied_or = copy_tp_slice_to_contiguous(tm, cfg, owner_rank, tensor_ptr, dst + owner_plan.dst_offset);
      if (!copied_or.ok()) {
        return copied_or.status();
      }
      const double inc = seconds_since(t_pack);
      pack_sec[static_cast<size_t>(owner_rank)] += inc;
      pack_inc += inc;
    }

    struct SendItem {
      int peer = -1;
      uint64_t dst_offset = 0;
      TpContiguousSlice slice;
    };

    std::vector<SendItem> send_items;
    send_items.reserve(static_cast<size_t>(world_size));
    for (int peer = 0; peer < world_size; ++peer) {
      if (peer == owner_rank) {
        continue;
      }
      const TensorSlicePlan& peer_plan = ranks[static_cast<size_t>(peer)].tensor_plans[ti];
      if (peer_plan.bytes == 0) {
        continue;
      }
      TpContiguousSlice peer_slice;
      TC_ASSIGN_OR_RETURN(peer_slice, compute_tp_contiguous_slice(tm, cfg, peer));
      if (peer_slice.bytes != peer_plan.bytes) {
        return absl::InternalError(absl::StrCat("Slice byte mismatch for tensor=", tm.name, " peer=", peer));
      }
      send_items.push_back(
          SendItem{
              .peer = peer,
              .dst_offset = peer_plan.dst_offset,
              .slice = peer_slice,
          });
    }

    if (!send_items.empty()) {
      TC_RETURN_IF_ERROR(clique.group_start());
      for (const auto& item : send_items) {
        uint8_t* dst = out_base[static_cast<size_t>(item.peer)];
        if (dst == nullptr) {
          return absl::InternalError(absl::StrCat("Output buffer not allocated for peer rank=", item.peer));
        }
        const uint8_t* src = tensor_ptr + item.slice.offset_in_tensor;
        TC_RETURN_IF_ERROR(clique.send_u8(owner_rank, src, static_cast<size_t>(item.slice.bytes), item.peer));
        TC_RETURN_IF_ERROR(
            clique.recv_u8(item.peer, dst + item.dst_offset, static_cast<size_t>(item.slice.bytes), owner_rank));
        ++n_collectives[static_cast<size_t>(owner_rank)];
        nccl_tx_bytes[static_cast<size_t>(owner_rank)] += item.slice.bytes;
        ++n_collectives[static_cast<size_t>(item.peer)];
        nccl_rx_bytes[static_cast<size_t>(item.peer)] += item.slice.bytes;
      }
      TC_RETURN_IF_ERROR(
          clique.group_end_and_wait(absl::StrCat("strategy_a:a4 tensor=", tm.name, " owner=", owner_rank)));

      const double elapsed = seconds_since(t_a4);
      net_sec[static_cast<size_t>(owner_rank)] += std::max(0.0, elapsed - pack_inc);
      for (const auto& item : send_items) {
        net_sec[static_cast<size_t>(item.peer)] += elapsed;
      }
    }
  }

  const double total_ready_sec = seconds_since(t_total);
  for (int rank = 0; rank < world_size; ++rank) {
    RunResult& r = out.results[static_cast<size_t>(rank)];
    RankCtx& ctx = ranks[static_cast<size_t>(rank)];
    r.t.pack = pack_sec[static_cast<size_t>(rank)];
    r.t.net = net_sec[static_cast<size_t>(rank)];
    r.bytes.nccl_tx_bytes = nccl_tx_bytes[static_cast<size_t>(rank)];
    r.bytes.nccl_rx_bytes = nccl_rx_bytes[static_cast<size_t>(rank)];
    r.n_collectives = n_collectives[static_cast<size_t>(rank)];
    r.res.gpu_alloc_bytes = ctx.owned_total_bytes + ctx.output_bytes;
    r.t.total_ready = total_ready_sec;
  }

  return out;
}

absl::StatusOr<RunResult> run_strategy_a(const LoaderConfig& cfg, const std::vector<std::filesystem::path>& shards) {
  if (!cfg.enable_collectives || cfg.tp_world_size <= 1) {
    return run_strategy_a_baseline(cfg, shards);
  }
  return absl::FailedPreconditionError(
      "Strategy A NCCL collectives require single-process multi-GPU runner; set --mode=loader --enable_collectives=true and do not use --tp_rank");
}

absl::StatusOr<std::pair<RunResult, StrategyAState>> run_strategy_a_with_state(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (!cfg.enable_collectives || cfg.tp_world_size <= 1) {
    return run_strategy_a_baseline_with_state(cfg, shards);
  }
  return absl::FailedPreconditionError(
      "Strategy A NCCL collectives require single-process multi-GPU runner; set --mode=loader --enable_collectives=true and do not use --tp_rank");
}

struct StrategyBState {
  std::vector<TensorMeta> metas;
  std::vector<const TensorMeta*> selected;
  std::vector<TensorSlicePlan> tensor_plans;
  std::vector<SegmentCopy> segments;
  uint64_t dst_bytes = 0;
  VmmAllocation vmm;
};

struct StrategyCState {
  std::vector<TensorMeta> metas;
  std::vector<PlannedCopyInstance> copies;
  uint64_t output_bytes = 0;
  VmmAllocation vmm;
  std::unique_ptr<common::memory::GpuDeviceMemory> staging;
  uint64_t staging_bytes = 0;
};

struct D2dCopyOp {
  uint64_t src_dst_offset = 0;
  uint64_t dst_dst_offset = 0;
  size_t bytes = 0;
};

struct StrategyCStats {
  uint64_t staged_tensors = 0;
  uint64_t staged_reads = 0;
  uint64_t direct_primary_reads = 0;
  uint64_t direct_dedup_copies = 0;
  uint64_t fallback_copies = 0;
};

absl::StatusOr<bool> is_row_major_contiguous(const TensorMeta& t) {
  if (t.stride.size() != t.shape.size()) {
    return false;
  }
  std::vector<int64_t> expect;
  TC_ASSIGN_OR_RETURN(expect, build_contiguous_stride_elems(t.shape));
  return t.stride == expect;
}

struct Slice2D {
  int64_t row_start = 0;
  int64_t row_size = 0;
  int64_t col_start = 0;
  int64_t col_size = 0;
};

absl::StatusOr<Slice2D> extract_slice_2d(const TensorMeta& t, const std::vector<AxisSlice>& slices) {
  if (t.shape.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat("extract_slice_2d requires 2D tensor: ", t.name));
  }
  Slice2D s;
  s.row_start = 0;
  s.row_size = t.shape[0];
  s.col_start = 0;
  s.col_size = t.shape[1];
  for (const auto& sl : slices) {
    if (sl.axis == 0) {
      s.row_start = sl.start;
      s.row_size = sl.size;
      continue;
    }
    if (sl.axis == 1) {
      s.col_start = sl.start;
      s.col_size = sl.size;
      continue;
    }
    return absl::InvalidArgumentError(absl::StrCat("Unsupported slice axis for 2D pack: axis=", sl.axis));
  }
  return s;
}

absl::StatusOr<std::pair<RunResult, StrategyCState>> run_strategy_c_with_state(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  RunResult r;
  r.strategy = StrategyKind::kC_BatchedOptimal;
  StrategyCState s;
  StrategyCStats st;
  const absl::Time t_total = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);

  if (!cfg.use_pinned_host_buffer) {
    return absl::FailedPreconditionError("Strategy C requires --use_pinned_host_buffer=true");
  }

  // Meta
  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  s.metas = std::move(meta_or->first);
  r.t.open_meta = open_meta;

  // Plan
  const absl::Time t_get = absl::Now();
  if (!cfg.load_plan_json_path.empty()) {
    uint64_t unique_tensors = 0;
    TC_ASSIGN_OR_RETURN(s.copies, parse_plan_copy_instances_for_rank(cfg, s.metas, &unique_tensors));
    r.selected_tensors = unique_tensors;
    r.selected_copies = s.copies.size();
  } else {
    auto selected_or = select_tensors_all(s.metas);
    if (!selected_or.ok()) {
      return selected_or.status();
    }
    std::vector<const TensorMeta*> selected = *selected_or;
    std::sort(selected.begin(), selected.end(), [](const TensorMeta* a, const TensorMeta* b) {
      if (a->offset != b->offset) {
        return a->offset < b->offset;
      }
      return a->name < b->name;
    });
    s.copies.clear();
    s.copies.reserve(selected.size());
    for (const TensorMeta* tm : selected) {
      PlannedCopyInstance inst;
      inst.meta = tm;
      inst.ckpt_name = tm->name;
      inst.dst_param = tm->name;
      inst.slices = {};
      inst.bytes = tm->size;
      inst.contiguous = true;
      inst.contiguous_src_offset = tm->offset;
      inst.src_sort_key = tm->offset;
      if (cfg.tp_world_size > 1) {
        TpContiguousSlice slice;
        TC_ASSIGN_OR_RETURN(slice, compute_tp_contiguous_slice(*tm, cfg, cfg.tp_rank));
        inst.bytes = slice.bytes;
        inst.contiguous_src_offset = tm->offset + slice.offset_in_tensor;
        inst.src_sort_key = inst.contiguous_src_offset;
      }
      s.copies.push_back(std::move(inst));
    }
    r.selected_tensors = selected.size();
    r.selected_copies = selected.size();
    std::sort(s.copies.begin(), s.copies.end(), [](const PlannedCopyInstance& a, const PlannedCopyInstance& b) {
      if (a.src_sort_key != b.src_sort_key) {
        return a.src_sort_key < b.src_sort_key;
      }
      if (a.ckpt_name != b.ckpt_name) {
        return a.ckpt_name < b.ckpt_name;
      }
      return a.dst_param < b.dst_param;
    });
  }

  uint64_t dst_cursor = 0;
  for (auto& c : s.copies) {
    c.dst_offset = dst_cursor;
    dst_cursor += c.bytes;
  }
  s.output_bytes = dst_cursor;
  r.output_bytes = dst_cursor;
  r.t.get_calls_total = seconds_since(t_get);

  // Output VMM allocation.
  const absl::Time t_commit = absl::Now();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  TC_RETURN_IF_ERROR(s.vmm.reserve_and_map(s.output_bytes, cfg.device_id));
  r.res.vmm_reserved_bytes = s.vmm.reserved_bytes();
  r.res.vmm_mapped_bytes = s.vmm.mapped_bytes();
  r.res.vmm_granularity_bytes = s.vmm.granularity_bytes();

  MultiSafetensorsSource backing_src(shards);

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));
  const int io_threads = std::max(1, cfg.io_threads);

  // Group copies by checkpoint tensor to enable batched decisions.
  struct TensorGroup {
    const TensorMeta* meta = nullptr;
    std::vector<size_t> idx;
    bool staged = false;
    int64_t row_start = 0;
    int64_t row_end = 0;
    size_t row_bytes = 0;
  };

  absl::flat_hash_map<const TensorMeta*, size_t> group_index;
  group_index.reserve(s.copies.size());
  std::vector<TensorGroup> groups;
  groups.reserve(s.copies.size());
  for (size_t i = 0; i < s.copies.size(); ++i) {
    const TensorMeta* tm = s.copies[i].meta;
    auto [it, inserted] = group_index.emplace(tm, groups.size());
    if (inserted) {
      TensorGroup g;
      g.meta = tm;
      groups.push_back(std::move(g));
    }
    groups[it->second].idx.push_back(i);
  }

  // Decide staged tensors and allocate staging buffer if needed.
  bool needs_staging = false;
  for (auto& g : groups) {
    const TensorMeta& tm = *g.meta;
    bool any_non_contig = false;
    bool axes_ok = true;
    for (const size_t ci : g.idx) {
      if (!s.copies[ci].contiguous) {
        any_non_contig = true;
      }
      for (const auto& sl : s.copies[ci].slices) {
        if (sl.axis < 0 || sl.axis > 1) {
          axes_ok = false;
        }
      }
    }
    bool can_pack_2d = false;
    if (tm.shape.size() == 2 && axes_ok) {
      TC_ASSIGN_OR_RETURN(can_pack_2d, is_row_major_contiguous(tm));
    }
    g.staged = any_non_contig && can_pack_2d;
    if (g.staged) {
      needs_staging = true;
    }
  }

  if (needs_staging) {
    s.staging_bytes = std::max<uint64_t>(1, cfg.strategy_c_staging_bytes);
    s.staging = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(s.staging->allocate(static_cast<size_t>(s.staging_bytes), cfg.device_id));
    r.res.gpu_alloc_bytes += s.staging_bytes;
  }

  // Pack stream for D2D copies and 2D pack.
  cudaStream_t pack_stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));
  double commit_sec = 0.0;
  double pack_sec = 0.0;

  auto flush_direct = [&](std::vector<SegmentCopy>& pending, std::vector<D2dCopyOp>& d2d_ops) -> absl::Status {
    if (pending.empty()) {
      if (!d2d_ops.empty()) {
        return absl::InternalError("Internal error: d2d_ops not empty while pending segments empty");
      }
      return absl::OkStatus();
    }
    auto [merged, planner_stats] = merge_adjacent_segments_by_src(std::move(pending));
    pending.clear();
    pending = std::move(merged);
    uint64_t max_dst_end = 0;
    for (const auto& seg : pending) {
      const uint64_t end = seg.dst_offset + static_cast<uint64_t>(seg.bytes);
      max_dst_end = std::max(max_dst_end, end);
    }
    if (max_dst_end > s.output_bytes) {
      return absl::OutOfRangeError("Strategy C: direct segment write exceeds output buffer bounds");
    }
    r.planned_segments += planner_stats.segments_post_merge;
    r.planner_segments_pre_merge += planner_stats.segments_pre_merge;
    r.planner_segments_merged += (planner_stats.segments_pre_merge >= planner_stats.segments_post_merge)
        ? (planner_stats.segments_pre_merge - planner_stats.segments_post_merge)
        : 0;
    r.planner_src_runs += planner_stats.src_runs_post_merge;
    r.planner_src_run_max_bytes = std::max(r.planner_src_run_max_bytes, planner_stats.src_run_max_bytes);

    std::vector<RemappedSource::Segment> remap;
    remap.reserve(pending.size());
    for (const auto& seg : pending) {
      remap.push_back(
          RemappedSource::Segment{
              .dst_offset = seg.dst_offset,
              .src_offset = seg.src_offset,
              .end_offset = seg.dst_offset + seg.bytes,
          });
    }
    RemappedSource src(gsl::not_null<SeekableSource*>(&backing_src), std::move(remap));

    loader::GpuMemorySink sink(
        loader::GpuMemorySink::Options{
            .gpu_base_ptr = gsl::not_null<void*>{s.vmm.base_ptr()},
            // Strategy C performs sparse writes into the final output buffer; do
            // not enforce "must write exactly total_size bytes" on close().
            .total_size = 0,
            .chunk_size = 128 * 1024 * 1024,
            .device_id = cfg.device_id,
            .allocation = nullptr,
            .gpu_sched_enabled = cfg.gpu_sched_enabled,
            .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
            .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
        });

    uint64_t total_requested_bytes = 0;
    uint64_t planned_ranges = 0;
    auto ranges_or = build_pump_ranges_for_copy(pending, io_threads, &total_requested_bytes, &planned_ranges);
    if (!ranges_or.ok()) {
      return ranges_or.status();
    }
    const auto& ranges = *ranges_or;
    r.planned_ranges += planned_ranges;
    r.bytes.disk_read_bytes += total_requested_bytes;
    r.bytes.h2d_bytes += total_requested_bytes;

    if (auto* adapter = dynamic_cast<StreamingBufferAdapter*>(pool_ptr.get()); adapter != nullptr) {
      TC_RETURN_IF_ERROR(adapter->get_buffer()->reset_for_new_production());
    }

    const absl::Time t = absl::Now();
    TC_RETURN_IF_ERROR(
        loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
    TC_RETURN_IF_ERROR(sink.close());
    commit_sec += seconds_since(t);

    if (!d2d_ops.empty()) {
      const absl::Time t_d2d = absl::Now();
      auto* base = static_cast<uint8_t*>(s.vmm.base_ptr());
      for (const auto& op : d2d_ops) {
        if (op.bytes == 0) {
          continue;
        }
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                base + op.dst_dst_offset, base + op.src_dst_offset, op.bytes, cudaMemcpyDeviceToDevice, pack_stream));
        r.bytes.d2d_bytes += static_cast<uint64_t>(op.bytes);
      }
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
      pack_sec += seconds_since(t_d2d);
      d2d_ops.clear();
    }
    pending.clear();
    return absl::OkStatus();
  };

  std::sort(groups.begin(), groups.end(), [](const TensorGroup& a, const TensorGroup& b) {
    return a.meta->offset < b.meta->offset;
  });

  std::vector<SegmentCopy> pending_direct;
  std::vector<D2dCopyOp> pending_d2d;
  pending_direct.reserve(s.copies.size());
  pending_d2d.reserve(s.copies.size());

  // Execute in on-disk tensor order to preserve sequential access.
  for (auto& g : groups) {
    const TensorMeta& tm = *g.meta;
    if (g.staged) {
      ++st.staged_tensors;
      TC_RETURN_IF_ERROR(flush_direct(pending_direct, pending_d2d));

      // Compute union row range for staged reads.
      Slice2D union_slice;
      union_slice.row_start = std::numeric_limits<int64_t>::max();
      union_slice.row_size = 0;
      union_slice.col_start = 0;
      union_slice.col_size = tm.shape[1];
      for (const size_t ci : g.idx) {
        Slice2D sl;
        TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, s.copies[ci].slices));
        union_slice.row_start = std::min(union_slice.row_start, sl.row_start);
        union_slice.row_size = std::max(union_slice.row_size, sl.row_start + sl.row_size);
      }
      if (union_slice.row_start == std::numeric_limits<int64_t>::max()) {
        union_slice.row_start = 0;
        union_slice.row_size = 0;
      }
      const int64_t row_start = std::max<int64_t>(0, union_slice.row_start);
      const int64_t row_end = std::min<int64_t>(tm.shape[0], union_slice.row_size);
      if (row_end <= row_start) {
        continue;
      }
      const uint64_t cols = static_cast<uint64_t>(tm.shape[1]);
      const uint64_t elem = static_cast<uint64_t>(tm.elem_size);
      const uint64_t row_bytes_u64 = cols * elem;
      if (row_bytes_u64 == 0 || row_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError(absl::StrCat("Row byte size overflow for tensor ", tm.name));
      }
      const uint64_t staging_bytes = s.staging_bytes;
      uint64_t max_rows_per_chunk = (staging_bytes / row_bytes_u64);
      if (max_rows_per_chunk == 0) {
        max_rows_per_chunk = 1;
      }

      const auto* staging_base = static_cast<uint8_t*>(s.staging->get());
      auto* out_base = static_cast<uint8_t*>(s.vmm.base_ptr());
      for (int64_t row = row_start; row < row_end;) {
        const int64_t chunk_rows = std::min<int64_t>(static_cast<int64_t>(max_rows_per_chunk), row_end - row);
        const uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * row_bytes_u64;
        if (chunk_bytes > s.staging_bytes) {
          return absl::InternalError("Strategy C: computed chunk_bytes exceeds staging buffer");
        }

        std::vector<RemappedSource::Segment> remap;
        remap.push_back(
            RemappedSource::Segment{
                .dst_offset = 0,
                .src_offset = tm.offset + static_cast<uint64_t>(row) * row_bytes_u64,
                .end_offset = chunk_bytes,
            });
        RemappedSource src(gsl::not_null<SeekableSource*>(&backing_src), std::move(remap));

        loader::GpuMemorySink sink(
            loader::GpuMemorySink::Options{
                .gpu_base_ptr = gsl::not_null<void*>{static_cast<void*>(s.staging->get())},
                // We only write chunk_bytes into staging (often < staging_bytes).
                .total_size = 0,
                .chunk_size = 128 * 1024 * 1024,
                .device_id = cfg.device_id,
                .allocation = nullptr,
                .gpu_sched_enabled = cfg.gpu_sched_enabled,
                .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
                .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
            });

        const auto ranges = split_even_ranges(/*base=*/0, chunk_bytes, io_threads);
        r.planned_ranges += ranges.size();
        r.planned_segments += 1;
        r.bytes.disk_read_bytes += chunk_bytes;
        r.bytes.h2d_bytes += chunk_bytes;

        if (auto* adapter = dynamic_cast<StreamingBufferAdapter*>(pool_ptr.get()); adapter != nullptr) {
          TC_RETURN_IF_ERROR(adapter->get_buffer()->reset_for_new_production());
        }

        const absl::Time t = absl::Now();
        TC_RETURN_IF_ERROR(
            loader::pump_ranges(
                src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
        TC_RETURN_IF_ERROR(sink.close());
        commit_sec += seconds_since(t);
        ++st.staged_reads;

        // GPU pack: staged buffer -> final output layout.
        const absl::Time t_pack = absl::Now();
        for (const size_t ci : g.idx) {
          const PlannedCopyInstance& copy = s.copies[ci];
          Slice2D sl;
          TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, copy.slices));

          const int64_t copy_row_start = sl.row_start;
          const int64_t copy_row_end = sl.row_start + sl.row_size;
          const int64_t chunk_row_start = row;
          const int64_t chunk_row_end = row + chunk_rows;
          const int64_t inter_start = std::max(copy_row_start, chunk_row_start);
          const int64_t inter_end = std::min(copy_row_end, chunk_row_end);
          if (inter_end <= inter_start || sl.col_size <= 0 || sl.row_size <= 0) {
            continue;
          }

          const uint64_t inter_rows = static_cast<uint64_t>(inter_end - inter_start);
          const uint64_t col_bytes = static_cast<uint64_t>(sl.col_size) * elem;
          const uint64_t src_pitch = row_bytes_u64;
          const uint64_t dst_pitch = col_bytes;

          const uint64_t src_row_off = static_cast<uint64_t>(inter_start - chunk_row_start);
          const uint64_t dst_row_off = static_cast<uint64_t>(inter_start - copy_row_start);
          const uint64_t src_col_off = static_cast<uint64_t>(sl.col_start) * elem;

          const uint8_t* src_ptr = staging_base + src_row_off * src_pitch + src_col_off;
          uint8_t* dst_ptr = out_base + copy.dst_offset + dst_row_off * dst_pitch;

          SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
              dst_ptr,
              static_cast<size_t>(dst_pitch),
              src_ptr,
              static_cast<size_t>(src_pitch),
              static_cast<size_t>(col_bytes),
              static_cast<size_t>(inter_rows),
              cudaMemcpyDeviceToDevice,
              pack_stream));
          r.bytes.d2d_bytes += col_bytes * inter_rows;
        }
        TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
        pack_sec += seconds_since(t_pack);

        row += chunk_rows;
      }
      continue;
    }

    // Direct (contiguous) reads; deduplicate identical src slices within a tensor.
    absl::flat_hash_map<std::string, std::vector<size_t>> by_key;
    by_key.reserve(g.idx.size());
    for (const size_t ci : g.idx) {
      const PlannedCopyInstance& copy = s.copies[ci];
      if (!copy.contiguous) {
        // Fallback to segment expansion for non-2D/non-contiguous slices.
        uint64_t copy_bytes = 0;
        uint64_t src_key = 0;
        std::vector<SegmentCopy> segs;
        TC_ASSIGN_OR_RETURN(
            segs, build_segments_for_slices(*copy.meta, copy.slices, copy.dst_offset, &copy_bytes, &src_key));
        pending_direct.insert(pending_direct.end(), segs.begin(), segs.end());
        ++st.fallback_copies;
        continue;
      }
      const std::string k = std::format("{}:{}", copy.contiguous_src_offset, copy.bytes);
      by_key[k].push_back(ci);
    }

    for (auto& it : by_key) {
      const std::vector<size_t>& copies = it.second;
      if (copies.empty()) {
        continue;
      }
      const PlannedCopyInstance& primary = s.copies[copies[0]];
      pending_direct.push_back(
          SegmentCopy{
              .src_offset = primary.contiguous_src_offset,
              .dst_offset = primary.dst_offset,
              .bytes = static_cast<size_t>(primary.bytes),
          });
      ++st.direct_primary_reads;

      for (size_t i = 1; i < copies.size(); ++i) {
        const PlannedCopyInstance& other = s.copies[copies[i]];
        pending_d2d.push_back(
            D2dCopyOp{
                .src_dst_offset = primary.dst_offset,
                .dst_dst_offset = other.dst_offset,
                .bytes = static_cast<size_t>(primary.bytes),
            });
        ++st.direct_dedup_copies;
      }
    }
  }

  TC_RETURN_IF_ERROR(flush_direct(pending_direct, pending_d2d));

  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  if (sched_after.waits >= sched_before.waits) {
    r.gpu_sched_waits = sched_after.waits - sched_before.waits;
  }
  if (sched_after.wait_sec >= sched_before.wait_sec) {
    r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
  }

  r.t.commit = commit_sec;
  r.t.pack = pack_sec;
  r.t.total_ready = seconds_since(t_total);
  (void)t_commit;

  (void)tensorcast::cuda::stream_destroy(pack_stream);

  r.c_staged_tensors = st.staged_tensors;
  r.c_staged_reads = st.staged_reads;
  r.c_direct_primary_reads = st.direct_primary_reads;
  r.c_direct_dedup_copies = st.direct_dedup_copies;
  r.c_fallback_copies = st.fallback_copies;

  return std::make_pair(r, std::move(s));
}

absl::StatusOr<std::pair<RunResult, StrategyCState>> run_strategy_c_host_pack_with_state(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  RunResult r;
  r.strategy = StrategyKind::kC_HostPack;
  StrategyCState s;
  StrategyCStats st;
  const absl::Time t_total = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);

  if (!cfg.use_pinned_host_buffer) {
    return absl::FailedPreconditionError("Strategy host_pack requires --use_pinned_host_buffer=true");
  }

  // Meta
  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  s.metas = std::move(meta_or->first);
  r.t.open_meta = open_meta;

  // Plan
  const absl::Time t_get = absl::Now();
  if (!cfg.load_plan_json_path.empty()) {
    uint64_t unique_tensors = 0;
    TC_ASSIGN_OR_RETURN(s.copies, parse_plan_copy_instances_for_rank(cfg, s.metas, &unique_tensors));
    r.selected_tensors = unique_tensors;
    r.selected_copies = s.copies.size();
  } else {
    auto selected_or = select_tensors_all(s.metas);
    if (!selected_or.ok()) {
      return selected_or.status();
    }
    std::vector<const TensorMeta*> selected = *selected_or;
    std::sort(selected.begin(), selected.end(), [](const TensorMeta* a, const TensorMeta* b) {
      if (a->offset != b->offset) {
        return a->offset < b->offset;
      }
      return a->name < b->name;
    });
    s.copies.clear();
    s.copies.reserve(selected.size());
    for (const TensorMeta* tm : selected) {
      PlannedCopyInstance inst;
      inst.meta = tm;
      inst.ckpt_name = tm->name;
      inst.dst_param = tm->name;
      inst.slices = {};
      inst.bytes = tm->size;
      inst.contiguous = true;
      inst.contiguous_src_offset = tm->offset;
      inst.src_sort_key = tm->offset;
      if (cfg.tp_world_size > 1) {
        TpContiguousSlice slice;
        TC_ASSIGN_OR_RETURN(slice, compute_tp_contiguous_slice(*tm, cfg, cfg.tp_rank));
        inst.bytes = slice.bytes;
        inst.contiguous_src_offset = tm->offset + slice.offset_in_tensor;
        inst.src_sort_key = inst.contiguous_src_offset;
      }
      s.copies.push_back(std::move(inst));
    }
    r.selected_tensors = selected.size();
    r.selected_copies = selected.size();
    std::sort(s.copies.begin(), s.copies.end(), [](const PlannedCopyInstance& a, const PlannedCopyInstance& b) {
      if (a.src_sort_key != b.src_sort_key) {
        return a.src_sort_key < b.src_sort_key;
      }
      if (a.ckpt_name != b.ckpt_name) {
        return a.ckpt_name < b.ckpt_name;
      }
      return a.dst_param < b.dst_param;
    });
  }

  uint64_t dst_cursor = 0;
  for (auto& c : s.copies) {
    c.dst_offset = dst_cursor;
    dst_cursor += c.bytes;
  }
  s.output_bytes = dst_cursor;
  r.output_bytes = dst_cursor;
  r.t.get_calls_total = seconds_since(t_get);

  // Output VMM allocation.
  const absl::Time t_commit = absl::Now();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  TC_RETURN_IF_ERROR(s.vmm.reserve_and_map(s.output_bytes, cfg.device_id));
  r.res.vmm_reserved_bytes = s.vmm.reserved_bytes();
  r.res.vmm_mapped_bytes = s.vmm.mapped_bytes();
  r.res.vmm_granularity_bytes = s.vmm.granularity_bytes();

  MultiSafetensorsSource backing_src(shards);

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));
  const int io_threads = std::max(1, cfg.io_threads);

  // Group copies by checkpoint tensor to enable batched decisions.
  struct TensorGroup {
    const TensorMeta* meta = nullptr;
    std::vector<size_t> idx;
    bool staged = false;
  };

  absl::flat_hash_map<const TensorMeta*, size_t> group_index;
  group_index.reserve(s.copies.size());
  std::vector<TensorGroup> groups;
  groups.reserve(s.copies.size());
  for (size_t i = 0; i < s.copies.size(); ++i) {
    const TensorMeta* tm = s.copies[i].meta;
    auto [it, inserted] = group_index.emplace(tm, groups.size());
    if (inserted) {
      TensorGroup g;
      g.meta = tm;
      groups.push_back(std::move(g));
    }
    groups[it->second].idx.push_back(i);
  }

  // Decide staged tensors; allocate host staging + pinned pack buffers if needed.
  bool needs_staging = false;
  for (auto& g : groups) {
    const TensorMeta& tm = *g.meta;
    bool any_non_contig = false;
    bool axes_ok = true;
    for (const size_t ci : g.idx) {
      if (!s.copies[ci].contiguous) {
        any_non_contig = true;
      }
      for (const auto& sl : s.copies[ci].slices) {
        if (sl.axis < 0 || sl.axis > 1) {
          axes_ok = false;
        }
      }
    }
    bool can_pack_2d = false;
    if (tm.shape.size() == 2 && axes_ok) {
      TC_ASSIGN_OR_RETURN(can_pack_2d, is_row_major_contiguous(tm));
    }
    g.staged = any_non_contig && can_pack_2d;
    if (g.staged) {
      needs_staging = true;
    }
  }

  std::unique_ptr<uint8_t, decltype(&free)> host_staging(nullptr, &free);
  uint64_t host_staging_bytes = 0;

  std::unique_ptr<common::memory::PinnedBufferPool> host_pack_pool;
  uint8_t* host_pack_buf0 = nullptr;
  uint8_t* host_pack_buf1 = nullptr;
  size_t host_pack_buf_bytes = 0;
  cudaStream_t h2d_stream = nullptr;
  cudaEvent_t h2d_done[2]{nullptr, nullptr};
  bool h2d_used[2]{false, false};

  if (needs_staging) {
    s.staging_bytes = std::max<uint64_t>(1, cfg.strategy_c_staging_bytes);
    host_staging_bytes = s.staging_bytes;
    if (host_staging_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return absl::InvalidArgumentError("host_pack: staging_bytes overflows size_t");
    }
    host_staging.reset(reinterpret_cast<uint8_t*>(aligned_alloc(4096, static_cast<size_t>(host_staging_bytes))));
    if (host_staging == nullptr) {
      return absl::ResourceExhaustedError("host_pack: failed to allocate host staging buffer");
    }

    // Compute an upper bound on the max packed output bytes per chunk, used to size pinned pack buffers.
    uint64_t max_packed_chunk_bytes = 0;
    for (const auto& g : groups) {
      if (!g.staged) {
        continue;
      }
      const TensorMeta& tm = *g.meta;
      if (tm.shape.size() != 2 || tm.elem_size == 0) {
        continue;
      }

      // Union row range across copies in this tensor.
      Slice2D union_slice;
      union_slice.row_start = std::numeric_limits<int64_t>::max();
      union_slice.row_size = 0;
      union_slice.col_start = 0;
      union_slice.col_size = tm.shape[1];
      for (const size_t ci : g.idx) {
        Slice2D sl;
        TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, s.copies[ci].slices));
        union_slice.row_start = std::min(union_slice.row_start, sl.row_start);
        union_slice.row_size = std::max(union_slice.row_size, sl.row_start + sl.row_size);
      }
      if (union_slice.row_start == std::numeric_limits<int64_t>::max()) {
        union_slice.row_start = 0;
        union_slice.row_size = 0;
      }
      const int64_t row_start = std::max<int64_t>(0, union_slice.row_start);
      const int64_t row_end = std::min<int64_t>(tm.shape[0], union_slice.row_size);
      if (row_end <= row_start) {
        continue;
      }

      const uint64_t cols = static_cast<uint64_t>(tm.shape[1]);
      const uint64_t elem = static_cast<uint64_t>(tm.elem_size);
      const uint64_t row_bytes_u64 = cols * elem;
      if (row_bytes_u64 == 0) {
        continue;
      }
      const uint64_t staging_bytes = s.staging_bytes;
      uint64_t max_rows_per_chunk = (staging_bytes / row_bytes_u64);
      if (max_rows_per_chunk == 0) {
        max_rows_per_chunk = 1;
      }
      for (const size_t ci : g.idx) {
        Slice2D sl;
        TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, s.copies[ci].slices));
        if (sl.row_size <= 0 || sl.col_size <= 0) {
          continue;
        }
        const uint64_t rows_u64 = std::min<uint64_t>(max_rows_per_chunk, static_cast<uint64_t>(sl.row_size));
        const uint64_t col_bytes = static_cast<uint64_t>(sl.col_size) * elem;
        if (col_bytes == 0) {
          continue;
        }
        if (rows_u64 > (std::numeric_limits<uint64_t>::max() / col_bytes)) {
          return absl::InvalidArgumentError("host_pack: packed bytes overflow");
        }
        max_packed_chunk_bytes = std::max(max_packed_chunk_bytes, rows_u64 * col_bytes);
      }
    }

    constexpr uint64_t kAlign = common::memory::PinnedBufferPool::kDirectIOAlignment;
    const uint64_t pack_chunk_u64 =
        std::max<uint64_t>(kAlign, ((max_packed_chunk_bytes + kAlign - 1) / kAlign) * kAlign);
    if (pack_chunk_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      return absl::InvalidArgumentError("host_pack: pinned pack chunk bytes overflow size_t");
    }
    const size_t pack_chunk_bytes = static_cast<size_t>(pack_chunk_u64);
    if (pack_chunk_bytes > (std::numeric_limits<size_t>::max() / 2u)) {
      return absl::InvalidArgumentError("host_pack: pinned pack pool bytes overflow size_t");
    }
    const size_t pack_total_bytes = pack_chunk_bytes * 2;

    common::memory::PinnedBufferPool::Options pool_opts;
    pool_opts.name = "host_pack";
    pool_opts.prefault = cfg.pinned_numa_prefault;
    TC_ASSIGN_OR_RETURN(pool_opts.numa_node, resolve_pinned_numa_node_for_device(cfg, cfg.device_id));
    host_pack_pool =
        std::make_unique<common::memory::PinnedBufferPool>(pack_total_bytes, pack_chunk_bytes, std::move(pool_opts));
    const auto slabs = host_pack_pool->list_slabs();
    if (slabs.empty()) {
      return absl::InternalError("host_pack: pinned pack pool has no slabs");
    }
    size_t pinned_total = 0;
    for (const auto& slab : slabs) {
      pinned_total += slab.bytes;
    }
    r.res.pinned_host_bytes += pinned_total;

    host_pack_buf_bytes = host_pack_pool->slice_bytes();
    if (slabs[0].bytes < (host_pack_buf_bytes * 2)) {
      return absl::InternalError("host_pack: pinned slab smaller than 2 slices");
    }
    auto* base = reinterpret_cast<uint8_t*>(slabs[0].base.get());
    host_pack_buf0 = base;
    host_pack_buf1 = base + host_pack_buf_bytes;

    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&h2d_stream, cudaStreamNonBlocking));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&h2d_done[0], cudaEventDisableTiming));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&h2d_done[1], cudaEventDisableTiming));
  }

  // Stream for D2D copies (dedup) and 2D pack on GPU (if needed).
  cudaStream_t pack_stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&pack_stream, cudaStreamNonBlocking));
  double commit_sec = 0.0;
  double pack_sec = 0.0;

  auto flush_direct = [&](std::vector<SegmentCopy>& pending, std::vector<D2dCopyOp>& d2d_ops) -> absl::Status {
    if (pending.empty()) {
      if (!d2d_ops.empty()) {
        return absl::InternalError("Internal error: d2d_ops not empty while pending segments empty");
      }
      return absl::OkStatus();
    }
    auto [merged, planner_stats] = merge_adjacent_segments_by_src(std::move(pending));
    pending.clear();
    pending = std::move(merged);
    uint64_t max_dst_end = 0;
    for (const auto& seg : pending) {
      const uint64_t end = seg.dst_offset + static_cast<uint64_t>(seg.bytes);
      max_dst_end = std::max(max_dst_end, end);
    }
    if (max_dst_end > s.output_bytes) {
      return absl::OutOfRangeError("Strategy host_pack: direct segment write exceeds output buffer bounds");
    }
    r.planned_segments += planner_stats.segments_post_merge;
    r.planner_segments_pre_merge += planner_stats.segments_pre_merge;
    r.planner_segments_merged += (planner_stats.segments_pre_merge >= planner_stats.segments_post_merge)
        ? (planner_stats.segments_pre_merge - planner_stats.segments_post_merge)
        : 0;
    r.planner_src_runs += planner_stats.src_runs_post_merge;
    r.planner_src_run_max_bytes = std::max(r.planner_src_run_max_bytes, planner_stats.src_run_max_bytes);

    std::vector<RemappedSource::Segment> remap;
    remap.reserve(pending.size());
    for (const auto& seg : pending) {
      remap.push_back(
          RemappedSource::Segment{
              .dst_offset = seg.dst_offset,
              .src_offset = seg.src_offset,
              .end_offset = seg.dst_offset + seg.bytes,
          });
    }
    RemappedSource src(gsl::not_null<SeekableSource*>(&backing_src), std::move(remap));

    loader::GpuMemorySink sink(
        loader::GpuMemorySink::Options{
            .gpu_base_ptr = gsl::not_null<void*>{s.vmm.base_ptr()},
            // Strategy C performs sparse writes into the final output buffer; do
            // not enforce "must write exactly total_size bytes" on close().
            .total_size = 0,
            .chunk_size = 128 * 1024 * 1024,
            .device_id = cfg.device_id,
            .allocation = nullptr,
            .gpu_sched_enabled = cfg.gpu_sched_enabled,
            .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
            .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
        });

    uint64_t total_requested_bytes = 0;
    uint64_t planned_ranges = 0;
    auto ranges_or = build_pump_ranges_for_copy(pending, io_threads, &total_requested_bytes, &planned_ranges);
    if (!ranges_or.ok()) {
      return ranges_or.status();
    }
    const auto& ranges = *ranges_or;
    r.planned_ranges += planned_ranges;
    r.bytes.disk_read_bytes += total_requested_bytes;
    r.bytes.h2d_bytes += total_requested_bytes;

    if (auto* adapter = dynamic_cast<StreamingBufferAdapter*>(pool_ptr.get()); adapter != nullptr) {
      TC_RETURN_IF_ERROR(adapter->get_buffer()->reset_for_new_production());
    }

    const absl::Time t = absl::Now();
    TC_RETURN_IF_ERROR(
        loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
    TC_RETURN_IF_ERROR(sink.close());
    commit_sec += seconds_since(t);

    if (!d2d_ops.empty()) {
      const absl::Time t_d2d = absl::Now();
      auto* base = static_cast<uint8_t*>(s.vmm.base_ptr());
      for (const auto& op : d2d_ops) {
        if (op.bytes == 0) {
          continue;
        }
        TC_RETURN_IF_ERROR(
            tensorcast::cuda::memcpy_async(
                base + op.dst_dst_offset, base + op.src_dst_offset, op.bytes, cudaMemcpyDeviceToDevice, pack_stream));
        r.bytes.d2d_bytes += static_cast<uint64_t>(op.bytes);
      }
      TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(pack_stream));
      pack_sec += seconds_since(t_d2d);
      d2d_ops.clear();
    }
    pending.clear();
    return absl::OkStatus();
  };

  std::sort(groups.begin(), groups.end(), [](const TensorGroup& a, const TensorGroup& b) {
    return a.meta->offset < b.meta->offset;
  });

  std::vector<SegmentCopy> pending_direct;
  std::vector<D2dCopyOp> pending_d2d;
  pending_direct.reserve(s.copies.size());
  pending_d2d.reserve(s.copies.size());

  size_t next_host_pack_buf = 0;

  // Execute in on-disk tensor order to preserve sequential access.
  for (auto& g : groups) {
    const TensorMeta& tm = *g.meta;
    if (g.staged) {
      ++st.staged_tensors;
      TC_RETURN_IF_ERROR(flush_direct(pending_direct, pending_d2d));

      if (!needs_staging || host_staging == nullptr || host_staging_bytes == 0) {
        return absl::InternalError("host_pack: staged tensor but host staging buffers are not initialized");
      }

      // Compute union row range for staged reads.
      Slice2D union_slice;
      union_slice.row_start = std::numeric_limits<int64_t>::max();
      union_slice.row_size = 0;
      union_slice.col_start = 0;
      union_slice.col_size = tm.shape[1];
      for (const size_t ci : g.idx) {
        Slice2D sl;
        TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, s.copies[ci].slices));
        union_slice.row_start = std::min(union_slice.row_start, sl.row_start);
        union_slice.row_size = std::max(union_slice.row_size, sl.row_start + sl.row_size);
      }
      if (union_slice.row_start == std::numeric_limits<int64_t>::max()) {
        union_slice.row_start = 0;
        union_slice.row_size = 0;
      }
      const int64_t row_start = std::max<int64_t>(0, union_slice.row_start);
      const int64_t row_end = std::min<int64_t>(tm.shape[0], union_slice.row_size);
      if (row_end <= row_start) {
        continue;
      }
      const uint64_t cols = static_cast<uint64_t>(tm.shape[1]);
      const uint64_t elem = static_cast<uint64_t>(tm.elem_size);
      const uint64_t row_bytes_u64 = cols * elem;
      if (row_bytes_u64 == 0 || row_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return absl::InvalidArgumentError(absl::StrCat("Row byte size overflow for tensor ", tm.name));
      }
      const uint64_t staging_bytes = host_staging_bytes;
      uint64_t max_rows_per_chunk = (staging_bytes / row_bytes_u64);
      if (max_rows_per_chunk == 0) {
        max_rows_per_chunk = 1;
      }

      auto* out_base = static_cast<uint8_t*>(s.vmm.base_ptr());
      auto* staging_base = host_staging.get();
      for (int64_t row = row_start; row < row_end;) {
        const int64_t chunk_rows = std::min<int64_t>(static_cast<int64_t>(max_rows_per_chunk), row_end - row);
        const uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * row_bytes_u64;
        if (chunk_bytes > host_staging_bytes) {
          return absl::InternalError("host_pack: computed chunk_bytes exceeds host staging buffer");
        }

        std::vector<RemappedSource::Segment> remap;
        remap.push_back(
            RemappedSource::Segment{
                .dst_offset = 0,
                .src_offset = tm.offset + static_cast<uint64_t>(row) * row_bytes_u64,
                .end_offset = chunk_bytes,
            });
        RemappedSource src(gsl::not_null<SeekableSource*>(&backing_src), std::move(remap));

        HostBufferSink sink(gsl::not_null<uint8_t*>{staging_base}, chunk_bytes);

        const auto ranges = split_even_ranges(/*base=*/0, chunk_bytes, io_threads);
        r.planned_ranges += ranges.size();
        r.planned_segments += 1;
        r.bytes.disk_read_bytes += chunk_bytes;

        if (auto* adapter = dynamic_cast<StreamingBufferAdapter*>(pool_ptr.get()); adapter != nullptr) {
          TC_RETURN_IF_ERROR(adapter->get_buffer()->reset_for_new_production());
        }

        const absl::Time t = absl::Now();
        TC_RETURN_IF_ERROR(
            loader::pump_ranges(
                src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
        commit_sec += seconds_since(t);
        ++st.staged_reads;

        // Host pack: staged host buffer -> pinned host -> H2D into final output layout.
        for (const size_t ci : g.idx) {
          const PlannedCopyInstance& copy = s.copies[ci];
          Slice2D sl;
          TC_ASSIGN_OR_RETURN(sl, extract_slice_2d(tm, copy.slices));

          const int64_t copy_row_start = sl.row_start;
          const int64_t copy_row_end = sl.row_start + sl.row_size;
          const int64_t chunk_row_start = row;
          const int64_t chunk_row_end = row + chunk_rows;
          const int64_t inter_start = std::max(copy_row_start, chunk_row_start);
          const int64_t inter_end = std::min(copy_row_end, chunk_row_end);
          if (inter_end <= inter_start || sl.col_size <= 0 || sl.row_size <= 0) {
            continue;
          }

          const uint64_t inter_rows = static_cast<uint64_t>(inter_end - inter_start);
          const uint64_t col_bytes_u64 = static_cast<uint64_t>(sl.col_size) * elem;
          if (col_bytes_u64 == 0 || col_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return absl::InvalidArgumentError("host_pack: col_bytes overflow");
          }
          if (inter_rows > (std::numeric_limits<uint64_t>::max() / col_bytes_u64)) {
            return absl::InvalidArgumentError("host_pack: packed bytes overflow");
          }
          const uint64_t packed_bytes_u64 = inter_rows * col_bytes_u64;
          if (packed_bytes_u64 == 0) {
            continue;
          }
          if (packed_bytes_u64 > static_cast<uint64_t>(host_pack_buf_bytes)) {
            return absl::OutOfRangeError("host_pack: packed_bytes exceeds pinned host pack buffer");
          }

          const uint64_t src_row_off = static_cast<uint64_t>(inter_start - chunk_row_start);
          const uint64_t dst_row_off = static_cast<uint64_t>(inter_start - copy_row_start);
          const uint64_t src_col_off = static_cast<uint64_t>(sl.col_start) * elem;

          const uint8_t* src_ptr = staging_base + src_row_off * row_bytes_u64 + src_col_off;
          uint8_t* dst_ptr = out_base + copy.dst_offset + dst_row_off * col_bytes_u64;

          // Wait for the selected pack buffer to be free before overwriting.
          const size_t buf_idx = next_host_pack_buf % 2;
          next_host_pack_buf += 1;
          uint8_t* pack_buf = (buf_idx == 0) ? host_pack_buf0 : host_pack_buf1;
          if (pack_buf == nullptr || h2d_done[buf_idx] == nullptr || h2d_stream == nullptr) {
            return absl::InternalError("host_pack: pinned pack buffers or CUDA resources are not initialized");
          }
          if (h2d_used[buf_idx]) {
            bool ready = false;
            TC_RETURN_IF_ERROR(tensorcast::cuda::event_query(h2d_done[buf_idx], &ready));
            if (!ready) {
              const absl::Time t_wait = absl::Now();
              TC_RETURN_IF_ERROR(tensorcast::cuda::event_synchronize(h2d_done[buf_idx]));
              commit_sec += seconds_since(t_wait);
            }
          }

          // CPU pack (row-by-row memcpy) into pinned host buffer.
          const absl::Time t_pack = absl::Now();
          const size_t col_bytes = static_cast<size_t>(col_bytes_u64);
          const size_t src_pitch = static_cast<size_t>(row_bytes_u64);
          const size_t rows = static_cast<size_t>(inter_rows);
          for (size_t rr = 0; rr < rows; ++rr) {
            const uint8_t* src_row = src_ptr + rr * src_pitch;
            std::memcpy(pack_buf + rr * col_bytes, src_row, col_bytes);
          }
          pack_sec += seconds_since(t_pack);

          TC_RETURN_IF_ERROR(
              tensorcast::cuda::memcpy_async(
                  dst_ptr, pack_buf, static_cast<size_t>(packed_bytes_u64), cudaMemcpyHostToDevice, h2d_stream));
          TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(h2d_done[buf_idx], h2d_stream));
          h2d_used[buf_idx] = true;
          r.bytes.h2d_bytes += packed_bytes_u64;
        }

        row += chunk_rows;
      }
      continue;
    }

    // Direct (contiguous) reads; deduplicate identical src slices within a tensor.
    absl::flat_hash_map<std::string, std::vector<size_t>> by_key;
    by_key.reserve(g.idx.size());
    for (const size_t ci : g.idx) {
      const PlannedCopyInstance& copy = s.copies[ci];
      if (!copy.contiguous) {
        // Fallback to segment expansion for non-2D/non-contiguous slices.
        uint64_t copy_bytes = 0;
        uint64_t src_key = 0;
        std::vector<SegmentCopy> segs;
        TC_ASSIGN_OR_RETURN(
            segs, build_segments_for_slices(*copy.meta, copy.slices, copy.dst_offset, &copy_bytes, &src_key));
        pending_direct.insert(pending_direct.end(), segs.begin(), segs.end());
        ++st.fallback_copies;
        continue;
      }
      const std::string k = std::format("{}:{}", copy.contiguous_src_offset, copy.bytes);
      by_key[k].push_back(ci);
    }

    for (auto& it : by_key) {
      const std::vector<size_t>& copies = it.second;
      if (copies.empty()) {
        continue;
      }
      const PlannedCopyInstance& primary = s.copies[copies[0]];
      pending_direct.push_back(
          SegmentCopy{
              .src_offset = primary.contiguous_src_offset,
              .dst_offset = primary.dst_offset,
              .bytes = static_cast<size_t>(primary.bytes),
          });
      ++st.direct_primary_reads;

      for (size_t i = 1; i < copies.size(); ++i) {
        const PlannedCopyInstance& other = s.copies[copies[i]];
        pending_d2d.push_back(
            D2dCopyOp{
                .src_dst_offset = primary.dst_offset,
                .dst_dst_offset = other.dst_offset,
                .bytes = static_cast<size_t>(primary.bytes),
            });
        ++st.direct_dedup_copies;
      }
    }
  }

  TC_RETURN_IF_ERROR(flush_direct(pending_direct, pending_d2d));

  if (h2d_stream != nullptr) {
    const absl::Time t_wait = absl::Now();
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(h2d_stream));
    commit_sec += seconds_since(t_wait);
  }

  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  if (sched_after.waits >= sched_before.waits) {
    r.gpu_sched_waits = sched_after.waits - sched_before.waits;
  }
  if (sched_after.wait_sec >= sched_before.wait_sec) {
    r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
  }

  if (h2d_done[0] != nullptr) {
    (void)tensorcast::cuda::event_destroy(h2d_done[0]);
  }
  if (h2d_done[1] != nullptr) {
    (void)tensorcast::cuda::event_destroy(h2d_done[1]);
  }
  if (h2d_stream != nullptr) {
    (void)tensorcast::cuda::stream_destroy(h2d_stream);
  }

  r.t.commit = commit_sec;
  r.t.pack = pack_sec;
  r.t.total_ready = seconds_since(t_total);
  (void)t_commit;

  (void)tensorcast::cuda::stream_destroy(pack_stream);

  r.c_staged_tensors = st.staged_tensors;
  r.c_staged_reads = st.staged_reads;
  r.c_direct_primary_reads = st.direct_primary_reads;
  r.c_direct_dedup_copies = st.direct_dedup_copies;
  r.c_fallback_copies = st.fallback_copies;

  return std::make_pair(r, std::move(s));
}

absl::StatusOr<std::pair<RunResult, StrategyBState>> run_strategy_b_with_state(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  RunResult r;
  r.strategy = StrategyKind::kB_LazyCommit;
  StrategyBState s;
  const absl::Time t_total = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);

  // Meta
  double open_meta = 0.0;
  auto meta_or = load_metas_from_safetensors(shards, &open_meta);
  if (!meta_or.ok()) {
    return meta_or.status();
  }
  s.metas = std::move(meta_or->first);
  const uint64_t total_payload_bytes = meta_or->second;
  r.t.open_meta = open_meta;

  if (!cfg.use_pinned_host_buffer) {
    return absl::FailedPreconditionError("Strategy B requires --use_pinned_host_buffer=true");
  }

  const absl::Time t_get = absl::Now();
  if (!cfg.load_plan_json_path.empty()) {
    RankLoadPlan plan;
    TC_ASSIGN_OR_RETURN(plan, build_rank_load_plan(cfg, s.metas));
    r.selected_tensors = plan.unique_tensors;
    r.selected_copies = plan.copies;
    r.output_bytes = plan.output_bytes;
    s.dst_bytes = plan.output_bytes;
    s.segments = std::move(plan.segments);
  } else {
    auto selected_or = select_tensors_all(s.metas);
    if (!selected_or.ok()) {
      return selected_or.status();
    }
    s.selected = *selected_or;
    r.selected_tensors = s.selected.size();
    r.selected_copies = s.selected.size();

    // Strategy B uses disk-only direct shard reads. To maximize disk locality, order
    // requests by on-disk source offset (i.e., shard file order) instead of tensor
    // name order.
    std::sort(s.selected.begin(), s.selected.end(), [](const TensorMeta* a, const TensorMeta* b) {
      if (a->offset != b->offset) {
        return a->offset < b->offset;
      }
      return a->name < b->name;
    });

    uint64_t dst_cursor = 0;
    s.segments.clear();
    s.tensor_plans.clear();
    s.tensor_plans.reserve(s.selected.size());
    for (const TensorMeta* tm : s.selected) {
      auto segs_or = plan_tensor_segments(*tm, cfg, dst_cursor);
      if (!segs_or.ok()) {
        return segs_or.status();
      }
      const auto& segs = *segs_or;
      uint64_t tensor_bytes = 0;
      for (const auto& seg : segs) {
        s.segments.push_back(seg);
        tensor_bytes += seg.bytes;
        dst_cursor += seg.bytes;
      }
      TensorSlicePlan tp;
      tp.name = tm->name;
      tp.dst_offset = dst_cursor - tensor_bytes;
      tp.bytes = tensor_bytes;
      if (tm->shape.size() >= 2) {
        tp.rows = tm->shape[0];
        tp.cols = tm->shape[1];
      }
      tp.elem_size = tm->elem_size;
      s.tensor_plans.push_back(std::move(tp));
    }
    s.dst_bytes = dst_cursor;
    r.output_bytes = dst_cursor;
  }
  r.t.get_calls_total = seconds_since(t_get);
  {
    auto [merged_segments, planner_stats] = merge_adjacent_segments_by_src(std::move(s.segments));
    s.segments = std::move(merged_segments);
    r.planned_segments = planner_stats.segments_post_merge;
    r.planner_segments_pre_merge = planner_stats.segments_pre_merge;
    r.planner_segments_merged = (planner_stats.segments_pre_merge >= planner_stats.segments_post_merge)
        ? (planner_stats.segments_pre_merge - planner_stats.segments_post_merge)
        : 0;
    r.planner_src_runs = planner_stats.src_runs_post_merge;
    r.planner_src_run_avg_bytes = planner_stats.src_run_avg_bytes;
    r.planner_src_run_max_bytes = planner_stats.src_run_max_bytes;
  }

  // commit: reserve target VA, then pump disk ranges into final mapped addresses.
  const absl::Time t_commit = absl::Now();
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  TC_RETURN_IF_ERROR(s.vmm.reserve_and_map(s.dst_bytes, cfg.device_id));
  r.res.vmm_reserved_bytes = s.vmm.reserved_bytes();
  r.res.vmm_mapped_bytes = s.vmm.mapped_bytes();
  r.res.vmm_granularity_bytes = s.vmm.granularity_bytes();

  MultiSafetensorsSource backing_src(shards);

  // Convert segment list into remapped segments and pump ranges in destination space.
  std::vector<RemappedSource::Segment> remap;
  remap.reserve(s.segments.size());
  for (const auto& seg : s.segments) {
    if (seg.bytes == 0) {
      continue;
    }
    remap.push_back(
        RemappedSource::Segment{
            .dst_offset = seg.dst_offset,
            .src_offset = seg.src_offset,
            .end_offset = seg.dst_offset + seg.bytes,
        });
  }
  RemappedSource src(gsl::not_null<SeekableSource*>(&backing_src), std::move(remap));

  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{s.vmm.base_ptr()},
          .total_size = s.dst_bytes,
          .chunk_size = 128 * 1024 * 1024,
          .device_id = cfg.device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = cfg.gpu_sched_enabled,
          .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
          .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
      });

  const int io_threads = std::max(1, cfg.io_threads);
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, &r.res));

  uint64_t total_requested_bytes = 0;
  uint64_t planned_ranges = 0;
  auto ranges_or = build_pump_ranges_for_copy(s.segments, io_threads, &total_requested_bytes, &planned_ranges);
  if (!ranges_or.ok()) {
    return ranges_or.status();
  }
  const auto& ranges = *ranges_or;
  r.planned_ranges = planned_ranges;
  r.bytes.disk_read_bytes = total_requested_bytes;
  r.bytes.h2d_bytes = total_requested_bytes;
  r.output_bytes = total_requested_bytes;
  (void)total_payload_bytes;

  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  r.t.commit = seconds_since(t_commit);
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  if (sched_after.waits >= sched_before.waits) {
    r.gpu_sched_waits = sched_after.waits - sched_before.waits;
  }
  if (sched_after.wait_sec >= sched_before.wait_sec) {
    r.gpu_sched_wait_sec = sched_after.wait_sec - sched_before.wait_sec;
  }

  r.t.total_ready = seconds_since(t_total);
  return std::make_pair(r, std::move(s));
}

absl::Status check_correctness_samples(const LoaderConfig& cfg, const StrategyAState& a, const StrategyBState& b) {
  if (cfg.check_correctness_samples == 0) {
    return absl::OkStatus();
  }
  if (b.vmm.base_ptr() == nullptr) {
    return absl::FailedPreconditionError("Strategy B VMM buffer not available for correctness check");
  }
  if (a.output == nullptr && !a.gpu_payload) {
    return absl::FailedPreconditionError("Strategy A buffers not available for correctness check");
  }

  std::vector<std::string> names;
  names.reserve(std::min<uint64_t>(cfg.check_correctness_samples, b.tensor_plans.size()));
  for (const auto& tp : b.tensor_plans) {
    names.push_back(tp.name);
    if (names.size() >= cfg.check_correctness_samples) {
      break;
    }
  }
  std::sort(names.begin(), names.end());

  const auto* a_payload_base = a.gpu_payload ? static_cast<const uint8_t*>(a.gpu_payload->get()) : nullptr;
  const auto* a_out_base = a.output ? static_cast<const uint8_t*>(a.output->get()) : nullptr;
  const auto* b_base = static_cast<const uint8_t*>(b.vmm.base_ptr());

  absl::flat_hash_map<std::string, const TensorMeta*> a_by_name;
  a_by_name.reserve(a.metas.size());
  for (const auto& tm : a.metas) {
    a_by_name.emplace(tm.name, &tm);
  }

  absl::flat_hash_map<std::string, const TensorSlicePlan*> b_plan;
  b_plan.reserve(b.tensor_plans.size());
  for (const auto& tp : b.tensor_plans) {
    b_plan.emplace(tp.name, &tp);
  }

  absl::flat_hash_map<std::string, const TensorSlicePlan*> a_plan;
  a_plan.reserve(a.tensor_plans.size());
  for (const auto& tp : a.tensor_plans) {
    a_plan.emplace(tp.name, &tp);
  }

  for (const auto& name : names) {
    const auto it_a = a_by_name.find(name);
    const auto it_b = b_plan.find(name);
    if (it_a == a_by_name.end() || it_b == b_plan.end()) {
      return absl::InternalError(absl::StrCat("Missing tensor in correctness mapping: ", name));
    }
    const TensorMeta& tm = *it_a->second;
    const TensorSlicePlan& tp = *it_b->second;
    if (tp.bytes > kCorrectnessSampleMaxBytes) {
      continue;
    }
    std::vector<uint8_t> expect(tp.bytes);
    std::vector<uint8_t> actual(tp.bytes);
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy(actual.data(), b_base + tp.dst_offset, tp.bytes, cudaMemcpyDeviceToHost));

    // Strategy A with A4 collectives: compare the final per-rank output buffer directly.
    if (a_out_base != nullptr) {
      const auto it_ap = a_plan.find(name);
      if (it_ap == a_plan.end()) {
        return absl::InternalError(absl::StrCat("Missing tensor plan in strategy A output mapping: ", name));
      }
      const TensorSlicePlan& ap = *it_ap->second;
      if (ap.bytes != tp.bytes) {
        return absl::InternalError(absl::StrCat("Slice size mismatch between A and B for tensor: ", name));
      }
      TC_RETURN_IF_ERROR(
          tensorcast::cuda::memcpy(expect.data(), a_out_base + ap.dst_offset, ap.bytes, cudaMemcpyDeviceToHost));
      if (expect != actual) {
        return absl::InternalError(absl::StrCat("Correctness mismatch for tensor: ", name));
      }
      continue;
    }

    if (a_payload_base == nullptr) {
      return absl::FailedPreconditionError("Strategy A GPU payload not available for correctness check");
    }

    // Construct expected bytes from strategy A payload buffer according to the same per-rank
    // contiguous slice rule (row-split for 2D, byte-split fallback otherwise).
    TpContiguousSlice slice;
    TC_ASSIGN_OR_RETURN(slice, compute_tp_contiguous_slice(tm, cfg, cfg.tp_rank));
    if (slice.bytes != tp.bytes) {
      return absl::InternalError(absl::StrCat("Slice byte mismatch for tensor: ", name));
    }
    const uint64_t src_off = tm.offset + slice.offset_in_tensor;
    TC_RETURN_IF_ERROR(
        tensorcast::cuda::memcpy(expect.data(), a_payload_base + src_off, slice.bytes, cudaMemcpyDeviceToHost));

    if (expect != actual) {
      return absl::InternalError(absl::StrCat("Correctness mismatch for tensor: ", name));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> compute_total_payload_bytes(const std::vector<std::filesystem::path>& shards) {
  uint64_t total = 0;
  for (const auto& p : shards) {
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open ", p.string()));
    }
    auto info_or = ParseSafetensorsHeader(fd);
    ::close(fd);
    if (!info_or.ok()) {
      return info_or.status();
    }
    total += info_or->data_size;
  }
  return total;
}

absl::Status run_safetensors_disk_baseline(const LoaderConfig& cfg, const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_disk_baseline: shards is empty");
  }
  uint64_t total_payload_bytes = 0;
  TC_ASSIGN_OR_RETURN(total_payload_bytes, compute_total_payload_bytes(shards));

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));

  common::memory::GpuDeviceMemory dst;
  TC_RETURN_IF_ERROR(dst.allocate(static_cast<size_t>(total_payload_bytes), cfg.device_id));

  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{dst.get()},
          .total_size = total_payload_bytes,
          .chunk_size = 128 * 1024 * 1024,
          .device_id = cfg.device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = cfg.gpu_sched_enabled,
          .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
          .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
      });

  MultiSafetensorsSource src(shards);
  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);
  const auto t0 = absl::Now();
  auto ranges = split_even_ranges(/*base=*/0, total_payload_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  const auto sched_waits = ((sched_after.waits >= sched_before.waits) ? (sched_after.waits - sched_before.waits) : 0);
  const double sched_wait_sec =
      (sched_after.wait_sec >= sched_before.wait_sec) ? (sched_after.wait_sec - sched_before.wait_sec) : 0.0;
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);

  LOG(INFO) << std::format(
      "safetensors_disk_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} gpu_sched_waits={} gpu_sched_wait_sec={:.6f}",
      total_payload_bytes,
      sec,
      gbps,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      sched_waits,
      sched_wait_sec);
  return absl::OkStatus();
}

absl::Status run_safetensors_host_baseline(const LoaderConfig& cfg, const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_host_baseline: shards is empty");
  }
  uint64_t total_payload_bytes = 0;
  TC_ASSIGN_OR_RETURN(total_payload_bytes, compute_total_payload_bytes(shards));

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  MultiSafetensorsSource src(shards);
  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  NullPositionedSink sink(total_payload_bytes);
  const auto t0 = absl::Now();
  auto ranges = split_even_ranges(/*base=*/0, total_payload_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);

  LOG(INFO) << std::format(
      "safetensors_host_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} bytes_written={}",
      total_payload_bytes,
      sec,
      gbps,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      sink.bytes_written());
  return absl::OkStatus();
}

absl::Status warmup_safetensors_page_cache(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards,
    uint64_t total_payload_bytes,
    const BounceBufferPlan& bb) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("warmup_safetensors_page_cache: shards is empty");
  }
  if (total_payload_bytes == 0) {
    return absl::InvalidArgumentError("warmup_safetensors_page_cache: total_payload_bytes is zero");
  }

  MultiSafetensorsSource src(shards);
  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  NullPositionedSink sink(total_payload_bytes);
  auto ranges = split_even_ranges(/*base=*/0, total_payload_bytes, io_threads);
  return loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor());
}

absl::Status run_safetensors_hot_host_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_hot_host_baseline: shards is empty");
  }
  uint64_t total_payload_bytes = 0;
  TC_ASSIGN_OR_RETURN(total_payload_bytes, compute_total_payload_bytes(shards));

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  LOG(INFO) << std::format(
      "safetensors_hot_host_baseline warmup: bytes={} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={}",
      total_payload_bytes,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      std::max(1, cfg.io_threads));
  TC_RETURN_IF_ERROR(warmup_safetensors_page_cache(cfg, shards, total_payload_bytes, bb));
  LOG(INFO) << "safetensors_hot_host_baseline warmup: done";

  return run_safetensors_host_baseline(cfg, shards);
}

absl::Status run_safetensors_hot_disk_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_hot_disk_baseline: shards is empty");
  }
  uint64_t total_payload_bytes = 0;
  TC_ASSIGN_OR_RETURN(total_payload_bytes, compute_total_payload_bytes(shards));

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  LOG(INFO) << std::format(
      "safetensors_hot_disk_baseline warmup: bytes={} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={}",
      total_payload_bytes,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      std::max(1, cfg.io_threads));
  TC_RETURN_IF_ERROR(warmup_safetensors_page_cache(cfg, shards, total_payload_bytes, bb));
  LOG(INFO) << "safetensors_hot_disk_baseline warmup: done";

  return run_safetensors_disk_baseline(cfg, shards);
}

class MmapRegion final {
 public:
  MmapRegion() = default;

  MmapRegion(MmapRegion&& other) noexcept {
    *this = std::move(other);
  }

  MmapRegion& operator=(MmapRegion&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    release();
    base_ = other.base_;
    bytes_ = other.bytes_;
    other.base_ = nullptr;
    other.bytes_ = 0;
    return *this;
  }

  ~MmapRegion() {
    release();
  }

  MmapRegion(const MmapRegion&) = delete;
  MmapRegion& operator=(const MmapRegion&) = delete;

  static absl::StatusOr<MmapRegion> allocate(size_t bytes) {
    if (bytes == 0) {
      return absl::InvalidArgumentError("MmapRegion::allocate: bytes is 0");
    }
    void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      return absl::ErrnoToStatus(errno, "mmap failed");
    }
    MmapRegion r;
    r.base_ = p;
    r.bytes_ = bytes;
    return r;
  }

  void* data() const {
    return base_;
  }

  size_t size() const {
    return bytes_;
  }

 private:
  void release() {
    if (base_ == nullptr || bytes_ == 0) {
      base_ = nullptr;
      bytes_ = 0;
      return;
    }
    (void)::munmap(base_, bytes_);
    base_ = nullptr;
    bytes_ = 0;
  }

  void* base_ = nullptr;
  size_t bytes_ = 0;
};

class MemorySpanSource final : public SeekableSource {
 public:
  MemorySpanSource(gsl::not_null<const uint8_t*> base, uint64_t bytes) : base_(base), bytes_(bytes) {}

  [[nodiscard]] uint64_t total_bytes() const override {
    return bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    absl::MutexLock lock(&offset_mu_);
    const uint64_t off = current_offset_;
    if (off >= bytes_) {
      return static_cast<size_t>(0);
    }
    const uint64_t remain = bytes_ - off;
    const size_t want = static_cast<size_t>(std::min<uint64_t>(remain, max_bytes));
    std::memcpy(dst, base_.get() + off, want);
    current_offset_ += want;
    return want;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    if (offset >= bytes_) {
      return static_cast<size_t>(0);
    }
    const uint64_t remain = bytes_ - offset;
    const size_t want = static_cast<size_t>(std::min<uint64_t>(remain, static_cast<uint64_t>(bytes)));
    std::memcpy(dst, base_.get() + offset, want);
    return want;
  }

 private:
  gsl::not_null<const uint8_t*> base_;
  uint64_t bytes_ = 0;
  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_) = 0;
};

absl::Status fill_dram_mirror_from_safetensors(
    const std::vector<std::filesystem::path>& shards,
    uint8_t* dst,
    uint64_t bytes,
    size_t fill_chunk_bytes) {
  if (dst == nullptr) {
    return absl::InvalidArgumentError("fill_dram_mirror_from_safetensors: dst is null");
  }
  if (bytes == 0) {
    return absl::InvalidArgumentError("fill_dram_mirror_from_safetensors: bytes is 0");
  }
  if (fill_chunk_bytes == 0) {
    return absl::InvalidArgumentError("fill_dram_mirror_from_safetensors: fill_chunk_bytes is 0");
  }

  MultiSafetensorsSource src(shards);
  uint64_t off = 0;
  while (off < bytes) {
    const size_t want = static_cast<size_t>(std::min<uint64_t>(bytes - off, static_cast<uint64_t>(fill_chunk_bytes)));
    size_t got = 0;
    TC_ASSIGN_OR_RETURN(got, src.read_at(off, dst + off, want));
    if (got != want) {
      return absl::OutOfRangeError(
          absl::StrCat("fill_dram_mirror_from_safetensors: short read got=", got, " want=", want, " off=", off));
    }
    off += static_cast<uint64_t>(got);
  }
  return absl::OkStatus();
}

absl::Status run_safetensors_dram_mirror_host_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_dram_mirror_host_baseline: shards is empty");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError(
        "--use_pinned_host_buffer=true is required for safetensors_dram_mirror_host_baseline");
  }

  uint64_t total_payload_bytes = 0;
  TC_ASSIGN_OR_RETURN(total_payload_bytes, compute_total_payload_bytes(shards));

  constexpr size_t kFillChunkBytes = 128ull * 1024ull * 1024ull;
  MmapRegion dram;
  TC_ASSIGN_OR_RETURN(dram, MmapRegion::allocate(static_cast<size_t>(total_payload_bytes)));
  auto* dram_base = static_cast<uint8_t*>(dram.data());

  // Build DRAM mirror (not part of the measured baseline).
  {
    const absl::Time t0 = absl::Now();
    TC_RETURN_IF_ERROR(fill_dram_mirror_from_safetensors(shards, dram_base, total_payload_bytes, kFillChunkBytes));
    const double sec = seconds_since(t0);
    LOG(INFO) << std::format(
        "safetensors_dram_mirror_host_baseline mirror_fill bytes={} sec={:.6f} GiB/s={:.3f} fill_chunk_bytes={}",
        total_payload_bytes,
        sec,
        (static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec),
        static_cast<uint64_t>(kFillChunkBytes));
  }

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  // Measured: DRAM (userspace) -> pinned host bounce buffer.
  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  MemorySpanSource src(gsl::not_null<const uint8_t*>{dram_base}, total_payload_bytes);
  NullPositionedSink sink(total_payload_bytes);
  const absl::Time t0 = absl::Now();
  auto ranges = split_even_ranges(/*base=*/0, total_payload_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_payload_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);
  LOG(INFO) << std::format(
      "safetensors_dram_mirror_host_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned=true bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={}",
      total_payload_bytes,
      sec,
      gbps,
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads);
  return absl::OkStatus();
}

absl::Status pwrite_fully(int fd, uint64_t off, const void* src, size_t bytes) {
  const char* ptr = static_cast<const char*>(src);
  size_t total = 0;
  while (total < bytes) {
    const ssize_t n = ::pwrite(fd, ptr + total, bytes - total, static_cast<off_t>(off + total));
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return absl::ErrnoToStatus(errno, "pwrite failed");
    }
    if (n == 0) {
      return absl::InternalError("pwrite returned 0");
    }
    total += static_cast<size_t>(n);
  }
  return absl::OkStatus();
}

absl::Status write_zero_pad(int fd, uint64_t off, uint64_t bytes) {
  if (bytes == 0) {
    return absl::OkStatus();
  }
  std::array<uint8_t, 4096> zeros{};
  uint64_t written = 0;
  while (written < bytes) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(bytes - written, zeros.size()));
    TC_RETURN_IF_ERROR(pwrite_fully(fd, off + written, zeros.data(), chunk));
    written += chunk;
  }
  return absl::OkStatus();
}

struct MaterializedMeta {
  uint64_t version = 1;
  int tp_world_size = 1;
  int tp_rank = 0;
  int device_id = 0;
  uint64_t output_bytes = 0;
  uint64_t output_bytes_aligned = 0;
  std::string safetensors_dir;
  std::string load_plan_json_path;
  std::string data_path;
};

absl::Status write_materialized_meta_json(const MaterializedMeta& m, const std::filesystem::path& path) {
  nlohmann::json j;
  j["version"] = m.version;
  j["tp_world_size"] = m.tp_world_size;
  j["tp_rank"] = m.tp_rank;
  j["device_id"] = m.device_id;
  j["output_bytes"] = m.output_bytes;
  j["output_bytes_aligned"] = m.output_bytes_aligned;
  j["safetensors_dir"] = m.safetensors_dir;
  j["load_plan_json_path"] = m.load_plan_json_path;
  j["data_path"] = m.data_path;
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open meta json for write: ", path.string()));
  }
  out << j.dump(2) << "\n";
  out.flush();
  if (!out.good()) {
    return absl::InternalError(absl::StrCat("Failed to write meta json: ", path.string()));
  }
  return absl::OkStatus();
}

absl::StatusOr<MaterializedMeta> read_materialized_meta_json(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open meta json: ", path.string()));
  }
  nlohmann::json j;
  in >> j;
  MaterializedMeta m;
  m.version = j.value("version", 0);
  m.tp_world_size = j.value("tp_world_size", 1);
  m.tp_rank = j.value("tp_rank", 0);
  m.device_id = j.value("device_id", 0);
  m.output_bytes = j.value("output_bytes", 0ull);
  m.output_bytes_aligned = j.value("output_bytes_aligned", 0ull);
  m.safetensors_dir = j.value("safetensors_dir", "");
  m.load_plan_json_path = j.value("load_plan_json_path", "");
  m.data_path = j.value("data_path", "");
  if (m.version != 1) {
    return absl::InvalidArgumentError(absl::StrCat("Unsupported materialized meta version: ", m.version));
  }
  if (m.output_bytes == 0 || m.output_bytes_aligned == 0 || m.data_path.empty()) {
    return absl::InvalidArgumentError(
        "Invalid materialized meta json (missing output_bytes/output_bytes_aligned/data_path)");
  }
  return m;
}

absl::Status run_materialize_d(const LoaderConfig& cfg, const std::vector<std::filesystem::path>& shards) {
  if (cfg.load_plan_json_path.empty()) {
    return absl::InvalidArgumentError("--load_plan_json_path is required for mode=materialize_d");
  }
  if (cfg.tp_world_size <= 0) {
    return absl::InvalidArgumentError("--tp_world_size must be > 0 for mode=materialize_d");
  }
  if (cfg.tp_rank < 0 || cfg.tp_rank >= cfg.tp_world_size) {
    return absl::InvalidArgumentError("--tp_rank must be within [0, tp_world_size) for mode=materialize_d");
  }
  if (cfg.safetensors_dir.empty()) {
    return absl::InvalidArgumentError("--safetensors_dir is required for mode=materialize_d");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError("--use_pinned_host_buffer=true is required for mode=materialize_d");
  }

  std::filesystem::path out_dir = cfg.materialized_dir;
  if (out_dir.empty()) {
    out_dir = cfg.safetensors_dir / "tensorcast_materialized";
  }
  out_dir /= std::format("tp{}", cfg.tp_world_size);
  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to create materialized_dir: ", out_dir.string()));
  }

  const std::filesystem::path data_path = out_dir / std::format("rank{}.bin", cfg.tp_rank);
  const std::filesystem::path meta_path = out_dir / std::format("rank{}.meta.json", cfg.tp_rank);

  // Materialize by running Strategy C once (reads from safetensors + pack into final layout).
  auto out_or = run_strategy_c_with_state(cfg, shards);
  if (!out_or.ok()) {
    return out_or.status();
  }
  const RunResult& r = out_or->first;
  const StrategyCState& s = out_or->second;
  if (s.vmm.base_ptr() == nullptr || s.output_bytes == 0) {
    return absl::InternalError("materialize_d: strategy C produced empty output");
  }
  if (r.output_bytes != s.output_bytes) {
    return absl::InternalError("materialize_d: inconsistent output_bytes between RunResult and state");
  }

  constexpr uint64_t kAlign = 512;
  const uint64_t out_bytes = s.output_bytes;
  const uint64_t out_bytes_aligned = ((out_bytes + kAlign - 1) / kAlign) * kAlign;

  // Stream D2H into pinned chunks and persist to disk.
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  auto pinned_pool =
      std::make_shared<common::memory::PinnedBufferPool>(static_cast<size_t>(bb.total_bytes), bb.chunk_bytes);
  const auto slabs = pinned_pool->list_slabs();
  if (slabs.empty()) {
    return absl::InternalError("materialize_d: no pinned slabs");
  }
  void* host_buf = slabs[0].base.get();
  const size_t host_cap = slabs[0].bytes;
  if (host_cap == 0) {
    return absl::InternalError("materialize_d: pinned slab has zero bytes");
  }

  const int fd = ::open(data_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("Failed to open output file: ", data_path.string()));
  }
  auto close_fd = [&]() { ::close(fd); };

  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  cudaStream_t stream = nullptr;
  absl::Status st = tensorcast::cuda::stream_create_with_flags(&stream, cudaStreamNonBlocking);
  if (!st.ok()) {
    close_fd();
    return st;
  }

  const uint8_t* src = static_cast<const uint8_t*>(s.vmm.base_ptr());
  uint64_t off = 0;
  const absl::Time t0 = absl::Now();
  while (off < out_bytes) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(out_bytes - off, host_cap));
    st = tensorcast::cuda::memcpy_async(host_buf, src + off, chunk, cudaMemcpyDeviceToHost, stream);
    if (!st.ok()) {
      break;
    }
    st = tensorcast::cuda::stream_synchronize(stream);
    if (!st.ok()) {
      break;
    }
    st = pwrite_fully(fd, off, host_buf, chunk);
    if (!st.ok()) {
      break;
    }
    off += static_cast<uint64_t>(chunk);
  }
  if (st.ok() && out_bytes_aligned > out_bytes) {
    st = write_zero_pad(fd, out_bytes, out_bytes_aligned - out_bytes);
  }
  if (st.ok()) {
    if (::fsync(fd) != 0) {
      st = absl::ErrnoToStatus(errno, "fsync failed");
    }
  }
  (void)tensorcast::cuda::stream_destroy(stream);
  close_fd();
  if (!st.ok()) {
    return st;
  }

  MaterializedMeta meta;
  meta.tp_world_size = cfg.tp_world_size;
  meta.tp_rank = cfg.tp_rank;
  meta.device_id = cfg.device_id;
  meta.output_bytes = out_bytes;
  meta.output_bytes_aligned = out_bytes_aligned;
  meta.safetensors_dir = cfg.safetensors_dir.string();
  meta.load_plan_json_path = cfg.load_plan_json_path.string();
  meta.data_path = data_path.string();
  TC_RETURN_IF_ERROR(write_materialized_meta_json(meta, meta_path));

  const double sec = seconds_since(t0);
  const double gib = static_cast<double>(out_bytes) / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << std::format(
      "materialize_d tp=(rank={}/{}) output_bytes={} aligned_bytes={} sec={:.6f} D2H+write_GiB/s={:.3f} data_path={} meta_path={}",
      cfg.tp_rank,
      cfg.tp_world_size,
      out_bytes,
      out_bytes_aligned,
      sec,
      gib / std::max(1e-9, sec),
      data_path.string(),
      meta_path.string());

  // Print the underlying Strategy C run result too (one-time cost).
  TC_RETURN_IF_ERROR(log_run_result(cfg, r));
  return absl::OkStatus();
}

absl::Status run_materialized_disk_baseline(const LoaderConfig& cfg) {
  if (cfg.materialized_meta_path.empty()) {
    return absl::InvalidArgumentError("--materialized_meta_path is required for mode=materialized_disk_baseline");
  }
  MaterializedMeta meta;
  TC_ASSIGN_OR_RETURN(meta, read_materialized_meta_json(cfg.materialized_meta_path));
  if (meta.device_id != cfg.device_id) {
    LOG(WARNING) << std::format(
        "materialized_disk_baseline: meta.device_id={} differs from --device_id={} (continuing)",
        meta.device_id,
        cfg.device_id);
  }
  LoaderConfig tmp = cfg;
  tmp.disk_bench_path = meta.data_path;
  tmp.disk_bench_bytes = meta.output_bytes;
  // Run disk_baseline (respects --disk_io_mode, bounce buffer flags, etc.).
  return run_disk_baseline(tmp);
}

absl::Status run_h2d_baseline(const LoaderConfig& cfg) {
  if (cfg.h2d_bench_bytes == 0) {
    return absl::InvalidArgumentError("--h2d_bench_bytes must be > 0 for mode=h2d_baseline");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError("--use_pinned_host_buffer=true is required for mode=h2d_baseline");
  }

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  if (cfg.tp_world_size <= 0) {
    return absl::InvalidArgumentError("--tp_world_size must be > 0 for mode=h2d_baseline");
  }
  std::vector<int> device_ids;
  TC_ASSIGN_OR_RETURN(device_ids, build_tp_device_ids(cfg));

  struct HostPool {
    std::shared_ptr<common::memory::PinnedBufferPool> pool;
    const uint8_t* base = nullptr;
    size_t bytes = 0;
    size_t chunk_bytes = 0;
    int numa_node = -1;
  };

  std::vector<HostPool> host_pools;
  host_pools.reserve(cfg.h2d_per_gpu_pinned_pool ? static_cast<size_t>(cfg.tp_world_size) : 1u);

  auto build_one_pool = [&](int device_id) -> absl::StatusOr<HostPool> {
    common::memory::PinnedBufferPool::Options pool_opts;
    pool_opts.name = "h2d_baseline";
    pool_opts.prefault = cfg.pinned_numa_prefault;
    TC_ASSIGN_OR_RETURN(pool_opts.numa_node, resolve_pinned_numa_node_for_device(cfg, device_id));

    auto pool = std::make_shared<common::memory::PinnedBufferPool>(
        static_cast<size_t>(bb.total_bytes), bb.chunk_bytes, std::move(pool_opts));
    const auto slabs = pool->list_slabs();
    if (slabs.empty()) {
      return absl::InternalError("h2d_baseline: no pinned slabs");
    }
    for (const auto& slab : slabs) {
      std::memset(slab.base.get(), 0, slab.bytes);
    }
    const size_t chunk_bytes = pool->slice_bytes();
    const uint8_t* host_base = reinterpret_cast<const uint8_t*>(slabs[0].base.get());
    const size_t host_bytes = slabs[0].bytes;
    if (host_bytes < chunk_bytes) {
      return absl::InternalError("h2d_baseline: pinned host slab smaller than one chunk");
    }
    return HostPool{
        .pool = std::move(pool),
        .base = host_base,
        .bytes = host_bytes,
        .chunk_bytes = chunk_bytes,
        .numa_node = pool_opts.numa_node,
    };
  };

  if (cfg.pinned_numa_node == -2 && !cfg.h2d_per_gpu_pinned_pool && cfg.tp_world_size > 1) {
    return absl::InvalidArgumentError(
        "--pinned_numa_node=-2 (auto) requires --h2d_per_gpu_pinned_pool=true when tp_world_size>1");
  }
  if (cfg.h2d_per_gpu_pinned_pool) {
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      HostPool host;
      TC_ASSIGN_OR_RETURN(host, build_one_pool(device_ids[static_cast<size_t>(rank)]));
      host_pools.push_back(std::move(host));
    }
  } else {
    HostPool host;
    TC_ASSIGN_OR_RETURN(host, build_one_pool(/*device_id=*/cfg.device_id));
    host_pools.push_back(std::move(host));
  }

  const uint64_t bytes_per_gpu = cfg.h2d_bench_bytes;

  struct PerGpuCtx {
    int device_id = -1;
    std::unique_ptr<common::memory::GpuDeviceMemory> dst;
    cudaStream_t stream = nullptr;
    cudaEvent_t done_event = nullptr;
    double sec = 0.0; // completion time since t0
  };

  std::vector<PerGpuCtx> gpus(static_cast<size_t>(cfg.tp_world_size));

  for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
    auto& ctx = gpus[static_cast<size_t>(rank)];
    ctx.device_id = device_ids[static_cast<size_t>(rank)];
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
    ctx.dst = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(ctx.dst->allocate(static_cast<size_t>(bytes_per_gpu), ctx.device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&ctx.stream, cudaStreamNonBlocking));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&ctx.done_event, cudaEventDisableTiming));
  }

  const absl::Time t0 = absl::Now();
  uint64_t copied = 0;
  while (copied < bytes_per_gpu) {
    const size_t bytes = static_cast<size_t>(std::min<uint64_t>(bytes_per_gpu - copied, host_pools[0].chunk_bytes));
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      auto& ctx = gpus[static_cast<size_t>(rank)];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
      const HostPool& host = host_pools[cfg.h2d_per_gpu_pinned_pool ? static_cast<size_t>(rank) : 0u];
      const uint64_t host_off = (host.bytes == bytes)
          ? 0
          : ((copied + static_cast<uint64_t>(rank) * 4096ull) % static_cast<uint64_t>(host.bytes - bytes));
      const void* src = host.base + host_off;
      void* dst_ptr = static_cast<uint8_t*>(ctx.dst->get()) + copied;
      TC_RETURN_IF_ERROR(tensorcast::cuda::memcpy_async(dst_ptr, src, bytes, cudaMemcpyHostToDevice, ctx.stream));
    }
    copied += static_cast<uint64_t>(bytes);
  }
  for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
    auto& ctx = gpus[static_cast<size_t>(rank)];
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ctx.done_event, ctx.stream));
  }

  std::vector<bool> done(static_cast<size_t>(cfg.tp_world_size), false);
  int remaining = cfg.tp_world_size;
  while (remaining > 0) {
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      if (done[static_cast<size_t>(rank)]) {
        continue;
      }
      auto& ctx = gpus[static_cast<size_t>(rank)];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
      bool ready = false;
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_query(ctx.done_event, &ready));
      if (!ready) {
        continue;
      }
      done[static_cast<size_t>(rank)] = true;
      remaining -= 1;
      ctx.sec = seconds_since(t0);
    }
    if (remaining > 0) {
      absl::SleepFor(absl::Milliseconds(1));
    }
  }

  double makespan = 0.0;
  for (const auto& ctx : gpus) {
    makespan = std::max(makespan, ctx.sec);
  }

  for (auto& ctx : gpus) {
    if (ctx.device_id >= 0) {
      (void)tensorcast::cuda::set_device(ctx.device_id);
    }
    if (ctx.stream != nullptr) {
      (void)tensorcast::cuda::stream_synchronize(ctx.stream);
    }
    if (ctx.done_event != nullptr) {
      (void)tensorcast::cuda::event_destroy(ctx.done_event);
      ctx.done_event = nullptr;
    }
    if (ctx.stream != nullptr) {
      (void)tensorcast::cuda::stream_destroy(ctx.stream);
      ctx.stream = nullptr;
    }
    ctx.dst.reset();
  }

  const double per_gpu_gib = static_cast<double>(bytes_per_gpu) / (1024.0 * 1024.0 * 1024.0);
  const double agg_gib = per_gpu_gib * static_cast<double>(cfg.tp_world_size);
  const double agg_gib_s = agg_gib / std::max(1e-9, makespan);
  std::string per_gpu_str;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&per_gpu_str, " ");
    }
    const auto& ctx = gpus[i];
    const double gib_s = per_gpu_gib / std::max(1e-9, ctx.sec);
    absl::StrAppend(&per_gpu_str, std::format("gpu{}={:.3f}GiB/s", ctx.device_id, gib_s));
  }
  std::string host_pool_str;
  for (size_t i = 0; i < host_pools.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&host_pool_str, " ");
    }
    absl::StrAppend(&host_pool_str, std::format("pool{}=numa{}", i, host_pools[i].numa_node));
  }
  LOG(INFO) << std::format(
      "h2d_baseline tp_world_size={} tp_devices={} bytes_per_gpu={} makespan_sec={:.6f} agg_GiB/s={:.3f} per_gpu=[{}] pinned=true bbuf_size_kb={} buffer_chunks={} chunk_bytes={} pinned_numa_node={} pinned_numa_prefault={} h2d_per_gpu_pinned_pool={} host_pools=[{}]",
      cfg.tp_world_size,
      join_device_ids(device_ids),
      static_cast<uint64_t>(bytes_per_gpu),
      makespan,
      agg_gib_s,
      per_gpu_str,
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      cfg.pinned_numa_node,
      cfg.pinned_numa_prefault,
      cfg.h2d_per_gpu_pinned_pool,
      host_pool_str);
  return absl::OkStatus();
}

absl::Status run_h2d_2d_baseline(const LoaderConfig& cfg) {
  if (cfg.h2d_bench_bytes == 0) {
    return absl::InvalidArgumentError("--h2d_bench_bytes must be > 0 for mode=h2d_2d_baseline");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError("--use_pinned_host_buffer=true is required for mode=h2d_2d_baseline");
  }
  if (cfg.tp_world_size <= 0) {
    return absl::InvalidArgumentError("--tp_world_size must be > 0 for mode=h2d_2d_baseline");
  }
  if (cfg.h2d_2d_width_bytes == 0) {
    return absl::InvalidArgumentError("--h2d_2d_width_bytes must be > 0 for mode=h2d_2d_baseline");
  }
  if (cfg.h2d_2d_height == 0) {
    return absl::InvalidArgumentError("--h2d_2d_height must be > 0 for mode=h2d_2d_baseline");
  }
  if (cfg.h2d_2d_src_pitch_bytes < cfg.h2d_2d_width_bytes) {
    return absl::InvalidArgumentError("--h2d_2d_src_pitch_bytes must be >= --h2d_2d_width_bytes");
  }
  if (cfg.h2d_2d_dst_pitch_bytes < cfg.h2d_2d_width_bytes) {
    return absl::InvalidArgumentError("--h2d_2d_dst_pitch_bytes must be >= --h2d_2d_width_bytes");
  }

  const uint64_t width_bytes = cfg.h2d_2d_width_bytes;
  const uint64_t height = cfg.h2d_2d_height;
  const uint64_t src_pitch = cfg.h2d_2d_src_pitch_bytes;
  const uint64_t dst_pitch = cfg.h2d_2d_dst_pitch_bytes;

  if (height > (std::numeric_limits<uint64_t>::max() / width_bytes)) {
    return absl::InvalidArgumentError("h2d_2d_baseline: width_bytes * height overflow");
  }
  if (height > (std::numeric_limits<uint64_t>::max() / src_pitch)) {
    return absl::InvalidArgumentError("h2d_2d_baseline: src_pitch * height overflow");
  }
  if (height > (std::numeric_limits<uint64_t>::max() / dst_pitch)) {
    return absl::InvalidArgumentError("h2d_2d_baseline: dst_pitch * height overflow");
  }
  const uint64_t bytes_per_call = width_bytes * height;
  const uint64_t src_bytes = src_pitch * height;
  const uint64_t dst_bytes = dst_pitch * height;

  if (bytes_per_call == 0 || src_bytes == 0 || dst_bytes == 0) {
    return absl::InvalidArgumentError("h2d_2d_baseline: computed bytes are zero");
  }
  if (src_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      dst_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      src_pitch > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      dst_pitch > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      width_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::InvalidArgumentError("h2d_2d_baseline: size_t overflow");
  }

  std::vector<int> device_ids;
  TC_ASSIGN_OR_RETURN(device_ids, build_tp_device_ids(cfg));

  struct HostPool {
    std::shared_ptr<common::memory::PinnedBufferPool> pool;
    const uint8_t* base = nullptr;
    size_t bytes = 0;
    int numa_node = -1;
  };

  std::vector<HostPool> host_pools;
  host_pools.reserve(cfg.h2d_per_gpu_pinned_pool ? static_cast<size_t>(cfg.tp_world_size) : 1u);

  const uint64_t src_bytes_per_rank = src_bytes;
  uint64_t src_pool_bytes_u64 = src_bytes_per_rank;
  if (!cfg.h2d_per_gpu_pinned_pool) {
    const uint64_t tp_u64 = static_cast<uint64_t>(cfg.tp_world_size);
    if (tp_u64 > 0 && src_bytes_per_rank > (std::numeric_limits<uint64_t>::max() / tp_u64)) {
      return absl::InvalidArgumentError("h2d_2d_baseline: src_bytes * tp_world_size overflow");
    }
    src_pool_bytes_u64 = src_bytes_per_rank * tp_u64;
  }
  if (src_pool_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return absl::InvalidArgumentError("h2d_2d_baseline: src_pool_bytes overflows size_t");
  }
  const size_t src_pool_bytes = static_cast<size_t>(src_pool_bytes_u64);

  auto build_one_pool = [&](int device_id) -> absl::StatusOr<HostPool> {
    common::memory::PinnedBufferPool::Options pool_opts;
    pool_opts.name = "h2d_2d_baseline";
    pool_opts.prefault = cfg.pinned_numa_prefault;
    TC_ASSIGN_OR_RETURN(pool_opts.numa_node, resolve_pinned_numa_node_for_device(cfg, device_id));

    auto pool = std::make_shared<common::memory::PinnedBufferPool>(
        src_pool_bytes, static_cast<size_t>(src_bytes_per_rank), std::move(pool_opts));
    const auto slabs = pool->list_slabs();
    if (slabs.empty()) {
      return absl::InternalError("h2d_2d_baseline: no pinned slabs");
    }
    for (const auto& slab : slabs) {
      std::memset(slab.base.get(), 0, slab.bytes);
    }
    const uint8_t* host_base = reinterpret_cast<const uint8_t*>(slabs[0].base.get());
    const size_t host_bytes = slabs[0].bytes;
    if (host_bytes < src_pool_bytes) {
      return absl::InternalError("h2d_2d_baseline: pinned host slab smaller than src_pool_bytes");
    }
    return HostPool{
        .pool = std::move(pool),
        .base = host_base,
        .bytes = host_bytes,
        .numa_node = pool_opts.numa_node,
    };
  };

  if (cfg.pinned_numa_node == -2 && !cfg.h2d_per_gpu_pinned_pool && cfg.tp_world_size > 1) {
    return absl::InvalidArgumentError(
        "--pinned_numa_node=-2 (auto) requires --h2d_per_gpu_pinned_pool=true when tp_world_size>1");
  }
  if (cfg.h2d_per_gpu_pinned_pool) {
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      HostPool host;
      TC_ASSIGN_OR_RETURN(host, build_one_pool(device_ids[static_cast<size_t>(rank)]));
      host_pools.push_back(std::move(host));
    }
  } else {
    HostPool host;
    TC_ASSIGN_OR_RETURN(host, build_one_pool(/*device_id=*/cfg.device_id));
    host_pools.push_back(std::move(host));
  }

  const uint64_t target_bytes_per_gpu = cfg.h2d_bench_bytes;
  const uint64_t iters = std::max<uint64_t>(1, (target_bytes_per_gpu + bytes_per_call - 1) / bytes_per_call);
  if (iters > (std::numeric_limits<uint64_t>::max() / bytes_per_call)) {
    return absl::InvalidArgumentError("h2d_2d_baseline: bytes_per_call * iters overflow");
  }
  const uint64_t bytes_per_gpu = bytes_per_call * iters;

  struct PerGpuCtx {
    int device_id = -1;
    std::unique_ptr<common::memory::GpuDeviceMemory> dst;
    cudaStream_t stream = nullptr;
    cudaEvent_t done_event = nullptr;
    double sec = 0.0; // completion time since t0
  };

  std::vector<PerGpuCtx> gpus(static_cast<size_t>(cfg.tp_world_size));
  for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
    auto& ctx = gpus[static_cast<size_t>(rank)];
    ctx.device_id = device_ids[static_cast<size_t>(rank)];
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
    ctx.dst = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(ctx.dst->allocate(static_cast<size_t>(dst_bytes), ctx.device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create_with_flags(&ctx.stream, cudaStreamNonBlocking));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_create_with_flags(&ctx.done_event, cudaEventDisableTiming));
  }

  const absl::Time t0 = absl::Now();
  for (uint64_t iter = 0; iter < iters; ++iter) {
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      auto& ctx = gpus[static_cast<size_t>(rank)];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
      const HostPool& host = host_pools[cfg.h2d_per_gpu_pinned_pool ? static_cast<size_t>(rank) : 0u];
      const uint64_t rank_off_u64 =
          cfg.h2d_per_gpu_pinned_pool ? 0ull : (static_cast<uint64_t>(rank) * src_bytes_per_rank);
      if (!cfg.h2d_per_gpu_pinned_pool) {
        const uint64_t host_bytes_u64 = static_cast<uint64_t>(host.bytes);
        if (rank_off_u64 > host_bytes_u64 || (host_bytes_u64 - rank_off_u64) < src_bytes_per_rank) {
          return absl::InternalError("h2d_2d_baseline: computed src offset exceeds host slab size");
        }
      }
      const void* src = host.base + static_cast<size_t>(rank_off_u64);
      void* dst_ptr = ctx.dst->get();
      SC_RETURN_IF_CUDA_ERROR(cudaMemcpy2DAsync(
          dst_ptr,
          static_cast<size_t>(dst_pitch),
          src,
          static_cast<size_t>(src_pitch),
          static_cast<size_t>(width_bytes),
          static_cast<size_t>(height),
          cudaMemcpyHostToDevice,
          ctx.stream));
    }
  }
  for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
    auto& ctx = gpus[static_cast<size_t>(rank)];
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
    TC_RETURN_IF_ERROR(tensorcast::cuda::event_record(ctx.done_event, ctx.stream));
  }

  std::vector<bool> done(static_cast<size_t>(cfg.tp_world_size), false);
  int remaining = cfg.tp_world_size;
  while (remaining > 0) {
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      if (done[static_cast<size_t>(rank)]) {
        continue;
      }
      auto& ctx = gpus[static_cast<size_t>(rank)];
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(ctx.device_id));
      bool ready = false;
      TC_RETURN_IF_ERROR(tensorcast::cuda::event_query(ctx.done_event, &ready));
      if (!ready) {
        continue;
      }
      done[static_cast<size_t>(rank)] = true;
      remaining -= 1;
      ctx.sec = seconds_since(t0);
    }
    if (remaining > 0) {
      absl::SleepFor(absl::Milliseconds(1));
    }
  }

  double makespan = 0.0;
  for (const auto& ctx : gpus) {
    makespan = std::max(makespan, ctx.sec);
  }

  for (auto& ctx : gpus) {
    if (ctx.device_id >= 0) {
      (void)tensorcast::cuda::set_device(ctx.device_id);
    }
    if (ctx.stream != nullptr) {
      (void)tensorcast::cuda::stream_synchronize(ctx.stream);
    }
    if (ctx.done_event != nullptr) {
      (void)tensorcast::cuda::event_destroy(ctx.done_event);
      ctx.done_event = nullptr;
    }
    if (ctx.stream != nullptr) {
      (void)tensorcast::cuda::stream_destroy(ctx.stream);
      ctx.stream = nullptr;
    }
    ctx.dst.reset();
  }

  const double per_gpu_gib = static_cast<double>(bytes_per_gpu) / (1024.0 * 1024.0 * 1024.0);
  const double agg_gib = per_gpu_gib * static_cast<double>(cfg.tp_world_size);
  const double agg_gib_s = agg_gib / std::max(1e-9, makespan);
  std::string per_gpu_str;
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&per_gpu_str, " ");
    }
    const auto& ctx = gpus[i];
    const double gib_s = per_gpu_gib / std::max(1e-9, ctx.sec);
    absl::StrAppend(&per_gpu_str, std::format("gpu{}={:.3f}GiB/s", ctx.device_id, gib_s));
  }
  std::string host_pool_str;
  for (size_t i = 0; i < host_pools.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&host_pool_str, " ");
    }
    absl::StrAppend(&host_pool_str, std::format("pool{}=numa{}", i, host_pools[i].numa_node));
  }

  LOG(INFO) << std::format(
      "h2d_2d_baseline tp_world_size={} tp_devices={} target_bytes_per_gpu={} bytes_per_gpu={} iters={} bytes_per_call={} width_bytes={} height={} src_pitch={} dst_pitch={} makespan_sec={:.6f} agg_GiB/s={:.3f} per_gpu=[{}] pinned=true pinned_numa_node={} pinned_numa_prefault={} h2d_per_gpu_pinned_pool={} host_pools=[{}] src_bytes={} dst_bytes={}",
      cfg.tp_world_size,
      join_device_ids(device_ids),
      static_cast<uint64_t>(target_bytes_per_gpu),
      static_cast<uint64_t>(bytes_per_gpu),
      static_cast<uint64_t>(iters),
      static_cast<uint64_t>(bytes_per_call),
      static_cast<uint64_t>(width_bytes),
      static_cast<uint64_t>(height),
      static_cast<uint64_t>(src_pitch),
      static_cast<uint64_t>(dst_pitch),
      makespan,
      agg_gib_s,
      per_gpu_str,
      cfg.pinned_numa_node,
      cfg.pinned_numa_prefault,
      cfg.h2d_per_gpu_pinned_pool,
      host_pool_str,
      static_cast<uint64_t>(src_bytes),
      static_cast<uint64_t>(dst_bytes));

  return absl::OkStatus();
}

class OdirectMultiFileSource final : public SeekableSource {
 public:
  explicit OdirectMultiFileSource(std::vector<std::filesystem::path> paths) : paths_(std::move(paths)) {}

  ~OdirectMultiFileSource() override {
    absl::MutexLock lock(&mu_);
    for (auto& s : segments_) {
      if (s.fd >= 0) {
        ::close(s.fd);
        s.fd = -1;
      }
    }
  }

  [[nodiscard]] uint64_t total_bytes() const override {
    auto* self = const_cast<OdirectMultiFileSource*>(this);
    if (auto st = self->open_files(); !st.ok()) {
      LOG(WARNING) << "OdirectMultiFileSource: open_files failed in total_bytes: " << st;
      return 0;
    }
    absl::MutexLock lock(&mu_);
    return total_size_aligned_;
  }

  absl::Status open_files() {
    absl::MutexLock lock(&mu_);
    if (initialized_) {
      return absl::OkStatus();
    }
    if (paths_.empty()) {
      return absl::InvalidArgumentError("OdirectMultiFileSource: paths is empty");
    }

    std::ranges::sort(paths_, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });
    segments_.clear();
    segments_.reserve(paths_.size());

    constexpr uint64_t kAlign = common::memory::PinnedBufferPool::kDirectIOAlignment;
    uint64_t base = 0;
    uint64_t skipped_tail = 0;
    for (const auto& p : paths_) {
      int fd = ::open(p.c_str(), O_RDONLY | O_DIRECT);
      if (fd < 0) {
        return absl::ErrnoToStatus(errno, absl::StrCat("O_DIRECT open failed for ", p.string()));
      }
      struct stat st{};
      if (::fstat(fd, &st) != 0) {
        const int err = errno;
        ::close(fd);
        return absl::ErrnoToStatus(err, absl::StrCat("fstat failed for ", p.string()));
      }
      if (st.st_size < 0) {
        ::close(fd);
        return absl::InvalidArgumentError(absl::StrCat("Negative file size for ", p.string()));
      }
      const uint64_t sz = static_cast<uint64_t>(st.st_size);
      const uint64_t aligned = (sz / kAlign) * kAlign;
      skipped_tail += (sz - aligned);
      segments_.push_back(
          Segment{
              .path = p.string(),
              .fd = fd,
              .base_offset = base,
              .size_aligned = aligned,
          });
      base += aligned;
    }
    total_size_aligned_ = base;
    skipped_tail_bytes_ = skipped_tail;
    initialized_ = true;
    return absl::OkStatus();
  }

  [[nodiscard]] uint64_t total_size_aligned() const {
    absl::MutexLock lock(&mu_);
    return total_size_aligned_;
  }

  [[nodiscard]] uint64_t skipped_tail_bytes() const {
    absl::MutexLock lock(&mu_);
    return skipped_tail_bytes_;
  }

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    TC_RETURN_IF_ERROR(open_files());
    absl::MutexLock lock(&offset_mu_);
    const uint64_t off = current_offset_;
    if (off >= total_size_aligned_) {
      return static_cast<size_t>(0);
    }
    const uint64_t remain = total_size_aligned_ - off;
    const size_t want = static_cast<size_t>(std::min<uint64_t>(remain, max_bytes));
    size_t got = 0;
    TC_ASSIGN_OR_RETURN(got, read_at(off, dst, want));
    current_offset_ += got;
    return got;
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    TC_RETURN_IF_ERROR(open_files());
    if (bytes == 0) {
      return static_cast<size_t>(0);
    }
    constexpr uint64_t kAlign = common::memory::PinnedBufferPool::kDirectIOAlignment;
    if ((reinterpret_cast<uintptr_t>(dst) % kAlign) != 0) {
      return absl::InvalidArgumentError("OdirectMultiFileSource: dst pointer must be 512B aligned");
    }
    if ((offset % kAlign) != 0 || (static_cast<uint64_t>(bytes) % kAlign) != 0) {
      return absl::InvalidArgumentError("OdirectMultiFileSource: offset/bytes must be 512B aligned");
    }
    if (offset >= total_size_aligned_) {
      return static_cast<size_t>(0);
    }
    const uint64_t remain = total_size_aligned_ - offset;
    const size_t to_read = static_cast<size_t>(std::min<uint64_t>(remain, bytes));
    char* out = static_cast<char*>(dst);

    size_t total = 0;
    uint64_t off = offset;
    while (total < to_read) {
      auto it =
          std::ranges::upper_bound(segments_, off, {}, [](const Segment& s) { return s.base_offset + s.size_aligned; });
      if (it == segments_.end()) {
        break;
      }
      if (it == segments_.begin() && off < it->base_offset) {
        break;
      }
      const Segment* seg = &(*it);
      if (seg == nullptr) {
        break;
      }
      const uint64_t within = off - seg->base_offset;
      const uint64_t seg_remain = seg->size_aligned - within;
      const size_t want = static_cast<size_t>(std::min<uint64_t>(seg_remain, to_read - total));
      // O_DIRECT read: rely on the caller to provide aligned dst pointer/len.
      size_t got = 0;
      TC_ASSIGN_OR_RETURN(got, pread_fully(seg->fd, within, out + total, want));
      if (got == 0) {
        break;
      }
      total += got;
      off += got;
    }
    return total;
  }

 private:
  struct Segment {
    std::string path;
    int fd = -1;
    uint64_t base_offset = 0;
    uint64_t size_aligned = 0;
  };

  mutable absl::Mutex mu_;
  std::vector<std::filesystem::path> paths_;
  std::vector<Segment> segments_;
  bool initialized_ = false;
  uint64_t total_size_aligned_ = 0;
  uint64_t skipped_tail_bytes_ = 0;

  absl::Mutex offset_mu_;
  uint64_t current_offset_ ABSL_GUARDED_BY(offset_mu_) = 0;
};

absl::Status run_safetensors_o_direct_host_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_o_direct_host_baseline: shards is empty");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError(
        "--use_pinned_host_buffer=true is required for safetensors_o_direct_host_baseline");
  }

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));

  OdirectMultiFileSource src(shards);
  TC_RETURN_IF_ERROR(src.open_files());
  const uint64_t total_bytes = src.total_size_aligned();
  const uint64_t skipped_tail = src.skipped_tail_bytes();

  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  NullPositionedSink sink(total_bytes);
  const auto t0 = absl::Now();
  auto ranges = split_even_ranges(/*base=*/0, total_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);

  LOG(INFO) << std::format(
      "safetensors_o_direct_host_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned=true bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} skipped_tail_bytes={} bytes_written={}",
      total_bytes,
      sec,
      gbps,
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      skipped_tail,
      sink.bytes_written());
  return absl::OkStatus();
}

absl::Status run_safetensors_o_direct_disk_baseline(
    const LoaderConfig& cfg,
    const std::vector<std::filesystem::path>& shards) {
  if (shards.empty()) {
    return absl::InvalidArgumentError("run_safetensors_o_direct_disk_baseline: shards is empty");
  }
  if (!cfg.use_pinned_host_buffer) {
    return absl::InvalidArgumentError(
        "--use_pinned_host_buffer=true is required for safetensors_o_direct_disk_baseline");
  }

  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));

  OdirectMultiFileSource src(shards);
  TC_RETURN_IF_ERROR(src.open_files());
  const uint64_t total_bytes = src.total_size_aligned();
  const uint64_t skipped_tail = src.skipped_tail_bytes();

  common::memory::GpuDeviceMemory dst;
  TC_RETURN_IF_ERROR(dst.allocate(static_cast<size_t>(total_bytes), cfg.device_id));
  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{dst.get()},
          .total_size = total_bytes,
          .chunk_size = 128 * 1024 * 1024,
          .device_id = cfg.device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = cfg.gpu_sched_enabled,
          .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
          .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
      });

  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);
  const auto t0 = absl::Now();
  auto ranges = split_even_ranges(/*base=*/0, total_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(src, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  const uint64_t sched_waits =
      ((sched_after.waits >= sched_before.waits) ? (sched_after.waits - sched_before.waits) : 0);
  const double sched_wait_sec =
      (sched_after.wait_sec >= sched_before.wait_sec) ? (sched_after.wait_sec - sched_before.wait_sec) : 0.0;

  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);
  LOG(INFO) << std::format(
      "safetensors_o_direct_disk_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned=true bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} skipped_tail_bytes={} gpu_sched_waits={} gpu_sched_wait_sec={:.6f}",
      total_bytes,
      sec,
      gbps,
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      skipped_tail,
      sched_waits,
      sched_wait_sec);
  return absl::OkStatus();
}

absl::Status run_nccl_baseline(const LoaderConfig& cfg) {
  if (cfg.tp_world_size <= 1) {
    return absl::InvalidArgumentError("--tp_world_size must be > 1 for nccl_baseline");
  }
  if (cfg.nccl_iters <= 0 || cfg.nccl_warmup < 0) {
    return absl::InvalidArgumentError("--nccl_iters must be > 0 and --nccl_warmup must be >= 0");
  }
  if (cfg.nccl_blocking_wait) {
    (void)::setenv("NCCL_BLOCKING_WAIT", "1", /*overwrite=*/1);
  }

  NcclOp op;
  TC_ASSIGN_OR_RETURN(op, parse_nccl_op(cfg.nccl_op));
  std::vector<uint64_t> sizes;
  if (!cfg.nccl_sizes.empty()) {
    TC_ASSIGN_OR_RETURN(sizes, parse_u64_csv(cfg.nccl_sizes, "--nccl_sizes"));
  }
  if (sizes.empty()) {
    TC_ASSIGN_OR_RETURN(sizes, build_power2_sweep(cfg.nccl_min_bytes, cfg.nccl_max_bytes));
  }
  if (sizes.empty()) {
    return absl::InvalidArgumentError("No NCCL sizes to benchmark");
  }

  std::vector<int> device_ids;
  TC_ASSIGN_OR_RETURN(device_ids, build_tp_device_ids(cfg));

  NcclClique clique;
  TC_ASSIGN_OR_RETURN(
      clique,
      NcclClique::create(
          NcclClique::Options{
              .device_ids = device_ids,
              .nccl_timeout_sec = cfg.nccl_timeout_sec,
          }));

  for (const uint64_t bytes : sizes) {
    if (bytes == 0) {
      continue;
    }
    const int world_size = clique.world_size();
    std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> send_bufs;
    send_bufs.reserve(static_cast<size_t>(world_size));
    std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> recv_bufs;
    if (op == NcclOp::kSendrecv) {
      recv_bufs.reserve(static_cast<size_t>(world_size));
    }

    for (int rank = 0; rank < world_size; ++rank) {
      auto send = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids[rank]));
      TC_RETURN_IF_ERROR(send->allocate(static_cast<size_t>(bytes), device_ids[rank]));
      send_bufs.push_back(std::move(send));

      if (op == NcclOp::kSendrecv) {
        auto recv = std::make_unique<common::memory::GpuDeviceMemory>();
        TC_RETURN_IF_ERROR(recv->allocate(static_cast<size_t>(bytes), device_ids[rank]));
        recv_bufs.push_back(std::move(recv));
      }
    }

    TC_RETURN_IF_ERROR(clique.barrier());

    for (int i = 0; i < cfg.nccl_warmup; ++i) {
      if (op == NcclOp::kBroadcast) {
        TC_RETURN_IF_ERROR(clique.group_start());
        for (int rank = 0; rank < world_size; ++rank) {
          TC_RETURN_IF_ERROR(clique.broadcast_u8(
              rank, send_bufs[rank]->get(), send_bufs[rank]->get(), static_cast<size_t>(bytes), /*root_rank=*/0));
        }
        TC_RETURN_IF_ERROR(
            clique.group_end_and_wait(absl::StrCat("nccl_baseline:broadcast bytes=", static_cast<uint64_t>(bytes))));
      } else {
        TC_RETURN_IF_ERROR(clique.group_start());
        for (int rank = 0; rank < world_size; ++rank) {
          const int send_peer = (rank + 1) % world_size;
          const int recv_peer = (rank + world_size - 1) % world_size;
          TC_RETURN_IF_ERROR(clique.send_u8(rank, send_bufs[rank]->get(), static_cast<size_t>(bytes), send_peer));
          TC_RETURN_IF_ERROR(clique.recv_u8(rank, recv_bufs[rank]->get(), static_cast<size_t>(bytes), recv_peer));
        }
        TC_RETURN_IF_ERROR(
            clique.group_end_and_wait(absl::StrCat("nccl_baseline:sendrecv bytes=", static_cast<uint64_t>(bytes))));
      }
    }

    TC_RETURN_IF_ERROR(clique.barrier());

    const absl::Time t0 = absl::Now();
    for (int i = 0; i < cfg.nccl_iters; ++i) {
      if (op == NcclOp::kBroadcast) {
        TC_RETURN_IF_ERROR(clique.group_start());
        for (int rank = 0; rank < world_size; ++rank) {
          TC_RETURN_IF_ERROR(clique.broadcast_u8(
              rank, send_bufs[rank]->get(), send_bufs[rank]->get(), static_cast<size_t>(bytes), /*root_rank=*/0));
        }
        TC_RETURN_IF_ERROR(
            clique.group_end_and_wait(absl::StrCat("nccl_baseline:broadcast bytes=", static_cast<uint64_t>(bytes))));
      } else {
        TC_RETURN_IF_ERROR(clique.group_start());
        for (int rank = 0; rank < world_size; ++rank) {
          const int send_peer = (rank + 1) % world_size;
          const int recv_peer = (rank + world_size - 1) % world_size;
          TC_RETURN_IF_ERROR(clique.send_u8(rank, send_bufs[rank]->get(), static_cast<size_t>(bytes), send_peer));
          TC_RETURN_IF_ERROR(clique.recv_u8(rank, recv_bufs[rank]->get(), static_cast<size_t>(bytes), recv_peer));
        }
        TC_RETURN_IF_ERROR(
            clique.group_end_and_wait(absl::StrCat("nccl_baseline:sendrecv bytes=", static_cast<uint64_t>(bytes))));
      }
    }
    const double sec = seconds_since(t0);
    const double avg_us = (sec / static_cast<double>(cfg.nccl_iters)) * 1e6;
    const double gib_s =
        (static_cast<double>(bytes) * static_cast<double>(cfg.nccl_iters) / (1024.0 * 1024.0 * 1024.0)) /
        std::max(1e-9, sec);

    LOG(INFO) << std::format(
        "nccl_baseline op={} tp_world_size={} bytes={} warmup={} iters={} avg_us={:.3f} GiB/s={:.3f} tp_devices={}",
        (op == NcclOp::kBroadcast) ? "broadcast" : "sendrecv",
        world_size,
        static_cast<uint64_t>(bytes),
        cfg.nccl_warmup,
        cfg.nccl_iters,
        avg_us,
        gib_s,
        join_device_ids(device_ids));
  }
  return absl::OkStatus();
}

absl::Status run_nccl_launch_tax(const LoaderConfig& cfg) {
  if (cfg.tp_world_size <= 1) {
    return absl::InvalidArgumentError("--tp_world_size must be > 1 for nccl_launch_tax");
  }
  if (cfg.nccl_overhead_iters <= 0 || cfg.nccl_warmup < 0) {
    return absl::InvalidArgumentError("--nccl_overhead_iters must be > 0 and --nccl_warmup must be >= 0");
  }
  if (cfg.nccl_blocking_wait) {
    (void)::setenv("NCCL_BLOCKING_WAIT", "1", /*overwrite=*/1);
  }

  NcclOp op;
  TC_ASSIGN_OR_RETURN(op, parse_nccl_op(cfg.nccl_op));

  constexpr uint64_t kBytes = 1;
  std::vector<int> device_ids;
  TC_ASSIGN_OR_RETURN(device_ids, build_tp_device_ids(cfg));

  NcclClique clique;
  TC_ASSIGN_OR_RETURN(
      clique,
      NcclClique::create(
          NcclClique::Options{
              .device_ids = device_ids,
              .nccl_timeout_sec = cfg.nccl_timeout_sec,
          }));

  const int world_size = clique.world_size();
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> send_bufs;
  send_bufs.reserve(static_cast<size_t>(world_size));
  std::vector<std::unique_ptr<common::memory::GpuDeviceMemory>> recv_bufs;
  if (op == NcclOp::kSendrecv) {
    recv_bufs.reserve(static_cast<size_t>(world_size));
  }
  for (int rank = 0; rank < world_size; ++rank) {
    auto send = std::make_unique<common::memory::GpuDeviceMemory>();
    TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(device_ids[rank]));
    TC_RETURN_IF_ERROR(send->allocate(static_cast<size_t>(kBytes), device_ids[rank]));
    send_bufs.push_back(std::move(send));

    if (op == NcclOp::kSendrecv) {
      auto recv = std::make_unique<common::memory::GpuDeviceMemory>();
      TC_RETURN_IF_ERROR(recv->allocate(static_cast<size_t>(kBytes), device_ids[rank]));
      recv_bufs.push_back(std::move(recv));
    }
  }

  TC_RETURN_IF_ERROR(clique.barrier());

  for (int i = 0; i < cfg.nccl_warmup; ++i) {
    if (op == NcclOp::kBroadcast) {
      TC_RETURN_IF_ERROR(clique.group_start());
      for (int rank = 0; rank < world_size; ++rank) {
        TC_RETURN_IF_ERROR(clique.broadcast_u8(
            rank, send_bufs[rank]->get(), send_bufs[rank]->get(), static_cast<size_t>(kBytes), /*root_rank=*/0));
      }
      TC_RETURN_IF_ERROR(clique.group_end_and_wait("nccl_launch_tax:warmup:broadcast"));
    } else {
      TC_RETURN_IF_ERROR(clique.group_start());
      for (int rank = 0; rank < world_size; ++rank) {
        const int send_peer = (rank + 1) % world_size;
        const int recv_peer = (rank + world_size - 1) % world_size;
        TC_RETURN_IF_ERROR(clique.send_u8(rank, send_bufs[rank]->get(), static_cast<size_t>(kBytes), send_peer));
        TC_RETURN_IF_ERROR(clique.recv_u8(rank, recv_bufs[rank]->get(), static_cast<size_t>(kBytes), recv_peer));
      }
      TC_RETURN_IF_ERROR(clique.group_end_and_wait("nccl_launch_tax:warmup:sendrecv"));
    }
  }

  TC_RETURN_IF_ERROR(clique.barrier());

  const absl::Time t0 = absl::Now();
  for (int i = 0; i < cfg.nccl_overhead_iters; ++i) {
    if (op == NcclOp::kBroadcast) {
      TC_RETURN_IF_ERROR(clique.group_start());
      for (int rank = 0; rank < world_size; ++rank) {
        TC_RETURN_IF_ERROR(clique.broadcast_u8(
            rank, send_bufs[rank]->get(), send_bufs[rank]->get(), static_cast<size_t>(kBytes), /*root_rank=*/0));
      }
      TC_RETURN_IF_ERROR(clique.group_end_and_wait("nccl_launch_tax:broadcast"));
    } else {
      TC_RETURN_IF_ERROR(clique.group_start());
      for (int rank = 0; rank < world_size; ++rank) {
        const int send_peer = (rank + 1) % world_size;
        const int recv_peer = (rank + world_size - 1) % world_size;
        TC_RETURN_IF_ERROR(clique.send_u8(rank, send_bufs[rank]->get(), static_cast<size_t>(kBytes), send_peer));
        TC_RETURN_IF_ERROR(clique.recv_u8(rank, recv_bufs[rank]->get(), static_cast<size_t>(kBytes), recv_peer));
      }
      TC_RETURN_IF_ERROR(clique.group_end_and_wait("nccl_launch_tax:sendrecv"));
    }
  }
  const double sec = seconds_since(t0);
  const double avg_us = (sec / static_cast<double>(cfg.nccl_overhead_iters)) * 1e6;
  LOG(INFO) << std::format(
      "nccl_launch_tax op={} tp_world_size={} bytes={} iters={} avg_us={:.3f} tp_devices={}",
      (op == NcclOp::kBroadcast) ? "broadcast" : "sendrecv",
      world_size,
      static_cast<uint64_t>(kBytes),
      cfg.nccl_overhead_iters,
      avg_us,
      join_device_ids(device_ids));
  return absl::OkStatus();
}

absl::Status run_disk_baseline(const LoaderConfig& cfg) {
  if (cfg.disk_bench_path.empty()) {
    return absl::InvalidArgumentError("--disk_bench_path is required for mode=disk_baseline");
  }
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  common::memory::GpuDeviceMemory dst;
  TC_RETURN_IF_ERROR(dst.allocate(static_cast<size_t>(cfg.disk_bench_bytes), cfg.device_id));
  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{dst.get()},
          .total_size = cfg.disk_bench_bytes,
          .chunk_size = 128 * 1024 * 1024,
          .device_id = cfg.device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = cfg.gpu_sched_enabled,
          .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
          .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
      });

  FilePartitionSource src(
      FilePartitionSource::Options{
          .partition_paths = {cfg.disk_bench_path},
          .partition_sizes = {static_cast<size_t>(cfg.disk_bench_bytes)},
          .total_size = cfg.disk_bench_bytes,
          .io_batch_bytes = bb.chunk_bytes,
          .direct_io_mode = cfg.disk_io_mode,
      });

  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);
  const auto t0 = absl::Now();
  RemappedSource remapped(
      gsl::not_null<SeekableSource*>{&src},
      {RemappedSource::Segment{
          .dst_offset = 0,
          .src_offset = 0,
          .end_offset = cfg.disk_bench_bytes,
      }});
  auto ranges = split_even_ranges(/*base=*/0, cfg.disk_bench_bytes, io_threads);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(remapped, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  const uint64_t sched_waits =
      ((sched_after.waits >= sched_before.waits) ? (sched_after.waits - sched_before.waits) : 0);
  const double sched_wait_sec =
      (sched_after.wait_sec >= sched_before.wait_sec) ? (sched_after.wait_sec - sched_before.wait_sec) : 0.0;
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(cfg.disk_bench_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);
  const char* direct_io = src.is_using_direct_io() ? "true" : "false";
  const uint64_t direct_io_fallback_errno = static_cast<uint64_t>(std::max(0, src.direct_io_fallback_errno()));
  const std::string direct_io_fallback_reason =
      src.direct_io_fallback_reason().empty() ? "none" : src.direct_io_fallback_reason();

  LOG(INFO) << std::format(
      "disk_baseline bytes={} sec={:.6f} GiB/s={:.3f} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} disk_io_mode={} direct_io={} direct_io_fallback_errno={} direct_io_fallback_reason={} gpu_sched_waits={} gpu_sched_wait_sec={:.6f}",
      static_cast<uint64_t>(cfg.disk_bench_bytes),
      sec,
      gbps,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      (cfg.disk_io_mode == DirectIoMode::kAuto)         ? "auto"
          : (cfg.disk_io_mode == DirectIoMode::kDirect) ? "direct"
                                                        : "buffered",
      direct_io,
      direct_io_fallback_errno,
      direct_io_fallback_reason,
      sched_waits,
      sched_wait_sec);
  return absl::OkStatus();
}

absl::Status run_disk_fragmentation(const LoaderConfig& cfg) {
  if (cfg.disk_bench_path.empty()) {
    return absl::InvalidArgumentError("--disk_bench_path is required for mode=disk_fragmentation");
  }
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(cfg.device_id));
  const uint64_t total_bytes = cfg.disk_frag_segment_bytes * cfg.disk_frag_segments;
  common::memory::GpuDeviceMemory dst;
  TC_RETURN_IF_ERROR(dst.allocate(static_cast<size_t>(total_bytes), cfg.device_id));
  loader::GpuMemorySink sink(
      loader::GpuMemorySink::Options{
          .gpu_base_ptr = gsl::not_null<void*>{dst.get()},
          .total_size = total_bytes,
          .chunk_size = 128 * 1024 * 1024,
          .device_id = cfg.device_id,
          .allocation = nullptr,
          .gpu_sched_enabled = cfg.gpu_sched_enabled,
          .gpu_sched_limit_bytes = cfg.gpu_sched_limit_bytes,
          .gpu_sched_limit_copies = cfg.gpu_sched_limit_copies,
      });

  FilePartitionSource src(
      FilePartitionSource::Options{
          .partition_paths = {cfg.disk_bench_path},
          .partition_sizes = {static_cast<size_t>(
              (cfg.disk_frag_segments - 1) * cfg.disk_frag_stride_bytes + cfg.disk_frag_segment_bytes)},
          .total_size = (cfg.disk_frag_segments - 1) * cfg.disk_frag_stride_bytes + cfg.disk_frag_segment_bytes,
          .io_batch_bytes = static_cast<size_t>(cfg.disk_frag_segment_bytes),
          .direct_io_mode = cfg.disk_io_mode,
      });

  const int io_threads = std::max(1, cfg.io_threads);
  std::unique_ptr<BufferPool> pool_ptr;
  TC_ASSIGN_OR_RETURN(pool_ptr, make_bounce_buffer_pool(cfg, bb, /*out_res=*/nullptr));

  // Build many small segments with a fixed stride.
  std::vector<SegmentCopy> segs;
  segs.reserve(static_cast<size_t>(cfg.disk_frag_segments));
  for (uint64_t i = 0; i < cfg.disk_frag_segments; ++i) {
    segs.push_back(
        SegmentCopy{
            .src_offset = i * cfg.disk_frag_stride_bytes,
            .dst_offset = i * cfg.disk_frag_segment_bytes,
            .bytes = static_cast<size_t>(cfg.disk_frag_segment_bytes),
        });
  }
  std::vector<RemappedSource::Segment> remap;
  remap.reserve(segs.size());
  for (const auto& seg : segs) {
    remap.push_back(
        RemappedSource::Segment{
            .dst_offset = seg.dst_offset,
            .src_offset = seg.src_offset,
            .end_offset = seg.dst_offset + seg.bytes,
        });
  }
  RemappedSource remapped(gsl::not_null<SeekableSource*>{&src}, std::move(remap));
  std::vector<std::pair<uint64_t, size_t>> ranges;
  ranges.reserve(segs.size());
  for (const auto& seg : segs) {
    ranges.push_back({seg.dst_offset, seg.bytes});
  }

  const auto t0 = absl::Now();
  const auto sched_before = loader::get_gpu_scheduler_stats(cfg.device_id);
  TC_RETURN_IF_ERROR(
      loader::pump_ranges(remapped, sink, *pool_ptr, ranges, io_threads, pump_benchmark_runtime().blocking_executor()));
  TC_RETURN_IF_ERROR(sink.close());
  const auto sched_after = loader::get_gpu_scheduler_stats(cfg.device_id);
  const uint64_t sched_waits =
      ((sched_after.waits >= sched_before.waits) ? (sched_after.waits - sched_before.waits) : 0);
  const double sched_wait_sec =
      (sched_after.wait_sec >= sched_before.wait_sec) ? (sched_after.wait_sec - sched_before.wait_sec) : 0.0;
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);
  const char* direct_io = src.is_using_direct_io() ? "true" : "false";
  const uint64_t direct_io_fallback_errno = static_cast<uint64_t>(std::max(0, src.direct_io_fallback_errno()));
  const std::string direct_io_fallback_reason =
      src.direct_io_fallback_reason().empty() ? "none" : src.direct_io_fallback_reason();
  LOG(INFO) << std::format(
      "disk_fragmentation segments={} seg_bytes={} total_bytes={} sec={:.6f} GiB/s={:.3f} pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} disk_io_mode={} direct_io={} direct_io_fallback_errno={} direct_io_fallback_reason={} gpu_sched_waits={} gpu_sched_wait_sec={:.6f}",
      static_cast<uint64_t>(cfg.disk_frag_segments),
      static_cast<uint64_t>(cfg.disk_frag_segment_bytes),
      static_cast<uint64_t>(total_bytes),
      sec,
      gbps,
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      io_threads,
      (cfg.disk_io_mode == DirectIoMode::kAuto)         ? "auto"
          : (cfg.disk_io_mode == DirectIoMode::kDirect) ? "direct"
                                                        : "buffered",
      direct_io,
      direct_io_fallback_errno,
      direct_io_fallback_reason,
      sched_waits,
      sched_wait_sec);
  return absl::OkStatus();
}

absl::Status run_gpu_peer_baseline(const LoaderConfig& cfg) {
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(kGpuPeerDefaultSrcDevice));
  common::memory::GpuDeviceMemory src;
  TC_RETURN_IF_ERROR(src.allocate(static_cast<size_t>(cfg.gpu_peer_bytes), kGpuPeerDefaultSrcDevice));
  TC_RETURN_IF_ERROR(tensorcast::cuda::set_device(kGpuPeerDefaultDstDevice));
  common::memory::GpuDeviceMemory dst;
  TC_RETURN_IF_ERROR(dst.allocate(static_cast<size_t>(cfg.gpu_peer_bytes), kGpuPeerDefaultDstDevice));

  int can_access = 0;
  TC_RETURN_IF_ERROR(
      tensorcast::cuda::device_can_access_peer(&can_access, kGpuPeerDefaultDstDevice, kGpuPeerDefaultSrcDevice));
  if (can_access) {
    TC_RETURN_IF_ERROR(tensorcast::cuda::enable_peer_access(kGpuPeerDefaultDstDevice, kGpuPeerDefaultSrcDevice));
  }

  cudaStream_t stream = nullptr;
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_create(&stream));
  const auto t0 = absl::Now();
  TC_RETURN_IF_ERROR(
      tensorcast::cuda::memcpy_peer_async(
          dst.get(), kGpuPeerDefaultDstDevice, src.get(), kGpuPeerDefaultSrcDevice, cfg.gpu_peer_bytes, stream));
  TC_RETURN_IF_ERROR(tensorcast::cuda::stream_synchronize(stream));
  const double sec = seconds_since(t0);
  const double gbps = (static_cast<double>(cfg.gpu_peer_bytes) / (1024.0 * 1024.0 * 1024.0)) / std::max(1e-9, sec);
  (void)tensorcast::cuda::stream_destroy(stream);

  LOG(INFO) << std::format(
      "gpu_peer_baseline src={} dst={} bytes={} sec={:.6f} GiB/s={:.3f} peer_access={}",
      kGpuPeerDefaultSrcDevice,
      kGpuPeerDefaultDstDevice,
      static_cast<uint64_t>(cfg.gpu_peer_bytes),
      sec,
      gbps,
      can_access ? "true" : "false");
  return absl::OkStatus();
}

absl::Status log_run_result(const LoaderConfig& cfg, const RunResult& r) {
  const char* strategy = "unknown";
  switch (r.strategy) {
    case StrategyKind::kA_Eager:
      strategy = "A_eager";
      break;
    case StrategyKind::kB_LazyCommit:
      strategy = "B_lazy_commit";
      break;
    case StrategyKind::kC_BatchedOptimal:
      strategy = "C_batched_optimal";
      break;
    case StrategyKind::kC_HostPack:
      strategy = "C_host_pack";
      break;
  }
  BounceBufferPlan bb;
  TC_ASSIGN_OR_RETURN(bb, plan_bounce_buffer(cfg));
  std::string planner_stats;
  if (r.strategy == StrategyKind::kB_LazyCommit) {
    planner_stats = std::format(
        " segments_pre_merge={} segments_merged={} src_runs={} src_run_avg_bytes={:.1f} src_run_max_bytes={}",
        static_cast<uint64_t>(r.planner_segments_pre_merge),
        static_cast<uint64_t>(r.planner_segments_merged),
        static_cast<uint64_t>(r.planner_src_runs),
        r.planner_src_run_avg_bytes,
        static_cast<uint64_t>(r.planner_src_run_max_bytes));
  } else if (r.strategy == StrategyKind::kC_BatchedOptimal) {
    planner_stats = std::format(
        " staged_tensors={} staged_reads={} direct_primary_reads={} direct_dedup_copies={} fallback_copies={}",
        static_cast<uint64_t>(r.c_staged_tensors),
        static_cast<uint64_t>(r.c_staged_reads),
        static_cast<uint64_t>(r.c_direct_primary_reads),
        static_cast<uint64_t>(r.c_direct_dedup_copies),
        static_cast<uint64_t>(r.c_fallback_copies));
  } else if (r.strategy == StrategyKind::kC_HostPack) {
    planner_stats = std::format(
        " staged_tensors={} staged_reads={} direct_primary_reads={} direct_dedup_copies={} fallback_copies={}",
        static_cast<uint64_t>(r.c_staged_tensors),
        static_cast<uint64_t>(r.c_staged_reads),
        static_cast<uint64_t>(r.c_direct_primary_reads),
        static_cast<uint64_t>(r.c_direct_dedup_copies),
        static_cast<uint64_t>(r.c_fallback_copies));
  }
  LOG(INFO) << std::format(
      "result strategy={} tp=(rank={}/{} mode=row_only) selection={} tensors copies={} segments={} ranges={}{} "
      "T(open_meta)={:.6f} T(open_copy)={:.6f} T(get)={:.6f} T(pack)={:.6f} T(net)={:.6f} T(commit)={:.6f} T(total_ready)={:.6f} "
      "bytes(disk)={} bytes(h2d)={} bytes(d2d)={} bytes(nccl_tx)={} bytes(nccl_rx)={} bytes(output)={} N_collectives={} "
      "pinned_bytes={} gpu_alloc_bytes={} vmm_reserved={} granularity={} gpu_sched_waits={} gpu_sched_wait_sec={:.6f}",
      strategy,
      cfg.tp_rank,
      cfg.tp_world_size,
      static_cast<uint64_t>(r.selected_tensors),
      static_cast<uint64_t>(r.selected_copies),
      static_cast<uint64_t>(r.planned_segments),
      static_cast<uint64_t>(r.planned_ranges),
      planner_stats,
      r.t.open_meta,
      r.t.open_copy,
      r.t.get_calls_total,
      r.t.pack,
      r.t.net,
      r.t.commit,
      r.t.total_ready,
      static_cast<uint64_t>(r.bytes.disk_read_bytes),
      static_cast<uint64_t>(r.bytes.h2d_bytes),
      static_cast<uint64_t>(r.bytes.d2d_bytes),
      static_cast<uint64_t>(r.bytes.nccl_tx_bytes),
      static_cast<uint64_t>(r.bytes.nccl_rx_bytes),
      static_cast<uint64_t>(r.output_bytes),
      static_cast<uint64_t>(r.n_collectives),
      static_cast<uint64_t>(r.res.pinned_host_bytes),
      static_cast<uint64_t>(r.res.gpu_alloc_bytes),
      static_cast<uint64_t>(r.res.vmm_reserved_bytes),
      r.res.vmm_granularity_bytes,
      static_cast<uint64_t>(r.gpu_sched_waits),
      r.gpu_sched_wait_sec);
  const std::string_view tp_devices = cfg.tp_devices.empty() ? "<default>" : std::string_view(cfg.tp_devices);
  LOG(INFO) << std::format(
      "config pinned={} bbuf_size_kb={} buffer_chunks={} chunk_bytes={} io_threads={} pinned_numa_node={} pinned_numa_prefault={} gpu_sched_enabled={} gpu_sched_limit_bytes={} gpu_sched_limit_copies={} plan_json_path={} "
      "collectives={} device_id={} tp_devices={} nccl_timeout_sec={} nccl_blocking_wait={}",
      cfg.use_pinned_host_buffer ? "true" : "false",
      static_cast<int64_t>(cfg.bbuf_size_kb),
      bb.chunks,
      bb.chunk_bytes,
      std::max(1, cfg.io_threads),
      cfg.pinned_numa_node,
      cfg.pinned_numa_prefault ? "true" : "false",
      cfg.gpu_sched_enabled ? "true" : "false",
      static_cast<uint64_t>(cfg.gpu_sched_limit_bytes),
      static_cast<uint64_t>(cfg.gpu_sched_limit_copies),
      cfg.load_plan_json_path.empty() ? "<none>" : cfg.load_plan_json_path.string(),
      cfg.enable_collectives ? "true" : "false",
      cfg.device_id,
      tp_devices,
      cfg.nccl_timeout_sec,
      cfg.nccl_blocking_wait ? "true" : "false");
  return absl::OkStatus();
}

} // namespace
} // namespace tensorcast::store::loader

namespace {
absl::Status log_proc_meminfo_summary(std::string_view label) {
  std::ifstream f("/proc/meminfo");
  if (!f.is_open()) {
    return absl::ErrnoToStatus(errno, "Failed to open /proc/meminfo");
  }

  absl::flat_hash_map<std::string, uint64_t> kb;
  std::string line;
  while (std::getline(f, line)) {
    const size_t pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, pos);
    const char* p = line.c_str() + pos + 1;
    while (*p == ' ' || *p == '\t') {
      ++p;
    }
    char* end = nullptr;
    errno = 0;
    const uint64_t v = std::strtoull(p, &end, 10);
    if (end == p || errno != 0) {
      continue;
    }
    kb.emplace(std::move(key), static_cast<uint64_t>(v));
  }

  auto get = [&](std::string_view k) -> uint64_t {
    const auto it = kb.find(std::string(k));
    if (it == kb.end()) {
      return 0;
    }
    return it->second;
  };

  LOG(INFO) << std::format(
      "meminfo({}) MemAvailable_kB={} Cached_kB={} Buffers_kB={} Active(file)_kB={} Inactive(file)_kB={}",
      label,
      static_cast<uint64_t>(get("MemAvailable")),
      static_cast<uint64_t>(get("Cached")),
      static_cast<uint64_t>(get("Buffers")),
      static_cast<uint64_t>(get("Active(file)")),
      static_cast<uint64_t>(get("Inactive(file)")));
  return absl::OkStatus();
}
} // namespace

ABSL_FLAG(
    std::string,
    mode,
    "loader",
    "One of: loader, disk_baseline, disk_fragmentation, gpu_peer_baseline, safetensors_disk_baseline, safetensors_host_baseline, safetensors_hot_disk_baseline, safetensors_hot_host_baseline, safetensors_dram_mirror_host_baseline, materialize_d, materialized_disk_baseline, h2d_baseline, h2d_2d_baseline, safetensors_o_direct_host_baseline, safetensors_o_direct_disk_baseline, nccl_baseline, nccl_launch_tax");

ABSL_FLAG(
    std::string,
    strategy,
    "a",
    "Strategy for loader mode: a|b|c|host_pack (aliases: hp,c_host_pack). Use --run_both_strategies to run both (A+B only).");
ABSL_FLAG(bool, run_both_strategies, false, "If true, run A then B in one invocation (enables A vs B checks).");

ABSL_FLAG(std::string, safetensors_dir, "", "Directory containing one or more .safetensors shards.");
ABSL_FLAG(
    std::string,
    load_plan_json_path,
    "",
    "Optional JSON load plan for --mode=loader. If set, selects tensors/slices/copies from the plan instead of the default tp slicing.");

ABSL_FLAG(
    std::string,
    materialized_dir,
    "",
    "Strategy D: output directory for materialize_d. Default: <safetensors_dir>/tensorcast_materialized/tpN/.");

ABSL_FLAG(
    std::string,
    materialized_meta_path,
    "",
    "Strategy D: meta json path produced by materialize_d; used by --mode=materialized_disk_baseline.");

ABSL_FLAG(int, device_id, 0, "CUDA device to use for GPU allocations and copies.");
ABSL_FLAG(int, io_threads, 4, "Producer threads for range pump.");
ABSL_FLAG(int, buffer_chunks, 8, "Buffer pool chunk capacity (slots).");
ABSL_FLAG(int64_t, bbuf_size_kb, 262144, "Total host bounce buffer size in KiB (pinned or pageable).");
ABSL_FLAG(bool, use_pinned_host_buffer, true, "Use pinned host bounce buffer; if false, use pageable host memory.");
ABSL_FLAG(
    int,
    pinned_numa_node,
    -1,
    "NUMA node for pinned host bounce buffer placement (-1=default OS policy; -2=auto from CUDA device sysfs).");
ABSL_FLAG(
    bool,
    pinned_numa_prefault,
    false,
    "If true and pinned_numa_node is set, touch each page before cudaHostRegister to make NUMA placement deterministic.");
ABSL_FLAG(bool, gpu_sched_enabled, true, "Enable per-GPU in-flight H2D scheduler limits in GpuMemorySink.");
ABSL_FLAG(
    uint64_t,
    gpu_sched_limit_bytes,
    tensorcast::store::loader::DEFAULT_GPU_SCHED_LIMIT_BYTES,
    "Per-GPU in-flight bytes limit for GpuMemorySink scheduler (0 disables bytes limit).");
ABSL_FLAG(
    uint64_t,
    gpu_sched_limit_copies,
    tensorcast::store::loader::DEFAULT_GPU_SCHED_LIMIT_COPIES,
    "Per-GPU in-flight copy count limit for GpuMemorySink scheduler (0 disables copy limit).");

ABSL_FLAG(
    uint64_t,
    strategy_c_staging_bytes,
    1024ull * 1024ull * 1024ull,
    "Strategy C: GPU staging buffer size in bytes for strided (e.g., axis=1) slices.");

ABSL_FLAG(
    bool,
    enable_collectives,
    false,
    "If true, Strategy A runs A4 cross-rank NCCL send/recv via NCCL C API (single-process, multi-GPU).");
ABSL_FLAG(
    std::string,
    master_addr,
    "127.0.0.1",
    "Deprecated/no-op (kept for old multi-process c10d runner): ignored in single-process NCCL mode.");
ABSL_FLAG(
    int,
    master_port,
    29500,
    "Deprecated/no-op (kept for old multi-process c10d runner): ignored in single-process NCCL mode.");
ABSL_FLAG(
    std::string,
    rendezvous_id,
    "tensorcast_safetensors_benchmark",
    "Deprecated/no-op (kept for old multi-process c10d runner): ignored in single-process NCCL mode.");
ABSL_FLAG(
    int,
    rendezvous_timeout_sec,
    300,
    "Deprecated/no-op (kept for old multi-process c10d runner): ignored in single-process NCCL mode.");
ABSL_FLAG(int, nccl_timeout_sec, 600, "NCCL watchdog timeout in seconds (abort comms on timeout).");
ABSL_FLAG(bool, nccl_blocking_wait, false, "If true, set NCCL_BLOCKING_WAIT=1 before initializing NCCL.");
ABSL_FLAG(std::string, nccl_op, "broadcast", "For nccl_* modes: broadcast|sendrecv.");
ABSL_FLAG(uint64_t, nccl_min_bytes, 1024, "For nccl_baseline: min message size in bytes (power-of-two sweep).");
ABSL_FLAG(
    uint64_t,
    nccl_max_bytes,
    1024ull * 1024ull * 1024ull,
    "For nccl_baseline: max message size in bytes (power-of-two sweep).");
ABSL_FLAG(
    std::string,
    nccl_sizes,
    "",
    "For nccl_baseline: comma-separated message sizes in bytes (overrides min/max).");
ABSL_FLAG(int, nccl_iters, tensorcast::store::loader::kDefaultNcclIters, "For nccl_baseline: iters per size.");
ABSL_FLAG(
    int,
    nccl_warmup,
    tensorcast::store::loader::kDefaultNcclWarmupIters,
    "For nccl_baseline/nccl_launch_tax: warmup iters.");
ABSL_FLAG(int, nccl_overhead_iters, 10000, "For nccl_launch_tax: iters for measuring per-call overhead.");

ABSL_FLAG(int, tp_world_size, 1, "TP world size (also NCCL process group size when --enable_collectives).");
ABSL_FLAG(
    int,
    tp_rank,
    0,
    "TP rank for per-rank shard planning (ignored in single-process collectives and nccl_* modes).");
ABSL_FLAG(
    std::string,
    tp_devices,
    "",
    "Optional rank->CUDA device mapping for single-process NCCL runs (e.g. 0,2,3). Default: 0..tp_world_size-1 (respects CUDA_VISIBLE_DEVICES).");

ABSL_FLAG(
    uint64_t,
    check_correctness_samples,
    0,
    "Compare first N tensor slices between strategy A and B (requires --run_both_strategies).");

ABSL_FLAG(std::string, disk_bench_path, "", "Path to a large file for disk baseline/fragmentation modes.");
ABSL_FLAG(uint64_t, disk_bench_bytes, 8ull * 1024ull * 1024ull * 1024ull, "Bytes to copy for disk_baseline.");
ABSL_FLAG(std::string, disk_io_mode, "auto", "Disk I/O mode for disk_*: auto|direct|buffered.");
ABSL_FLAG(uint64_t, disk_frag_segment_bytes, 262144, "Segment size for disk_fragmentation.");
ABSL_FLAG(uint64_t, disk_frag_segments, 32768, "Number of segments for disk_fragmentation.");
ABSL_FLAG(
    uint64_t,
    disk_frag_stride_bytes,
    4ull * 1024ull * 1024ull,
    "Stride between segments for disk_fragmentation.");

ABSL_FLAG(uint64_t, gpu_peer_bytes, 1024ull * 1024ull * 1024ull, "Bytes to copy for gpu_peer_baseline.");
ABSL_FLAG(
    uint64_t,
    h2d_bench_bytes,
    8ull * 1024ull * 1024ull * 1024ull,
    "Bytes to copy per GPU for h2d_baseline and h2d_2d_baseline.");
ABSL_FLAG(
    uint64_t,
    h2d_2d_width_bytes,
    13824,
    "For h2d_2d_baseline: bytes per row to copy (widthBytes). Defaults to 6912 fp16 elems (Qwen2.5-32B TP=4 MLP gate/up).");
ABSL_FLAG(
    uint64_t,
    h2d_2d_height,
    5120,
    "For h2d_2d_baseline: number of rows to copy (height). Defaults to hidden size 5120 (Qwen2.5-32B).");
ABSL_FLAG(
    uint64_t,
    h2d_2d_src_pitch_bytes,
    55296,
    "For h2d_2d_baseline: source pitch in bytes (srcPitch). Defaults to 27648 fp16 elems (Qwen2.5-32B MLP gate/up).");
ABSL_FLAG(
    uint64_t,
    h2d_2d_dst_pitch_bytes,
    13824,
    "For h2d_2d_baseline: destination pitch in bytes (dstPitch). Default equals h2d_2d_width_bytes (packed output).");
ABSL_FLAG(
    bool,
    h2d_per_gpu_pinned_pool,
    false,
    "For h2d_baseline/h2d_2d_baseline: allocate a separate pinned host pool per GPU (recommended with pinned_numa_node=-2).");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  tensorcast::common::ensure_logging_initialized();

  using tensorcast::store::loader::BenchMode;
  using tensorcast::store::loader::LoaderConfig;
  using tensorcast::store::loader::RunResult;

  LoaderConfig cfg;
  {
    auto mode_or = tensorcast::store::loader::parse_mode(absl::GetFlag(FLAGS_mode));
    if (!mode_or.ok()) {
      LOG(ERROR) << mode_or.status();
      return 2;
    }
    cfg.mode = *mode_or;
  }
  {
    auto strategy_or = tensorcast::store::loader::parse_strategy(absl::GetFlag(FLAGS_strategy));
    if (!strategy_or.ok()) {
      LOG(ERROR) << strategy_or.status();
      return 2;
    }
    cfg.strategy = *strategy_or;
  }
  cfg.run_both_strategies = absl::GetFlag(FLAGS_run_both_strategies);

  cfg.device_id = absl::GetFlag(FLAGS_device_id);
  cfg.io_threads = absl::GetFlag(FLAGS_io_threads);
  cfg.buffer_chunks = absl::GetFlag(FLAGS_buffer_chunks);
  cfg.bbuf_size_kb = absl::GetFlag(FLAGS_bbuf_size_kb);
  cfg.use_pinned_host_buffer = absl::GetFlag(FLAGS_use_pinned_host_buffer);
  cfg.pinned_numa_node = absl::GetFlag(FLAGS_pinned_numa_node);
  cfg.pinned_numa_prefault = absl::GetFlag(FLAGS_pinned_numa_prefault);
  cfg.gpu_sched_enabled = absl::GetFlag(FLAGS_gpu_sched_enabled);
  cfg.gpu_sched_limit_bytes = absl::GetFlag(FLAGS_gpu_sched_limit_bytes);
  cfg.gpu_sched_limit_copies = absl::GetFlag(FLAGS_gpu_sched_limit_copies);
  cfg.strategy_c_staging_bytes = absl::GetFlag(FLAGS_strategy_c_staging_bytes);

  cfg.enable_collectives = absl::GetFlag(FLAGS_enable_collectives);
  cfg.master_addr = absl::GetFlag(FLAGS_master_addr);
  cfg.master_port = absl::GetFlag(FLAGS_master_port);
  cfg.rendezvous_id = absl::GetFlag(FLAGS_rendezvous_id);
  cfg.rendezvous_timeout_sec = absl::GetFlag(FLAGS_rendezvous_timeout_sec);
  cfg.nccl_timeout_sec = absl::GetFlag(FLAGS_nccl_timeout_sec);
  cfg.nccl_blocking_wait = absl::GetFlag(FLAGS_nccl_blocking_wait);
  cfg.nccl_op = absl::GetFlag(FLAGS_nccl_op);
  cfg.nccl_min_bytes = absl::GetFlag(FLAGS_nccl_min_bytes);
  cfg.nccl_max_bytes = absl::GetFlag(FLAGS_nccl_max_bytes);
  cfg.nccl_sizes = absl::GetFlag(FLAGS_nccl_sizes);
  cfg.nccl_iters = absl::GetFlag(FLAGS_nccl_iters);
  cfg.nccl_warmup = absl::GetFlag(FLAGS_nccl_warmup);
  cfg.nccl_overhead_iters = absl::GetFlag(FLAGS_nccl_overhead_iters);

  cfg.tp_world_size = absl::GetFlag(FLAGS_tp_world_size);
  cfg.tp_rank = absl::GetFlag(FLAGS_tp_rank);
  cfg.tp_devices = absl::GetFlag(FLAGS_tp_devices);
  cfg.check_correctness_samples = absl::GetFlag(FLAGS_check_correctness_samples);
  cfg.disk_bench_path = absl::GetFlag(FLAGS_disk_bench_path);
  cfg.disk_bench_bytes = absl::GetFlag(FLAGS_disk_bench_bytes);
  {
    auto io_or = tensorcast::store::loader::parse_direct_io_mode(absl::GetFlag(FLAGS_disk_io_mode));
    if (!io_or.ok()) {
      LOG(ERROR) << io_or.status();
      return 2;
    }
    cfg.disk_io_mode = *io_or;
  }
  cfg.disk_frag_segment_bytes = absl::GetFlag(FLAGS_disk_frag_segment_bytes);
  cfg.disk_frag_segments = absl::GetFlag(FLAGS_disk_frag_segments);
  cfg.disk_frag_stride_bytes = absl::GetFlag(FLAGS_disk_frag_stride_bytes);
  cfg.gpu_peer_bytes = absl::GetFlag(FLAGS_gpu_peer_bytes);
  cfg.h2d_bench_bytes = absl::GetFlag(FLAGS_h2d_bench_bytes);
  cfg.h2d_per_gpu_pinned_pool = absl::GetFlag(FLAGS_h2d_per_gpu_pinned_pool);
  cfg.h2d_2d_width_bytes = absl::GetFlag(FLAGS_h2d_2d_width_bytes);
  cfg.h2d_2d_height = absl::GetFlag(FLAGS_h2d_2d_height);
  cfg.h2d_2d_src_pitch_bytes = absl::GetFlag(FLAGS_h2d_2d_src_pitch_bytes);
  cfg.h2d_2d_dst_pitch_bytes = absl::GetFlag(FLAGS_h2d_2d_dst_pitch_bytes);

  cfg.safetensors_dir = absl::GetFlag(FLAGS_safetensors_dir);
  cfg.load_plan_json_path = absl::GetFlag(FLAGS_load_plan_json_path);
  cfg.materialized_dir = absl::GetFlag(FLAGS_materialized_dir);
  cfg.materialized_meta_path = absl::GetFlag(FLAGS_materialized_meta_path);

  if (cfg.enable_collectives && cfg.mode != BenchMode::kLoader) {
    LOG(WARNING) << "--enable_collectives is ignored unless --mode=loader";
  }

  if (cfg.enable_collectives && cfg.mode == BenchMode::kLoader) {
    if (cfg.tp_world_size <= 1) {
      LOG(WARNING) << "--enable_collectives requires --tp_world_size>1; running without collectives";
    } else {
      if (cfg.run_both_strategies) {
        LOG(ERROR) << "--run_both_strategies is not supported with --enable_collectives and tp_world_size>1";
        return 2;
      }
      if (cfg.strategy != tensorcast::store::loader::StrategyKind::kA_Eager) {
        LOG(ERROR) << "--enable_collectives only applies to strategy A (use --strategy=a)";
        return 2;
      }
      if (cfg.tp_rank != 0) {
        LOG(WARNING) << "--tp_rank is ignored in single-process collectives mode";
      }
      if (cfg.device_id != 0 && cfg.tp_devices.empty()) {
        LOG(WARNING)
            << "--device_id is ignored in single-process collectives mode; use --tp_devices or CUDA_VISIBLE_DEVICES";
      }
      if (cfg.master_addr != "127.0.0.1" || cfg.master_port != 29500 ||
          cfg.rendezvous_id != "tensorcast_safetensors_benchmark" || cfg.rendezvous_timeout_sec != 300) {
        LOG(WARNING)
            << "--master_addr/--master_port/--rendezvous_id/--rendezvous_timeout_sec are deprecated and ignored in single-process NCCL mode";
      }
    }
  }

  if (cfg.mode == BenchMode::kNcclBaseline) {
    auto st = tensorcast::store::loader::run_nccl_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kNcclLaunchTax) {
    auto st = tensorcast::store::loader::run_nccl_launch_tax(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }

  if (cfg.mode == BenchMode::kDiskBaseline) {
    auto st = tensorcast::store::loader::run_disk_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kDiskFragmentation) {
    auto st = tensorcast::store::loader::run_disk_fragmentation(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kGpuPeerBaseline) {
    auto st = tensorcast::store::loader::run_gpu_peer_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kH2dBaseline) {
    auto st = tensorcast::store::loader::run_h2d_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kH2d2dBaseline) {
    auto st = tensorcast::store::loader::run_h2d_2d_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kMaterializedDiskBaseline) {
    auto st = tensorcast::store::loader::run_materialized_disk_baseline(cfg);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }

  auto shards_or = tensorcast::store::loader::collect_safetensors_inputs(cfg.safetensors_dir);
  if (!shards_or.ok()) {
    LOG(ERROR) << shards_or.status();
    return 2;
  }
  const auto& shards = *shards_or;

  {
    const absl::Status st = log_proc_meminfo_summary("before");
    if (!st.ok()) {
      LOG(WARNING) << st;
    }
  }

  if (cfg.mode == BenchMode::kSafetensorsDiskBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_disk_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsHostBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_host_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsHotHostBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_hot_host_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsHotDiskBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_hot_disk_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsDramMirrorHostBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_dram_mirror_host_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kMaterializeD) {
    auto st = tensorcast::store::loader::run_materialize_d(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsODirectHostBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_o_direct_host_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }
  if (cfg.mode == BenchMode::kSafetensorsODirectDiskBaseline) {
    auto st = tensorcast::store::loader::run_safetensors_o_direct_disk_baseline(cfg, shards);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }

  if (!cfg.load_plan_json_path.empty()) {
    if (cfg.mode != BenchMode::kLoader) {
      LOG(ERROR) << "--load_plan_json_path only applies to --mode=loader";
      return 2;
    }
    if (cfg.enable_collectives) {
      LOG(ERROR) << "--load_plan_json_path is not supported with --enable_collectives";
      return 2;
    }
    if (cfg.run_both_strategies) {
      LOG(ERROR)
          << "--load_plan_json_path is not supported with --run_both_strategies (likely OOM); run --strategy=a and --strategy=b separately";
      return 2;
    }
    if (cfg.check_correctness_samples != 0) {
      LOG(ERROR)
          << "--check_correctness_samples requires --run_both_strategies; not supported with --load_plan_json_path";
      return 2;
    }
  }

  if (cfg.enable_collectives && cfg.tp_world_size > 1) {
    if (cfg.run_both_strategies) {
      LOG(ERROR) << "--run_both_strategies is not supported with --enable_collectives and tp_world_size>1";
      return 2;
    }
    if (cfg.strategy != tensorcast::store::loader::StrategyKind::kA_Eager) {
      LOG(ERROR) << "--enable_collectives only applies to strategy A (use --strategy=a)";
      return 2;
    }

    auto multi_or = tensorcast::store::loader::run_strategy_a_collectives_single_process(cfg, shards);
    if (!multi_or.ok()) {
      LOG(ERROR) << multi_or.status();
      return 1;
    }
    const auto& multi = *multi_or;
    double makespan_sec = 0.0;
    for (int rank = 0; rank < cfg.tp_world_size; ++rank) {
      tensorcast::store::loader::LoaderConfig rank_cfg = cfg;
      rank_cfg.tp_rank = rank;
      rank_cfg.device_id = multi.device_ids[rank];
      const auto st = tensorcast::store::loader::log_run_result(rank_cfg, multi.results[static_cast<size_t>(rank)]);
      if (!st.ok()) {
        LOG(ERROR) << st;
        return 1;
      }
      makespan_sec = std::max(makespan_sec, multi.results[static_cast<size_t>(rank)].t.total_ready);
    }
    std::string tp_devices_joined;
    for (size_t i = 0; i < multi.device_ids.size(); ++i) {
      if (i != 0) {
        absl::StrAppend(&tp_devices_joined, ",");
      }
      absl::StrAppend(&tp_devices_joined, multi.device_ids[i]);
    }
    LOG(INFO) << std::format(
        "makespan tp_world_size={} max_T_total_ready_sec={:.6f} tp_devices={}",
        cfg.tp_world_size,
        makespan_sec,
        tp_devices_joined);
    return 0;
  }

  if (cfg.run_both_strategies) {
    auto a_or = tensorcast::store::loader::run_strategy_a_with_state(cfg, shards);
    if (!a_or.ok()) {
      LOG(ERROR) << a_or.status();
      return 1;
    }
    auto& [a_res, a_state] = *a_or;
    auto st = tensorcast::store::loader::log_run_result(cfg, a_res);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }

    auto b_or = tensorcast::store::loader::run_strategy_b_with_state(cfg, shards);
    if (!b_or.ok()) {
      LOG(ERROR) << b_or.status();
      return 1;
    }
    auto& [b_res, b_state] = *b_or;
    st = tensorcast::store::loader::log_run_result(cfg, b_res);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }

    st = tensorcast::store::loader::check_correctness_samples(cfg, a_state, b_state);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }

  if (cfg.strategy == tensorcast::store::loader::StrategyKind::kA_Eager) {
    auto res_or = tensorcast::store::loader::run_strategy_a(cfg, shards);
    if (!res_or.ok()) {
      LOG(ERROR) << res_or.status();
      return 1;
    }
    auto st = tensorcast::store::loader::log_run_result(cfg, *res_or);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    return 0;
  }

  if (cfg.strategy == tensorcast::store::loader::StrategyKind::kC_BatchedOptimal) {
    auto c_or = tensorcast::store::loader::run_strategy_c_with_state(cfg, shards);
    if (!c_or.ok()) {
      LOG(ERROR) << c_or.status();
      return 1;
    }
    auto& [c_res, c_state] = *c_or;
    auto st = tensorcast::store::loader::log_run_result(cfg, c_res);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    (void)c_state;
    return 0;
  }

  if (cfg.strategy == tensorcast::store::loader::StrategyKind::kC_HostPack) {
    auto c_or = tensorcast::store::loader::run_strategy_c_host_pack_with_state(cfg, shards);
    if (!c_or.ok()) {
      LOG(ERROR) << c_or.status();
      return 1;
    }
    auto& [c_res, c_state] = *c_or;
    auto st = tensorcast::store::loader::log_run_result(cfg, c_res);
    if (!st.ok()) {
      LOG(ERROR) << st;
      return 1;
    }
    (void)c_state;
    return 0;
  }

  auto b_or = tensorcast::store::loader::run_strategy_b_with_state(cfg, shards);
  if (!b_or.ok()) {
    LOG(ERROR) << b_or.status();
    return 1;
  }
  auto& [b_res, b_state] = *b_or;
  auto st = tensorcast::store::loader::log_run_result(cfg, b_res);
  if (!st.ok()) {
    LOG(ERROR) << st;
    return 1;
  }
  (void)b_state;
  return 0;
}
