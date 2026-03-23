// Copyright (c) 2025-2026, TensorCast Team.

#include <getopt.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <syncstream>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "core/communicator/base/constants.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/cuda/cuda_api.h"

namespace tc = tensorcast::communicator;
namespace base = tensorcast::communicator::base;
namespace engine = tensorcast::communicator::engine;
namespace transport = tensorcast::communicator::transport;
namespace cuda = tensorcast::cuda;

namespace {

using SteadyClock = std::chrono::steady_clock;

uint64_t elapsed_us(SteadyClock::time_point start, SteadyClock::time_point end) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
}

void emit_machine_line(const std::string& line) {
  std::osyncstream synced(std::cout);
  synced << line << '\n';
}

struct Options {
  std::string role = "inspect";
  std::string listen_ip = "0.0.0.0";
  uint16_t listen_port = 6606;
  std::string peer_ip = "127.0.0.1";
  uint16_t peer_port = 6606;
  std::string tensor_key = "communicator-bench-tensor";
  std::string memory = "gpu";
  uint64_t bytes = 64ULL * 1024 * 1024;
  int gpu_id = 0;
  bool enable_rdma = false;
  std::string rdma_nic;
  std::string expected_remote_nic;
  bool strict_nic = false;
  bool direct_rdma = false;
  bool strict_direct_rdma = false;
  bool probe_gpu_mr = false;
  bool verify = true;
  int iterations = 1;
  int warmup_iterations = 0;
  int duration_sec = 0;
  int threads = 1;
  int batch_size = 1;
  int qp_count = 1;
  int outstanding_wr = 64;
};

struct RdmPreflight {
  std::vector<std::string> visible_nics;
  std::string selected_nic;
  int16_t selected_rail_id = -1;
  std::string gpu_name;
  int gpu_bus = -1;
  int gpu_device = -1;
};

struct AllocationProfile {
  uint64_t total_us = 0;
  uint64_t set_device_us = 0;
  uint64_t allocate_call_us = 0;
};

struct PatternFillProfile {
  uint64_t total_us = 0;
  uint64_t buffer_alloc_us = 0;
  uint64_t pattern_fill_us = 0;
  uint64_t copy_us = 0;
};

struct VerifyProfile {
  uint64_t total_us = 0;
  uint64_t buffer_alloc_us = 0;
  uint64_t copy_us = 0;
  uint64_t checksum_us = 0;
};

std::string tensor_key_for_thread(const std::string& base_key, int thread_id, int threads) {
  if (threads <= 1) {
    return base_key;
  }
  return std::format("{}-t{}", base_key, thread_id);
}

absl::StatusOr<uint64_t> get_total_buffer_bytes(const Options& options);

class Buffer {
 public:
  ~Buffer() {
    reset();
  }

  absl::Status allocate(const Options& options, AllocationProfile* profile = nullptr) {
    const auto total_start = SteadyClock::now();
    auto record_profile = [&](uint64_t set_device_us, uint64_t allocate_call_us) {
      if (profile == nullptr) {
        return;
      }
      profile->set_device_us = set_device_us;
      profile->allocate_call_us = allocate_call_us;
      profile->total_us = elapsed_us(total_start, SteadyClock::now());
    };

    auto total_bytes_or = get_total_buffer_bytes(options);
    if (!total_bytes_or.ok()) {
      return total_bytes_or.status();
    }
    bytes_ = static_cast<size_t>(*total_bytes_or);
    if (options.memory == "cpu") {
      const auto alloc_start = SteadyClock::now();
      cpu_ptr_ = static_cast<uint8_t*>(std::malloc(bytes_));
      const auto alloc_us = elapsed_us(alloc_start, SteadyClock::now());
      if (cpu_ptr_ == nullptr) {
        return absl::ResourceExhaustedError("malloc failed");
      }
      record_profile(/*set_device_us=*/0, alloc_us);
      return absl::OkStatus();
    }

    const auto set_device_start = SteadyClock::now();
    auto set_status = cuda::set_device(options.gpu_id);
    const auto set_device_us = elapsed_us(set_device_start, SteadyClock::now());
    if (!set_status.ok()) {
      return set_status;
    }
    gpu_id_ = options.gpu_id;
    const auto alloc_start = SteadyClock::now();
    auto alloc_status = cuda::malloc(reinterpret_cast<void**>(&gpu_ptr_), bytes_);
    const auto alloc_us = elapsed_us(alloc_start, SteadyClock::now());
    if (!alloc_status.ok()) {
      return alloc_status;
    }
    record_profile(set_device_us, alloc_us);
    return absl::OkStatus();
  }

  void reset() {
    if (gpu_ptr_ != nullptr) {
      if (gpu_id_ >= 0) {
        cuda::set_device(gpu_id_).IgnoreError();
      }
      cuda::free(gpu_ptr_).IgnoreError();
      gpu_ptr_ = nullptr;
    }
    if (cpu_ptr_ != nullptr) {
      std::free(cpu_ptr_);
      cpu_ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  [[nodiscard]] uint64_t addr() const {
    return reinterpret_cast<uint64_t>(cpu_ptr_ != nullptr ? cpu_ptr_ : gpu_ptr_);
  }

  [[nodiscard]] size_t bytes() const {
    return bytes_;
  }

  absl::Status fill_pattern(const Options& options, uint8_t seed, PatternFillProfile* profile = nullptr) {
    const auto total_start = SteadyClock::now();
    auto record_profile = [&](uint64_t buffer_alloc_us, uint64_t pattern_fill_us, uint64_t copy_us) {
      if (profile == nullptr) {
        return;
      }
      profile->buffer_alloc_us = buffer_alloc_us;
      profile->pattern_fill_us = pattern_fill_us;
      profile->copy_us = copy_us;
      profile->total_us = elapsed_us(total_start, SteadyClock::now());
    };

    const auto alloc_start = SteadyClock::now();
    std::vector<uint8_t> pattern(bytes_);
    const auto pattern_alloc_us = elapsed_us(alloc_start, SteadyClock::now());
    const auto fill_start = SteadyClock::now();
    for (size_t i = 0; i < bytes_; ++i) {
      pattern[i] = static_cast<uint8_t>((i + seed) % 251);
    }
    const auto pattern_fill_us = elapsed_us(fill_start, SteadyClock::now());
    uint64_t copy_us = 0;
    if (options.memory == "cpu") {
      const auto copy_start = SteadyClock::now();
      std::memcpy(cpu_ptr_, pattern.data(), bytes_);
      copy_us = elapsed_us(copy_start, SteadyClock::now());
      record_profile(pattern_alloc_us, pattern_fill_us, copy_us);
      return absl::OkStatus();
    }
    const auto set_device_start = SteadyClock::now();
    auto set_status = cuda::set_device(options.gpu_id);
    copy_us = elapsed_us(set_device_start, SteadyClock::now());
    if (!set_status.ok()) {
      record_profile(pattern_alloc_us, pattern_fill_us, copy_us);
      return set_status;
    }
    const auto copy_start = SteadyClock::now();
    auto copy_status = cuda::memcpy(gpu_ptr_, pattern.data(), bytes_, cudaMemcpyHostToDevice);
    copy_us += elapsed_us(copy_start, SteadyClock::now());
    record_profile(pattern_alloc_us, pattern_fill_us, copy_us);
    return copy_status;
  }

  absl::Status clear(const Options& options, uint64_t* clear_us = nullptr) {
    const auto clear_start = SteadyClock::now();
    auto record_clear_us = [&]() {
      if (clear_us != nullptr) {
        *clear_us = elapsed_us(clear_start, SteadyClock::now());
      }
    };
    if (options.memory == "cpu") {
      std::memset(cpu_ptr_, 0, bytes_);
      record_clear_us();
      return absl::OkStatus();
    }
    auto set_status = cuda::set_device(options.gpu_id);
    if (!set_status.ok()) {
      record_clear_us();
      return set_status;
    }
    auto clear_status = cuda::memset(gpu_ptr_, 0, bytes_);
    record_clear_us();
    return clear_status;
  }

  absl::Status verify_pattern(const Options& options, uint8_t seed, VerifyProfile* profile = nullptr) const {
    const auto total_start = SteadyClock::now();
    auto record_profile = [&](uint64_t buffer_alloc_us, uint64_t copy_us, uint64_t checksum_us) {
      if (profile == nullptr) {
        return;
      }
      profile->buffer_alloc_us = buffer_alloc_us;
      profile->copy_us = copy_us;
      profile->checksum_us = checksum_us;
      profile->total_us = elapsed_us(total_start, SteadyClock::now());
    };

    const auto alloc_start = SteadyClock::now();
    std::vector<uint8_t> data(bytes_);
    const auto data_alloc_us = elapsed_us(alloc_start, SteadyClock::now());
    uint64_t copy_us = 0;
    if (options.memory == "cpu") {
      const auto copy_start = SteadyClock::now();
      std::memcpy(data.data(), cpu_ptr_, bytes_);
      copy_us = elapsed_us(copy_start, SteadyClock::now());
    } else {
      const auto set_device_start = SteadyClock::now();
      auto set_status = cuda::set_device(options.gpu_id);
      copy_us = elapsed_us(set_device_start, SteadyClock::now());
      if (!set_status.ok()) {
        record_profile(data_alloc_us, copy_us, /*checksum_us=*/0);
        return set_status;
      }
      const auto copy_start = SteadyClock::now();
      auto copy_status = cuda::memcpy(data.data(), gpu_ptr_, bytes_, cudaMemcpyDeviceToHost);
      copy_us += elapsed_us(copy_start, SteadyClock::now());
      if (!copy_status.ok()) {
        record_profile(data_alloc_us, copy_us, /*checksum_us=*/0);
        return copy_status;
      }
    }
    const auto checksum_start = SteadyClock::now();
    for (size_t i = 0; i < bytes_; ++i) {
      const auto expected = static_cast<uint8_t>((i + seed) % 251);
      if (data[i] != expected) {
        const auto checksum_us = elapsed_us(checksum_start, SteadyClock::now());
        record_profile(data_alloc_us, copy_us, checksum_us);
        return absl::DataLossError(
            absl::StrCat("data mismatch at offset=", i, " expected=", expected, " actual=", data[i]));
      }
    }
    const auto checksum_us = elapsed_us(checksum_start, SteadyClock::now());
    record_profile(data_alloc_us, copy_us, checksum_us);
    return absl::OkStatus();
  }

 private:
  size_t bytes_ = 0;
  int gpu_id_ = -1;
  uint8_t* cpu_ptr_ = nullptr;
  uint8_t* gpu_ptr_ = nullptr;
};

absl::StatusOr<uint64_t> get_total_buffer_bytes(const Options& options) {
  if (options.batch_size <= 0) {
    return absl::InvalidArgumentError("--batch-size must be > 0");
  }
  if (options.bytes == 0) {
    return absl::InvalidArgumentError("--bytes must be > 0");
  }
  constexpr auto kMaxBytes = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  const auto batch_size = static_cast<uint64_t>(options.batch_size);
  if (options.bytes > (kMaxBytes / batch_size)) {
    return absl::InvalidArgumentError("buffer size overflow");
  }
  return options.bytes * batch_size;
}

void print_usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " --role inspect|target|initiator [options]\n"
            << "  --listen-ip <ip>            listen ip for target/initiator local control socket\n"
            << "  --listen-port <port>        listen port for target/initiator local control socket\n"
            << "  --peer-ip <ip>              target ip for initiator\n"
            << "  --peer-port <port>          target port for initiator\n"
            << "  --tensor-key <key>          tensor key\n"
            << "  --memory cpu|gpu            backing memory type\n"
            << "  --bytes <n>                 transfer size in bytes\n"
            << "  --gpu-id <id>               gpu id when --memory=gpu\n"
            << "  --rdma                      enable rdma\n"
            << "  --rdma-nic <mlx5_x>         restrict process-visible RDMA NICs\n"
            << "  --expected-remote-nic <n>   strict check on server-advertised RDMA NIC\n"
            << "  --strict-nic                fail if selected NIC does not match requested NIC\n"
            << "  --direct-rdma               request direct GPU RDMA\n"
            << "  --strict-direct-rdma        fail instead of falling back when direct RDMA is unavailable\n"
            << "  --iterations <n>            measured iterations\n"
            << "  --warmup-iterations <n>     warmup iterations\n"
            << "  --threads <n>               concurrent initiator threads\n"
            << "  --batch-size <n>            requests issued per thread before waiting\n"
            << "  --qp-count <n>              RDMA QPs per peer transport\n"
            << "  --outstanding-wr <n>        max send WRs per QP\n"
            << "  --duration-sec <n>          run until duration instead of a fixed iteration count\n"
            << "  --no-verify                 skip payload verification\n";
}

absl::Status parse_uint16(const char* text, uint16_t* out) {
  char* end = nullptr;
  const auto value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value > 65535) {
    return absl::InvalidArgumentError(absl::StrCat("invalid uint16 value: ", text));
  }
  *out = static_cast<uint16_t>(value);
  return absl::OkStatus();
}

absl::Status parse_uint64(const char* text, uint64_t* out) {
  char* end = nullptr;
  const auto value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    return absl::InvalidArgumentError(absl::StrCat("invalid uint64 value: ", text));
  }
  *out = static_cast<uint64_t>(value);
  return absl::OkStatus();
}

absl::Status parse_int(const char* text, int* out) {
  char* end = nullptr;
  const auto value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0') {
    return absl::InvalidArgumentError(absl::StrCat("invalid int value: ", text));
  }
  *out = static_cast<int>(value);
  return absl::OkStatus();
}

absl::Status parse_options(int argc, char** argv, Options* options) {
  if (options == nullptr) {
    return absl::InvalidArgumentError("options is null");
  }
  static option long_options[] = {
      {"role", required_argument, nullptr, 'r'},
      {"listen-ip", required_argument, nullptr, 'l'},
      {"listen-port", required_argument, nullptr, 'p'},
      {"peer-ip", required_argument, nullptr, 'i'},
      {"peer-port", required_argument, nullptr, 'q'},
      {"tensor-key", required_argument, nullptr, 'k'},
      {"memory", required_argument, nullptr, 'm'},
      {"bytes", required_argument, nullptr, 'b'},
      {"gpu-id", required_argument, nullptr, 'g'},
      {"rdma", no_argument, nullptr, 'd'},
      {"rdma-nic", required_argument, nullptr, 'n'},
      {"expected-remote-nic", required_argument, nullptr, 'e'},
      {"strict-nic", no_argument, nullptr, 's'},
      {"direct-rdma", no_argument, nullptr, 'x'},
      {"strict-direct-rdma", no_argument, nullptr, 'X'},
      {"probe-gpu-mr", no_argument, nullptr, 'P'},
      {"iterations", required_argument, nullptr, 't'},
      {"warmup-iterations", required_argument, nullptr, 'w'},
      {"threads", required_argument, nullptr, 'T'},
      {"batch-size", required_argument, nullptr, 'B'},
      {"qp-count", required_argument, nullptr, 'Q'},
      {"outstanding-wr", required_argument, nullptr, 'W'},
      {"duration-sec", required_argument, nullptr, 'u'},
      {"no-verify", no_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0},
  };

  int opt = 0;
  while ((opt = getopt_long(argc, argv, "r:l:p:i:q:k:m:b:g:dn:e:sxXPt:w:T:B:Q:W:u:vh", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'r':
        options->role = optarg;
        break;
      case 'l':
        options->listen_ip = optarg;
        break;
      case 'p': {
        auto st = parse_uint16(optarg, &options->listen_port);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'i':
        options->peer_ip = optarg;
        break;
      case 'q': {
        auto st = parse_uint16(optarg, &options->peer_port);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'k':
        options->tensor_key = optarg;
        break;
      case 'm':
        options->memory = optarg;
        break;
      case 'b': {
        auto st = parse_uint64(optarg, &options->bytes);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'g': {
        auto st = parse_int(optarg, &options->gpu_id);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'd':
        options->enable_rdma = true;
        break;
      case 'n':
        options->rdma_nic = optarg;
        break;
      case 'e':
        options->expected_remote_nic = optarg;
        break;
      case 's':
        options->strict_nic = true;
        break;
      case 'x':
        options->direct_rdma = true;
        break;
      case 'X':
        options->strict_direct_rdma = true;
        options->direct_rdma = true;
        break;
      case 'P':
        options->probe_gpu_mr = true;
        break;
      case 't': {
        auto st = parse_int(optarg, &options->iterations);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'w': {
        auto st = parse_int(optarg, &options->warmup_iterations);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'T': {
        auto st = parse_int(optarg, &options->threads);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'B': {
        auto st = parse_int(optarg, &options->batch_size);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'Q': {
        auto st = parse_int(optarg, &options->qp_count);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'W': {
        auto st = parse_int(optarg, &options->outstanding_wr);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'u': {
        auto st = parse_int(optarg, &options->duration_sec);
        if (!st.ok()) {
          return st;
        }
        break;
      }
      case 'v':
        options->verify = false;
        break;
      case 'h':
        print_usage(argv[0]);
        std::exit(0);
      default:
        return absl::InvalidArgumentError("unknown argument");
    }
  }

  if (options->memory != "cpu" && options->memory != "gpu") {
    return absl::InvalidArgumentError("--memory must be cpu or gpu");
  }
  if ((options->direct_rdma || options->strict_direct_rdma) && options->memory != "gpu") {
    return absl::InvalidArgumentError("direct RDMA requires --memory=gpu");
  }
  if ((options->direct_rdma || options->strict_direct_rdma) && !options->enable_rdma) {
    return absl::InvalidArgumentError("direct RDMA requires --rdma");
  }
  if (options->strict_nic && options->rdma_nic.empty()) {
    return absl::InvalidArgumentError("--strict-nic requires --rdma-nic");
  }
  if (options->threads <= 0) {
    return absl::InvalidArgumentError("--threads must be > 0");
  }
  if (options->qp_count <= 0) {
    return absl::InvalidArgumentError("--qp-count must be > 0");
  }
  if (options->outstanding_wr <= 0) {
    return absl::InvalidArgumentError("--outstanding-wr must be > 0");
  }
  auto total_bytes_or = get_total_buffer_bytes(*options);
  if (!total_bytes_or.ok()) {
    return total_bytes_or.status();
  }
  return absl::OkStatus();
}

void apply_rdm_nic_filter(const Options& options) {
  if (!options.rdma_nic.empty()) {
    setenv("TENSORCAST_IB_HCA", options.rdma_nic.c_str(), 1);
  }
}

tc::v1::CommunicatorConfig make_config(const Options& options) {
  tc::v1::CommunicatorConfig cfg;
  cfg.set_enable_rdma(options.enable_rdma);
  cfg.mutable_transport()->set_tcp_conn_count(2);
  cfg.mutable_transport()->set_connect_timeout_sec(10);
  cfg.mutable_transport()->set_so_reuseport(false);
  cfg.mutable_rdma()->set_qp_count(options.qp_count);
  cfg.mutable_rdma()->set_outstanding_wr(options.outstanding_wr);
  cfg.mutable_stager()->set_buffers_per_flow(4);
  cfg.mutable_stager()->set_max_window_segments(4);
  cfg.mutable_stager()->set_expected_gpu_channels(1);
  return cfg;
}

absl::StatusOr<RdmPreflight> inspect_rdma_selection(const Options& options) {
  RdmPreflight preflight;
  if (!options.enable_rdma) {
    return preflight;
  }

  transport::RdmaContext context;
  for (const auto& dev : context.list_devs()) {
    preflight.visible_nics.push_back(dev->get_name());
  }

  if (options.memory == "gpu") {
    cudaDeviceProp prop{};
    auto prop_status = cuda::get_device_properties(options.gpu_id, &prop);
    if (!prop_status.ok()) {
      return prop_status;
    }
    preflight.gpu_name = prop.name;
    preflight.gpu_bus = prop.pciBusID;
    preflight.gpu_device = prop.pciDeviceID;
    auto selected = context.get_best_dev(options.gpu_id);
    if (selected != nullptr) {
      preflight.selected_nic = selected->get_name();
      preflight.selected_rail_id = selected->get_rail_id();
    }
  } else if (!options.rdma_nic.empty()) {
    auto selected = context.get_dev(options.rdma_nic);
    if (selected != nullptr) {
      preflight.selected_nic = selected->get_name();
      preflight.selected_rail_id = selected->get_rail_id();
    }
  }

  if (options.strict_nic) {
    if (preflight.selected_nic != options.rdma_nic) {
      return absl::FailedPreconditionError(
          absl::StrCat("requested NIC ", options.rdma_nic, " but selected ", preflight.selected_nic));
    }
    if (preflight.visible_nics.size() != 1 || preflight.visible_nics.front() != options.rdma_nic) {
      return absl::FailedPreconditionError(
          absl::StrCat("strict NIC selection expected only ", options.rdma_nic, " to be visible"));
    }
  }

  return preflight;
}

absl::Status probe_gpu_mr_registration(const Options& options, const RdmPreflight& preflight) {
  if (!options.enable_rdma || options.memory != "gpu") {
    return absl::OkStatus();
  }
  const std::string probe_nic = !options.rdma_nic.empty() ? options.rdma_nic : preflight.selected_nic;
  if (probe_nic.empty()) {
    return absl::FailedPreconditionError("gpu MR probe could not resolve NIC");
  }

  transport::RdmaContext context;
  auto dev = context.get_dev(probe_nic);
  if (dev == nullptr) {
    return absl::NotFoundError(absl::StrCat("gpu MR probe could not find NIC ", probe_nic));
  }

  Options probe_options = options;
  probe_options.bytes = std::max<uint64_t>(1ULL << 20, std::min<uint64_t>(options.bytes, 4ULL << 20));
  Buffer probe_buffer;
  auto alloc_status = probe_buffer.allocate(probe_options);
  if (!alloc_status.ok()) {
    return alloc_status;
  }

  ibv_mr* mr = nullptr;
  constexpr int kAccess = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_RELAXED_ORDERING;
  const auto res = dev->reg_mr(&mr, reinterpret_cast<void*>(probe_buffer.addr()), probe_buffer.bytes(), kAccess);
  if (res != tc::misc::SUCCESS || mr == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("gpu MR probe failed for nic=", probe_nic, " bytes=", probe_buffer.bytes(), " res=", res));
  }
  tc::misc::wrap_ibv_dereg_mr(mr);
  std::ostringstream line;
  line << "GPU_MR_PROBE"
       << " nic=" << probe_nic << " rail=" << dev->get_rail_id() << " bytes=" << probe_buffer.bytes()
       << " status=ok";
  emit_machine_line(line.str());
  return absl::OkStatus();
}

void print_preflight(const Options& options, const RdmPreflight& preflight) {
  std::ostringstream line;
  line << "PRECHECK"
       << " role=" << options.role << " memory=" << options.memory << " rdma=" << (options.enable_rdma ? 1 : 0)
       << " requested_nic=" << (options.rdma_nic.empty() ? "-" : options.rdma_nic) << " visible_nics=";
  if (preflight.visible_nics.empty()) {
    line << "-";
  } else {
    for (size_t i = 0; i < preflight.visible_nics.size(); ++i) {
      if (i > 0) {
        line << ",";
      }
      line << preflight.visible_nics[i];
    }
  }
  line << " selected_nic=" << (preflight.selected_nic.empty() ? "-" : preflight.selected_nic)
       << " selected_rail=" << preflight.selected_rail_id;
  if (!preflight.gpu_name.empty()) {
    line << " gpu=" << preflight.gpu_name << " gpu_bus=" << preflight.gpu_bus
         << " gpu_device=" << preflight.gpu_device;
  }
  emit_machine_line(line.str());
}

struct PrepareBufferProfile {
  AllocationProfile allocation;
  PatternFillProfile pattern_fill;
  uint64_t clear_us = 0;
};

absl::Status prepare_buffer(Buffer* buffer, const Options& options, bool fill_pattern, PrepareBufferProfile* profile) {
  if (buffer == nullptr) {
    return absl::InvalidArgumentError("buffer is null");
  }
  auto alloc_status = buffer->allocate(options, profile == nullptr ? nullptr : &profile->allocation);
  if (!alloc_status.ok()) {
    return alloc_status;
  }
  if (fill_pattern) {
    return buffer->fill_pattern(options, /*seed=*/0x5A, profile == nullptr ? nullptr : &profile->pattern_fill);
  }
  return buffer->clear(options, profile == nullptr ? nullptr : &profile->clear_us);
}

absl::Status run_target(const Options& options) {
  struct TargetStartupProfile {
    uint64_t communicator_init_us = 0;
    uint64_t buffer_alloc_us = 0;
    uint64_t buffer_alloc_set_device_us = 0;
    uint64_t buffer_alloc_call_us = 0;
    uint64_t buffer_fill_us = 0;
    uint64_t buffer_fill_alloc_us = 0;
    uint64_t buffer_fill_pattern_us = 0;
    uint64_t buffer_fill_copy_us = 0;
    uint64_t register_tensor_us = 0;
  };
  TargetStartupProfile startup;

  auto cfg = make_config(options);
  engine::Communicator communicator(cfg);
  const auto init_start = SteadyClock::now();
  auto init_status = communicator.init(options.listen_ip, options.listen_port);
  startup.communicator_init_us = elapsed_us(init_start, SteadyClock::now());
  if (!init_status.ok()) {
    return init_status;
  }

  std::vector<std::unique_ptr<Buffer>> buffers;
  buffers.reserve(static_cast<size_t>(options.threads));
  const int dev_type = options.memory == "gpu" ? base::COMMUNICATE_ENGINE_DEV_GPU : base::COMMUNICATE_ENGINE_DEV_CPU;
  const int dev_id = options.memory == "gpu" ? options.gpu_id : -1;
  for (int thread_id = 0; thread_id < options.threads; ++thread_id) {
    auto buffer = std::make_unique<Buffer>();
    PrepareBufferProfile profile;
    auto buffer_status = prepare_buffer(buffer.get(), options, /*fill_pattern=*/true, &profile);
    if (!buffer_status.ok()) {
      return buffer_status;
    }
    startup.buffer_alloc_us += profile.allocation.total_us;
    startup.buffer_alloc_set_device_us += profile.allocation.set_device_us;
    startup.buffer_alloc_call_us += profile.allocation.allocate_call_us;
    startup.buffer_fill_us += profile.pattern_fill.total_us;
    startup.buffer_fill_alloc_us += profile.pattern_fill.buffer_alloc_us;
    startup.buffer_fill_pattern_us += profile.pattern_fill.pattern_fill_us;
    startup.buffer_fill_copy_us += profile.pattern_fill.copy_us;

    engine::Communicator::RegisterTensorOptions register_options;
    register_options.register_mr = options.enable_rdma;
    register_options.needs_staging = !options.enable_rdma && options.memory == "gpu";
    register_options.async = false;
    register_options.direct_rdma_enabled = options.direct_rdma;
    register_options.direct_rdma_required = options.strict_direct_rdma;
    const auto register_start = SteadyClock::now();
    auto register_status = communicator.register_tensor_ex(
        tensor_key_for_thread(options.tensor_key, thread_id, options.threads),
        buffer->addr(),
        buffer->bytes(),
        dev_type,
        dev_id,
        register_options);
    startup.register_tensor_us += elapsed_us(register_start, SteadyClock::now());
    if (!register_status.ok()) {
      return register_status;
    }
    buffers.push_back(std::move(buffer));
  }

  std::ostringstream ready_line;
  ready_line << "READY"
             << " role=target"
             << " listen_ip=" << options.listen_ip << " listen_port=" << communicator.listening_port()
             << " tensor_key=" << options.tensor_key << " bytes=" << options.bytes
             << " batch_size=" << options.batch_size << " threads=" << options.threads
             << " qp_count=" << options.qp_count << " outstanding_wr=" << options.outstanding_wr
             << " memory=" << options.memory << " direct_rdma=" << (options.direct_rdma ? 1 : 0)
             << " strict_direct_rdma=" << (options.strict_direct_rdma ? 1 : 0)
             << " init_communicator_us=" << startup.communicator_init_us
             << " init_buffer_alloc_us=" << startup.buffer_alloc_us
             << " init_buffer_alloc_set_device_us=" << startup.buffer_alloc_set_device_us
             << " init_buffer_alloc_call_us=" << startup.buffer_alloc_call_us
             << " init_buffer_fill_us=" << startup.buffer_fill_us
             << " init_buffer_fill_alloc_us=" << startup.buffer_fill_alloc_us
             << " init_buffer_fill_pattern_us=" << startup.buffer_fill_pattern_us
             << " init_buffer_fill_copy_us=" << startup.buffer_fill_copy_us
             << " init_register_tensor_us=" << startup.register_tensor_us
             << " init_total_us="
             << (startup.communicator_init_us + startup.buffer_alloc_us + startup.buffer_fill_us + startup.register_tensor_us);
  emit_machine_line(ready_line.str());

  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

absl::Status validate_result(const Options& options, const transport::read_result_t& result) {
  if (!result.status.ok()) {
    return result.status;
  }
  if (options.strict_nic && !options.rdma_nic.empty() && result.local_nic != options.rdma_nic) {
    return absl::FailedPreconditionError(
        absl::StrCat("local NIC mismatch: expected=", options.rdma_nic, " actual=", result.local_nic));
  }
  if (!options.expected_remote_nic.empty() && result.remote_nic != options.expected_remote_nic) {
    return absl::FailedPreconditionError(
        absl::StrCat("remote NIC mismatch: expected=", options.expected_remote_nic, " actual=", result.remote_nic));
  }
  if (options.strict_direct_rdma && !result.rdma_zero_copy_response) {
    return absl::FailedPreconditionError("strict direct RDMA requested but response was not zero-copy");
  }
  return absl::OkStatus();
}

absl::Status run_initiator(const Options& options) {
  struct InitiatorStartupProfile {
    uint64_t communicator_init_us = 0;
    uint64_t buffer_alloc_us = 0;
    uint64_t buffer_alloc_set_device_us = 0;
    uint64_t buffer_alloc_call_us = 0;
    uint64_t buffer_initial_clear_us = 0;
    uint64_t warmup_total_us = 0;
  };
  InitiatorStartupProfile startup;

  auto cfg = make_config(options);
  engine::Communicator communicator(cfg);
  const auto init_start = SteadyClock::now();
  auto init_status = communicator.init(options.listen_ip, options.listen_port);
  startup.communicator_init_us = elapsed_us(init_start, SteadyClock::now());
  if (!init_status.ok()) {
    return init_status;
  }

  std::vector<std::unique_ptr<Buffer>> buffers;
  buffers.reserve(static_cast<size_t>(options.threads));
  for (int thread_id = 0; thread_id < options.threads; ++thread_id) {
    auto buffer = std::make_unique<Buffer>();
    PrepareBufferProfile profile;
    auto buffer_status = prepare_buffer(buffer.get(), options, /*fill_pattern=*/false, &profile);
    if (!buffer_status.ok()) {
      return buffer_status;
    }
    startup.buffer_alloc_us += profile.allocation.total_us;
    startup.buffer_alloc_set_device_us += profile.allocation.set_device_us;
    startup.buffer_alloc_call_us += profile.allocation.allocate_call_us;
    startup.buffer_initial_clear_us += profile.clear_us;
    buffers.push_back(std::move(buffer));
  }

  const int dev_type = options.memory == "gpu" ? base::COMMUNICATE_ENGINE_DEV_GPU : base::COMMUNICATE_ENGINE_DEV_CPU;
  const int dev_id = options.memory == "gpu" ? options.gpu_id : -1;

  struct BatchResult {
    transport::read_result_t first_result;
    uint64_t batch_wall_us = 0;
    uint64_t batch_total_us = 0;
    uint64_t clear_us = 0;
    uint64_t issue_us = 0;
    uint64_t wait_us = 0;
    uint64_t verify_total_us = 0;
    uint64_t verify_buffer_alloc_us = 0;
    uint64_t verify_copy_us = 0;
    uint64_t verify_checksum_us = 0;
    double avg_request_us = 0.0;
    uint64_t max_request_us = 0;
    uint64_t sum_request_us = 0;
    uint64_t request_count = 0;
    uint64_t sum_request_first_response_us = 0;
    uint64_t sum_rdma_first_post_us = 0;
    uint64_t sum_rdma_post_to_last_completion_us = 0;
    uint64_t sum_rdma_post_after_response_us = 0;
    uint64_t sum_tail_after_last_completion_us = 0;
    uint64_t sum_rdma_handshake_queue_wait_us = 0;
    uint64_t sum_rdma_response_windows = 0;
    uint64_t sum_rdma_response_segments = 0;
    uint64_t sum_rdma_wr_posted = 0;
    uint64_t sum_rdma_wc_completed = 0;
    uint64_t sum_rdma_ack_windows = 0;
    uint64_t sum_rdma_ack_segments = 0;
  };

  std::mutex io_mu;
  auto run_single =
      [&](Buffer& buffer, int iteration, int thread_id, bool verify, bool emit_result) -> absl::StatusOr<BatchResult> {
    BatchResult batch_result;
    const auto iteration_start = SteadyClock::now();
    auto clear_status = buffer.clear(options, &batch_result.clear_us);
    if (!clear_status.ok()) {
      return clear_status;
    }
    const auto issue_start = SteadyClock::now();
    std::vector<transport::future_read_result_t> futures;
    futures.reserve(static_cast<size_t>(options.batch_size));
    for (int batch_idx = 0; batch_idx < options.batch_size; ++batch_idx) {
      const auto local_addr = buffer.addr() + (options.bytes * static_cast<uint64_t>(batch_idx));
      const auto remote_offset = options.bytes * static_cast<uint64_t>(batch_idx);
      futures.push_back(communicator.read_tensor(
          tensor_key_for_thread(options.tensor_key, thread_id, options.threads),
          local_addr,
          options.bytes,
          dev_type,
          dev_id,
          options.peer_ip,
          options.peer_port,
          remote_offset));
    }
    const auto issue_end = SteadyClock::now();
    batch_result.issue_us = elapsed_us(issue_start, issue_end);

    bool saw_result = false;
    const auto wait_start = SteadyClock::now();
    for (auto& future : futures) {
      auto result = future.get();
      auto validate_status = validate_result(options, result);
      if (!validate_status.ok()) {
        return validate_status;
      }
      batch_result.sum_request_us += result.read_cost;
      batch_result.max_request_us = std::max(batch_result.max_request_us, result.read_cost);
      batch_result.request_count += 1;
      batch_result.sum_request_first_response_us += result.request_first_response_us;
      batch_result.sum_rdma_first_post_us += result.rdma_first_post_us;
      batch_result.sum_rdma_post_to_last_completion_us += result.rdma_post_to_last_completion_us;
      if (result.rdma_first_post_us > result.request_first_response_us) {
        batch_result.sum_rdma_post_after_response_us += (result.rdma_first_post_us - result.request_first_response_us);
      }
      if (result.rdma_last_completion_us > 0 && result.read_cost > result.rdma_last_completion_us) {
        batch_result.sum_tail_after_last_completion_us += (result.read_cost - result.rdma_last_completion_us);
      }
      batch_result.sum_rdma_handshake_queue_wait_us += result.rdma_handshake_queue_wait_us;
      batch_result.sum_rdma_response_windows += result.rdma_response_windows;
      batch_result.sum_rdma_response_segments += result.rdma_response_segments;
      batch_result.sum_rdma_wr_posted += result.rdma_wr_posted;
      batch_result.sum_rdma_wc_completed += result.rdma_wc_completed;
      batch_result.sum_rdma_ack_windows += result.rdma_ack_windows;
      batch_result.sum_rdma_ack_segments += result.rdma_ack_segments;
      if (!saw_result) {
        batch_result.first_result = result;
        saw_result = true;
      }
    }
    const auto wait_end = SteadyClock::now();
    batch_result.wait_us = elapsed_us(wait_start, wait_end);
    batch_result.batch_wall_us = elapsed_us(issue_start, wait_end);
    if (batch_result.request_count > 0) {
      batch_result.avg_request_us =
          static_cast<double>(batch_result.sum_request_us) / static_cast<double>(batch_result.request_count);
    }

    if (verify) {
      VerifyProfile verify_profile;
      auto verify_status = buffer.verify_pattern(options, /*seed=*/0x5A, &verify_profile);
      if (!verify_status.ok()) {
        return verify_status;
      }
      batch_result.verify_total_us = verify_profile.total_us;
      batch_result.verify_buffer_alloc_us = verify_profile.buffer_alloc_us;
      batch_result.verify_copy_us = verify_profile.copy_us;
      batch_result.verify_checksum_us = verify_profile.checksum_us;
    }
    batch_result.batch_total_us = elapsed_us(iteration_start, SteadyClock::now());

    if (emit_result) {
      auto avg_from_sum = [&](uint64_t sum) -> double {
        if (batch_result.request_count == 0) {
          return 0.0;
        }
        return static_cast<double>(sum) / static_cast<double>(batch_result.request_count);
      };
      std::ostringstream iter_line;
      iter_line << "ITER"
                << " thread=" << thread_id << " idx=" << iteration << " batch_size=" << options.batch_size
                << " status=" << static_cast<int>(batch_result.first_result.status.code()) << " local_nic="
                << (batch_result.first_result.local_nic.empty() ? "-" : batch_result.first_result.local_nic)
                << " remote_nic="
                << (batch_result.first_result.remote_nic.empty() ? "-" : batch_result.first_result.remote_nic)
                << " local_rail=" << batch_result.first_result.local_rail_id
                << " remote_rail=" << batch_result.first_result.remote_rail_id
                << " rdma=" << (batch_result.first_result.transport_is_rdma ? 1 : 0)
                << " staged=" << (batch_result.first_result.rdma_staged_response ? 1 : 0)
                << " zero_copy=" << (batch_result.first_result.rdma_zero_copy_response ? 1 : 0)
                << " request_us=" << batch_result.first_result.request_cost
                << " rdma_queue_us=" << batch_result.first_result.rdma_queue_cost
                << " rdma_regmr_us=" << batch_result.first_result.rdma_regmr_cost
                << " request_first_response_us=" << batch_result.first_result.request_first_response_us
                << " rdma_first_post_us=" << batch_result.first_result.rdma_first_post_us
                << " rdma_first_completion_us=" << batch_result.first_result.rdma_first_completion_us
                << " rdma_last_completion_us=" << batch_result.first_result.rdma_last_completion_us
                << " rdma_post_to_last_completion_us=" << batch_result.first_result.rdma_post_to_last_completion_us
                << " rdma_response_windows=" << batch_result.first_result.rdma_response_windows
                << " rdma_response_segments=" << batch_result.first_result.rdma_response_segments
                << " rdma_wr_posted=" << batch_result.first_result.rdma_wr_posted
                << " rdma_wc_completed=" << batch_result.first_result.rdma_wc_completed
                << " rdma_ack_windows=" << batch_result.first_result.rdma_ack_windows
                << " rdma_ack_segments=" << batch_result.first_result.rdma_ack_segments
                << " rdma_handshake_queue_wait_us=" << batch_result.first_result.rdma_handshake_queue_wait_us
                << " batch_avg_request_us=" << batch_result.avg_request_us
                << " batch_avg_response_wait_us=" << avg_from_sum(batch_result.sum_request_first_response_us)
                << " batch_avg_post_after_response_us=" << avg_from_sum(batch_result.sum_rdma_post_after_response_us)
                << " batch_avg_data_phase_us=" << avg_from_sum(batch_result.sum_rdma_post_to_last_completion_us)
                << " batch_avg_tail_after_last_completion_us="
                << avg_from_sum(batch_result.sum_tail_after_last_completion_us)
                << " batch_avg_handshake_wait_us=" << avg_from_sum(batch_result.sum_rdma_handshake_queue_wait_us)
                << " batch_avg_response_windows=" << avg_from_sum(batch_result.sum_rdma_response_windows)
                << " batch_avg_response_segments=" << avg_from_sum(batch_result.sum_rdma_response_segments)
                << " batch_avg_wr_posted=" << avg_from_sum(batch_result.sum_rdma_wr_posted)
                << " batch_avg_wc_completed=" << avg_from_sum(batch_result.sum_rdma_wc_completed)
                << " batch_avg_ack_windows=" << avg_from_sum(batch_result.sum_rdma_ack_windows)
                << " batch_avg_ack_segments=" << avg_from_sum(batch_result.sum_rdma_ack_segments)
                << " clear_us=" << batch_result.clear_us << " issue_us=" << batch_result.issue_us
                << " wait_us=" << batch_result.wait_us << " verify_total_us=" << batch_result.verify_total_us
                << " verify_buffer_alloc_us=" << batch_result.verify_buffer_alloc_us
                << " verify_copy_us=" << batch_result.verify_copy_us
                << " verify_checksum_us=" << batch_result.verify_checksum_us
                << " batch_avg_clear_per_request_us=" << avg_from_sum(batch_result.clear_us)
                << " batch_avg_verify_per_request_us=" << avg_from_sum(batch_result.verify_total_us)
                << " batch_avg_verify_copy_per_request_us=" << avg_from_sum(batch_result.verify_copy_us)
                << " batch_avg_verify_checksum_per_request_us=" << avg_from_sum(batch_result.verify_checksum_us)
                << " iteration_total_us=" << batch_result.batch_total_us
                << " batch_max_request_us=" << batch_result.max_request_us << " total_us=" << batch_result.batch_wall_us;
      std::lock_guard<std::mutex> lock(io_mu);
      emit_machine_line(iter_line.str());
    }
    return batch_result;
  };

  const auto warmup_start = SteadyClock::now();
  for (int thread_id = 0; thread_id < options.threads; ++thread_id) {
    for (int i = 0; i < options.warmup_iterations; ++i) {
      auto warmup_or = run_single(
          *buffers[static_cast<size_t>(thread_id)], i, thread_id, /*verify=*/options.verify, /*emit_result=*/false);
      if (!warmup_or.ok()) {
        return warmup_or.status();
      }
    }
  }
  startup.warmup_total_us = elapsed_us(warmup_start, SteadyClock::now());

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(0, options.duration_sec));
  std::vector<double> latencies_us;
  struct SummaryProfile {
    uint64_t iterations = 0;
    uint64_t requests = 0;
    uint64_t sum_batch_total_us = 0;
    uint64_t sum_clear_us = 0;
    uint64_t sum_issue_us = 0;
    uint64_t sum_wait_us = 0;
    uint64_t sum_verify_total_us = 0;
    uint64_t sum_verify_buffer_alloc_us = 0;
    uint64_t sum_verify_copy_us = 0;
    uint64_t sum_verify_checksum_us = 0;
    uint64_t sum_request_first_response_us = 0;
    uint64_t sum_rdma_first_post_us = 0;
    uint64_t sum_rdma_post_to_last_completion_us = 0;
    uint64_t sum_rdma_post_after_response_us = 0;
    uint64_t sum_tail_after_last_completion_us = 0;
    uint64_t sum_rdma_handshake_queue_wait_us = 0;
    uint64_t sum_rdma_response_windows = 0;
    uint64_t sum_rdma_response_segments = 0;
    uint64_t sum_rdma_wr_posted = 0;
    uint64_t sum_rdma_wc_completed = 0;
    uint64_t sum_rdma_ack_windows = 0;
    uint64_t sum_rdma_ack_segments = 0;
  };
  SummaryProfile summary_profile;
  std::mutex stats_mu;
  std::mutex error_mu;
  absl::Status first_error = absl::OkStatus();
  std::atomic<int> next_iteration{0};
  std::atomic<int> completed{0};
  const auto measure_start = std::chrono::steady_clock::now();

  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(options.threads));
  for (int thread_id = 0; thread_id < options.threads; ++thread_id) {
    workers.emplace_back([&, thread_id]() {
      Buffer& buffer = *buffers[static_cast<size_t>(thread_id)];
      while (true) {
        if (!first_error.ok()) {
          return;
        }
        if (options.duration_sec > 0) {
          if (std::chrono::steady_clock::now() >= deadline) {
            return;
          }
        }
        const int iteration = next_iteration.fetch_add(1, std::memory_order_relaxed);
        if (options.duration_sec <= 0 && iteration >= options.iterations) {
          return;
        }
        auto result_or = run_single(buffer, iteration, thread_id, /*verify=*/options.verify, /*emit_result=*/true);
        if (!result_or.ok()) {
          std::lock_guard<std::mutex> error_lock(error_mu);
          if (first_error.ok()) {
            first_error = result_or.status();
          }
          return;
        }
        {
          std::lock_guard<std::mutex> stats_lock(stats_mu);
          latencies_us.push_back(static_cast<double>(result_or->batch_wall_us));
          summary_profile.iterations += 1;
          summary_profile.requests += result_or->request_count;
          summary_profile.sum_batch_total_us += result_or->batch_total_us;
          summary_profile.sum_clear_us += result_or->clear_us;
          summary_profile.sum_issue_us += result_or->issue_us;
          summary_profile.sum_wait_us += result_or->wait_us;
          summary_profile.sum_verify_total_us += result_or->verify_total_us;
          summary_profile.sum_verify_buffer_alloc_us += result_or->verify_buffer_alloc_us;
          summary_profile.sum_verify_copy_us += result_or->verify_copy_us;
          summary_profile.sum_verify_checksum_us += result_or->verify_checksum_us;
          summary_profile.sum_request_first_response_us += result_or->sum_request_first_response_us;
          summary_profile.sum_rdma_first_post_us += result_or->sum_rdma_first_post_us;
          summary_profile.sum_rdma_post_to_last_completion_us += result_or->sum_rdma_post_to_last_completion_us;
          summary_profile.sum_rdma_post_after_response_us += result_or->sum_rdma_post_after_response_us;
          summary_profile.sum_tail_after_last_completion_us += result_or->sum_tail_after_last_completion_us;
          summary_profile.sum_rdma_handshake_queue_wait_us += result_or->sum_rdma_handshake_queue_wait_us;
          summary_profile.sum_rdma_response_windows += result_or->sum_rdma_response_windows;
          summary_profile.sum_rdma_response_segments += result_or->sum_rdma_response_segments;
          summary_profile.sum_rdma_wr_posted += result_or->sum_rdma_wr_posted;
          summary_profile.sum_rdma_wc_completed += result_or->sum_rdma_wc_completed;
          summary_profile.sum_rdma_ack_windows += result_or->sum_rdma_ack_windows;
          summary_profile.sum_rdma_ack_segments += result_or->sum_rdma_ack_segments;
        }
        completed.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  const auto measure_end = std::chrono::steady_clock::now();
  if (!first_error.ok()) {
    return first_error;
  }

  const int completed_count = completed.load(std::memory_order_relaxed);
  const uint64_t total_bytes =
      options.bytes * static_cast<uint64_t>(options.batch_size) * static_cast<uint64_t>(completed_count);
  const uint64_t total_requests = static_cast<uint64_t>(options.batch_size) * static_cast<uint64_t>(completed_count);
  const auto total_us = latencies_us.empty() ? 0.0 : std::accumulate(latencies_us.begin(), latencies_us.end(), 0.0);
  const auto avg_us = latencies_us.empty() ? 0.0 : total_us / static_cast<double>(latencies_us.size());
  const auto wall_us =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(measure_end - measure_start).count());
  const auto bw_GBps = wall_us <= 0.0 ? 0.0 : (static_cast<double>(total_bytes) / 1.0e9) / (wall_us / 1.0e6);
  const auto bw_gbps = bw_GBps; // Legacy alias retained for existing parsers.
  std::sort(latencies_us.begin(), latencies_us.end());
  auto percentile = [&](double p) -> double {
    if (latencies_us.empty()) {
      return 0.0;
    }
    const auto pos = std::clamp(
        static_cast<size_t>(std::ceil((p / 100.0) * static_cast<double>(latencies_us.size()))) - 1,
        static_cast<size_t>(0),
        latencies_us.size() - 1);
    return latencies_us[pos];
  };
  auto avg_request_metric = [&](uint64_t sum) -> double {
    if (summary_profile.requests == 0) {
      return 0.0;
    }
    return static_cast<double>(sum) / static_cast<double>(summary_profile.requests);
  };
  auto avg_iteration_metric = [&](uint64_t sum) -> double {
    if (summary_profile.iterations == 0) {
      return 0.0;
    }
    return static_cast<double>(sum) / static_cast<double>(summary_profile.iterations);
  };
  const uint64_t amortizable_init_total_us =
      startup.communicator_init_us + startup.buffer_alloc_us + startup.buffer_initial_clear_us;
  const uint64_t amortizable_total_us = amortizable_init_total_us + startup.warmup_total_us;
  const auto amortized_per_iteration_us =
      summary_profile.iterations == 0
          ? 0.0
          : static_cast<double>(amortizable_total_us) / static_cast<double>(summary_profile.iterations);
  const auto amortized_per_request_us =
      total_requests == 0 ? 0.0 : static_cast<double>(amortizable_total_us) / static_cast<double>(total_requests);
  std::ostringstream summary_line;
  summary_line << "SUMMARY"
               << " iterations=" << completed_count << " requests=" << total_requests << " threads=" << options.threads
               << " batch_size=" << options.batch_size << " qp_count=" << options.qp_count
               << " outstanding_wr=" << options.outstanding_wr << " bytes=" << total_bytes << " wall_us=" << wall_us
               << " avg_us=" << avg_us << " p50_us=" << percentile(50.0) << " p95_us=" << percentile(95.0)
               << " p99_us=" << percentile(99.0) << " bw_GBps=" << bw_GBps << " bw_gbps=" << bw_gbps
               << " amortizable_init_communicator_us=" << startup.communicator_init_us
               << " amortizable_init_buffer_alloc_us=" << startup.buffer_alloc_us
               << " amortizable_init_buffer_alloc_set_device_us=" << startup.buffer_alloc_set_device_us
               << " amortizable_init_buffer_alloc_call_us=" << startup.buffer_alloc_call_us
               << " amortizable_init_buffer_clear_us=" << startup.buffer_initial_clear_us
               << " amortizable_warmup_total_us=" << startup.warmup_total_us
               << " amortizable_init_total_us=" << amortizable_init_total_us
               << " amortizable_total_us=" << amortizable_total_us
               << " amortized_per_iteration_us=" << amortized_per_iteration_us
               << " amortized_per_request_us=" << amortized_per_request_us
               << " recurring_avg_iteration_total_us=" << avg_iteration_metric(summary_profile.sum_batch_total_us)
               << " recurring_avg_clear_us=" << avg_iteration_metric(summary_profile.sum_clear_us)
               << " recurring_avg_issue_us=" << avg_iteration_metric(summary_profile.sum_issue_us)
               << " recurring_avg_wait_us=" << avg_iteration_metric(summary_profile.sum_wait_us)
               << " recurring_avg_verify_total_us=" << avg_iteration_metric(summary_profile.sum_verify_total_us)
               << " recurring_avg_verify_buffer_alloc_us="
               << avg_iteration_metric(summary_profile.sum_verify_buffer_alloc_us)
               << " recurring_avg_verify_copy_us=" << avg_iteration_metric(summary_profile.sum_verify_copy_us)
               << " recurring_avg_verify_checksum_us=" << avg_iteration_metric(summary_profile.sum_verify_checksum_us)
               << " recurring_avg_clear_per_request_us=" << avg_request_metric(summary_profile.sum_clear_us)
               << " recurring_avg_verify_per_request_us=" << avg_request_metric(summary_profile.sum_verify_total_us)
               << " recurring_avg_verify_buffer_alloc_per_request_us="
               << avg_request_metric(summary_profile.sum_verify_buffer_alloc_us)
               << " recurring_avg_verify_copy_per_request_us=" << avg_request_metric(summary_profile.sum_verify_copy_us)
               << " recurring_avg_verify_checksum_per_request_us="
               << avg_request_metric(summary_profile.sum_verify_checksum_us)
               << " avg_response_wait_us=" << avg_request_metric(summary_profile.sum_request_first_response_us)
               << " avg_first_post_us=" << avg_request_metric(summary_profile.sum_rdma_first_post_us)
               << " avg_post_after_response_us=" << avg_request_metric(summary_profile.sum_rdma_post_after_response_us)
               << " avg_data_phase_us=" << avg_request_metric(summary_profile.sum_rdma_post_to_last_completion_us)
               << " avg_tail_after_last_completion_us="
               << avg_request_metric(summary_profile.sum_tail_after_last_completion_us)
               << " avg_handshake_wait_us=" << avg_request_metric(summary_profile.sum_rdma_handshake_queue_wait_us)
               << " avg_response_windows=" << avg_request_metric(summary_profile.sum_rdma_response_windows)
               << " avg_response_segments=" << avg_request_metric(summary_profile.sum_rdma_response_segments)
               << " avg_wr_posted=" << avg_request_metric(summary_profile.sum_rdma_wr_posted)
               << " avg_wc_completed=" << avg_request_metric(summary_profile.sum_rdma_wc_completed)
               << " avg_ack_windows=" << avg_request_metric(summary_profile.sum_rdma_ack_windows)
               << " avg_ack_segments=" << avg_request_metric(summary_profile.sum_rdma_ack_segments);
  emit_machine_line(summary_line.str());
  return absl::OkStatus();
}

} // namespace

int main(int argc, char** argv) {
  std::cout << std::unitbuf;
  Options options;
  const auto parse_status = parse_options(argc, argv, &options);
  if (!parse_status.ok()) {
    std::cerr << parse_status << "\n";
    print_usage(argv[0]);
    return 2;
  }

  apply_rdm_nic_filter(options);
  auto preflight_or = inspect_rdma_selection(options);
  if (!preflight_or.ok()) {
    std::cerr << preflight_or.status() << "\n";
    return 3;
  }
  print_preflight(options, *preflight_or);
  if (options.probe_gpu_mr || options.direct_rdma || options.strict_direct_rdma) {
    const auto probe_status = probe_gpu_mr_registration(options, *preflight_or);
    if (!probe_status.ok()) {
      std::cerr << probe_status << "\n";
      return 4;
    }
  }

  absl::Status status = absl::OkStatus();
  if (options.role == "inspect") {
    return 0;
  }
  if (options.role == "target") {
    status = run_target(options);
  } else if (options.role == "initiator") {
    status = run_initiator(options);
  } else {
    status = absl::InvalidArgumentError(std::format("unknown role: {}", options.role));
  }

  if (!status.ok()) {
    std::cerr << status << "\n";
    return 1;
  }
  return 0;
}
