// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <ranges>
#include <string>

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <grpc/grpc.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/config/daemon_config_io.h"
#include "core/common/logging_init.h"
#include "core/common/memory/pinned_memory_authority.h"
#include "core/common/otel/init.h"
#include "core/common/trace/trace_manager.h"
#include "core/cuda/cuda_api.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/grpc_service_impl.h"
#include "daemon/worker_lifecycle_manager.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gsl/pointers"

#include <pthread.h>
#include <csignal>
#include <fstream>
#include <sstream>

ABSL_FLAG(std::string, config, "", "Path to unified daemon config (YAML/JSON)");
ABSL_FLAG(bool, use_cursor_pagination, false, "Enable opaque cursor pagination for GetLoadedReplicasV2");
using namespace tensorcast;

namespace {

absl::Status probe_memfd_shared_mapping() {
  constexpr size_t kProbeBytes = 4096;
  int fd = static_cast<int>(::syscall(SYS_memfd_create, "tensorcast_probe", MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, "memfd_create probe failed");
  }
  if (::ftruncate(fd, static_cast<off_t>(kProbeBytes)) != 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, "ftruncate probe failed");
  }
  if (::fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK) != 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, "fcntl(F_ADD_SEALS) probe failed");
  }
  void* addr = ::mmap(nullptr, kProbeBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, "mmap(MAP_SHARED) probe failed");
  }
  // Touch at least one byte to force fault.
  static_cast<volatile char*>(addr)[0] = 0;
  ::munmap(addr, kProbeBytes);
  ::close(fd);
  return absl::OkStatus();
}

absl::Status ensure_local_handle_parent_dir(const std::filesystem::path& dir) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(dir, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("stat failed for ", dir.string()));
  }
  if (!exists) {
    std::filesystem::create_directories(dir, ec);
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), absl::StrCat("create_directories failed for ", dir.string()));
    }
    if (::chmod(dir.c_str(), 0700) < 0) {
      return absl::ErrnoToStatus(errno, absl::StrCat("chmod(0700) failed for ", dir.string()));
    }
  }
  if (!std::filesystem::is_directory(dir, ec)) {
    if (ec) {
      return absl::ErrnoToStatus(ec.value(), absl::StrCat("stat failed for ", dir.string()));
    }
    return absl::InvalidArgumentError(absl::StrCat("local handle parent is not a directory: ", dir.string()));
  }
  struct stat st{};
  if (::stat(dir.c_str(), &st) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("stat failed for ", dir.string()));
  }
  const uid_t uid = ::geteuid();
  if (st.st_uid != uid) {
    return absl::PermissionDeniedError(absl::StrCat("local handle parent dir not owned by daemon uid: ", dir.string()));
  }
  if ((st.st_mode & S_IWOTH) != 0 && (st.st_mode & S_ISVTX) == 0) {
    return absl::PermissionDeniedError(
        absl::StrCat("local handle parent dir is world-writable without sticky bit: ", dir.string()));
  }
  return absl::OkStatus();
}

std::optional<std::string> read_machine_id() {
  std::ifstream in("/etc/machine-id");
  if (!in.is_open()) {
    return std::nullopt;
  }
  std::string line;
  std::getline(in, line);
  const auto trimmed = [&line]() {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto begin = std::ranges::find_if(line, not_space);
    auto end = std::ranges::find_if(std::ranges::reverse_view(line), not_space).base();
    if (begin >= end) {
      return std::string();
    }
    return std::string(begin, end);
  }();
  if (trimmed.empty()) {
    return std::nullopt;
  }
  return std::make_optional(trimmed);
}

std::string sanitize_component(std::string value) {
  for (char& c : value) {
    if (c == '/' || c == '\\') {
      c = '_';
    }
  }
  return value;
}

std::string host_id() {
  char hostname[256];
  if (::gethostname(hostname, sizeof(hostname)) != 0) {
    return "unknown";
  }
  hostname[sizeof(hostname) - 1] = '\0';
  std::string host = hostname[0] ? hostname : "unknown";
  if (auto machine_id = read_machine_id(); machine_id.has_value()) {
    host = absl::StrCat(host, "-", *machine_id);
  }
  return sanitize_component(std::move(host));
}

absl::StatusOr<std::filesystem::path> tensorcast_home_dir() {
  if (const char* override = std::getenv("TENSORCAST_HOME"); override && *override) {
    return std::filesystem::path(override);
  }
  const char* home = std::getenv("HOME");
  if (!home || !*home) {
    return absl::InvalidArgumentError("HOME is not set; cannot resolve TensorCast runtime root");
  }
  return std::filesystem::path(home) / ".tensorcast";
}

absl::StatusOr<std::filesystem::path> discover_daemon_state_dir() {
  const char* instance = std::getenv("TENSORCAST_INSTANCE");
  if (!instance || !*instance) {
    return absl::InvalidArgumentError("TENSORCAST_INSTANCE is not set; auto-discovery requires a daemon session id");
  }
  auto home_or = tensorcast_home_dir();
  if (!home_or.ok()) {
    return home_or.status();
  }
  const std::string hid = host_id();
  if (hid.empty()) {
    return absl::InvalidArgumentError("Host id is empty; cannot resolve TensorCast runtime root");
  }
  return *home_or / "hosts" / hid / "sessions" / instance / "session";
}

absl::StatusOr<std::string> discover_local_handle_socket_path() {
  auto state_dir_or = discover_daemon_state_dir();
  if (!state_dir_or.ok()) {
    return state_dir_or.status();
  }
  const std::filesystem::path& dir = *state_dir_or;
  absl::Status st = ensure_local_handle_parent_dir(dir);
  if (!st.ok()) {
    return st;
  }
  std::filesystem::path sock = dir / "local_handle.sock";
  return sock.string();
}

absl::StatusOr<std::optional<uint64_t>> read_cgroup_v2_memory_max() {
  std::ifstream in("/proc/self/cgroup");
  if (!in.is_open()) {
    return std::optional<uint64_t>{};
  }
  std::string line;
  std::string rel;
  while (std::getline(in, line)) {
    // cgroup v2 line format: 0::/some/path
    if (line.rfind("0::", 0) == 0) {
      rel = line.substr(3);
      break;
    }
  }
  if (rel.empty()) {
    return std::optional<uint64_t>{};
  }
  std::filesystem::path cg_root("/sys/fs/cgroup");
  std::filesystem::path cg_path = cg_root / std::filesystem::path(rel).relative_path();
  std::filesystem::path max_path = cg_path / "memory.max";
  std::ifstream max_in(max_path);
  if (!max_in.is_open()) {
    return std::optional<uint64_t>{};
  }
  std::string raw;
  std::getline(max_in, raw);
  if (raw.empty() || raw == "max") {
    return std::optional<uint64_t>{};
  }
  try {
    const uint64_t limit = std::stoull(raw);
    return std::optional<uint64_t>{limit};
  } catch (...) {
    return std::optional<uint64_t>{};
  }
}

} // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  common::ensure_logging_initialized();
  // Avoid global using-directives per project guidelines
  // Note: config loading happens below; defer OTel/log-sink init until then.
  const std::string cfg_path = absl::GetFlag(FLAGS_config);
  if (cfg_path.empty()) {
    LOG(ERROR) << "Missing required --config=/path/to/store_daemon_config.{yaml,json}";
    return 2;
  }
  absl::StatusOr<config::v1::DaemonConfig> cfg_or = common::config::load_daemon_config_from_file(cfg_path);
  if (!cfg_or.ok()) {
    LOG(ERROR) << "Failed to load config: " << cfg_or.status();
    return 2;
  }
  auto cfg = *cfg_or;
  const std::filesystem::path storage_root_cfg = cfg.server().storage_path();
  std::filesystem::path storage_root;
  if (storage_root_cfg.empty()) {
    LOG(INFO) << "server.storage_path is empty; disk materialization is disabled";
  } else {
    std::error_code storage_ec;
    const bool storage_exists = std::filesystem::exists(storage_root_cfg, storage_ec);
    if (storage_ec) {
      LOG(ERROR) << "INVALID_ARGUMENT: Failed to stat server.storage_path (" << storage_root_cfg.string()
                 << "): " << storage_ec.message();
      return 2;
    }
    if (!storage_exists) {
      LOG(ERROR) << "INVALID_ARGUMENT: server.storage_path does not exist: " << storage_root_cfg.string();
      return 2;
    }
    const bool storage_is_dir = std::filesystem::is_directory(storage_root_cfg, storage_ec);
    if (storage_ec) {
      LOG(ERROR) << "INVALID_ARGUMENT: Failed to stat server.storage_path (" << storage_root_cfg.string()
                 << "): " << storage_ec.message();
      return 2;
    }
    if (!storage_is_dir) {
      LOG(ERROR) << "INVALID_ARGUMENT: server.storage_path must be a directory: " << storage_root_cfg.string();
      return 2;
    }
    storage_root = std::filesystem::weakly_canonical(storage_root_cfg, storage_ec);
    if (storage_ec) {
      LOG(ERROR) << "INVALID_ARGUMENT: Failed to canonicalize server.storage_path (" << storage_root_cfg.string()
                 << "): " << storage_ec.message();
      return 2;
    }
  }

  // Block SIGINT/SIGTERM in this main thread BEFORE starting any threads so that
  // all subsequently created threads inherit the blocked mask. We'll handle
  // these signals via sigwait in a dedicated thread later to perform a
  // cooperative shutdown (gRPC shutdown + worker unregistration).
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &set, nullptr);

  // Configure CUDA debug toggles
  cuda::configure_same_process_ipc_fallback(cfg.debug().cuda().enable_same_process_ipc_fallback());

  LOG(INFO) << "Config: " << cfg.DebugString();

  if (!cfg.has_pinned_memory()) {
    LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory is required (no legacy pinned pool fields exist)";
    return 2;
  }

  auto duration_to_millis = [](const google::protobuf::Duration& d) -> std::chrono::milliseconds {
    return std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  };

  auto duration_to_absl = [](const google::protobuf::Duration& d) -> absl::Duration {
    return absl::Seconds(d.seconds()) + absl::Nanoseconds(d.nanos());
  };

  const auto pinned_allocation_timeout_ms = [&]() -> std::chrono::milliseconds {
    if (cfg.pinned_memory().has_allocation_timeout()) {
      return duration_to_millis(cfg.pinned_memory().allocation_timeout());
    }
    return std::chrono::milliseconds(30000);
  }();

  common::memory::PinnedMemoryAuthority::Config pm_cfg;
  if (cfg.pinned_memory().has_allocation_timeout()) {
    pm_cfg.allocation_timeout = duration_to_absl(cfg.pinned_memory().allocation_timeout());
  }
  pm_cfg.classes.reserve(static_cast<size_t>(cfg.pinned_memory().classes_size()));
  uint64_t pinned_total_bytes = 0;
  for (const auto& cls : cfg.pinned_memory().classes()) {
    common::memory::PinnedMemoryAuthority::ClassConfig cc;
    cc.name = cls.name();
    cc.slice_bytes = cls.slice_bytes();
    cc.pool_bytes = cls.pool_bytes();
    cc.rdma_preregister = cls.rdma_preregister();
    pinned_total_bytes += cls.pool_bytes();
    pm_cfg.classes.push_back(std::move(cc));
  }

  absl::StatusOr<std::shared_ptr<common::memory::PinnedMemoryAuthority>> pma_or =
      common::memory::PinnedMemoryAuthority::create(std::move(pm_cfg));
  if (!pma_or.ok()) {
    LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory invalid: " << pma_or.status();
    return 2;
  }
  auto pma = std::move(*pma_or);

  auto engine_pool_or = pma->get_class_pool("engine");
  auto comm_gpu_pool_or = pma->get_class_pool("comm_gpu");
  auto comm_cpu_pool_or = pma->get_class_pool("comm_cpu");
  if (!engine_pool_or.ok() || !comm_gpu_pool_or.ok() || !comm_cpu_pool_or.ok()) {
    LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory must define required classes: engine, comm_gpu, comm_cpu";
    return 2;
  }
  auto engine_pool = std::move(*engine_pool_or);
  auto comm_gpu_pool = std::move(*comm_gpu_pool_or);
  auto comm_cpu_pool = std::move(*comm_cpu_pool_or);

  if (cfg.engine().artifact_chunk_bytes() % engine_pool->slice_bytes() != 0) {
    LOG(ERROR)
        << "INVALID_ARGUMENT: engine.artifact_chunk_bytes must be a multiple of pinned_memory.classes[name=engine].slice_bytes";
    return 2;
  }

  uint32_t streaming_buffer_chunks = cfg.engine().streaming_buffer_chunks();
  if (streaming_buffer_chunks == 0) {
    LOG(WARNING) << "engine.streaming_buffer_chunks is unset/0; defaulting to 16";
    streaming_buffer_chunks = 16;
  }

  // Fail fast on communicator pinned sizing before starting communicator threads.
  const int buffers_per_flow = cfg.communicator().stager().buffers_per_flow();
  if (buffers_per_flow <= 0) {
    LOG(ERROR) << "INVALID_ARGUMENT: communicator.stager.buffers_per_flow must be > 0";
    return 2;
  }
  int tcp_conn_count = cfg.communicator().transport().tcp_conn_count();
  if (tcp_conn_count <= 0) {
    // Keep consistent with Communicator defaults (proto comment: default 8).
    tcp_conn_count = 8;
  }
  tcp_conn_count = std::max(2, tcp_conn_count);

  const size_t num_buffers = static_cast<size_t>(buffers_per_flow);
  const size_t required_gpu_slices = num_buffers + (num_buffers * static_cast<size_t>(tcp_conn_count));
  const size_t capacity_gpu_slices = comm_gpu_pool->capacity_slices();
  if (capacity_gpu_slices < required_gpu_slices) {
    LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory.classes[name=comm_gpu] too small: capacity_slices="
               << capacity_gpu_slices << " required_slices=" << required_gpu_slices
               << " slice_bytes=" << comm_gpu_pool->slice_bytes() << " (buffers_per_flow=" << buffers_per_flow
               << " tcp_conn_count=" << tcp_conn_count << ")";
    return 2;
  }

  if (comm_cpu_pool && comm_cpu_pool.get() != comm_gpu_pool.get()) {
    const size_t required_cpu_slices = num_buffers;
    const size_t capacity_cpu_slices = comm_cpu_pool->capacity_slices();
    if (capacity_cpu_slices < required_cpu_slices) {
      LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory.classes[name=comm_cpu] too small: capacity_slices="
                 << capacity_cpu_slices << " required_slices=" << required_cpu_slices
                 << " slice_bytes=" << comm_cpu_pool->slice_bytes() << " (buffers_per_flow=" << buffers_per_flow << ")";
      return 2;
    }
  }

  const uint32_t expected_gpu_channels = cfg.communicator().stager().expected_gpu_channels();
  if (expected_gpu_channels > 0) {
    const size_t stager_reserve = num_buffers;
    const size_t available_gpu_slices =
        (capacity_gpu_slices > stager_reserve) ? (capacity_gpu_slices - stager_reserve) : 0;
    const size_t computed_limit =
        (buffers_per_flow > 0) ? (available_gpu_slices / static_cast<size_t>(buffers_per_flow)) : 0;
    if (static_cast<size_t>(expected_gpu_channels) > computed_limit) {
      LOG(ERROR) << "INVALID_ARGUMENT: communicator.stager.expected_gpu_channels=" << expected_gpu_channels
                 << " exceeds staging capacity: computed_limit=" << computed_limit
                 << " (gpu_pool_slices=" << capacity_gpu_slices << " reserve=" << stager_reserve
                 << " buffers_per_flow=" << buffers_per_flow
                 << "). Increase pinned_memory.classes[name=comm_gpu].pool_bytes "
                 << "or reduce expected_gpu_channels.";
      return 2;
    }
  }

  // Map config to StoreEngineOptions
  store::StoreEngineOptions opts;
  opts.storage_path = storage_root.string();
  opts.num_thread = static_cast<int>(cfg.server().num_threads());
  opts.memory_pool_size = static_cast<size_t>(engine_pool->capacity_slices() * engine_pool->slice_bytes());
  opts.tx_slice_bytes = static_cast<size_t>(engine_pool->slice_bytes());
  opts.pinned_buffer_pool_override = engine_pool;
  opts.pinned_memory_authority = pma;
  opts.pinned_total_bytes = static_cast<size_t>(pinned_total_bytes);
  opts.artifact_chunk_bytes = static_cast<size_t>(cfg.engine().artifact_chunk_bytes());
  opts.streaming_buffer_chunks = streaming_buffer_chunks;
  opts.pinned_memory_timeout = pinned_allocation_timeout_ms;
  opts.cpu_shared_memory_enabled = cfg.engine().cpu_shared_memory().enabled();
  if (opts.cpu_shared_memory_enabled && cfg.lifecycle().handle_leases().local_handle_socket_path().empty()) {
    auto path_or = discover_local_handle_socket_path();
    if (!path_or.ok()) {
      LOG(ERROR) << "INVALID_ARGUMENT: lifecycle.handle_leases.local_handle_socket_path is empty and auto-discovery "
                    "failed: "
                 << path_or.status();
      return 2;
    }
    cfg.mutable_lifecycle()->mutable_handle_leases()->set_local_handle_socket_path(*path_or);
    LOG(INFO) << "Auto-selected lifecycle.handle_leases.local_handle_socket_path=" << *path_or;
  }
  if (cfg.engine().has_memory_tiers()) {
    store::MemoryTierConfig tiers;
    const auto& mt = cfg.engine().memory_tiers();
    tiers.enable_preemptible_memory = mt.enable_preemptible();
    tiers.stable_bytes = mt.stable_bytes();
    tiers.preemptible_limit_bytes = mt.preemptible_limit_bytes();
    tiers.preemptible_low_watermark_ratio = mt.preemptible_low_watermark_ratio();
    opts.memory_tier_config = tiers;
  }

  if (opts.cpu_shared_memory_enabled) {
    if (!cfg.engine().has_memory_tiers() || cfg.engine().memory_tiers().stable_bytes() == 0) {
      LOG(ERROR) << "INVALID_ARGUMENT: engine.cpu_shared_memory.enabled requires engine.memory_tiers.stable_bytes > 0";
      return 2;
    }
    if (cfg.lifecycle().handle_leases().local_handle_socket_path().empty()) {
      LOG(ERROR) << "INVALID_ARGUMENT: engine.cpu_shared_memory.enabled requires "
                    "lifecycle.handle_leases.local_handle_socket_path (auto-discovery needs TENSORCAST_INSTANCE; "
                    "set explicitly when daemon and client run in different pods)";
      return 2;
    }
    const absl::Status memfd_probe = probe_memfd_shared_mapping();
    if (!memfd_probe.ok()) {
      LOG(ERROR) << "INVALID_ARGUMENT: CPU shared memory enabled but memfd probe failed: " << memfd_probe;
      return 2;
    }
    auto limit_or = read_cgroup_v2_memory_max();
    if (limit_or.ok() && limit_or->has_value()) {
      const uint64_t limit = **limit_or;
      const uint64_t required = static_cast<uint64_t>(pinned_total_bytes) + cfg.engine().memory_tiers().stable_bytes();
      if (required > limit) {
        LOG(ERROR) << "INVALID_ARGUMENT: cgroup memory.max too small for pinned+stable: required=" << required
                   << " memory.max=" << limit;
        return 2;
      }
    }
  }

  // Communicator setup (always create; RDMA enable is a config toggle inside engine)
  std::shared_ptr<store::components::CommunicationManager> comm_mgr;
  uint16_t p2p_port = 0;
  std::string p2p_host = cfg.server().listen().host();
  if (cfg.server().has_p2p_listen()) {
    p2p_host = cfg.server().p2p_listen().host().empty() ? p2p_host : cfg.server().p2p_listen().host();
    p2p_port = static_cast<uint16_t>(cfg.server().p2p_listen().port());
  }
  comm_mgr = std::make_shared<store::components::CommunicationManager>();
  {
    bool prereg_gpu = false;
    bool prereg_cpu = false;
    if (cfg.communicator().enable_rdma()) {
      auto cfg_gpu_or = pma->get_class_config("comm_gpu");
      auto cfg_cpu_or = pma->get_class_config("comm_cpu");
      prereg_gpu = cfg_gpu_or.ok() && cfg_gpu_or->rdma_preregister;
      prereg_cpu = cfg_cpu_or.ok() && cfg_cpu_or->rdma_preregister;
    }
    tensorcast::communicator::engine::Communicator::PinnedStagingPools pools{
        .gpu_pool = comm_gpu_pool,
        .cpu_pool = comm_cpu_pool,
        .preregister_gpu = prereg_gpu,
        .preregister_cpu = prereg_cpu,
        .staging_wait_timeout = pma->config().allocation_timeout,
    };
    auto st = comm_mgr->initialize_with_config_and_pools(p2p_host, p2p_port, cfg.communicator(), std::move(pools));
    if (!st.ok()) {
      LOG(WARNING) << "Failed to initialize communication engine: " << st.message();
    } else {
      const uint16_t actual_port = comm_mgr->listen_port();
      if (actual_port != 0) {
        p2p_port = actual_port;
      }
      opts.comm_manager = comm_mgr;
    }
  }
  opts.p2p_port = p2p_port;
  opts.p2p_listen_host = p2p_host;
  opts.enable_rdma = cfg.communicator().enable_rdma();

  if (cfg.high_availability().enabled() && p2p_port == 0) {
    LOG(ERROR) << "Global Store high availability requires server.p2p_listen.port to be non-zero;"
               << " configure a valid P2P port before starting the daemon.";
    return 2;
  }

  // Global Store HA - pick first endpoint if enabled
  std::string gs_addr;
  if (cfg.high_availability().enabled() && !cfg.high_availability().global_store_endpoints().empty()) {
    const auto& ep = cfg.high_availability().global_store_endpoints(0);
    gs_addr = absl::StrCat(ep.host(), ":", ep.port());
  }
  opts.global_store_address = gs_addr;
  std::shared_ptr<store::components::IGlobalStoreClient> shared_global_store_client;
  if (!gs_addr.empty()) {
    store::components::GlobalStoreClientConfig gs_client_cfg;
    gs_client_cfg.global_store_address = gs_addr;
    gs_client_cfg.cluster_token = cfg.meta().cluster_token();
    shared_global_store_client = std::make_shared<store::components::GlobalStoreClient>(std::move(gs_client_cfg));
    opts.global_store_client = shared_global_store_client;
  }

  // Configure logging level/VLOG and optional sinks, then initialize OTel
  common::initialize_logging_from_config(cfg.observability().logging());
  if (!common::otel::init_from_config(cfg.observability(), "store-daemon")) {
    LOG(WARNING) << "OpenTelemetry initialization failed; continuing without telemetry";
  }
  // Configure Chrome trace directory (optional)
  if (!cfg.observability().tracing().chrome_trace_dir().empty()) {
    common::trace::TraceManager::set_chrome_trace_dir(cfg.observability().tracing().chrome_trace_dir());
  }

  // Async runtime shared by daemon + embedded store.
  auto async_runtime = std::make_shared<common::AsyncRuntime>(common::AsyncRuntime::Options{
      .cpu_threads = static_cast<size_t>(std::max<int>(1, opts.num_thread)),
      .blocking_threads = static_cast<size_t>(std::max<int>(2, opts.num_thread)),
      .thread_name_prefix = "tensorcast",
  });
  opts.async_runtime = async_runtime;

  auto engine = std::make_shared<store::StoreEngine>(opts);

  // Map lifecycle options into service options
  daemon::StoreDaemonServiceImpl::Options svc_opts;
  if (cfg.lifecycle().has_sessions_ttl()) {
    const auto& d = cfg.lifecycle().sessions_ttl();
    svc_opts.sessions_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_locks_ttl()) {
    const auto& d = cfg.lifecycle().locks_ttl();
    svc_opts.locks_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_sessions_sweep_interval()) {
    const auto& d = cfg.lifecycle().sessions_sweep_interval();
    svc_opts.sessions_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_locks_sweep_interval()) {
    const auto& d = cfg.lifecycle().locks_sweep_interval();
    svc_opts.locks_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_verification_sweep_interval()) {
    const auto& d = cfg.lifecycle().verification_sweep_interval();
    svc_opts.verification_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_proc_check_interval()) {
    const auto& d = cfg.lifecycle().proc_check_interval();
    svc_opts.proc_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  svc_opts.enable_periodic_eviction = cfg.lifecycle().enable_periodic_eviction();
  svc_opts.gpu_memory_limit_fraction = cfg.lifecycle().gpu_memory_limit_fraction();
  if (cfg.lifecycle().has_eviction_loop_interval()) {
    const auto& d = cfg.lifecycle().eviction_loop_interval();
    svc_opts.eviction_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  svc_opts.storage_path = storage_root;
  svc_opts.local_handle_socket_path = cfg.lifecycle().handle_leases().local_handle_socket_path();
  if (cfg.lifecycle().handle_leases().has_ttl()) {
    svc_opts.handle_lease_ttl = duration_to_millis(cfg.lifecycle().handle_leases().ttl());
  }
  svc_opts.handle_lease_max_mints_per_second = cfg.lifecycle().handle_leases().max_mints_per_second();
  svc_opts.cpu_shared_memory_enabled = opts.cpu_shared_memory_enabled;
  // Observability high-cardinality attributes: default off (config hook TBD)
  svc_opts.allow_high_card_attrs = false;
  // Feature flags (override via flags for now)
  svc_opts.use_cursor_pagination = absl::GetFlag(FLAGS_use_cursor_pagination);

  daemon::StoreDaemonServiceImpl service(engine, svc_opts, async_runtime);

  // gRPC server
  const std::string listen_addr = absl::StrCat(cfg.server().listen().host(), ":", cfg.server().listen().port());
  grpc::ServerBuilder builder;
  // Do not override gRPC max receive size; defaults are sufficient for metadata-only RPCs
  if (cfg.server().grpc().max_concurrent_streams() > 0) {
    builder.AddChannelArgument("grpc.max_concurrent_streams", cfg.server().grpc().max_concurrent_streams());
  }
  std::shared_ptr<grpc::ServerCredentials> creds;
  if (cfg.server().grpc().tls().enabled()) {
    std::string cert;
    std::string key;
    std::string ca;
    auto read_all = [](const std::string& path) -> std::string {
      std::ifstream f(path);
      if (!f.is_open())
        return {};
      std::stringstream ss;
      ss << f.rdbuf();
      return ss.str();
    };
    cert = read_all(cfg.server().grpc().tls().cert_file());
    key = read_all(cfg.server().grpc().tls().key_file());
    if (!cfg.server().grpc().tls().client_ca_file().empty()) {
      ca = read_all(cfg.server().grpc().tls().client_ca_file());
    }
    grpc::SslServerCredentialsOptions ssl_opts;
    if (!ca.empty()) {
      ssl_opts.pem_root_certs = ca;
    }
    grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp{.private_key = key, .cert_chain = cert};
    ssl_opts.pem_key_cert_pairs.push_back(pkcp);
    creds = grpc::SslServerCredentials(ssl_opts);
  } else {
    creds = grpc::InsecureServerCredentials();
  }
  builder.AddListeningPort(listen_addr, creds);

  // Apply channel/server args for keepalive and connection lifetimes
  auto to_ms = [](const google::protobuf::Duration& d) -> int {
    return static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
  };
  if (cfg.server().grpc().has_keepalive_time()) {
    builder.AddChannelArgument("grpc.keepalive_time_ms", to_ms(cfg.server().grpc().keepalive_time()));
  }
  if (cfg.server().grpc().has_keepalive_timeout()) {
    builder.AddChannelArgument("grpc.keepalive_timeout_ms", to_ms(cfg.server().grpc().keepalive_timeout()));
  }
  if (cfg.server().grpc().has_max_connection_idle()) {
    builder.AddChannelArgument("grpc.max_connection_idle_ms", to_ms(cfg.server().grpc().max_connection_idle()));
  }
  if (cfg.server().grpc().has_max_connection_age()) {
    builder.AddChannelArgument("grpc.max_connection_age_ms", to_ms(cfg.server().grpc().max_connection_age()));
  }
  // tcp_nodelay / so_reuseport toggles
  builder.AddChannelArgument("grpc.tcp_nodelay", cfg.server().grpc().tcp_nodelay() ? 1 : 0);
  builder.AddChannelArgument("grpc.so_reuseport", cfg.server().grpc().so_reuseport() ? 1 : 0);
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

  if (server == nullptr) {
    LOG(ERROR) << "Failed to start gRPC server";
    return 2;
  }

  LOG(INFO) << "tensorcast-daemon listening on " << listen_addr;

  // Start worker lifecycle if configured
  std::unique_ptr<daemon::WorkerLifecycleManager> lifecycle;
  if (!gs_addr.empty() && cfg.high_availability().enabled()) {
    daemon::WorkerLifecycleManager::Options lopts;
    lopts.global_store_addr = gs_addr;
    lopts.listen_addr = listen_addr;
    // Prefer explicit advertise.host if provided
    if (cfg.server().has_advertise() && !cfg.server().advertise().host().empty()) {
      lopts.advertise_host = cfg.server().advertise().host();
    }
    lopts.p2p_port = p2p_port;
    if (cfg.high_availability().has_heartbeat_interval()) {
      const auto& d = cfg.high_availability().heartbeat_interval();
      lopts.heartbeat_interval_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_heartbeat_rpc_timeout()) {
      const auto& d = cfg.high_availability().heartbeat_rpc_timeout();
      lopts.heartbeat_rpc_timeout_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_heartbeat_rpc_max_retries()) {
      lopts.heartbeat_rpc_max_retries = cfg.high_availability().heartbeat_rpc_max_retries();
    }
    if (cfg.high_availability().has_periodic_sync_interval()) {
      const auto& d = cfg.high_availability().periodic_sync_interval();
      lopts.chunk_sync_interval_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_state_sync_rpc_timeout()) {
      const auto& d = cfg.high_availability().state_sync_rpc_timeout();
      lopts.state_sync_rpc_timeout_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_state_sync_rpc_max_retries()) {
      lopts.state_sync_rpc_max_retries = cfg.high_availability().state_sync_rpc_max_retries();
    }
    if (cfg.high_availability().has_full_sync_rpc_timeout()) {
      const auto& d = cfg.high_availability().full_sync_rpc_timeout();
      lopts.full_sync_rpc_timeout_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_full_sync_rpc_max_retries()) {
      lopts.full_sync_rpc_max_retries = cfg.high_availability().full_sync_rpc_max_retries();
    }
    lopts.force_full_sync_on_empty_inventory = cfg.high_availability().force_full_sync_on_empty_inventory();
    lopts.cluster_token = cfg.meta().cluster_token();
    lopts.global_store_client = shared_global_store_client;
    lifecycle = std::make_unique<daemon::WorkerLifecycleManager>(
        gsl::not_null<std::shared_ptr<store::StoreEngine>>{engine},
        gsl::not_null<daemon::StoreDaemonServiceImpl*>{&service},
        lopts);
    auto st = lifecycle->start();
    if (!st.ok()) {
      LOG(WARNING) << "Worker lifecycle start failed: " << st.message();
    }
  }

  constexpr absl::Duration kShutdownTimeout = absl::Seconds(30);

  std::thread sig_thread([&server, &service, set]() mutable {
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      LOG(INFO) << "Received signal " << sig << ", initiating gRPC shutdown...";
      service.begin_shutdown();
      const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(30);
      server->Shutdown(deadline);
    }
  });

  server->Wait();
  if (sig_thread.joinable()) {
    sig_thread.join();
  }
  if (lifecycle) {
    lifecycle->stop();
  }
  service.begin_shutdown();
  const absl::Status drain_status = async_runtime->drain(absl::Now() + kShutdownTimeout);
  if (!drain_status.ok()) {
    LOG(FATAL) << "AsyncRuntime drain failed during shutdown: " << drain_status;
  }

  return 0;
}
