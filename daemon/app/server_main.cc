// Copyright (c) 2025-2026, TensorCast Team.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>

#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <grpc/grpc.h>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/common/async_runtime.h"
#include "core/common/capability_token.h"
#include "core/common/config/daemon_config_io.h"
#include "core/common/logging_init.h"
#include "core/common/memory/pinned_memory_authority.h"
#include "core/common/otel/init.h"
#include "core/common/trace/trace_manager.h"
#include "core/cuda/cuda_api.h"
#include "core/store/components/communication_manager.h"
#include "core/store/components/global_store_client.h"
#include "core/store/replica/collective_disk_loader.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/app/daemon_app.h"
#include "daemon/app/startup_memory_preflight.h"
#include "daemon/util/identity_utils.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "gsl/pointers"
#include "tensorcast/global_store/v1/global_store.pb.h"

#include <pthread.h>
#include <csignal>
#include <fstream>
#include <sstream>

ABSL_FLAG(std::string, config, "", "Path to unified daemon config (YAML/JSON)");
ABSL_FLAG(bool, use_cursor_pagination, false, "Enable opaque cursor pagination for GetLoadedReplicasV2");
using namespace tensorcast;

namespace {

absl::StatusOr<std::filesystem::path> normalize_storage_root_for_config(const std::filesystem::path& storage_root) {
  std::error_code ec;
  const bool exists = std::filesystem::exists(storage_root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat storage root: ", storage_root.string()));
  }
  if (!exists) {
    return absl::InvalidArgumentError(absl::StrCat("storage root does not exist: ", storage_root.string()));
  }
  const bool is_dir = std::filesystem::is_directory(storage_root, ec);
  if (ec) {
    return absl::ErrnoToStatus(ec.value(), absl::StrCat("Failed to stat storage root: ", storage_root.string()));
  }
  if (!is_dir) {
    return absl::InvalidArgumentError(absl::StrCat("storage root must be a directory: ", storage_root.string()));
  }
  auto normalized = std::filesystem::weakly_canonical(storage_root, ec);
  if (ec) {
    return absl::ErrnoToStatus(
        ec.value(), absl::StrCat("Failed to canonicalize storage root: ", storage_root.string()));
  }
  return normalized;
}

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

absl::Status ensure_local_handle_socket_path_ready(const std::string& socket_path) {
  if (socket_path.empty()) {
    return absl::InvalidArgumentError("local_handle_socket_path is empty");
  }
  if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
    return absl::InvalidArgumentError(
        absl::StrCat("local_handle_socket_path is too long for AF_UNIX (len=", socket_path.size(), "): ", socket_path));
  }
  const std::filesystem::path parent = std::filesystem::path(socket_path).parent_path();
  if (parent.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("local_handle_socket_path must include a parent directory: ", socket_path));
  }
  return ensure_local_handle_parent_dir(parent);
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

std::string trim_copy(std::string_view value) {
  size_t begin = 0;
  size_t end = value.size();
  while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

uint64_t fnv1a_hash_64(std::string_view value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) {
    hash ^= c;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string short_socket_name(std::string_view seed) {
  const uint64_t hash = fnv1a_hash_64(seed);
  return std::format("lh-{:016x}.sock", hash);
}

uint64_t saturating_mul_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs == 0 || rhs == 0) {
    return 0;
  }
  if (lhs > (std::numeric_limits<uint64_t>::max() / rhs)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs * rhs;
}

uint64_t saturating_add_u64(uint64_t lhs, uint64_t rhs) {
  if (lhs > (std::numeric_limits<uint64_t>::max() - rhs)) {
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}

std::string format_binary_bytes(uint64_t bytes) {
  constexpr double kKiB = 1024.0;
  constexpr double kMiB = kKiB * 1024.0;
  constexpr double kGiB = kMiB * 1024.0;
  const double value = static_cast<double>(bytes);
  if (value >= kGiB) {
    return std::format("{:.2f}GiB", value / kGiB);
  }
  if (value >= kMiB) {
    return std::format("{:.2f}MiB", value / kMiB);
  }
  if (value >= kKiB) {
    return std::format("{:.2f}KiB", value / kKiB);
  }
  return std::format("{}B", bytes);
}

absl::StatusOr<std::string> shorten_socket_path_if_needed(const std::filesystem::path& preferred) {
  const std::string preferred_str = preferred.string();
  if (preferred_str.size() < sizeof(sockaddr_un::sun_path)) {
    return preferred_str;
  }
  auto home_or = tensorcast_home_dir();
  if (!home_or.ok()) {
    return home_or.status();
  }
  const std::filesystem::path fallback = *home_or / "uds" / short_socket_name(preferred_str);
  const std::string fallback_str = fallback.string();
  if (fallback_str.size() >= sizeof(sockaddr_un::sun_path)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "local_handle_socket_path too long even after shortening (len=", fallback_str.size(), "): ", fallback_str));
  }
  LOG(WARNING) << "local_handle_socket_path too long (len=" << preferred_str.size()
               << "); using shortened path=" << fallback_str;
  return fallback_str;
}

absl::StatusOr<std::filesystem::path> tensorcast_host_root_dir() {
  auto home_or = tensorcast_home_dir();
  if (!home_or.ok()) {
    return home_or.status();
  }
  const std::string hid = daemon::derive_host_id();
  if (hid.empty()) {
    return absl::InvalidArgumentError("Host id is empty; cannot resolve TensorCast runtime root");
  }
  auto root = *home_or / "hosts" / hid;
  absl::Status st = ensure_local_handle_parent_dir(root);
  if (!st.ok()) {
    return st;
  }
  return root;
}

absl::StatusOr<std::filesystem::path> tensorcast_runtime_root_dir() {
  auto host_root_or = tensorcast_host_root_dir();
  if (!host_root_or.ok()) {
    return host_root_or.status();
  }
  auto runtime_root = *host_root_or / "runtime";
  absl::Status st = ensure_local_handle_parent_dir(runtime_root);
  if (!st.ok()) {
    return st;
  }
  return runtime_root;
}

absl::StatusOr<std::filesystem::path> daemon_id_state_path() {
  auto runtime_root_or = tensorcast_runtime_root_dir();
  if (!runtime_root_or.ok()) {
    return runtime_root_or.status();
  }
  return *runtime_root_or / "daemon_id";
}

absl::StatusOr<std::string> read_daemon_id_file(const std::filesystem::path& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) {
      return absl::NotFoundError(absl::StrCat("daemon_id file not found: ", path.string()));
    }
    return absl::ErrnoToStatus(errno, absl::StrCat("open failed for ", path.string()));
  }
  std::string contents;
  char buf[256];
  for (;;) {
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n < 0) {
      const int err = errno;
      ::close(fd);
      return absl::ErrnoToStatus(err, absl::StrCat("read failed for ", path.string()));
    }
    if (n == 0) {
      break;
    }
    contents.append(buf, static_cast<size_t>(n));
  }
  if (::close(fd) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close failed for ", path.string()));
  }
  std::string trimmed = trim_copy(contents);
  if (trimmed.empty()) {
    return absl::NotFoundError(absl::StrCat("daemon_id file empty: ", path.string()));
  }
  return trimmed;
}

absl::Status write_daemon_id_file(const std::filesystem::path& path, std::string_view value) {
  const std::filesystem::path parent = path.parent_path();
  if (parent.empty()) {
    return absl::InvalidArgumentError("daemon_id file path missing parent directory");
  }
  absl::Status st = ensure_local_handle_parent_dir(parent);
  if (!st.ok()) {
    return st;
  }
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("open failed for ", path.string()));
  }
  size_t written = 0;
  const size_t total = value.size();
  while (written < total) {
    const ssize_t n = ::write(fd, value.data() + written, total - written);
    if (n < 0) {
      const int err = errno;
      ::close(fd);
      return absl::ErrnoToStatus(err, absl::StrCat("write failed for ", path.string()));
    }
    written += static_cast<size_t>(n);
  }
  if (::fsync(fd) < 0) {
    const int err = errno;
    ::close(fd);
    return absl::ErrnoToStatus(err, absl::StrCat("fsync failed for ", path.string()));
  }
  if (::close(fd) < 0) {
    return absl::ErrnoToStatus(errno, absl::StrCat("close failed for ", path.string()));
  }
  return absl::OkStatus();
}

std::string generate_random_daemon_id() {
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::uniform_int_distribution<uint64_t> dist;
  const uint64_t hi = dist(rng);
  const uint64_t lo = dist(rng);
  return std::format("daemon-{:016x}{:016x}", hi, lo);
}

absl::StatusOr<std::string> resolve_daemon_id(std::string_view configured) {
  const std::string configured_trimmed = trim_copy(configured);
  auto path_or = daemon_id_state_path();
  if (!path_or.ok()) {
    return path_or.status();
  }
  const auto& path = *path_or;
  if (!configured_trimmed.empty()) {
    absl::Status st = write_daemon_id_file(path, configured_trimmed);
    if (!st.ok()) {
      return st;
    }
    return configured_trimmed;
  }
  auto stored_or = read_daemon_id_file(path);
  if (stored_or.ok()) {
    return *stored_or;
  }
  if (stored_or.status().code() != absl::StatusCode::kNotFound) {
    return stored_or.status();
  }
  const std::string generated = generate_random_daemon_id();
  absl::Status st = write_daemon_id_file(path, generated);
  if (!st.ok()) {
    return st;
  }
  return generated;
}

absl::StatusOr<std::filesystem::path> discover_session_state_dir() {
  const char* instance = std::getenv("TENSORCAST_INSTANCE");
  if (!instance || !*instance) {
    return absl::InvalidArgumentError("TENSORCAST_INSTANCE is not set; auto-discovery requires a daemon session id");
  }
  auto host_root_or = tensorcast_host_root_dir();
  if (!host_root_or.ok()) {
    return host_root_or.status();
  }
  return *host_root_or / "sessions" / instance / "session";
}

absl::StatusOr<std::filesystem::path> discover_daemon_runtime_dir(std::string_view daemon_id) {
  if (daemon_id.empty()) {
    return absl::InvalidArgumentError("daemon_id is empty; cannot resolve daemon runtime directory");
  }
  auto runtime_root_or = tensorcast_runtime_root_dir();
  if (!runtime_root_or.ok()) {
    return runtime_root_or.status();
  }
  return *runtime_root_or / "daemons" / std::string(daemon_id);
}

absl::StatusOr<std::string> discover_local_handle_socket_path(std::string_view daemon_id) {
  if (const char* instance = std::getenv("TENSORCAST_INSTANCE"); instance && *instance) {
    auto state_dir_or = discover_session_state_dir();
    if (!state_dir_or.ok()) {
      return state_dir_or.status();
    }
    const std::filesystem::path& dir = *state_dir_or;
    absl::Status st = ensure_local_handle_parent_dir(dir);
    if (!st.ok()) {
      return st;
    }
    std::filesystem::path sock = dir / "local_handle.sock";
    return shorten_socket_path_if_needed(sock);
  }
  auto runtime_dir_or = discover_daemon_runtime_dir(daemon_id);
  if (!runtime_dir_or.ok()) {
    return runtime_dir_or.status();
  }
  const std::filesystem::path& dir = *runtime_dir_or;
  absl::Status st = ensure_local_handle_parent_dir(dir);
  if (!st.ok()) {
    return st;
  }
  std::filesystem::path sock = dir / "local_handle.sock";
  return shorten_socket_path_if_needed(sock);
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

absl::StatusOr<std::optional<uint64_t>> read_memlock_limit_bytes() {
  struct rlimit lim{};
  if (::getrlimit(RLIMIT_MEMLOCK, &lim) != 0) {
    return absl::ErrnoToStatus(errno, "getrlimit(RLIMIT_MEMLOCK) failed");
  }
  if (lim.rlim_cur == RLIM_INFINITY) {
    return std::optional<uint64_t>{};
  }
  return std::optional<uint64_t>{static_cast<uint64_t>(lim.rlim_cur)};
}

} // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  common::ensure_logging_initialized();
  // Broken pipes from log sinks or peer sockets must surface as write/send
  // errors instead of terminating the daemon process.
  std::signal(SIGPIPE, SIG_IGN);
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
  const std::string configured_daemon_id = trim_copy(cfg.daemon_id());
  auto daemon_id_or = resolve_daemon_id(configured_daemon_id);
  if (!daemon_id_or.ok()) {
    LOG(ERROR) << "Failed to resolve daemon_id: " << daemon_id_or.status();
    return 2;
  }
  if (configured_daemon_id.empty()) {
    LOG(INFO) << "Auto-selected daemon_id=" << *daemon_id_or;
  } else {
    LOG(INFO) << "Using configured daemon_id=" << *daemon_id_or;
  }
  if (cfg.daemon_id() != *daemon_id_or) {
    cfg.set_daemon_id(*daemon_id_or);
  }
  const std::filesystem::path storage_root_cfg = cfg.server().storage_path();
  std::filesystem::path storage_root;
  if (storage_root_cfg.empty()) {
    LOG(INFO) << "server.storage_path is empty; disk materialization allows absolute disk_path only";
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
  pm_cfg.defer_host_registration = true;
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

  const uint64_t stable_bytes =
      (cfg.engine().has_memory_tiers() ? static_cast<uint64_t>(cfg.engine().memory_tiers().stable_bytes()) : 0);
  const uint64_t startup_required_bytes = saturating_add_u64(pinned_total_bytes, stable_bytes);
  auto cgroup_limit_or = read_cgroup_v2_memory_max();
  if (cgroup_limit_or.ok() && cgroup_limit_or->has_value() && startup_required_bytes > **cgroup_limit_or) {
    LOG(ERROR) << "INVALID_ARGUMENT: cgroup memory.max too small for pinned+stable startup reservation: required="
               << startup_required_bytes << " memory.max=" << **cgroup_limit_or;
    return 2;
  }
  auto memlock_limit_or = read_memlock_limit_bytes();
  if (memlock_limit_or.ok() && memlock_limit_or->has_value() && pinned_total_bytes > **memlock_limit_or) {
    LOG(WARNING) << "RLIMIT_MEMLOCK may be too small for pinned pool registration: pinned_total=" << pinned_total_bytes
                 << " memlock_limit=" << **memlock_limit_or << ". cudaHostRegister may fail during startup.";
  }
  const absl::Status startup_mem = daemon::preflight_startup_memory(pinned_total_bytes, stable_bytes);
  if (!startup_mem.ok()) {
    LOG(ERROR) << "RESOURCE_EXHAUSTED: startup memory preflight failed: " << startup_mem;
    return 2;
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

  int detected_gpu_count = 0;
  const absl::Status gpu_count_status = cuda::get_device_count(&detected_gpu_count);
  if (!gpu_count_status.ok()) {
    LOG(ERROR) << "INVALID_ARGUMENT: failed to query CUDA device count for pinned_memory sizing: " << gpu_count_status;
    return 2;
  }
  if (detected_gpu_count < 0) {
    LOG(ERROR) << "INTERNAL: cuda::get_device_count returned a negative value: " << detected_gpu_count;
    return 2;
  }
  const bool fake_cuda_backend = cuda::is_fake();
  const daemon::EnginePinnedConcurrencySizing engine_sizing =
      daemon::compute_engine_pinned_concurrency_sizing(streaming_buffer_chunks, detected_gpu_count, fake_cuda_backend);
  if (engine_sizing.required_slices > 0) {
    const uint64_t capacity_engine_slices = engine_pool->capacity_slices();
    if (capacity_engine_slices < engine_sizing.required_slices) {
      LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory.classes[name=engine] cannot cover GPU concurrency: "
                 << "capacity_slices=" << capacity_engine_slices << " required_slices=" << engine_sizing.required_slices
                 << " (detected_gpu_count=" << detected_gpu_count
                 << " effective_gpu_count=" << engine_sizing.effective_gpu_count
                 << " fake_cuda_backend=" << (fake_cuda_backend ? "true" : "false")
                 << " streaming_buffer_chunks=" << streaming_buffer_chunks
                 << " slice_bytes=" << engine_pool->slice_bytes()
                 << "). Increase pinned_memory.classes[name=engine].pool_bytes "
                 << "or lower engine.streaming_buffer_chunks.";
      return 2;
    }
    LOG(INFO) << "Engine pinned pool startup check passed: capacity_slices=" << capacity_engine_slices
              << " required_slices=" << engine_sizing.required_slices << " (detected_gpu_count=" << detected_gpu_count
              << ", effective_gpu_count=" << engine_sizing.effective_gpu_count
              << ", fake_cuda_backend=" << (fake_cuda_backend ? "true" : "false")
              << ", streaming_buffer_chunks=" << streaming_buffer_chunks << ")";
  } else {
    LOG(WARNING) << "No CUDA devices detected at startup; skipping engine pinned concurrency coverage check";
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
  const size_t stager_reserve_slices = num_buffers;
  const size_t tcp_transport_slices = num_buffers * static_cast<size_t>(tcp_conn_count);
  const size_t required_gpu_slices = stager_reserve_slices + tcp_transport_slices;
  const size_t capacity_gpu_slices = comm_gpu_pool->capacity_slices();
  const uint32_t expected_gpu_channels = cfg.communicator().stager().expected_gpu_channels();
  const size_t expected_channel_required_slices = expected_gpu_channels > 0
      ? (stager_reserve_slices + (num_buffers * static_cast<size_t>(expected_gpu_channels)))
      : 0;
  const size_t recommended_gpu_slices = std::max(required_gpu_slices, expected_channel_required_slices);
  const uint64_t recommended_pool_bytes =
      saturating_mul_u64(static_cast<uint64_t>(recommended_gpu_slices), comm_gpu_pool->slice_bytes());
  const uint64_t required_pool_bytes =
      saturating_mul_u64(static_cast<uint64_t>(required_gpu_slices), comm_gpu_pool->slice_bytes());
  const uint64_t capacity_pool_bytes =
      saturating_mul_u64(static_cast<uint64_t>(capacity_gpu_slices), comm_gpu_pool->slice_bytes());

  LOG(INFO) << "comm_gpu startup sizing: capacity_slices=" << capacity_gpu_slices << " ("
            << format_binary_bytes(capacity_pool_bytes) << ")"
            << " required_slices=" << required_gpu_slices << " (" << format_binary_bytes(required_pool_bytes)
            << ") [stager_reserve_slices=" << stager_reserve_slices << ", tcp_transport_slices=" << tcp_transport_slices
            << ", buffers_per_flow=" << buffers_per_flow << ", tcp_conn_count=" << tcp_conn_count << "]"
            << " expected_gpu_channels=" << expected_gpu_channels
            << (expected_gpu_channels > 0 ? std::format(
                                                " expected_channel_required_slices={} recommended_pool_bytes>={} ({})",
                                                expected_channel_required_slices,
                                                recommended_pool_bytes,
                                                format_binary_bytes(recommended_pool_bytes))
                                          : std::format(
                                                " recommended_pool_bytes>={} ({})",
                                                required_pool_bytes,
                                                format_binary_bytes(required_pool_bytes)));

  if (capacity_gpu_slices < required_gpu_slices) {
    const size_t missing_slices = required_gpu_slices - capacity_gpu_slices;
    const uint64_t missing_bytes =
        saturating_mul_u64(static_cast<uint64_t>(missing_slices), comm_gpu_pool->slice_bytes());
    LOG(ERROR) << "INVALID_ARGUMENT: pinned_memory.classes[name=comm_gpu] too small: capacity_slices="
               << capacity_gpu_slices << " required_slices=" << required_gpu_slices
               << " slice_bytes=" << comm_gpu_pool->slice_bytes() << " missing_slices=" << missing_slices << " ("
               << format_binary_bytes(missing_bytes) << ")"
               << " [stager_reserve_slices=" << stager_reserve_slices
               << ", tcp_transport_slices=" << tcp_transport_slices << ", buffers_per_flow=" << buffers_per_flow
               << ", tcp_conn_count=" << tcp_conn_count << "]"
               << " recommended_pool_bytes>=" << required_pool_bytes << " (" << format_binary_bytes(required_pool_bytes)
               << ")";
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

  if (expected_gpu_channels > 0) {
    const size_t stager_reserve = num_buffers;
    const size_t available_gpu_slices =
        (capacity_gpu_slices > stager_reserve) ? (capacity_gpu_slices - stager_reserve) : 0;
    const size_t computed_limit =
        (buffers_per_flow > 0) ? (available_gpu_slices / static_cast<size_t>(buffers_per_flow)) : 0;
    if (static_cast<size_t>(expected_gpu_channels) > computed_limit) {
      const uint64_t recommended_expected_pool_bytes =
          saturating_mul_u64(static_cast<uint64_t>(expected_channel_required_slices), comm_gpu_pool->slice_bytes());
      LOG(ERROR) << "INVALID_ARGUMENT: communicator.stager.expected_gpu_channels=" << expected_gpu_channels
                 << " exceeds staging capacity: computed_limit=" << computed_limit
                 << " (gpu_pool_slices=" << capacity_gpu_slices << " reserve=" << stager_reserve
                 << " buffers_per_flow=" << buffers_per_flow
                 << " required_slices_for_expected_channels=" << expected_channel_required_slices
                 << " recommended_pool_bytes>=" << recommended_expected_pool_bytes << " ("
                 << format_binary_bytes(recommended_expected_pool_bytes)
                 << ")). Increase pinned_memory.classes[name=comm_gpu].pool_bytes or reduce expected_gpu_channels.";
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
    auto path_or = discover_local_handle_socket_path(cfg.daemon_id());
    if (!path_or.ok()) {
      LOG(ERROR) << "INVALID_ARGUMENT: lifecycle.handle_leases.local_handle_socket_path is empty and auto-discovery "
                    "failed: "
                 << path_or.status();
      return 2;
    }
    cfg.mutable_lifecycle()->mutable_handle_leases()->set_local_handle_socket_path(*path_or);
    LOG(INFO) << "Auto-selected lifecycle.handle_leases.local_handle_socket_path=" << *path_or;
  }
  if (!cfg.lifecycle().handle_leases().local_handle_socket_path().empty()) {
    const std::string socket_path = cfg.lifecycle().handle_leases().local_handle_socket_path();
    const absl::Status st = ensure_local_handle_socket_path_ready(socket_path);
    if (!st.ok()) {
      LOG(ERROR) << "INVALID_ARGUMENT: local handle socket path is invalid: " << st;
      return 2;
    }
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

  if (cfg.engine().has_byte_mapping()) {
    const auto& bm = cfg.engine().byte_mapping();
    opts.byte_mapping.enable_strided_execution =
        bm.has_enable_strided_execution() ? bm.enable_strided_execution() : true;
    opts.byte_mapping.enable_direct_write_at = bm.has_enable_direct_write_at() ? bm.enable_direct_write_at() : true;
    if (bm.program_cache_entries() > 0) {
      opts.byte_mapping.program_cache_entries = bm.program_cache_entries();
    }
    if (bm.strided_run_min_ranges() > 0) {
      opts.byte_mapping.strided_run_min_ranges = bm.strided_run_min_ranges();
    }
    if (bm.strided_min_row_len_bytes() > 0) {
      opts.byte_mapping.strided_min_row_len_bytes = bm.strided_min_row_len_bytes();
    }
    if (bm.strided_max_amplification() > 0) {
      opts.byte_mapping.strided_max_amplification = bm.strided_max_amplification();
    }
    if (bm.strided_block_target_bytes() > 0) {
      opts.byte_mapping.strided_block_target_bytes = bm.strided_block_target_bytes();
    }
    if (bm.strided_block_max_bytes() > 0) {
      opts.byte_mapping.strided_block_max_bytes = bm.strided_block_max_bytes();
    }
    opts.byte_mapping.disk_source_ordered_read =
        bm.has_disk_source_ordered_read() ? bm.disk_source_ordered_read() : true;
    if (bm.disk_source_merge_max_gap_bytes() > 0) {
      opts.byte_mapping.disk_source_merge_max_gap_bytes = bm.disk_source_merge_max_gap_bytes();
    }
    if (bm.disk_source_merge_max_amplification() > 0) {
      opts.byte_mapping.disk_source_merge_max_amplification = bm.disk_source_merge_max_amplification();
    }
    if (bm.disk_source_prefetch_depth() > 0) {
      opts.byte_mapping.disk_source_prefetch_depth = bm.disk_source_prefetch_depth();
    }
  }

  if (cfg.engine().has_materialization_strategy()) {
    const auto& ms = cfg.engine().materialization_strategy();
    auto& strategy = opts.materialization_strategy;
    strategy.enable_tensor_aware_mapped_executor =
        ms.has_enable_tensor_aware_mapped_executor() ? ms.enable_tensor_aware_mapped_executor() : true;
    strategy.enable_local_batched_disk_load =
        ms.has_enable_local_batched_disk_load() ? ms.enable_local_batched_disk_load() : true;
    strategy.enable_owner_file_collective =
        ms.has_enable_owner_file_collective() ? ms.enable_owner_file_collective() : false;
    strategy.allow_mixed_execution = ms.has_allow_mixed_execution() ? ms.allow_mixed_execution() : true;
    strategy.prefer_local_canonical_for_mapped = ms.prefer_local_canonical_for_mapped();
    strategy.allow_source_ordered_for_mapped =
        ms.has_allow_source_ordered_for_mapped() ? ms.allow_source_ordered_for_mapped() : true;
    strategy.enable_mapped_dim0_tensor_jobs =
        ms.has_enable_mapped_dim0_tensor_jobs() ? ms.enable_mapped_dim0_tensor_jobs() : true;
    strategy.enable_mapped_dim1_tensor_jobs =
        ms.has_enable_mapped_dim1_tensor_jobs() ? ms.enable_mapped_dim1_tensor_jobs() : true;
    strategy.enable_mapped_concat_jobs = ms.has_enable_mapped_concat_jobs() ? ms.enable_mapped_concat_jobs() : true;
    strategy.enable_mapped_concat_execution =
        ms.has_enable_mapped_concat_execution() ? ms.enable_mapped_concat_execution() : true;
    strategy.enable_mapped_single_range_concat_jobs =
        ms.has_enable_mapped_single_range_concat_jobs() ? ms.enable_mapped_single_range_concat_jobs() : true;
    strategy.enable_mapped_multirange_concat_jobs =
        ms.has_enable_mapped_multirange_concat_jobs() ? ms.enable_mapped_multirange_concat_jobs() : true;
    strategy.sync_after_single_range_concat_job = ms.sync_after_single_range_concat_job();
    strategy.use_dedicated_single_range_concat_stream = ms.use_dedicated_single_range_concat_stream();
    if (ms.owner_file_collective_peak_bytes_budget() > 0) {
      strategy.owner_file_collective_peak_bytes_budget = ms.owner_file_collective_peak_bytes_budget();
    }
    if (ms.owner_file_collective_batch_bytes() > 0) {
      strategy.owner_file_collective_batch_bytes = ms.owner_file_collective_batch_bytes();
    }
    if (ms.owner_file_collective_dim1_staging_bytes() > 0) {
      strategy.owner_file_collective_dim1_staging_bytes = ms.owner_file_collective_dim1_staging_bytes();
    }
    if (ms.owner_file_collective_max_inflight_batches() > 0) {
      strategy.owner_file_collective_max_inflight_batches = ms.owner_file_collective_max_inflight_batches();
    }
    strategy.owner_file_collective_shared_fs_only =
        ms.has_owner_file_collective_shared_fs_only() ? ms.owner_file_collective_shared_fs_only() : true;
    if (ms.owner_file_collective_max_owner_skew_ratio() > 0.0) {
      strategy.owner_file_collective_max_owner_skew_ratio = ms.owner_file_collective_max_owner_skew_ratio();
    }
    if (ms.owner_file_collective_min_dedup_saving_bytes() > 0) {
      strategy.owner_file_collective_min_dedup_saving_bytes = ms.owner_file_collective_min_dedup_saving_bytes();
    }
    if (ms.has_owner_file_collective_group_assemble_timeout()) {
      strategy.owner_file_collective_group_assemble_timeout =
          duration_to_millis(ms.owner_file_collective_group_assemble_timeout());
    }
    strategy.owner_file_collective_allow_mixed_residual =
        ms.has_owner_file_collective_allow_mixed_residual() ? ms.owner_file_collective_allow_mixed_residual() : false;
    if (ms.owner_file_collective_planner_cache_entries() > 0) {
      strategy.owner_file_collective_planner_cache_entries = ms.owner_file_collective_planner_cache_entries();
    }
    switch (ms.executor_preference()) {
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_GENERIC_BYTE_RANGE:
        strategy.executor_preference =
            store::StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kGenericByteRange;
        break;
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_TENSOR_AWARE_LOCAL:
        strategy.executor_preference =
            store::StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kTensorAwareLocal;
        break;
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_OWNER_FILE_COLLECTIVE:
        strategy.executor_preference =
            store::StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kOwnerFileCollective;
        break;
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_AUTO:
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_EXECUTOR_PREFERENCE_UNSPECIFIED:
      default:
        strategy.executor_preference =
            store::StoreEngineOptions::MaterializationStrategyConfig::ExecutorPreference::kAuto;
        break;
    }
    switch (ms.diagnostics_verbosity()) {
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_OFF:
        strategy.diagnostics_verbosity =
            store::StoreEngineOptions::MaterializationStrategyConfig::DiagnosticsVerbosity::kOff;
        break;
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_VERBOSE:
        strategy.diagnostics_verbosity =
            store::StoreEngineOptions::MaterializationStrategyConfig::DiagnosticsVerbosity::kVerbose;
        break;
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_BASIC:
      case tensorcast::config::v1::Engine::MATERIALIZATION_STRATEGY_DIAGNOSTICS_VERBOSITY_UNSPECIFIED:
      default:
        strategy.diagnostics_verbosity =
            store::StoreEngineOptions::MaterializationStrategyConfig::DiagnosticsVerbosity::kBasic;
        break;
    }
  }

  if (cfg.has_promotion()) {
    const auto& promo = cfg.promotion();
    switch (promo.policy()) {
      case tensorcast::config::v1::PROMOTION_POLICY_ON_MATERIALIZE:
        opts.promotion.policy = store::StoreEngineOptions::PromotionPolicy::kOnMaterialize;
        break;
      case tensorcast::config::v1::PROMOTION_POLICY_ON_HOTNESS:
        opts.promotion.policy = store::StoreEngineOptions::PromotionPolicy::kOnHotness;
        break;
      case tensorcast::config::v1::PROMOTION_POLICY_ON_POLICY:
        opts.promotion.policy = store::StoreEngineOptions::PromotionPolicy::kOnPolicy;
        break;
      case tensorcast::config::v1::PROMOTION_POLICY_NEVER:
      case tensorcast::config::v1::PROMOTION_POLICY_UNSPECIFIED:
      default:
        opts.promotion.policy = store::StoreEngineOptions::PromotionPolicy::kNever;
        break;
    }
    opts.promotion.require_verified = promo.require_verified();
    if (promo.has_demotion_drain_timeout()) {
      opts.promotion.demotion_drain_timeout = duration_to_millis(promo.demotion_drain_timeout());
    }
    if (promo.max_concurrency() > 0) {
      opts.promotion.max_concurrency = promo.max_concurrency();
    }
  }

  if (opts.cpu_shared_memory_enabled) {
    if (!cfg.engine().has_memory_tiers() || cfg.engine().memory_tiers().stable_bytes() == 0) {
      LOG(ERROR) << "INVALID_ARGUMENT: engine.cpu_shared_memory.enabled requires engine.memory_tiers.stable_bytes > 0";
      return 2;
    }
    if (cfg.lifecycle().handle_leases().local_handle_socket_path().empty()) {
      LOG(ERROR) << "INVALID_ARGUMENT: engine.cpu_shared_memory.enabled requires "
                    "lifecycle.handle_leases.local_handle_socket_path (auto-discovery uses TENSORCAST_INSTANCE when "
                    "set, otherwise falls back to the daemon_id runtime directory)";
      return 2;
    }
    const absl::Status memfd_probe = probe_memfd_shared_mapping();
    if (!memfd_probe.ok()) {
      LOG(ERROR) << "INVALID_ARGUMENT: CPU shared memory enabled but memfd probe failed: " << memfd_probe;
      return 2;
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
      LOG(FATAL) << "Failed to initialize communication engine: " << st.message();
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
      // blocking_executor() may schedule nested fan-out (e.g., pump producers) while the
      // parent task blocks waiting for completion. Keep a small headroom to avoid
      // thread-pool starvation deadlocks for low thread-count configurations.
      .blocking_threads = static_cast<size_t>(std::max<int>(4, opts.num_thread)),
      .thread_name_prefix = "tensorcast",
  });
  opts.async_runtime = async_runtime;

  auto engine = std::make_shared<store::StoreEngine>(opts);

  // Map lifecycle options into daemon options
  daemon::DaemonOptions daemon_opts;
  if (cfg.lifecycle().has_sessions_ttl()) {
    const auto& d = cfg.lifecycle().sessions_ttl();
    daemon_opts.sessions_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_locks_ttl()) {
    const auto& d = cfg.lifecycle().locks_ttl();
    daemon_opts.locks_ttl = std::chrono::seconds(d.seconds());
  }
  if (cfg.lifecycle().has_sessions_sweep_interval()) {
    const auto& d = cfg.lifecycle().sessions_sweep_interval();
    daemon_opts.sessions_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_locks_sweep_interval()) {
    const auto& d = cfg.lifecycle().locks_sweep_interval();
    daemon_opts.locks_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_verification_sweep_interval()) {
    const auto& d = cfg.lifecycle().verification_sweep_interval();
    daemon_opts.verification_sweep_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  if (cfg.lifecycle().has_proc_check_interval()) {
    const auto& d = cfg.lifecycle().proc_check_interval();
    daemon_opts.proc_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  daemon_opts.enable_periodic_eviction = cfg.lifecycle().enable_periodic_eviction();
  daemon_opts.gpu_memory_limit_fraction = cfg.lifecycle().gpu_memory_limit_fraction();
  if (cfg.lifecycle().has_eviction_loop_interval()) {
    const auto& d = cfg.lifecycle().eviction_loop_interval();
    daemon_opts.eviction_check_interval = std::chrono::milliseconds(d.seconds() * 1000 + d.nanos() / 1000000);
  }
  daemon_opts.storage_path = storage_root;
  const auto& public_disk_source = cfg.public_disk_source();
  daemon_opts.public_disk_source_policy.unmatched_path_mode = public_disk_source.unmatched_path_mode() ==
          tensorcast::config::v1::DaemonConfig::PublicDiskSource::
              PUBLIC_DISK_SOURCE_UNMATCHED_PATH_MODE_ALLOW_ABSOLUTE_FALLBACK
      ? daemon::DaemonOptions::PublicDiskSourcePolicy::UnmatchedPathMode::kAllowAbsoluteFallback
      : daemon::DaemonOptions::PublicDiskSourcePolicy::UnmatchedPathMode::kReject;
  for (const auto& trusted_root : public_disk_source.trusted_root_policies()) {
    daemon::DaemonOptions::PublicDiskSourcePolicy::TrustedRootPolicy policy;
    policy.policy_id = trusted_root.policy_id();
    if (!trusted_root.root_path().empty()) {
      auto normalized_root_or = normalize_storage_root_for_config(std::filesystem::path(trusted_root.root_path()));
      if (!normalized_root_or.ok()) {
        LOG(ERROR) << "INVALID_ARGUMENT: public_disk_source.trusted_root_policies.root_path invalid ("
                   << trusted_root.root_path() << "): " << normalized_root_or.status();
        return 2;
      }
      policy.root_path = *normalized_root_or;
    }
    for (const auto format : trusted_root.allowed_formats()) {
      switch (format) {
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_FORMAT_PARTITIONED:
          policy.allowed_formats.push_back(daemon::DaemonOptions::PublicDiskSourcePolicy::Format::kPartitioned);
          break;
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_FORMAT_SAFETENSORS:
          policy.allowed_formats.push_back(daemon::DaemonOptions::PublicDiskSourcePolicy::Format::kSafetensors);
          break;
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_FORMAT_UNSPECIFIED:
        default:
          break;
      }
    }
    for (const auto capability : trusted_root.allowed_metadata_capabilities()) {
      switch (capability) {
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_METADATA_CAPABILITY_TENSOR_AWARE:
          policy.allowed_metadata_capabilities.push_back(
              daemon::DaemonOptions::PublicDiskSourcePolicy::MetadataCapability::kTensorAware);
          break;
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_METADATA_CAPABILITY_BYTE_ONLY:
          policy.allowed_metadata_capabilities.push_back(
              daemon::DaemonOptions::PublicDiskSourcePolicy::MetadataCapability::kByteOnly);
          break;
        case tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_METADATA_CAPABILITY_UNSPECIFIED:
        default:
          break;
      }
    }
    policy.descriptor_reuse_mode = trusted_root.descriptor_reuse_mode() ==
            tensorcast::config::v1::DaemonConfig::PUBLIC_DISK_SOURCE_DESCRIPTOR_REUSE_MODE_DISABLED
        ? daemon::DaemonOptions::PublicDiskSourcePolicy::DescriptorReuseMode::kDisabled
        : daemon::DaemonOptions::PublicDiskSourcePolicy::DescriptorReuseMode::kTrustedHintOnly;
    policy.validation_mode = daemon::DaemonOptions::PublicDiskSourcePolicy::ValidationMode::kValidateBeforeRead;
    policy.lightweight_attestation_enabled =
        !trusted_root.has_lightweight_attestation_enabled() || trusted_root.lightweight_attestation_enabled();
    daemon_opts.public_disk_source_policy.trusted_root_policies.push_back(std::move(policy));
  }
  daemon_opts.local_handle_socket_path = cfg.lifecycle().handle_leases().local_handle_socket_path();
  if (cfg.lifecycle().handle_leases().has_ttl()) {
    daemon_opts.handle_lease_ttl = duration_to_millis(cfg.lifecycle().handle_leases().ttl());
  }
  daemon_opts.handle_lease_max_mints_per_second = cfg.lifecycle().handle_leases().max_mints_per_second();
  daemon_opts.cpu_shared_memory_enabled = opts.cpu_shared_memory_enabled;
  daemon_opts.external_target_verification_enabled = cfg.engine().enable_external_target_verification();
  daemon_opts.max_concurrency = std::max<uint32_t>(1, opts.promotion.max_concurrency);
  const auto& post_seal = cfg.post_seal();
  daemon_opts.post_seal_policy.migrate_views = post_seal.migrate_views();
  daemon_opts.post_seal_policy.migrate_transpose_only = post_seal.migrate_transpose_only();
  daemon_opts.post_seal_policy.reuse_views_if_safe = post_seal.reuse_views_if_safe();
  daemon_opts.post_seal_policy.retire_pieces = post_seal.retire_pieces();
  if (cfg.daemon_id().empty()) {
    LOG(ERROR) << "Resolved daemon_id is empty; cannot start daemon.";
    return 1;
  }
  daemon_opts.daemon_id = cfg.daemon_id();
  auto daemon_runtime_dir_or = discover_daemon_runtime_dir(daemon_opts.daemon_id);
  if (!daemon_runtime_dir_or.ok()) {
    LOG(ERROR) << "Failed to resolve daemon runtime dir for import root: " << daemon_runtime_dir_or.status();
    return 1;
  }
  daemon_opts.import_root = *daemon_runtime_dir_or / "import";

  if (cfg.has_capability_tokens() && cfg.capability_tokens().has_active()) {
    const auto& active = cfg.capability_tokens().active();
    if (active.version() != 0 && !active.secret().empty()) {
      daemon_opts.capability_tokens.active.version = active.version();
      daemon_opts.capability_tokens.active.secret = active.secret();
      for (const auto& prev : cfg.capability_tokens().previous()) {
        if (prev.version() == 0 || prev.secret().empty()) {
          continue;
        }
        if (prev.version() == daemon_opts.capability_tokens.active.version) {
          continue;
        }
        daemon_opts.capability_tokens.previous.push_back(
            common::CapabilityTokenKey{.version = prev.version(), .secret = prev.secret()});
      }
    }
  }
  if (cfg.has_retention_handles()) {
    const auto& retention_cfg = cfg.retention_handles();
    daemon_opts.retention_handles.enabled = retention_cfg.enabled();
    if (retention_cfg.has_default_ttl()) {
      daemon_opts.retention_handles.default_ttl = duration_to_millis(retention_cfg.default_ttl());
    }
    if (retention_cfg.has_max_ttl()) {
      daemon_opts.retention_handles.max_ttl = duration_to_millis(retention_cfg.max_ttl());
    }
  }

  if (cfg.has_byte_artifact_routing()) {
    const auto& bar = cfg.byte_artifact_routing();
    if (bar.shard_count() > 0) {
      daemon_opts.byte_artifact_routing.shard_count = bar.shard_count();
    }
    if (bar.inline_payload_threshold_bytes() > 0) {
      daemon_opts.byte_artifact_routing.inline_payload_threshold_bytes = bar.inline_payload_threshold_bytes();
    }
    if (bar.has_route_staleness_budget()) {
      daemon_opts.byte_artifact_routing.route_staleness_budget = duration_to_millis(bar.route_staleness_budget());
    }
    if (bar.has_lease_ttl()) {
      daemon_opts.byte_artifact_routing.lease_ttl = duration_to_millis(bar.lease_ttl());
    }
    if (bar.has_keepalive_interval()) {
      daemon_opts.byte_artifact_routing.keepalive_interval = duration_to_millis(bar.keepalive_interval());
    }
    if (bar.has_worker_directory_staleness_budget()) {
      daemon_opts.byte_artifact_routing.worker_directory_staleness_budget =
          duration_to_millis(bar.worker_directory_staleness_budget());
    }
    if (bar.routing_epoch() != 0) {
      daemon_opts.byte_artifact_routing.routing_epoch = bar.routing_epoch();
    }
    if (bar.has_payload_transport()) {
      const auto& pt = bar.payload_transport();
      if (pt.has_ref_ttl()) {
        daemon_opts.byte_artifact_routing.payload_transport.ref_ttl = duration_to_absl(pt.ref_ttl());
      }
      if (pt.max_chunk_bytes() > 0) {
        daemon_opts.byte_artifact_routing.payload_transport.max_chunk_bytes = pt.max_chunk_bytes();
      }
      if (pt.has_fetch_deadline()) {
        daemon_opts.byte_artifact_routing.payload_transport.fetch_deadline = duration_to_absl(pt.fetch_deadline());
      }
      if (pt.has_cleanup_interval()) {
        daemon_opts.byte_artifact_routing.payload_transport.cleanup_interval = duration_to_absl(pt.cleanup_interval());
      }
      if (pt.max_batch_payload_bytes() > 0) {
        daemon_opts.byte_artifact_routing.payload_transport.max_batch_payload_bytes = pt.max_batch_payload_bytes();
      }
      if (pt.max_batch_items() > 0) {
        daemon_opts.byte_artifact_routing.payload_transport.max_batch_items = pt.max_batch_items();
      }
      if (pt.max_batch_stage_bytes_per_peer() > 0) {
        daemon_opts.byte_artifact_routing.payload_transport.max_batch_stage_bytes_per_peer =
            pt.max_batch_stage_bytes_per_peer();
      }
      if (pt.has_batch_transport_protocol_version()) {
        daemon_opts.byte_artifact_routing.payload_transport.batch_transport_protocol_version =
            pt.batch_transport_protocol_version();
      }
      if (pt.has_communicator_source_enabled()) {
        daemon_opts.byte_artifact_routing.payload_transport.communicator_source_enabled =
            pt.communicator_source_enabled();
      }
      if (pt.has_host_memory_export_enabled()) {
        daemon_opts.byte_artifact_routing.payload_transport.host_memory_export_enabled =
            pt.host_memory_export_enabled();
      }
      if (pt.has_minimum_batch_transport_ttl()) {
        daemon_opts.byte_artifact_routing.payload_transport.minimum_batch_transport_ttl =
            duration_to_absl(pt.minimum_batch_transport_ttl());
      }
      if (pt.has_transport_release_guard()) {
        daemon_opts.byte_artifact_routing.payload_transport.transport_release_guard =
            duration_to_absl(pt.transport_release_guard());
      }
    }
  }

  uint64_t capability_flags = 0;
  const bool cap_dir_enabled = cfg.has_capability_directory() && cfg.capability_directory().enabled();
  if (cfg.has_capability_directory()) {
    const auto& cap_cfg = cfg.capability_directory();
    if (cap_cfg.has_gateway_ingress_enabled()) {
      daemon_opts.gateway_ingress_enabled = cap_cfg.gateway_ingress_enabled();
    }
    if (cap_cfg.has_shard_home_eligible()) {
      daemon_opts.byte_artifact_routing.shard_home_eligible = cap_cfg.shard_home_eligible();
    }
  }
  const bool tokens_configured =
      daemon_opts.capability_tokens.active.version != 0 && !daemon_opts.capability_tokens.active.secret.empty();
  if (cap_dir_enabled) {
    if (tokens_configured) {
      capability_flags |= (1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_CAPABILITY_TOKENS_V2_ENABLED);
    }
    if (daemon_opts.retention_handles.enabled && tokens_configured) {
      capability_flags |= (1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_RETENTION_HANDLES_ENABLED);
    }
    if (daemon_opts.gateway_ingress_enabled) {
      capability_flags |= (1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_GATEWAY_INGRESS_ENABLED);
    }
    if (daemon_opts.byte_artifact_routing.shard_home_eligible) {
      capability_flags |= (1ULL << tensorcast::global_store::v1::WORKER_CAPABILITY_FLAG_SHARD_HOME_ELIGIBLE);
    }
  }
  // Observability high-cardinality attributes: default off (config hook TBD)
  daemon_opts.allow_high_card_attrs = false;
  // Feature flags (override via flags for now)
  daemon_opts.use_cursor_pagination = absl::GetFlag(FLAGS_use_cursor_pagination);

  const std::string listen_addr = absl::StrCat(cfg.server().listen().host(), ":", cfg.server().listen().port());

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
      ssl_opts.client_certificate_request = GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
    }
    grpc::SslServerCredentialsOptions::PemKeyCertPair pkcp{.private_key = key, .cert_chain = cert};
    ssl_opts.pem_key_cert_pairs.push_back(pkcp);
    creds = grpc::SslServerCredentials(ssl_opts);
    daemon_opts.inter_daemon_grpc_security = tensorcast::daemon::DaemonOptions::InterDaemonGrpcSecurity{
        .tls_enabled = true,
        .mutual_auth_enabled = !ca.empty(),
        .cert_chain_pem = cert,
        .private_key_pem = key,
        .root_cert_pem = ca,
    };
  } else {
    creds = grpc::InsecureServerCredentials();
  }

  daemon::DaemonApp::GrpcOptions grpc_opts;
  grpc_opts.listen_addr = listen_addr;
  grpc_opts.credentials = creds;
  grpc_opts.max_concurrent_streams = cfg.server().grpc().max_concurrent_streams();
  auto to_ms = [](const google::protobuf::Duration& d) -> int {
    return static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
  };
  auto assign_grpc_duration_if_enabled = [&to_ms](
                                             const google::protobuf::Duration& duration, std::optional<int>* target) {
    const int duration_ms = to_ms(duration);
    if (duration_ms > 0) {
      *target = duration_ms;
    }
  };
  if (cfg.server().grpc().has_keepalive_time()) {
    assign_grpc_duration_if_enabled(cfg.server().grpc().keepalive_time(), &grpc_opts.keepalive_time_ms);
  }
  if (cfg.server().grpc().has_keepalive_timeout()) {
    assign_grpc_duration_if_enabled(cfg.server().grpc().keepalive_timeout(), &grpc_opts.keepalive_timeout_ms);
  }
  if (cfg.server().grpc().has_max_connection_idle()) {
    assign_grpc_duration_if_enabled(cfg.server().grpc().max_connection_idle(), &grpc_opts.max_connection_idle_ms);
  }
  if (cfg.server().grpc().has_max_connection_age()) {
    assign_grpc_duration_if_enabled(cfg.server().grpc().max_connection_age(), &grpc_opts.max_connection_age_ms);
  }
  grpc_opts.tcp_nodelay = cfg.server().grpc().tcp_nodelay();
  grpc_opts.so_reuseport = cfg.server().grpc().so_reuseport();

  std::optional<daemon::WorkerLifecycleManager::Options> lifecycle_opts;
  if (!gs_addr.empty() && cfg.high_availability().enabled()) {
    daemon::WorkerLifecycleManager::Options lopts;
    lopts.global_store_addr = gs_addr;
    lopts.listen_addr = listen_addr;
    // Prefer explicit advertise.host if provided
    if (cfg.server().has_advertise() && !cfg.server().advertise().host().empty()) {
      lopts.advertise_host = cfg.server().advertise().host();
    }
    lopts.p2p_port = p2p_port;
    lopts.max_concurrency = std::max<uint32_t>(1, daemon_opts.max_concurrency);
    if (cfg.high_availability().has_heartbeat_interval()) {
      const auto& d = cfg.high_availability().heartbeat_interval();
      lopts.heartbeat_interval_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    if (cfg.high_availability().has_heartbeat_rpc_timeout()) {
      const auto& d = cfg.high_availability().heartbeat_rpc_timeout();
      lopts.heartbeat_rpc_timeout_ms = static_cast<int>(d.seconds() * 1000 + d.nanos() / 1000000);
    }
    lopts.capability_flags = capability_flags;
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
    lopts.cluster_token = cfg.meta().cluster_token();
    lopts.global_store_client = shared_global_store_client;
    lifecycle_opts = lopts;
  }

  daemon::DaemonApp::Options app_opts;
  app_opts.engine = engine;
  app_opts.async_runtime = async_runtime;
  app_opts.daemon_options = std::move(daemon_opts);
  app_opts.grpc = std::move(grpc_opts);
  app_opts.worker_lifecycle = lifecycle_opts;
  app_opts.global_store_client = shared_global_store_client;
  app_opts.startup_coordinator = std::make_shared<daemon::StartupCoordinator>();
  app_opts.deferred_startup_work = [pma, fake_cuda_backend, detected_gpu_count]() -> absl::Status {
    const absl::Status register_status = pma->register_all_pools();
    if (!register_status.ok()) {
      return absl::Status(
          register_status.code(),
          absl::StrCat("deferred pinned pool host registration failed: ", register_status.message()));
    }
    LOG(INFO) << "Deferred pinned pool host registration complete";

    if (!fake_cuda_backend && detected_gpu_count > 0) {
      const absl::Status gpu_hash_prewarm = common::prewarm_gpu_hash_nvrtc_for_visible_devices();
      if (!gpu_hash_prewarm.ok()) {
        return absl::Status(
            gpu_hash_prewarm.code(),
            absl::StrCat("GPU hash NVRTC prewarm failed during deferred startup: ", gpu_hash_prewarm.message()));
      }
      LOG(INFO) << "GPU hash NVRTC prewarm complete for detected_gpu_count=" << detected_gpu_count;
      if (detected_gpu_count > 1) {
        std::vector<int> clique_devices;
        clique_devices.reserve(static_cast<size_t>(detected_gpu_count));
        for (int device_id = 0; device_id < detected_gpu_count; ++device_id) {
          clique_devices.push_back(device_id);
        }
        const absl::Status clique_prewarm = store::replica::warm_collective_clique_cache(clique_devices);
        if (!clique_prewarm.ok()) {
          return absl::Status(
              clique_prewarm.code(),
              absl::StrCat("collective clique prewarm failed during deferred startup: ", clique_prewarm.message()));
        }
      }
    }
    return absl::OkStatus();
  };

  auto app_or = daemon::DaemonApp::create(std::move(app_opts));
  if (!app_or.ok()) {
    LOG(ERROR) << "Failed to create daemon app: " << app_or.status();
    return 2;
  }
  auto app = std::move(*app_or);
  auto start_st = app->start();
  if (!start_st.ok()) {
    LOG(ERROR) << "Failed to start daemon app: " << start_st;
    return 2;
  }

  constexpr absl::Duration kShutdownTimeout = absl::Seconds(30);

  std::thread sig_thread([&app, set, kShutdownTimeout]() mutable {
    int sig = 0;
    if (sigwait(&set, &sig) == 0) {
      LOG(INFO) << "Received signal " << sig << ", initiating shutdown...";
      const absl::Status stop_status = app->stop(absl::Now() + kShutdownTimeout);
      if (!stop_status.ok()) {
        LOG(ERROR) << "Shutdown request failed: " << stop_status;
      }
    }
  });

  app->wait();
  if (sig_thread.joinable()) {
    sig_thread.join();
  }
  const absl::Status drain_status = app->stop(absl::Now() + kShutdownTimeout);
  if (!drain_status.ok()) {
    LOG(FATAL) << "AsyncRuntime drain failed during shutdown: " << drain_status;
  }

  return 0;
}
