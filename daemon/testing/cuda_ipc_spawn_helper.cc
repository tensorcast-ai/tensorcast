// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/cuda_ipc_spawn_helper.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string_view>

#include "absl/status/status.h"

extern char** environ;

namespace tensorcast::daemon::testing {
namespace {

absl::Status close_fd(int fd) {
  if (fd < 0) {
    return absl::OkStatus();
  }
  if (close(fd) == 0) {
    return absl::OkStatus();
  }
  return absl::ErrnoToStatus(errno, "close failed");
}

absl::StatusOr<std::string> read_line(int fd) {
  std::string out;
  char buffer[256];
  while (true) {
    ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n < 0) {
      return absl::ErrnoToStatus(errno, "read failed");
    }
    if (n == 0) {
      break;
    }
    for (ssize_t i = 0; i < n; ++i) {
      if (buffer[i] == '\n') {
        return out;
      }
      out.push_back(buffer[i]);
    }
  }
  if (out.empty()) {
    return absl::InternalError("helper produced no output");
  }
  return out;
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return 10 + (c - 'a');
  }
  if (c >= 'A' && c <= 'F') {
    return 10 + (c - 'A');
  }
  return -1;
}

absl::StatusOr<std::string> decode_hex(std::string_view hex) {
  if (hex.size() % 2 != 0) {
    return absl::InvalidArgumentError("hex string has odd length");
  }
  std::string bytes;
  bytes.resize(hex.size() / 2);
  for (size_t i = 0; i < bytes.size(); ++i) {
    const int hi = hex_value(hex[i * 2]);
    const int lo = hex_value(hex[i * 2 + 1]);
    if (hi < 0 || lo < 0) {
      return absl::InvalidArgumentError("invalid hex character");
    }
    bytes[i] = static_cast<char>((hi << 4) | lo);
  }
  return bytes;
}

absl::StatusOr<pid_t> parse_pid_token(std::string_view token) {
  if (token.rfind("PID=", 0) != 0) {
    return absl::InvalidArgumentError("missing PID token");
  }
  std::string_view value = token.substr(4);
  if (value.empty()) {
    return absl::InvalidArgumentError("empty PID token");
  }
  char* end = nullptr;
  errno = 0;
  long parsed = std::strtol(std::string(value).c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0') {
    return absl::InvalidArgumentError("invalid PID token");
  }
  return static_cast<pid_t>(parsed);
}

absl::StatusOr<std::vector<std::string>> parse_handles_token(std::string_view token) {
  if (token.rfind("HANDLES_HEX=", 0) != 0) {
    return absl::InvalidArgumentError("missing HANDLES_HEX token");
  }
  std::string_view list = token.substr(std::string_view("HANDLES_HEX=").size());
  if (list.empty()) {
    return absl::InvalidArgumentError("empty HANDLES_HEX token");
  }
  std::vector<std::string> handles;
  size_t start = 0;
  while (start < list.size()) {
    size_t end = list.find(',', start);
    if (end == std::string_view::npos) {
      end = list.size();
    }
    std::string_view entry = list.substr(start, end - start);
    if (entry.empty()) {
      return absl::InvalidArgumentError("empty handle hex entry");
    }
    auto bytes_or = decode_hex(entry);
    if (!bytes_or.ok()) {
      return bytes_or.status();
    }
    handles.push_back(*bytes_or);
    start = end + 1;
  }
  return handles;
}

absl::StatusOr<std::pair<pid_t, std::vector<std::string>>> parse_helper_output(std::string_view line) {
  std::istringstream iss(std::string(line));
  std::string pid_token;
  std::string handles_token;
  iss >> pid_token >> handles_token;
  if (pid_token.empty() || handles_token.empty()) {
    return absl::InvalidArgumentError("helper output missing tokens");
  }
  auto pid_or = parse_pid_token(pid_token);
  if (!pid_or.ok()) {
    return pid_or.status();
  }
  auto handles_or = parse_handles_token(handles_token);
  if (!handles_or.ok()) {
    return handles_or.status();
  }
  return std::make_pair(*pid_or, *handles_or);
}

} // namespace

absl::StatusOr<std::string> resolve_cuda_ipc_helper_path() {
  if (const char* env = std::getenv("TENSORCAST_CUDA_IPC_HELPER"); env != nullptr && *env != '\0') {
    std::filesystem::path path(env);
    if (std::filesystem::exists(path)) {
      return path.string();
    }
    return absl::NotFoundError("TENSORCAST_CUDA_IPC_HELPER path does not exist");
  }
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (srcdir != nullptr && workspace != nullptr && *srcdir != '\0' && *workspace != '\0') {
    std::filesystem::path runfile = std::filesystem::path(srcdir) / workspace / "daemon" / "cuda_ipc_helper";
    if (std::filesystem::exists(runfile)) {
      return runfile.string();
    }
  }
  const std::vector<std::filesystem::path> candidates = {
      std::filesystem::path("bazel-bin") / "daemon" / "cuda_ipc_helper",
      std::filesystem::path("./daemon/cuda_ipc_helper"),
  };
  for (const auto& path : candidates) {
    if (std::filesystem::exists(path)) {
      return path.string();
    }
  }
  return absl::NotFoundError("cuda_ipc_helper not found in runfiles or bazel-bin");
}

CudaIpcChild::CudaIpcChild(CudaIpcChild&& other) noexcept
    : pid_(other.pid_),
      stdin_fd_(other.stdin_fd_),
      stdout_fd_(other.stdout_fd_),
      closed_(other.closed_),
      handle_bytes_(std::move(other.handle_bytes_)) {
  other.reset_no_wait();
}

CudaIpcChild& CudaIpcChild::operator=(CudaIpcChild&& other) noexcept {
  if (this != &other) {
    (void)Shutdown();
    pid_ = other.pid_;
    stdin_fd_ = other.stdin_fd_;
    stdout_fd_ = other.stdout_fd_;
    closed_ = other.closed_;
    handle_bytes_ = std::move(other.handle_bytes_);
    other.reset_no_wait();
  }
  return *this;
}

CudaIpcChild::~CudaIpcChild() {
  (void)Shutdown();
}

void CudaIpcChild::reset_no_wait() {
  pid_ = -1;
  stdin_fd_ = -1;
  stdout_fd_ = -1;
  closed_ = true;
  handle_bytes_.clear();
}

absl::StatusOr<CudaIpcChild> CudaIpcChild::Spawn(
    std::string helper_path,
    int device_id,
    const std::vector<CudaIpcBufferSpec>& buffers) {
  if (helper_path.empty()) {
    return absl::InvalidArgumentError("helper_path is empty");
  }
  if (buffers.empty()) {
    return absl::InvalidArgumentError("buffers list is empty");
  }
  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  if (pipe(stdin_pipe) != 0) {
    return absl::ErrnoToStatus(errno, "pipe failed for stdin");
  }
  if (pipe(stdout_pipe) != 0) {
    (void)close_fd(stdin_pipe[0]);
    (void)close_fd(stdin_pipe[1]);
    return absl::ErrnoToStatus(errno, "pipe failed for stdout");
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, stdin_pipe[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, stdout_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&actions, stdin_pipe[1]);
  posix_spawn_file_actions_addclose(&actions, stdout_pipe[0]);

  std::vector<std::string> args;
  args.reserve(buffers.size() + 3);
  args.push_back(helper_path);
  args.push_back(std::string("--device=") + std::to_string(device_id));
  for (const auto& buffer : buffers) {
    if (buffer.size_bytes == 0) {
      posix_spawn_file_actions_destroy(&actions);
      (void)close_fd(stdin_pipe[0]);
      (void)close_fd(stdin_pipe[1]);
      (void)close_fd(stdout_pipe[0]);
      (void)close_fd(stdout_pipe[1]);
      return absl::InvalidArgumentError("buffer size must be non-zero");
    }
    std::string spec = std::string("--buffer=") + std::to_string(buffer.size_bytes);
    if (buffer.fill_byte >= 0) {
      spec.append(",").append(std::to_string(buffer.fill_byte));
    }
    args.push_back(std::move(spec));
  }

  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  pid_t pid = -1;
  int spawn_rc = posix_spawn(&pid, helper_path.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (spawn_rc != 0) {
    (void)close_fd(stdin_pipe[0]);
    (void)close_fd(stdin_pipe[1]);
    (void)close_fd(stdout_pipe[0]);
    (void)close_fd(stdout_pipe[1]);
    return absl::ErrnoToStatus(spawn_rc, "posix_spawn failed");
  }

  (void)close_fd(stdin_pipe[0]);
  (void)close_fd(stdout_pipe[1]);

  auto line_or = read_line(stdout_pipe[0]);
  if (!line_or.ok()) {
    (void)close_fd(stdin_pipe[1]);
    (void)close_fd(stdout_pipe[0]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    return line_or.status();
  }
  auto parsed_or = parse_helper_output(*line_or);
  if (!parsed_or.ok()) {
    (void)close_fd(stdin_pipe[1]);
    (void)close_fd(stdout_pipe[0]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    return parsed_or.status();
  }

  const auto& [reported_pid, handles] = *parsed_or;
  if (reported_pid <= 0) {
    (void)close_fd(stdin_pipe[1]);
    (void)close_fd(stdout_pipe[0]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    return absl::InvalidArgumentError("helper reported invalid PID");
  }
  if (handles.size() != buffers.size()) {
    (void)close_fd(stdin_pipe[1]);
    (void)close_fd(stdout_pipe[0]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    return absl::InvalidArgumentError("helper handle count mismatch");
  }

  CudaIpcChild child;
  child.pid_ = pid;
  child.stdin_fd_ = stdin_pipe[1];
  child.stdout_fd_ = stdout_pipe[0];
  child.closed_ = false;
  child.handle_bytes_ = handles;
  return child;
}

absl::Status CudaIpcChild::Shutdown() {
  if (closed_) {
    return absl::OkStatus();
  }
  closed_ = true;
  if (stdin_fd_ >= 0) {
    ssize_t written = write(stdin_fd_, "q", 1);
    if (written < 0 && errno != EPIPE) {
      (void)close_fd(stdin_fd_);
      stdin_fd_ = -1;
      return absl::ErrnoToStatus(errno, "write failed while shutting down helper");
    }
    (void)close_fd(stdin_fd_);
    stdin_fd_ = -1;
  }
  if (stdout_fd_ >= 0) {
    (void)close_fd(stdout_fd_);
    stdout_fd_ = -1;
  }
  if (pid_ > 0) {
    int status = 0;
    if (waitpid(pid_, &status, 0) < 0) {
      return absl::ErrnoToStatus(errno, "waitpid failed");
    }
    pid_ = -1;
  }
  handle_bytes_.clear();
  return absl::OkStatus();
}

} // namespace tensorcast::daemon::testing
