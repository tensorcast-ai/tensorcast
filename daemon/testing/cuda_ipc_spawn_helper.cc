// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/testing/cuda_ipc_spawn_helper.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string_view>

#include "absl/status/status.h"

extern char** environ;

namespace tensorcast::daemon::testing {
namespace {

absl::Status close_fd(int file_descriptor) {
  if (file_descriptor < 0) {
    return absl::OkStatus();
  }
  if (close(file_descriptor) == 0) {
    return absl::OkStatus();
  }
  return absl::ErrnoToStatus(errno, "close failed");
}

absl::StatusOr<std::string> read_pid_line(int file_descriptor) {
  std::string buffer;
  buffer.reserve(512);
  constexpr size_t kMaxBytes = 8192;
  const std::string pid_token = "PID=";
  const std::string handles_token = "HANDLES_HEX=";
  while (buffer.size() < kMaxBytes) {
    char chunk[256];
    ssize_t bytes_read = read(file_descriptor, chunk, sizeof(chunk));
    if (bytes_read < 0) {
      return absl::ErrnoToStatus(errno, "read failed");
    }
    if (bytes_read == 0) {
      break;
    }
    buffer.append(chunk, static_cast<size_t>(bytes_read));
    const size_t pid_pos = buffer.find(pid_token);
    if (pid_pos == std::string::npos) {
      continue;
    }
    const size_t handles_pos = buffer.find(handles_token, pid_pos);
    if (handles_pos == std::string::npos) {
      continue;
    }
    const size_t newline_pos = buffer.find('\n', handles_pos);
    if (newline_pos != std::string::npos) {
      return buffer.substr(pid_pos, newline_pos - pid_pos);
    }
  }
  return absl::InternalError("helper output missing PID/HANDLES line");
}

int hex_value(char hex_char) {
  if (hex_char >= '0' && hex_char <= '9') {
    return hex_char - '0';
  }
  if (hex_char >= 'a' && hex_char <= 'f') {
    return 10 + (hex_char - 'a');
  }
  if (hex_char >= 'A' && hex_char <= 'F') {
    return 10 + (hex_char - 'A');
  }
  return -1;
}

absl::StatusOr<std::string> decode_hex(std::string_view hex) {
  if (hex.size() % 2 != 0) {
    return absl::InvalidArgumentError("hex string has odd length");
  }
  std::string decoded_bytes;
  decoded_bytes.resize(hex.size() / 2);
  for (size_t byte_index = 0; byte_index < decoded_bytes.size(); ++byte_index) {
    const int high_nibble = hex_value(hex[byte_index * 2]);
    const int low_nibble = hex_value(hex[byte_index * 2 + 1]);
    if (high_nibble < 0 || low_nibble < 0) {
      return absl::InvalidArgumentError("invalid hex character");
    }
    decoded_bytes[byte_index] = static_cast<char>((high_nibble << 4) | low_nibble);
  }
  return decoded_bytes;
}

absl::StatusOr<std::vector<std::string>> parse_handles_token(std::string_view token) {
  if (token.rfind("HANDLES_HEX=", 0) != 0) {
    return absl::InvalidArgumentError("missing HANDLES_HEX token");
  }
  std::string_view handle_list = token.substr(std::string_view("HANDLES_HEX=").size());
  if (handle_list.empty()) {
    return absl::InvalidArgumentError("empty HANDLES_HEX token");
  }
  std::vector<std::string> handles;
  size_t cursor = 0;
  while (cursor < handle_list.size()) {
    size_t separator = handle_list.find(',', cursor);
    if (separator == std::string_view::npos) {
      separator = handle_list.size();
    }
    std::string_view entry_view = handle_list.substr(cursor, separator - cursor);
    if (entry_view.empty()) {
      return absl::InvalidArgumentError("empty handle hex entry");
    }
    auto bytes_or = decode_hex(entry_view);
    if (!bytes_or.ok()) {
      return bytes_or.status();
    }
    handles.push_back(*bytes_or);
    cursor = separator + 1;
  }
  return handles;
}

std::string_view trim_whitespace(std::string_view value) {
  size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
    ++start;
  }
  size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return value.substr(start, end - start);
}

absl::StatusOr<std::pair<pid_t, std::vector<std::string>>> parse_helper_output(std::string_view line) {
  const std::string_view pid_prefix = "PID=";
  const std::string_view handles_prefix = "HANDLES_HEX=";
  const size_t pid_pos = line.find(pid_prefix);
  if (pid_pos == std::string_view::npos) {
    return absl::InvalidArgumentError(std::string("helper output missing PID prefix: ") + std::string(line));
  }
  size_t pid_value_pos = pid_pos + pid_prefix.size();
  while (pid_value_pos < line.size() && std::isspace(static_cast<unsigned char>(line[pid_value_pos])) != 0) {
    ++pid_value_pos;
  }
  size_t pid_end = pid_value_pos;
  while (pid_end < line.size() && std::isdigit(static_cast<unsigned char>(line[pid_end])) != 0) {
    ++pid_end;
  }
  if (pid_end == pid_value_pos) {
    return absl::InvalidArgumentError(std::string("invalid PID token: ") + std::string(line));
  }
  std::string pid_value(line.substr(pid_value_pos, pid_end - pid_value_pos));
  char* end_ptr = nullptr;
  errno = 0;
  long parsed = std::strtol(pid_value.c_str(), &end_ptr, 10);
  if (errno != 0 || end_ptr == nullptr || *end_ptr != '\0') {
    return absl::InvalidArgumentError(std::string("invalid PID token: ") + pid_value);
  }
  const size_t handles_pos = line.find(handles_prefix, pid_end);
  if (handles_pos == std::string_view::npos) {
    return absl::InvalidArgumentError(std::string("helper output missing handles token: ") + std::string(line));
  }
  std::string_view handles_view = trim_whitespace(line.substr(handles_pos));
  auto handles_or = parse_handles_token(handles_view);
  if (!handles_or.ok()) {
    return handles_or.status();
  }
  return std::make_pair(static_cast<pid_t>(parsed), *handles_or);
}

absl::StatusOr<std::string> resolve_runfiles_path(std::string_view runfiles_root, std::string_view workspace) {
  if (runfiles_root.empty()) {
    return absl::InvalidArgumentError("runfiles root is empty");
  }
  std::filesystem::path base(runfiles_root);
  if (!workspace.empty()) {
    base /= std::filesystem::path(workspace);
  }
  base /= std::filesystem::path("daemon") / "cuda_ipc_helper";
  if (std::filesystem::exists(base)) {
    return base.string();
  }
  return absl::NotFoundError("cuda_ipc_helper not found under runfiles root");
}

absl::StatusOr<std::string> resolve_from_manifest(std::string_view manifest_path) {
  if (manifest_path.empty()) {
    return absl::InvalidArgumentError("manifest path is empty");
  }
  std::ifstream stream{std::string(manifest_path)};
  if (!stream.is_open()) {
    return absl::NotFoundError("runfiles manifest could not be opened");
  }
  const std::string suffix = std::string("daemon/cuda_ipc_helper");
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    const size_t space_pos = line.find(' ');
    if (space_pos == std::string::npos) {
      continue;
    }
    std::string_view line_view(line);
    std::string_view runfile_path = line_view.substr(0, space_pos);
    std::string_view real_path = line_view.substr(space_pos + 1);
    if (runfile_path.size() < suffix.size()) {
      continue;
    }
    if (runfile_path.compare(runfile_path.size() - suffix.size(), suffix.size(), suffix) != 0) {
      continue;
    }
    std::filesystem::path resolved(real_path);
    if (std::filesystem::exists(resolved)) {
      return resolved.string();
    }
  }
  return absl::NotFoundError("cuda_ipc_helper not found in runfiles manifest");
}

} // namespace

absl::StatusOr<std::string> resolve_cuda_ipc_helper_path() {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  if (srcdir != nullptr && *srcdir != '\0') {
    const std::string_view workspace_view = workspace != nullptr ? std::string_view(workspace) : std::string_view();
    auto runfiles_or = resolve_runfiles_path(srcdir, workspace_view);
    if (runfiles_or.ok()) {
      return runfiles_or;
    }
  }
  const char* runfiles_dir = std::getenv("RUNFILES_DIR");
  if (runfiles_dir != nullptr && *runfiles_dir != '\0') {
    const std::string_view workspace_view = workspace != nullptr ? std::string_view(workspace) : std::string_view();
    auto runfiles_or = resolve_runfiles_path(runfiles_dir, workspace_view);
    if (runfiles_or.ok()) {
      return runfiles_or;
    }
  }
  const char* runfiles_manifest = std::getenv("RUNFILES_MANIFEST_FILE");
  if (runfiles_manifest != nullptr && *runfiles_manifest != '\0') {
    auto manifest_or = resolve_from_manifest(runfiles_manifest);
    if (manifest_or.ok()) {
      return manifest_or;
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
  int stdin_pipe_fds[2] = {-1, -1};
  int stdout_pipe_fds[2] = {-1, -1};
  if (pipe(stdin_pipe_fds) != 0) {
    return absl::ErrnoToStatus(errno, "pipe failed for stdin");
  }
  if (pipe(stdout_pipe_fds) != 0) {
    (void)close_fd(stdin_pipe_fds[0]);
    (void)close_fd(stdin_pipe_fds[1]);
    return absl::ErrnoToStatus(errno, "pipe failed for stdout");
  }

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, stdin_pipe_fds[0], STDIN_FILENO);
  posix_spawn_file_actions_adddup2(&actions, stdout_pipe_fds[1], STDOUT_FILENO);
  posix_spawn_file_actions_addclose(&actions, stdin_pipe_fds[1]);
  posix_spawn_file_actions_addclose(&actions, stdout_pipe_fds[0]);

  std::vector<std::string> args;
  args.reserve(buffers.size() + 3);
  args.push_back(helper_path);
  args.push_back(std::string("--device=") + std::to_string(device_id));
  for (const auto& buffer : buffers) {
    if (buffer.size_bytes == 0) {
      posix_spawn_file_actions_destroy(&actions);
      (void)close_fd(stdin_pipe_fds[0]);
      (void)close_fd(stdin_pipe_fds[1]);
      (void)close_fd(stdout_pipe_fds[0]);
      (void)close_fd(stdout_pipe_fds[1]);
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
    (void)close_fd(stdin_pipe_fds[0]);
    (void)close_fd(stdin_pipe_fds[1]);
    (void)close_fd(stdout_pipe_fds[0]);
    (void)close_fd(stdout_pipe_fds[1]);
    return absl::ErrnoToStatus(spawn_rc, "posix_spawn failed");
  }

  (void)close_fd(stdin_pipe_fds[0]);
  (void)close_fd(stdout_pipe_fds[1]);

  auto line_or = read_pid_line(stdout_pipe_fds[0]);
  if (!line_or.ok()) {
    (void)close_fd(stdin_pipe_fds[1]);
    (void)close_fd(stdout_pipe_fds[0]);
    int wait_status = 0;
    (void)waitpid(pid, &wait_status, 0);
    return line_or.status();
  }
  auto parsed_or = parse_helper_output(*line_or);
  if (!parsed_or.ok()) {
    (void)close_fd(stdin_pipe_fds[1]);
    (void)close_fd(stdout_pipe_fds[0]);
    int wait_status = 0;
    (void)waitpid(pid, &wait_status, 0);
    return parsed_or.status();
  }

  const auto& [reported_pid, handles] = *parsed_or;
  if (reported_pid <= 0) {
    (void)close_fd(stdin_pipe_fds[1]);
    (void)close_fd(stdout_pipe_fds[0]);
    int wait_status = 0;
    (void)waitpid(pid, &wait_status, 0);
    return absl::InvalidArgumentError("helper reported invalid PID");
  }
  if (handles.size() != buffers.size()) {
    (void)close_fd(stdin_pipe_fds[1]);
    (void)close_fd(stdout_pipe_fds[0]);
    int wait_status = 0;
    (void)waitpid(pid, &wait_status, 0);
    return absl::InvalidArgumentError("helper handle count mismatch");
  }

  CudaIpcChild child;
  child.pid_ = pid;
  child.stdin_fd_ = stdin_pipe_fds[1];
  child.stdout_fd_ = stdout_pipe_fds[0];
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
