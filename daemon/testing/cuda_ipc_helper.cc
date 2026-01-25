// Copyright (c) 2025-2026, TensorCast Team.

#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "core/cuda/cuda_api.h"
#include "core/cuda/device_guard.h"

namespace {

struct BufferSpec {
  size_t size_bytes = 0;
  int fill_byte = -1;
};

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.rfind(prefix, 0) == 0;
}

bool parse_int(std::string_view value, int* out_value) {
  if (out_value == nullptr) {
    return false;
  }
  char* end_ptr = nullptr;
  errno = 0;
  long parsed = std::strtol(std::string(value).c_str(), &end_ptr, 10);
  if (errno != 0 || end_ptr == nullptr || *end_ptr != '\0') {
    return false;
  }
  *out_value = static_cast<int>(parsed);
  return true;
}

bool parse_size(std::string_view value, size_t* out_value) {
  if (out_value == nullptr) {
    return false;
  }
  char* end_ptr = nullptr;
  errno = 0;
  unsigned long long parsed = std::strtoull(std::string(value).c_str(), &end_ptr, 10);
  if (errno != 0 || end_ptr == nullptr || *end_ptr != '\0') {
    return false;
  }
  *out_value = static_cast<size_t>(parsed);
  return true;
}

absl::Status parse_buffer_spec(std::string_view value, BufferSpec* out_spec) {
  if (out_spec == nullptr) {
    return absl::InvalidArgumentError("buffer spec output is null");
  }
  const size_t comma_pos = value.find(',');
  std::string_view size_part = value;
  std::string_view fill_part;
  if (comma_pos != std::string_view::npos) {
    size_part = value.substr(0, comma_pos);
    fill_part = value.substr(comma_pos + 1);
  }
  size_t size_bytes = 0;
  if (!parse_size(size_part, &size_bytes) || size_bytes == 0) {
    return absl::InvalidArgumentError("invalid buffer size");
  }
  int fill = -1;
  if (!fill_part.empty()) {
    if (!parse_int(fill_part, &fill) || fill < 0 || fill > 255) {
      return absl::InvalidArgumentError("invalid fill byte");
    }
  }
  out_spec->size_bytes = size_bytes;
  out_spec->fill_byte = fill;
  return absl::OkStatus();
}

std::string handle_to_hex(const cudaIpcMemHandle_t& handle) {
  const auto* handle_bytes = reinterpret_cast<const std::uint8_t*>(&handle);
  std::string out;
  out.reserve(sizeof(cudaIpcMemHandle_t) * 2);
  constexpr char kHexDigits[] = "0123456789abcdef";
  for (size_t index = 0; index < sizeof(cudaIpcMemHandle_t); ++index) {
    const std::uint8_t value = handle_bytes[index];
    out.push_back(kHexDigits[(value >> 4) & 0x0F]);
    out.push_back(kHexDigits[value & 0x0F]);
  }
  return out;
}

void print_usage() {
  std::cerr << "Usage: cuda_ipc_helper --device=<id> [--size=<bytes> --fill=<byte>] "
               "[--buffer=<bytes>[,<fill>]]...\n";
}

} // namespace

int main(int argc, char** argv) {
  int device_id = 0;
  size_t fallback_size = 0;
  int fallback_fill = -1;
  std::vector<BufferSpec> buffers;
  for (int arg_index = 1; arg_index < argc; ++arg_index) {
    const std::string_view arg(argv[arg_index]);
    if (arg == "--help") {
      print_usage();
      return 0;
    }
    if (starts_with(arg, "--device=")) {
      int parsed = 0;
      if (!parse_int(arg.substr(std::string_view("--device=").size()), &parsed)) {
        std::cerr << "Invalid --device value\n";
        return 1;
      }
      device_id = parsed;
      continue;
    }
    if (starts_with(arg, "--size=")) {
      if (!parse_size(arg.substr(std::string_view("--size=").size()), &fallback_size)) {
        std::cerr << "Invalid --size value\n";
        return 1;
      }
      continue;
    }
    if (starts_with(arg, "--fill=")) {
      int parsed = 0;
      if (!parse_int(arg.substr(std::string_view("--fill=").size()), &parsed) || parsed < 0 || parsed > 255) {
        std::cerr << "Invalid --fill value\n";
        return 1;
      }
      fallback_fill = parsed;
      continue;
    }
    if (starts_with(arg, "--buffer=")) {
      BufferSpec spec;
      absl::Status st = parse_buffer_spec(arg.substr(std::string_view("--buffer=").size()), &spec);
      if (!st.ok()) {
        std::cerr << "Invalid --buffer value: " << st.ToString() << "\n";
        return 1;
      }
      buffers.push_back(spec);
      continue;
    }
    std::cerr << "Unknown argument: " << arg << "\n";
    print_usage();
    return 1;
  }

  if (buffers.empty()) {
    if (fallback_size == 0) {
      std::cerr << "No buffers requested.\n";
      print_usage();
      return 1;
    }
    buffers.push_back(BufferSpec{.size_bytes = fallback_size, .fill_byte = fallback_fill});
  }

  if (!tensorcast::cuda::is_available()) {
    std::cerr << "CUDA is not available\n";
    return 2;
  }

  tensorcast::cuda::DeviceGuard guard(device_id);
  if (!guard.status().ok()) {
    std::cerr << "Failed to select device: " << guard.status().ToString() << "\n";
    return 2;
  }

  std::vector<void*> allocations;
  allocations.reserve(buffers.size());
  std::vector<std::string> handles_hex;
  handles_hex.reserve(buffers.size());

  for (const auto& spec : buffers) {
    void* device_ptr = nullptr;
    absl::Status st = tensorcast::cuda::malloc(&device_ptr, spec.size_bytes);
    if (!st.ok()) {
      std::cerr << "cuda::malloc failed: " << st.ToString() << "\n";
      return 2;
    }
    if (spec.fill_byte >= 0) {
      st = tensorcast::cuda::memset(device_ptr, spec.fill_byte, spec.size_bytes);
      if (!st.ok()) {
        std::cerr << "cuda::memset failed: " << st.ToString() << "\n";
        return 2;
      }
    }
    cudaIpcMemHandle_t handle{};
    st = tensorcast::cuda::get_ipc_mem_handle(&handle, device_ptr);
    if (!st.ok()) {
      std::cerr << "get_ipc_mem_handle failed: " << st.ToString() << "\n";
      return 2;
    }
    allocations.push_back(device_ptr);
    handles_hex.push_back(handle_to_hex(handle));
  }

  std::cout << "PID=" << getpid() << " HANDLES_HEX=";
  for (size_t handle_index = 0; handle_index < handles_hex.size(); ++handle_index) {
    if (handle_index > 0) {
      std::cout << ",";
    }
    std::cout << handles_hex[handle_index];
  }
  std::cout << "\n";
  std::cout.flush();

  char signal_byte = 0;
  (void)read(STDIN_FILENO, &signal_byte, 1);

  for (void* device_ptr : allocations) {
    absl::Status st = tensorcast::cuda::free(device_ptr);
    if (!st.ok()) {
      std::cerr << "cuda::free failed: " << st.ToString() << "\n";
      return 3;
    }
  }

  return 0;
}
