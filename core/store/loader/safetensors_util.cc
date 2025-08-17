// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/safetensors_util.h"

#include <endian.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "absl/strings/str_format.h"

namespace stepcast::store::loader {

absl::StatusOr<SafetensorsHeaderInfo> ParseSafetensorsHeader(int fd) {
  // Read the 8-byte header length (little-endian)
  uint64_t header_len_le = 0;
  ssize_t n = ::pread(fd, &header_len_le, sizeof(header_len_le), 0);
  if (n != static_cast<ssize_t>(sizeof(header_len_le))) {
    return absl::InvalidArgumentError("Invalid safetensors file: cannot read header length");
  }
  
  // Convert from little-endian to host byte order
  uint64_t header_length = le64toh(header_len_le);
  
  // Validate header length (max 1GB for safety)
  if (header_length > (1ULL << 30)) {
    return absl::InvalidArgumentError("Safetensors header too large");
  }
  
  // Get file size to calculate data size
  struct stat stbuf{};
  if (::fstat(fd, &stbuf) != 0) {
    return absl::InternalError(absl::StrFormat("fstat failed: %s", std::strerror(errno)));
  }
  uint64_t file_size = static_cast<uint64_t>(stbuf.st_size);
  
  // Calculate offsets and sizes
  SafetensorsHeaderInfo info;
  info.header_length = header_length;
  info.data_start = sizeof(uint64_t) + header_length;
  
  // Validate that data start is within file bounds
  if (info.data_start > file_size) {
    return absl::InvalidArgumentError("Invalid safetensors file: data starts beyond EOF");
  }
  
  info.data_size = file_size - info.data_start;
  
  return info;
}

}  // namespace stepcast::store::loader