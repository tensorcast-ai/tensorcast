// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/safetensors_util.h"

#include <endian.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fcntl.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
// Use nlohmann/json for header parsing and canonical index serialization
#include <nlohmann/json.hpp>

namespace tensorcast::store::loader {

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
    return absl::ErrnoToStatus(errno, "fstat failed");
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

namespace {
// Map safetensors dtype token to torch dtype string used across the system
std::string map_safetensors_dtype_to_torch(const std::string& dtype) {
  static const std::map<std::string, std::string> kMap = {
      {"F16", "torch.float16"},
      {"BF16", "torch.bfloat16"},
      {"F32", "torch.float32"},
      {"F64", "torch.float64"},
      {"I8", "torch.int8"},
      {"I16", "torch.int16"},
      {"I32", "torch.int32"},
      {"I64", "torch.int64"},
      {"U8", "torch.uint8"},
      {"BOOL", "torch.uint8"},
  };
  auto it = kMap.find(dtype);
  if (it == kMap.end()) {
    return std::string();
  }
  return it->second;
}

std::vector<int64_t> compute_row_major_stride(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return {};
  }
  std::vector<int64_t> stride(shape.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    stride[i] = acc;
    acc *= shape[i];
  }
  return stride;
}
} // namespace

absl::StatusOr<std::string> BuildCanonicalIndexFromSafetensors(const std::vector<std::filesystem::path>& files) {
  using nlohmann::json;
  if (files.empty()) {
    return absl::InvalidArgumentError("No safetensors files provided");
  }
  // Sort by filename for deterministic base_offset accumulation
  std::vector<std::filesystem::path> paths = files;
  std::ranges::sort(paths, [](const auto& a, const auto& b) { return a.filename() < b.filename(); });

  // Collect entries: name -> [offset, size, shape, stride, dtype, storage_offset]
  struct Entry {
    uint64_t offset;
    uint64_t size;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    std::string dtype;
    int64_t storage_offset;
  };
  std::map<std::string, Entry> entries; // sorted by key

  uint64_t base_offset = 0;
  for (const auto& p : paths) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      return absl::ErrnoToStatus(errno, absl::StrFormat("Failed to open %s", p.string()));
    }
    auto header_info = ParseSafetensorsHeader(fd);
    if (!header_info.ok()) {
      ::close(fd);
      return header_info.status();
    }
    // Read header JSON
    std::string header;
    header.resize(static_cast<size_t>(header_info->header_length));
    ssize_t got = ::pread(fd, header.data(), header.size(), sizeof(uint64_t));
    ::close(fd);
    if (got != static_cast<ssize_t>(header.size())) {
      return absl::InvalidArgumentError("Truncated safetensors header");
    }
    json h = json::parse(header, nullptr, true);
    const uint64_t data_start = header_info->data_start;
    const uint64_t file_size = static_cast<uint64_t>(std::filesystem::file_size(p));
    if (data_start > file_size) {
      return absl::InvalidArgumentError("Invalid safetensors layout: data_start > file_size");
    }
    for (auto it = h.begin(); it != h.end(); ++it) {
      const auto& name = it.key();
      if (name == "__metadata__") {
        continue;
      }
      if (name.empty() || name.front() == '.' || name.find('/') != std::string::npos ||
          name.find('\\') != std::string::npos) {
        return absl::InvalidArgumentError("Invalid tensor name in safetensors header");
      }
      if (!it.value().is_object()) {
        return absl::InvalidArgumentError("Malformed safetensors tensor entry");
      }
      const json& meta = it.value();
      if (!meta.contains("dtype") || !meta.contains("shape") || !meta.contains("data_offsets")) {
        return absl::InvalidArgumentError("Missing required fields in safetensors tensor meta");
      }
      const std::string dtype_token = meta.at("dtype").get<std::string>();
      const std::string torch_dtype = map_safetensors_dtype_to_torch(dtype_token);
      if (torch_dtype.empty()) {
        return absl::InvalidArgumentError("Unsupported safetensors dtype");
      }
      const auto offsets = meta.at("data_offsets").get<std::vector<uint64_t>>();
      if (offsets.size() != 2 || offsets[1] < offsets[0]) {
        return absl::InvalidArgumentError("Invalid data_offsets in safetensors header");
      }
      std::vector<int64_t> shape;
      for (const auto& dim : meta.at("shape")) {
        shape.push_back(static_cast<int64_t>(dim.get<int64_t>()));
      }
      auto stride = compute_row_major_stride(shape);

      Entry e;
      e.offset = base_offset + offsets[0];
      e.size = offsets[1] - offsets[0];
      e.shape = std::move(shape);
      e.stride = std::move(stride);
      e.dtype = torch_dtype;
      e.storage_offset = 0;

      if (!entries.emplace(name, std::move(e)).second) {
        return absl::InvalidArgumentError("Duplicate tensor key across safetensors files");
      }
    }
    base_offset += (file_size - data_start);
  }

  // Serialize canonical index JSON with sorted keys and compact separators
  nlohmann::json out;
  for (const auto& [name, e] : entries) {
    out[name] = nlohmann::json::array();
    out[name].push_back(e.offset);
    out[name].push_back(e.size);
    out[name].push_back(e.shape);
    out[name].push_back(e.stride);
    out[name].push_back(e.dtype);
    out[name].push_back(e.storage_offset);
  }
  return out.dump();
}

} // namespace tensorcast::store::loader
