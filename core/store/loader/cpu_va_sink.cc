// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/cpu_va_sink.h"

#include <cstring>

#include "absl/log/log.h"
#include "core/common/memory/virtual_address_space.h"

namespace tensorcast::store::loader {

CpuVaSink::CpuVaSink(Options options) : options_(std::move(options)) {
  if (options_.total_size == 0) {
    LOG(WARNING) << "CpuVaSink: total_size is 0";
  }
}

absl::Status CpuVaSink::write(const void* src, size_t bytes) {
  auto st = write_at(current_offset_, src, bytes);
  if (st.ok()) {
    current_offset_ += bytes;
  }
  return st;
}

absl::Status CpuVaSink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("CpuVaSink: write would exceed total size");
  }

  return options_.region.write_at(offset, src, bytes);
}

} // namespace tensorcast::store::loader
