// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/dataplane/sinks/cpu_va_sink.h"

#include <cstring>

#include "absl/log/log.h"

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

  if (!options_.uma) {
    return absl::FailedPreconditionError("CpuVaSink: UMA handle is null");
  }
  return options_.uma->write_cpu_span(options_.replica_key, offset, src, bytes);
}

} // namespace tensorcast::store::loader
