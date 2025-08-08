// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/dvmp_region_sink.h"

#include <cstring>

#include "absl/log/log.h"
#include "core/common/memory/distributed_memory_pool.h"

namespace stepcast::store::loader {

DVMPRegionSink::DVMPRegionSink(Options options) : options_(std::move(options)) {
  if (!options_.memory_manager) {
    LOG(ERROR) << "DVMPRegionSink: memory_manager is null";
  }
}

absl::Status DVMPRegionSink::write(const void* src, size_t bytes) {
  auto st = write_at(current_offset_, src, bytes);
  if (st.ok()) {
    current_offset_ += bytes;
  }
  return st;
}

absl::Status DVMPRegionSink::write_at(uint64_t offset, const void* src, size_t bytes) {
  if (!options_.memory_manager) {
    return absl::InvalidArgumentError("DVMPRegionSink: memory_manager is null");
  }

  auto* dvmp = options_.memory_manager->get_dvmp();
  if (dvmp == nullptr) {
    return absl::FailedPreconditionError("DVMPRegionSink: DVMP not available");
  }

  if (offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("DVMPRegionSink: write would exceed total size");
  }

  return dvmp->write_at(options_.memory_manager->get_model_id(), offset, src, bytes);
}

} // namespace stepcast::store::loader
