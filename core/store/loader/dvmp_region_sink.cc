// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/dvmp_region_sink.h"

#include <cstring>

#include "absl/log/log.h"
#include "core/common/memory/distributed_memory_pool.h"

namespace stepcast::store::loader {

DVMPRegionSink::DVMPRegionSink(Options options) : options_(std::move(options)) {
  if (options_.total_size == 0) {
    LOG(WARNING) << "DVMPRegionSink: total_size is 0";
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
  if (offset + bytes > options_.total_size) {
    return absl::InvalidArgumentError("DVMPRegionSink: write would exceed total size");
  }

  return options_.region.write_at(offset, src, bytes);
}

} // namespace stepcast::store::loader
