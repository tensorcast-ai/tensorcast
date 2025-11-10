// Copyright (c) 2025, TensorCast Team.

#include "core/local/loader/dram_chunk_loader.h"
#include "core/local/chunk/data_chunk.h"

namespace tensorcast::local::data {

absl::Status DramChunkLoader::load() {
  if (data_chunk_ == nullptr) {
    return absl::FailedPreconditionError("DataChunk pointer is null");
  }
  if (dram_src_ == nullptr) {
    return absl::FailedPreconditionError("DRAM source pointer is null");
  }
  if (data_chunk_->get_base_addr() == nullptr) {
    return absl::FailedPreconditionError("DataChunk CPU base not mapped");
  }
  if (::memcpy(data_chunk_->get_base_addr(), dram_src_, data_chunk_->get_size()) != 0) {
    return absl::InternalError("memcpy failed");
  }
  return absl::OkStatus();
}

} // namespace tensorcast::local::data