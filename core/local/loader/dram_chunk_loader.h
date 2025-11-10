// Copyright (c) 2025, TensorCast Team.

#include <future>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "core/local/loader/chunk_loader.h"

namespace tensorcast::local::data {
class DramChunkLoader : public ChunkLoader {
 public:
  explicit DramChunkLoader(DataChunk* data_chunk, void* dram_src) : ChunkLoader(data_chunk), dram_src_(dram_src) {}

  absl::Status load() override;
  // std::future<absl::Status> load_async() override;

 private:
  void* dram_src_{nullptr};
  // absl::Status load_();
};
} // namespace tensorcast::local::data