// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "gsl/pointers"

#include "absl/status/statusor.h"
#include "core/communicator/engine/engine.h"
#include "core/store/loader/source.h"

namespace tensorcast::store::loader {

class RemoteKeySource : public SeekableSource {
 public:
  struct Options {
    gsl::not_null<std::shared_ptr<communicator::engine::CommunicateEngine>> comm_engine; // Communicator instance
    std::vector<std::string> memory_keys; // Remote tensor keys in order
    std::vector<size_t> buffer_sizes; // Corresponding sizes for each key
    std::string ip; // Remote peer IP
    uint16_t port = 0; // Remote peer port
    uint64_t total_size = 0; // Aggregate size across all keys
  };

  explicit RemoteKeySource(Options options);
  ~RemoteKeySource() override = default;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;

  // Random access read (required for pump_ranges)
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  // Enable direct-write when RDMA is available on the engine.
  [[nodiscard]] bool supports_direct_write() const override;
  absl::StatusOr<size_t> read_into(uint64_t dest_va_offset, size_t bytes, const DirectWriteToken& token) override;

 private:
  Options options_;
  size_t current_key_index_ = 0;
  size_t current_key_offset_ = 0;
  size_t total_bytes_read_ = 0;
};

} // namespace tensorcast::store::loader
