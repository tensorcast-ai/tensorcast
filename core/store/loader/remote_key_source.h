// Copyright (c) 2025, StepCast Team. All rights reserved.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "core/communicator/engine/engine.h"
#include "core/store/loader/source.h"

// Forward declaration for CommunicateEngine (needed for some tooling without full header parse)

namespace stepcast::communicator {
class CommunicateEngine;
} // namespace stepcast::communicator

namespace stepcast::store::loader {

class RemoteKeySource : public SeekableSource {
 public:
  struct Options {
    std::shared_ptr<stepcast::communicator::CommunicateEngine> comm_engine; // Communicator instance
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

 private:
  Options options_;
  size_t current_key_index_ = 0;
  size_t current_key_offset_ = 0;
  size_t total_bytes_read_ = 0;
};

} // namespace stepcast::store::loader