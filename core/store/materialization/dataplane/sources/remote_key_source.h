// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "gsl/pointers"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/routing_context.h"
#include "core/store/materialization/dataplane/contracts/source.h"

namespace tensorcast::store::loader {

class RemoteKeySource : public SeekableSource {
 public:
  struct Options {
    gsl::not_null<std::shared_ptr<communicator::engine::Communicator>> comm_engine; // Communicator instance
    std::vector<std::string> memory_keys; // Remote tensor keys in order
    std::vector<size_t> buffer_sizes; // Corresponding sizes for each key
    std::string ip; // Remote peer IP
    uint16_t port = 0; // Remote peer port
    std::string local_endpoint_id;
    std::string remote_endpoint_id;
    std::shared_ptr<communicator::routing::RoutingContext> routing_context;
    uint64_t total_size = 0; // Aggregate size across all keys
    // Request-level budget propagated from RPC/request scope. When >0, all
    // remote read waits are bounded by this budget.
    std::chrono::milliseconds request_budget{0};
    // Poll interval while waiting for communicator read futures.
    std::chrono::milliseconds wait_slice{std::chrono::seconds(5)};
    // Periodic stalled-read diagnostics cadence.
    std::chrono::milliseconds stalled_log_interval{std::chrono::seconds(30)};
    // Optional artifact id used only for diagnostics.
    std::string artifact_id;
  };

  explicit RemoteKeySource(Options options);
  ~RemoteKeySource() override = default;

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override;

  // Random access read (required for pump_ranges)
  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override;

  // Enable direct-write when RDMA is available on the engine.
  [[nodiscard]] uint64_t total_bytes() const override {
    return options_.total_size;
  }

  [[nodiscard]] bool supports_direct_write_at() const override;
  absl::StatusOr<size_t> read_into_at(
      uint64_t src_offset,
      uint64_t dest_va_offset,
      size_t bytes,
      const DirectWriteGrant& grant) override;

 private:
  absl::StatusOr<communicator::transport::read_result_t> await_read_result(
      communicator::transport::future_read_result_t& future,
      std::string_view key,
      uint64_t remote_offset,
      size_t bytes);
  absl::StatusOr<communicator::transport::read_result_t> read_with_strict_fallback(
      const std::string& key,
      uint64_t local_addr,
      size_t bytes,
      uint64_t remote_offset,
      int dev_type,
      int dev_id);
  void abort_timed_out_channel(std::string_view key, uint64_t remote_offset, size_t bytes) const;
  std::chrono::milliseconds remaining_request_budget() const;

  Options options_;
  std::chrono::steady_clock::time_point request_start_{std::chrono::steady_clock::now()};
  std::chrono::steady_clock::time_point last_stalled_log_{request_start_};
  std::chrono::steady_clock::time_point last_cost_log_{request_start_};
  size_t current_key_index_ = 0;
  size_t current_key_offset_ = 0;
  size_t total_bytes_read_ = 0;
  bool routed_path_enabled_ = true;
};

} // namespace tensorcast::store::loader
