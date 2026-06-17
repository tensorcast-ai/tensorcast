// Copyright (c) 2025-2026, TensorCast Team.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/communicator/engine/engine.h"
#include "core/communicator/routing/routing_context.h"
#include "core/store/communication_types.h"
#include "core/store/materialization/contracts/stable_local_backing.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace tensorcast::store::components {

/**
 * @brief Manages P2P communication and RDMA operations.
 *
 * This component handles:
 * - Communication engine initialization
 * - P2P transfer coordination
 * - Memory registration for RDMA
 */
class CommunicationManager {
 public:
  CommunicationManager() = default;
  explicit CommunicationManager(std::shared_ptr<tensorcast::communicator::engine::Communicator> external_engine);
  ~CommunicationManager() = default;

  // Disable copy and move
  CommunicationManager(const CommunicationManager&) = delete;
  CommunicationManager& operator=(const CommunicationManager&) = delete;
  CommunicationManager(CommunicationManager&&) = delete;
  CommunicationManager& operator=(CommunicationManager&&) = delete;

  /**
   * @brief Initialize the communication engine.
   *
   * Phase-5 change: the RDMA enable/disable flag is now provided explicitly by
   * configuration rather than being inferred from environment variables.  The
   * parameter has a default value of `true` so that existing call-sites that
   * do not yet pass the flag continue to compile and retain previous
   * behaviour.
   *
   * @param listen_addr  Address to listen on (e.g., "0.0.0.0").
   * @param listen_port  Port to listen on.
   * @param enable_rdma  Whether to enable RDMA transports.  Defaults to true.
   * @return Status indicating success or failure.
   */
  absl::Status initialize(const std::string& listen_addr, uint16_t listen_port, bool enable_rdma = false);

  // Initialize with typed communicator config
  absl::Status initialize_with_config(
      const std::string& listen_addr,
      uint16_t listen_port,
      const tensorcast::communicator::v1::CommunicatorConfig& config);

  // Initialize with typed config and externally provided pinned staging pools.
  absl::Status initialize_with_config_and_pools(
      const std::string& listen_addr,
      uint16_t listen_port,
      const tensorcast::communicator::v1::CommunicatorConfig& config,
      tensorcast::communicator::engine::Communicator::PinnedStagingPools pools);

  /**
   * @brief Check if communication is enabled.
   */
  [[nodiscard]] bool is_enabled() const {
    return enabled_ && comm_engine_ != nullptr;
  }

  /**
   * @brief Get the communication engine.
   * @return Pointer to the engine or nullptr if not initialized
   */
  tensorcast::communicator::engine::Communicator& get_engine() {
    return *comm_engine_;
  }

  // Return shared pointer for cases needing shared ownership (e.g., ReplicaConfig)
  [[nodiscard]] std::shared_ptr<tensorcast::communicator::engine::Communicator> get_shared_engine() const {
    return comm_engine_;
  }

  /**
   * @brief Register memory for communication access.
   * @param buffer_addresses Vector of buffer addresses
   * @param buffer_sizes Vector of buffer sizes
   * @param device_id Device ID for GPU memory
   * @return Registration info or error
   */
  absl::StatusOr<ExportRegistration> register_memory(
      const std::vector<void*>& buffer_addresses,
      const std::vector<size_t>& buffer_sizes,
      int device_id);

  struct StableLocalBackingSourceView {
    uint64_t address = 0;
    size_t size_bytes = 0;
    store::StableLocalBackingRef backing;
    std::shared_ptr<void> keepalive;
  };

  absl::StatusOr<ExportRegistration> register_stable_local_backing_source_views(
      const std::vector<StableLocalBackingSourceView>& views);

  absl::Status activate_stable_local_backing(
      const store::StableLocalBackingRef& backing,
      std::shared_ptr<void> keepalive = nullptr);

  absl::Status deactivate_stable_local_backing(std::string_view backing_id);

  [[nodiscard]] bool stable_local_backing_supported_for_test() const;

  [[nodiscard]] bool stable_local_backing_active_for_test(std::string_view backing_id) const;

  uint16_t listen_port() const {
    return listen_port_;
  }

  void set_routing_context(std::shared_ptr<communicator::routing::RoutingContext> routing_context);

  [[nodiscard]] std::shared_ptr<communicator::routing::RoutingContext> routing_context() const;

 private:
  bool enabled_ = false;
  uint16_t listen_port_{0};
  std::shared_ptr<tensorcast::communicator::engine::Communicator> comm_engine_;
  mutable absl::Mutex routing_context_mu_;
  std::shared_ptr<communicator::routing::RoutingContext> routing_context_ ABSL_GUARDED_BY(routing_context_mu_);
  std::atomic<uint64_t> next_registration_id_{1};
};

} // namespace tensorcast::store::components
