// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/communicator/engine/engine.h"
#include "core/store/communication_types.h"
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

 private:
  bool enabled_ = false;
  std::shared_ptr<tensorcast::communicator::engine::Communicator> comm_engine_;
};

} // namespace tensorcast::store::components
