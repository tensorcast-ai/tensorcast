// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_ENGINE_H_
#define CORE_COMMUNICATOR_ENGINE_ENGINE_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"

#include "core/communicator/base/constants.h"
#include "core/communicator/misc/queue.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/request.h"
#include "core/communicator/transport/tcp_context.h"

#include "core/common/memory/pinned_memory_pool.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/gpu_tcp_stager.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/engine/mr_cache.h"
#include "core/communicator/engine/dram_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/communicator_config.h"
#include "core/communicator/engine/store.h"
#include "core/communicator/misc/ibv_wrap.h"

namespace tensorcast::communicator {

 class CommunicateEngine {
 public:
  explicit CommunicateEngine(bool enable_rdma = false, uint32_t channel_expire_sec = 0);
  explicit CommunicateEngine(const CommunicatorConfig& config, uint32_t channel_expire_sec = 0);

  ~CommunicateEngine();

  /**
   * Init the communicate engine with the ip and port
   * @param local_ip local ip
   * @param local_port local port
   * @return  init result
   */
  absl::Status init(const std::string& local_ip, uint16_t local_port, int conn_count = kMTcpConnCount);

  /**
   * Read a remote tensor
   * @param key remote tensor key
   * @param addr local tensor addr
   * @param bytes local tensor bytes
   * @param dev_type tensor in CPU or GPU
   * @param dev_id the GPU id if dev_type is GPU
   * @param dst_ip remote ip
   * @param dst_port remote port
   * @param remote_offset remote tensor addr offset
   * @return
   */
  future_read_result_t read_tensor(
      const std::string& key,
      uint64_t addr,
      uint64_t bytes,
      int dev_type,
      int dev_id,
      const std::string& dst_ip,
      uint16_t dst_port,
      uint64_t remote_offset = 0);

  /**
   * Register a partition tensor
   * @param tensor_key the unique tensor key
   * @param addr uint64 address
   * @param bytes tensor length in bytes
   * @param dev_type tensor in CPU or GPU
   * @param dev_id the GPU id if dev_type is GPU
   * @param async whether enabling async tensor registration
   * @return if async, the status is always Ok,
   *    otherwise return the ib registration status
   */
  absl::Status register_tensor(
      const std::string& tensor_key,
      uint64_t tensor_addr,
      uint64_t tensor_bytes,
      int dev_type,
      int dev_id,
      bool async = false);

  struct RegisterTensorOptions {
    bool register_mr = true;     // Skip MR registration when false
    bool needs_staging = false;  // Hint for transports that staging is required
    bool async = false;          // Async MR registration when applicable
  };

  // Extended registration with options.
  absl::Status register_tensor_ex(
      const std::string& tensor_key,
      uint64_t tensor_addr,
      uint64_t tensor_bytes,
      int dev_type,
      int dev_id,
      const RegisterTensorOptions& opts);

  /**
   * Unregister a tensor from communication engine
   * @param tensor_key the unique tensor key
   * @return
   */
  absl::Status unregister_tensor(const std::string& tensor_key);

  /**
   * Close a connection
   * @param dst_ip remote ip
   * @param dst_port remote port
   * @return
   */
  absl::Status close_connection(const std::string& dst_ip, uint16_t dst_port);

  // Inject a UMA-backed lease provider for DRAM staging (optional).
  void set_dram_lease_provider(std::shared_ptr<DRAMStager::LeaseProvider> provider);

  // Lightweight UMA residency provider for DirectMR escape hatch
  struct ResidencyProvider {
    virtual ~ResidencyProvider() = default;
    virtual bool is_hot(const std::string& tensor_key, uint64_t offset, uint64_t bytes) = 0;
  };
  void set_residency_provider(std::shared_ptr<ResidencyProvider> provider) { residency_provider_ = std::move(provider); }

  /**
   * @brief Whether this engine instance was configured with RDMA support.
   * When false the engine falls back to pure TCP transports and only supports
   * CPU memory registration / transfer.
   */
  inline bool is_rdma_enabled() const {
    return enable_rdma_;
  }

  // Test-only accessors (safe to call in unit tests)
  int inflight_direct_mr_for_test() const { return inflight_direct_mr_.load(); }
  size_t staged_segments_count_for_test() {
    absl::MutexLock lk(&staged_mu_);
    return staged_segments_.pairs().size();
  }

 private:
  net_dev_t get_net_dev(int dev_type, int dev_id);

  result_t on_receive_request(const channel_t& channel, const tcp_transport_t& t, const engine_message_t& msg);
  result_t on_receive_response(const channel_t& channel, const tcp_transport_t& t, const engine_message_t& msg);

  void do_read_request_loop();
  void do_channel_gc_loop();

  result_t on_new_client(const tcp_transport_t& t);

  absl::StatusOr<channel_t> do_create_channel(const std::string& ip, uint16_t port);

  std::atomic_bool stop_;
  std::atomic_bool inited_;
  tcp_context_t server_context_;
  tcp_context_t client_context_;
  rdma_context_t rdma_context_ = nullptr;
  Queue<read_request_t> request_queue_;
  Map<std::string, read_request_t> pending_requests_;
  Map<std::string, channel_t> channels_;
  PartitionTensorStore store_;
  std::thread request_thread_;
  std::thread gc_thread_;
  bool enable_rdma_;
  int mtcp_conn_count_;
  uint32_t ack_ttl_ms_ = 30000;
  CommunicatorConfig config_{}; // defaults unless provided
  std::shared_ptr<ResidencyProvider> residency_provider_ = nullptr;

  uint64_t channel_expire_;

  // GPU->CPU staging for TCP transport (legacy adapter) and unified GPU MemoryStager
  std::shared_ptr<GpuTcpStager> gpu_tcp_stager_;
  std::shared_ptr<MemoryStager> gpu_memory_stager_;

  // Shared pinned memory pool for GPU operations
  std::shared_ptr<store::PinnedMemoryPool> gpu_memory_pool_;
  // Dedicated pinned memory pool for CPU staging when configured with a
  // different chunk size than GPU. Falls back to gpu_memory_pool_ when null.
  std::shared_ptr<store::PinnedMemoryPool> cpu_memory_pool_;

  // Unified memory stager (CPU staging in TCP path)
  std::shared_ptr<MemoryStager> memory_stager_;
  std::unique_ptr<MrCache> mr_cache_;

  // --- RDMA staged response tracking (server-side) ---
  struct StagedRdmaSegment {
    void* ptr = nullptr;
    size_t bytes = 0;
    ibv_mr* mr = nullptr;
    enum class Kind { CPU, GPU } kind = Kind::CPU;
    uint64_t ts_us = 0;
    bool deregister_mr = true;
    // Remember which stager produced the buffer (CPU or GPU)
    MemoryStager* stager_ptr = nullptr;
  };
  // key: request key "<tensor_key>:<offset>"
  absl::Mutex staged_mu_;
  Map<std::string, StagedRdmaSegment> staged_segments_;

  // Serialize channel creation to avoid duplicate control connections to same peer
  mutable absl::Mutex create_channel_mu_;

  // --- Simple NUMA mapping (Phase 3) ---
  // Mapping from NIC name -> CPU MemoryStager (pool per NUMA node)
  std::unordered_map<std::string, std::shared_ptr<MemoryStager>> nic_cpu_stagers_;
  // Mapping from GPU id -> GpuTcpStager (pool per NUMA node)
  std::unordered_map<int, std::shared_ptr<GpuTcpStager>> gpu_stagers_;
  // Mapping from GPU id -> MemoryStager (GpuNetStager adapter per NUMA node)
  std::unordered_map<int, std::shared_ptr<MemoryStager>> gpu_mem_stagers_;
  // Keep pools alive and accessible for MR preregistration
  std::vector<std::shared_ptr<store::PinnedMemoryPool>> numa_pools_;

  // Helpers to select NUMA-aware stagers
  std::shared_ptr<MemoryStager> get_cpu_stager_for_nic(const std::string& nic_name) const;
  std::shared_ptr<GpuTcpStager> get_gpu_stager_for_id(int gpu_id) const;
  std::shared_ptr<MemoryStager> get_gpu_mem_stager_for_id(int gpu_id) const;

  // Track inflight direct-MR responses (CPU escape hatch)
  std::atomic<int> inflight_direct_mr_{0};
};

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_ENGINE_H_
