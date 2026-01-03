// Copyright (c) 2025-2026, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_ENGINE_H_
#define CORE_COMMUNICATOR_ENGINE_ENGINE_H_

#include <atomic>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"

#include "core/communicator/base/constants.h"
#include "core/communicator/misc/queue.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/request.h"
#include "core/communicator/transport/tcp_context.h"

#include "core/common/memory/pinned_buffer_pool.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/host_pinned_cpu_stager.h"
#include "core/communicator/engine/memory_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/mr_cache.h"
#include "core/communicator/engine/store.h"
#include "tensorcast/communicator/v1/communicator_config.pb.h"

namespace tensorcast::communicator::engine {

class CommunicatorTestPeer;
class GpuVramStagingPool;

class Communicator {
 public:
  struct PinnedStagingPools {
    // Host-pinned pool backing GPU-side staging (MTCP and staged RDMA).
    std::shared_ptr<common::memory::PinnedBufferPool> gpu_pool;
    // Optional host-pinned pool for CPU staging. When null, cpu staging uses gpu_pool.
    std::shared_ptr<common::memory::PinnedBufferPool> cpu_pool;
    bool preregister_gpu = false;
    bool preregister_cpu = false;
    // Upper bound for staging-buffer wait loops (e.g., MTCP staged transfers).
    // Daemon wiring should set this from pinned_memory.allocation_timeout.
    absl::Duration staging_wait_timeout = absl::Seconds(30);
  };

  // Convenience constructor for non-daemon use (tests and standalone tools).
  // Uses conservative default slice sizes and derives pool sizes from config
  // fan-out settings.
  explicit Communicator(const v1::CommunicatorConfig& config, uint32_t channel_expire_sec = 0);

  Communicator(const v1::CommunicatorConfig& config, PinnedStagingPools pools, uint32_t channel_expire_sec = 0);

  ~Communicator();

  /**
   * Init the communicate engine with the ip and port
   * @param local_ip local ip
   * @param local_port local port
   * @return  init result
   */
  absl::Status init(const std::string& local_ip, uint16_t local_port, int conn_count = -1);

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
  transport::future_read_result_t read_tensor(
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
  struct RegisterTensorOptions {
    bool register_mr = true; // Skip MR registration when false
    bool needs_staging = false; // Hint for transports that staging is required
    bool async = false; // Async MR registration when applicable
    bool direct_rdma_enabled = false; // Allow zero-copy RDMA when preconditions hold
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
  void set_dram_lease_provider(const std::shared_ptr<HostPinnedCpuStager::LeaseProvider>& provider);

  // Lightweight UMA residency provider (reserved)
  struct ResidencyProvider {
    virtual ~ResidencyProvider() = default;
    virtual bool is_hot(const std::string& tensor_key, uint64_t offset, uint64_t bytes) = 0;
  };

  void set_residency_provider(std::shared_ptr<ResidencyProvider> provider) {
    residency_provider_ = std::move(provider);
  }

  /**
   * @brief Whether this engine instance was configured with RDMA support.
   * When false the engine falls back to pure TCP transports and only supports
   * CPU memory registration / transfer.
   */
  bool is_rdma_enabled() const {
    return enable_rdma_;
  }

  uint16_t listening_port() const;

 private:
  friend class CommunicatorTestPeer;

  transport::net_dev_t get_net_dev(int dev_type, int dev_id, const std::string& key = "", int rail_id = -1);

  misc::result_t on_receive_request(
      const channel_t& channel,
      const transport::tcp_transport_t& t,
      const engine_message_t& msg);
  misc::result_t on_receive_response(
      const channel_t& channel,
      const transport::tcp_transport_t& t,
      const engine_message_t& msg);

  void do_read_request_loop();
  void do_channel_gc_loop();

  misc::result_t on_new_client(const transport::tcp_transport_t& t);

  absl::StatusOr<channel_t> do_create_channel(const std::string& ip, uint16_t port);

  absl::Status handle_rdma_read_request(
      const channel_t& channel,
      const transport::tcp_transport_t& control_transport,
      ProtoReadRequest& request,
      const std::shared_ptr<transport::PartitionTensor>& tensor);
  absl::Status handle_mtcp_read_request(
      const channel_t& channel,
      const transport::tcp_transport_t& control_transport,
      const ProtoReadRequest& request,
      const std::shared_ptr<transport::PartitionTensor>& tensor);
  static absl::Status resume_rdma_reads(const channel_t& channel);
  void schedule_handshake_retry(
      const channel_t& channel,
      const std::string& local_dev_name,
      const std::string& peer_dev_name,
      absl::Duration delay);
  void handshake_retry_loop();
  void process_handshake_retry_task(
      const std::weak_ptr<Channel>& channel_weak,
      const std::string& local_dev_name,
      const std::string& peer_dev_name);
  void start_pending_rdma_handshake(
      const channel_t& channel,
      const std::shared_ptr<Channel::RdmaEndpoint>& endpoint,
      const std::string& local_dev_name,
      const std::string& peer_dev_name);

  struct MtcpReadTask {
    channel_t channel;
    transport::tcp_transport_t control_transport;
    ProtoReadRequest request;
    std::shared_ptr<transport::PartitionTensor> tensor;
  };

  void mtcp_staging_loop();
  void process_mtcp_read_task(MtcpReadTask task);
  static void fail_mtcp_read_task(const MtcpReadTask& task, absl::Status status);

  struct GpuChannelLease;
  absl::StatusOr<std::shared_ptr<void>> acquire_gpu_channel_slot();
  void release_gpu_channel_slot();

  struct HandshakeRetryTask {
    absl::Time resume_at;
    std::weak_ptr<Channel> channel;
    std::string local_dev_name;
    std::string peer_dev_name;
  };

  struct HandshakeRetryCompare {
    bool operator()(const HandshakeRetryTask& lhs, const HandshakeRetryTask& rhs) const {
      return lhs.resume_at > rhs.resume_at;
    }
  };

  std::atomic_bool stop_;
  std::atomic_bool inited_;
  transport::tcp_context_t server_context_;
  transport::tcp_context_t client_context_;
  transport::rdma_context_t rdma_context_ = nullptr;
  misc::Queue<transport::read_request_t> request_queue_;
  misc::Map<std::string, transport::read_request_t> pending_requests_;
  misc::Map<std::string, channel_t> channels_;
  PartitionTensorStore store_;
  std::thread request_thread_;
  std::thread gc_thread_;
  bool enable_rdma_;
  int mtcp_conn_count_;
  uint32_t ack_ttl_ms_ = 30000;
  v1::CommunicatorConfig config_{}; // defaults unless provided
  std::shared_ptr<ResidencyProvider> residency_provider_ = nullptr;

  uint64_t channel_expire_;
  int buffers_per_flow_ = 4;
  uint32_t max_window_segments_ = 0;
  uint64_t direct_rdma_chunk_bytes_ = 0;

  // Host-pinned GPU staging uses unified GPU MemoryStager only.
  std::shared_ptr<engine::MemoryStager> gpu_memory_stager_;
  // GPU VRAM staged-RDMA backend (when enabled).
  std::unordered_map<int, std::shared_ptr<MemoryStager>> gpu_vram_stagers_;
  std::unordered_map<int, std::shared_ptr<GpuVramStagingPool>> gpu_vram_pools_;
  bool use_gpu_vram_staging_ = false;

  // Shared pinned memory pool for GPU operations
  std::shared_ptr<common::memory::PinnedBufferPool> gpu_memory_pool_;
  // Dedicated pinned memory pool for CPU staging when configured with a
  // different chunk size than GPU. Falls back to gpu_memory_pool_ when null.
  std::shared_ptr<common::memory::PinnedBufferPool> cpu_memory_pool_;
  bool preregister_gpu_pool_ = false;
  bool preregister_cpu_pool_ = false;
  absl::Duration staging_wait_timeout_ = absl::Seconds(30);

  // Unified memory stager (CPU staging in TCP path)
  std::shared_ptr<engine::MemoryStager> memory_stager_;

  // MR cache for meta data on CPU
  std::unique_ptr<MrCache> meta_mr_cache_;

  // Serialize channel creation to avoid duplicate control connections to same peer
  mutable absl::Mutex create_channel_mu_;

  // --- Simple NUMA mapping (Phase 3) ---
  // Mapping from NIC name -> CPU MemoryStager (pool per NUMA node)
  std::unordered_map<std::string, std::shared_ptr<MemoryStager>> nic_cpu_stagers_;
  // Mapping from GPU id -> MemoryStager (host-pinned GPU staging per NUMA node)
  std::unordered_map<int, std::shared_ptr<MemoryStager>> gpu_mem_stagers_;

  // Helpers to select NUMA-aware stagers
  std::shared_ptr<engine::MemoryStager> get_cpu_stager_for_nic(const std::string& nic_name) const;
  std::shared_ptr<engine::MemoryStager> get_gpu_mem_stager_for_id(int gpu_id) const;
  std::shared_ptr<engine::MemoryStager> get_gpu_vram_stager_for_id(int gpu_id) const;

  absl::Mutex handshake_retry_mu_;
  absl::CondVar handshake_retry_cv_;
  std::priority_queue<HandshakeRetryTask, std::vector<HandshakeRetryTask>, HandshakeRetryCompare> handshake_retry_queue_
      ABSL_GUARDED_BY(handshake_retry_mu_);
  std::thread handshake_retry_thread_;
  std::atomic_bool handshake_retry_stop_{false};
  bool handshake_retry_thread_started_ = false;

  misc::Queue<MtcpReadTask> mtcp_staging_queue_;
  std::thread mtcp_staging_thread_;
  std::atomic<int> active_gpu_channels_{0};
  int max_gpu_channels_ = 0;
  bool enforce_gpu_channel_limit_ = false;
};

} // namespace tensorcast::communicator::engine

#endif // CORE_COMMUNICATOR_ENGINE_ENGINE_H_
