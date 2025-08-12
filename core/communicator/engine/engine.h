// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_ENGINE_ENGINE_H_
#define CORE_COMMUNICATOR_ENGINE_ENGINE_H_

#include <string>

#include "absl/status/status.h"

#include "core/communicator/base/constants.h"
#include "core/communicator/misc/queue.h"
#include "core/communicator/transport/rdma_context.h"
#include "core/communicator/transport/request.h"
#include "core/communicator/transport/tcp_context.h"

#include "core/common/memory/pinned_memory_pool.h"
#include "core/communicator/engine/channel.h"
#include "core/communicator/engine/gpu_tcp_stager.h"
#include "core/communicator/engine/message.h"
#include "core/communicator/engine/store.h"

namespace stepcast::communicator {

class CommunicateEngine {
 public:
  explicit CommunicateEngine(bool enable_rdma = false, uint32_t channel_expire_sec = 0);

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

  /**
   * @brief Whether this engine instance was configured with RDMA support.
   * When false the engine falls back to pure TCP transports and only supports
   * CPU memory registration / transfer.
   */
  inline bool is_rdma_enabled() const {
    return enable_rdma_;
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

  uint64_t channel_expire_;

  // GPU->CPU staging for TCP transport
  std::shared_ptr<GpuTcpStager> gpu_tcp_stager_;

  // Shared pinned memory pool for GPU operations
  std::shared_ptr<store::PinnedMemoryPool> gpu_memory_pool_;

  // Serialize channel creation to avoid duplicate control connections to same peer
  mutable absl::Mutex create_channel_mu_;
};

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_ENGINE_H_
