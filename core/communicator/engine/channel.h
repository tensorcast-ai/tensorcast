// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_CHANNEL_H_
#define CORE_COMMUNICATOR_ENGINE_CHANNEL_H_

#include <memory>
#include <string>

#include "core/communicator/transport/mtcp_transport.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/communicator/transport/tcp_transport.h"

namespace tensorcast::communicator::engine {

class Channel {
 public:
  explicit Channel(communicator::transport::tcp_transport_t control, int type);
  ~Channel();

  transport::tcp_transport_t get_control();
  transport::mtcp_transport_t get_mtcp();
  transport::rdma_transport_t get_rdma(const std::string& local_dev_name, const std::string& remote_dev_name);

  void set_channel_type(int type);

  void set_transport(
      const std::string& local_dev_name,
      const std::string& remote_dev_name,
      transport::rdma_transport_t t);
  void set_transport(communicator::transport::mtcp_transport_t t);
  void del_transport(const std::string& local_dev_name, const std::string& remote_dev_name);

  misc::result_t close();

  void record_expire(uint64_t now);

  bool is_expired(uint64_t now);

 private:
  int type_;
  transport::tcp_transport_t control_;
  misc::Map<std::string, transport::rdma_transport_t> rdma_;
  transport::mtcp_transport_t mtcp_;
  uint64_t expired_time_;
};

using channel_t = std::shared_ptr<Channel>;

} // namespace tensorcast::communicator::engine

#endif // COMMUNICATOR_ENGINE_CHANNEL_H_
