// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_ENGINE_CHANNEL_H_
#define CORE_COMMUNICATOR_ENGINE_CHANNEL_H_

#include <memory>
#include <string>

#include "core/communicator/transport/mtcp_transport.h"
#include "core/communicator/transport/rdma_transport.h"
#include "core/communicator/transport/tcp_transport.h"

namespace stepcast::communicator {

enum {
  CHANNEL_RDMA = 0,
  CHANNEL_MTCP = 1,
};

class Channel {
 public:
  explicit Channel(tcp_transport_t control, int type);
  ~Channel();

  tcp_transport_t get_control();
  mtcp_transport_t get_mtcp();
  rdma_transport_t get_rdma(const std::string& local_dev_name, const std::string& remote_dev_name);

  void set_channel_type(int type);

  void set_transport(const std::string& local_dev_name, const std::string& remote_dev_name, rdma_transport_t t);
  void set_transport(mtcp_transport_t t);
  void del_transport(const std::string& local_dev_name, const std::string& remote_dev_name);

  result_t close();

  void record_expire(uint64_t now);

  bool is_expired(uint64_t now);

 private:
  int type_;
  tcp_transport_t control_;
  Map<std::string, rdma_transport_t> rdma_;
  mtcp_transport_t mtcp_;
  uint64_t expired_time_;
};

typedef std::shared_ptr<Channel> channel_t;

} // namespace stepcast::communicator

#endif // COMMUNICATOR_ENGINE_CHANNEL_H_
