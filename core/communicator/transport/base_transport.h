// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_

#include "core/communicator/misc/common.h"

namespace tensorcast::communicator::transport {

misc::result_t send_bytes(int sock_fd, uint8_t* buf, int size);

misc::result_t recv_bytes(int sock_fd, uint8_t* buf, int size);

} // namespace tensorcast::communicator::transport

#endif // COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_
