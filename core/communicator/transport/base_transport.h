// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_
#define CORE_COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_

#include "core/communicator/misc/common.h"

namespace stepcast::communicator {

result_t send_bytes(int sock_fd, uint8_t* buf, int size);

result_t recv_bytes(int sock_fd, uint8_t* buf, int size);

} // namespace stepcast::communicator

#endif // COMMUNICATOR_TRANSPORT_BASE_TRANSPORT_H_
