// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_BASE_CONSTANTS_H_
#define CORE_COMMUNICATOR_BASE_CONSTANTS_H_

namespace stepcast::communicator {

// Device type constants
enum {
  COMMUNICATE_ENGINE_DEV_CPU = 0,
  COMMUNICATE_ENGINE_DEV_GPU = 1,
};

// Channel type constants
enum {
  CHANNEL_RDMA = 0,
  CHANNEL_MTCP = 1,
};

// TCP connection constants
constexpr int kMTcpConnCount = 8;
constexpr int kMaxTcpConns = 32;
constexpr int kMaxFd = 32;

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_BASE_CONSTANTS_H_