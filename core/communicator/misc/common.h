// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_MISC_COMMON_H_
#define CORE_COMMUNICATOR_MISC_COMMON_H_

extern "C" {
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
}

#include "absl/log/log.h"

#include <future>

namespace stepcast::communicator {

enum {
  SUCCESS = 0,
  FAILED = -1,
  SYS_ERROR = -2,
  NULL_ERROR = -3,
  INTERNAL_ERROR = -4,
  INVALID_ARGUMENT = -5,
  REMOTE_ERROR = -6,
  CUDA_ERROR = -7,
  REMOTE_NO_TENSOR = -8,
  REMOTE_RDMA_READ_FAILED = -9,
  REMOTE_RDMA_CONNECT_FAILED = -10,
  READ_REQUEST_FAILED = -11,
  TRANSPORT_FAILED = -12,
  IN_PROGRESS = 1,
};

using result_t = int;
using future_result_t = std::future<result_t>;

constexpr size_t KB = (1ull << 10);
constexpr size_t MB = (1ull << 20);
constexpr size_t GB = (1ull << 30);

#define COMM_CHECK(call)                                                         \
  do {                                                                           \
    result_t res = call;                                                         \
    if (res != SUCCESS) {                                                        \
      LOG(WARNING) << __FILE__ << ":" << __LINE__ << " " << res << " " << errno; \
      return res;                                                                \
    }                                                                            \
  } while (0)

#define CHECK_WARN(call, msg)                                                  \
  do {                                                                         \
    result_t res = call;                                                       \
    if (res != SUCCESS) {                                                      \
      LOG(WARNING) << __FILE__ << ":" << __LINE__ << " " << res << " " << msg; \
    }                                                                          \
  } while (0)

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_MISC_COMMON_H_
