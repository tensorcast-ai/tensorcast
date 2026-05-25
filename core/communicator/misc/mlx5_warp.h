// Copyright (c) 2026, TensorCast Team.

#include <dlfcn.h>
#include <errno.h>
extern "C" {
#ifdef USE_MOCK_IBV
#include "core/communicator/misc/ibv_mock.h" // NOLINT
}
#else
}
#include <infiniband/verbs.h>
#endif

#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#ifndef MLX5_WARP_H_
#define MLX5_WARP_H_

namespace tensorcast::communicator::misc {

//  static pthread_once_t initOnceControl;
//  int initResult;

int wrap_mlx5dv_query_qp_lag_port(struct ibv_qp* qp, uint8_t* port_num, uint8_t* active_port_num);

int wrap_mlx5dv_modify_qp_lag_port(struct ibv_qp* qp, uint8_t port_num);

int wrap_mlx5dv_symbols(void);

} // namespace tensorcast::communicator::misc

#endif // IBVWARP_H_
