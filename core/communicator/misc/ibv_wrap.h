
// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_MISC_IBV_WRAP_H_
#define CORE_COMMUNICATOR_MISC_IBV_WRAP_H_

extern "C" {
#include <stdint.h>
#include <unistd.h>

#ifdef USE_FAKE_CUDA
#include "core/communicator/misc/ibv_mock.h" // NOLINT
}
#else
}
#include <infiniband/verbs.h>
#endif

#include "core/communicator/misc/common.h"

namespace tensorcast::communicator::misc {

enum {
  IBV_SUCCESS = 0, //!< The operation was successful
};

result_t wrap_ibv_symbols();
result_t wrap_ibv_fork_init();
result_t wrap_ibv_get_device_list(struct ibv_device*** ret, int* num_devices);
result_t wrap_ibv_free_device_list(struct ibv_device** list);
const char* wrap_ibv_get_device_name(struct ibv_device* device);
result_t wrap_ibv_open_device(struct ibv_context** ret, struct ibv_device* device);
result_t wrap_ibv_close_device(struct ibv_context* context);
result_t wrap_ibv_get_async_event(struct ibv_context* context, struct ibv_async_event* event);
result_t wrap_ibv_ack_async_event(struct ibv_async_event* event);
result_t wrap_ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr);
result_t wrap_ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr);
result_t wrap_ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid);
result_t wrap_ibv_query_qp(
    struct ibv_qp* qp,
    struct ibv_qp_attr* attr,
    int attr_mask,
    struct ibv_qp_init_attr* init_attr);
result_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context* context);
result_t wrap_ibv_dealloc_pd(struct ibv_pd* pd);
result_t wrap_ibv_reg_mr(struct ibv_mr** ret, struct ibv_pd* pd, void* addr, size_t length, int access);
struct ibv_mr* wrap_direct_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access);
struct ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(
    struct ibv_pd* pd,
    uint64_t offset,
    size_t length,
    uint64_t iova,
    int fd,
    int access);
result_t wrap_ibv_dereg_mr(struct ibv_mr* mr);
result_t wrap_ibv_create_cq(
    struct ibv_cq** ret,
    struct ibv_context* context,
    int cqe,
    void* cq_context,
    struct ibv_comp_channel* channel,
    int comp_vector);
result_t wrap_ibv_destroy_cq(struct ibv_cq* cq);
result_t wrap_ibv_create_qp(struct ibv_qp** ret, struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr);
result_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask);
result_t wrap_ibv_destroy_qp(struct ibv_qp* qp);

static inline result_t wrap_ibv_post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) {
  int ret = qp->context->ops.post_send(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

static inline result_t wrap_ibv_poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc, int* num_done) {
  int done = cq->context->ops.poll_cq(cq, num_entries, wc);
  /*returns the number of wcs or 0 on success, a negative number otherwise*/
  if (done < 0) {
    return INTERNAL_ERROR;
  }
  *num_done = done;
  return SUCCESS;
}

static inline result_t wrap_ibv_post_recv(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) {
  int ret = qp->context->ops.post_recv(qp, wr, bad_wr);
  if (ret != IBV_SUCCESS) {
    return SYS_ERROR;
  }
  return SUCCESS;
}

} // namespace tensorcast::communicator::misc

#endif // STEPUCX_SRC_MISC_IBV_WRAP_H_
