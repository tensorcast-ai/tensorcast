
// Copyright (c) 2025, TensorCast Team.

extern "C" {
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
}

#include "core/communicator/misc/ibv_wrap.h"

namespace tensorcast::communicator {
#define IBVERBS_VERSION "IBVERBS_1.1"

#define CHECK_NULL(internal)                        \
  if (g_symbols.internal == NULL) {                 \
    LOG(WARNING) << "lib wrapper not initialized."; \
    return NULL_ERROR;                              \
  }

#define IBV_PTR_CHECK_ERRNO(internal, call, retval, error_retval, name)      \
  CHECK_NULL(internal);                                                      \
  retval = g_symbols.call;                                                   \
  if (retval == error_retval) {                                              \
    LOG(WARNING) << "Call to " name " failed with error" << strerror(errno); \
    return SYS_ERROR;                                                        \
  }                                                                          \
  return SUCCESS;

#define IBV_PTR_CHECK(internal_name, call, retval, error_retval, name) \
  CHECK_NULL(internal_name);                                           \
  retval = g_symbols.call;                                             \
  if (retval == error_retval) {                                        \
    LOG(WARNING) << "Call to " name " failed";                         \
    return SYS_ERROR;                                                  \
  }                                                                    \
  return SUCCESS;

#define IBV_INT_CHECK_RET_ERRNO(internal, call, success_retval, name)       \
  CHECK_NULL(internal);                                                     \
  int ret = g_symbols.call;                                                 \
  if (ret != success_retval) {                                              \
    LOG(WARNING) << "Call to " name " failed with error " << strerror(ret); \
    return SYS_ERROR;                                                       \
  }                                                                         \
  return SUCCESS;

#define IBV_INT_CHECK(internal_name, call, error_retval, name) \
  CHECK_NULL(internal_name);                                   \
  int ret = g_symbols.call;                                    \
  if (ret == error_retval) {                                   \
    LOG(WARNING) << "Call to " name " failed";                 \
    return SYS_ERROR;                                          \
  }                                                            \
  return SUCCESS;

static pthread_once_t g_init_once_control = PTHREAD_ONCE_INIT;
static result_t g_init_result;

/* IB Verbs Function Pointers*/
struct ibv_symbols_t {
  int (*fork_init)() = NULL;
  struct ibv_device** (*get_device_list)(int* num_devices) = NULL;
  void (*free_device_list)(struct ibv_device** list) = NULL;
  const char* (*get_device_name)(struct ibv_device* device) = NULL;
  struct ibv_context* (*open_device)(struct ibv_device* device) = NULL;
  int (*close_device)(struct ibv_context* context) = NULL;
  int (*get_async_event)(struct ibv_context* context, struct ibv_async_event* event) = NULL;
  void (*ack_async_event)(struct ibv_async_event* event) = NULL;
  int (*query_device)(struct ibv_context* context, struct ibv_device_attr* device_attr) = NULL;
  int (*query_port)(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr) = NULL;
  int (*query_gid)(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) = NULL;
  int (*query_qp)(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr) =
      NULL;
  struct ibv_pd* (*alloc_pd)(struct ibv_context* context) = NULL;
  int (*dealloc_pd)(struct ibv_pd* pd) = NULL;
  struct ibv_mr* (*reg_mr)(struct ibv_pd* pd, void* addr, size_t length, int access) = NULL;
  int (*dereg_mr)(struct ibv_mr* mr) = NULL;
  struct ibv_cq* (*create_cq)(
      struct ibv_context* context,
      int cqe,
      void* cq_context,
      struct ibv_comp_channel* channel,
      int comp_vector) = NULL;
  int (*destroy_cq)(struct ibv_cq* cq) = NULL;
  struct ibv_qp* (*create_qp)(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) = NULL;
  int (*modify_qp)(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) = NULL;
  int (*destroy_qp)(struct ibv_qp* qp) = NULL;
  const char* (*event_type_str)(enum ibv_event_type event) = NULL;
} g_symbols;

#define LOAD_SYM(symbol, func)                               \
  do {                                                       \
    void** cast = reinterpret_cast<void**>(&g_symbols.func); \
    void* tmp = dlvsym(ibvhandle, symbol, IBVERBS_VERSION);  \
    if (tmp == NULL) {                                       \
      goto teardown;                                         \
    }                                                        \
    *cast = tmp;                                             \
  } while (0)

result_t init_symbols() {
  static void* ibvhandle = nullptr;
  ibvhandle = dlopen("libibverbs.so", RTLD_NOW);
  if (!ibvhandle) {
    ibvhandle = dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibvhandle) {
      LOG(WARNING) << "failed to open libibverbs.so[.1]";
      goto teardown;
    }
  }

  LOAD_SYM("ibv_get_device_list", get_device_list);
  LOAD_SYM("ibv_free_device_list", free_device_list);
  LOAD_SYM("ibv_get_device_name", get_device_name);
  LOAD_SYM("ibv_open_device", open_device);
  LOAD_SYM("ibv_close_device", close_device);
  LOAD_SYM("ibv_get_async_event", get_async_event);
  LOAD_SYM("ibv_ack_async_event", ack_async_event);
  LOAD_SYM("ibv_query_device", query_device);
  LOAD_SYM("ibv_query_port", query_port);
  LOAD_SYM("ibv_query_gid", query_gid);
  LOAD_SYM("ibv_query_qp", query_qp);
  LOAD_SYM("ibv_alloc_pd", alloc_pd);
  LOAD_SYM("ibv_dealloc_pd", dealloc_pd);
  LOAD_SYM("ibv_reg_mr", reg_mr);
  LOAD_SYM("ibv_dereg_mr", dereg_mr);
  LOAD_SYM("ibv_create_cq", create_cq);
  LOAD_SYM("ibv_destroy_cq", destroy_cq);
  LOAD_SYM("ibv_create_qp", create_qp);
  LOAD_SYM("ibv_modify_qp", modify_qp);
  LOAD_SYM("ibv_destroy_qp", destroy_qp);
  LOAD_SYM("ibv_fork_init", fork_init);
  return SUCCESS;

teardown:
  if (ibvhandle != nullptr) {
    dlclose(ibvhandle);
  }
  return SYS_ERROR;
}

result_t wrap_ibv_symbols() {
  pthread_once(&g_init_once_control, []() { g_init_result = init_symbols(); });
  return g_init_result;
}

result_t wrap_ibv_fork_init() {
  IBV_INT_CHECK(fork_init, fork_init(), -1, "ibv_fork_init");
}

result_t wrap_ibv_get_device_list(struct ibv_device*** ret, int* num_devices) {
  *ret = g_symbols.get_device_list(num_devices);
  if (*ret == NULL) {
    *num_devices = 0;
  }
  return SUCCESS;
}

result_t wrap_ibv_free_device_list(struct ibv_device** list) {
  CHECK_NULL(free_device_list);
  g_symbols.free_device_list(list);
  return SUCCESS;
}

const char* wrap_ibv_get_device_name(struct ibv_device* device) {
  if (g_symbols.get_device_name == NULL) {
    LOG(FATAL) << "lib wrapper not initialized.";
    exit(-1);
  }
  return g_symbols.get_device_name(device);
}

result_t wrap_ibv_open_device(struct ibv_context** ret, struct ibv_device* device) {
  /*returns 0 on success, -1 on failure*/
  IBV_PTR_CHECK(open_device, open_device(device), *ret, nullptr, "ibv_open_device");
}

result_t wrap_ibv_close_device(struct ibv_context* context) {
  /*returns 0 on success, -1 on failure*/
  IBV_INT_CHECK(close_device, close_device(context), -1, "ibv_close_device");
}

result_t wrap_ibv_get_async_event(struct ibv_context* context, struct ibv_async_event* event) {
  /*returns 0 on success, and -1 on error*/
  IBV_INT_CHECK(get_async_event, get_async_event(context, event), -1, "ibv_get_async_event");
}

result_t wrap_ibv_ack_async_event(struct ibv_async_event* event) {
  CHECK_NULL(ack_async_event);
  g_symbols.ack_async_event(event);
  return SUCCESS;
}

result_t wrap_ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
  /*returns 0 on success, or the value of errno on failure
   * (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(query_device, query_device(context, device_attr), 0, "ibv_query_device");
}

result_t wrap_ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr) {
  /*returns 0 on success, or the value of errno on failure
   * (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(query_port, query_port(context, port_num, port_attr), 0, "ibv_query_port");
}

result_t wrap_ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
  IBV_INT_CHECK_RET_ERRNO(query_gid, query_gid(context, port_num, index, gid), 0, "ibv_query_gid");
}

result_t wrap_ibv_query_qp(
    struct ibv_qp* qp,
    struct ibv_qp_attr* attr,
    int attr_mask,
    struct ibv_qp_init_attr* init_attr) {
  IBV_INT_CHECK_RET_ERRNO(query_qp, query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

result_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context* context) {
  IBV_PTR_CHECK_ERRNO(alloc_pd, alloc_pd(context), *ret, nullptr, "ibv_alloc_pd");
}

result_t wrap_ibv_dealloc_pd(struct ibv_pd* pd) {
  /*returns 0 on success, or the value of errno on
   * failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(dealloc_pd, dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

result_t wrap_ibv_reg_mr(struct ibv_mr** ret, struct ibv_pd* pd, void* addr, size_t length, int access) {
  IBV_PTR_CHECK_ERRNO(reg_mr, reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr* wrap_direct_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  if (g_symbols.reg_mr == nullptr) {
    LOG(WARNING) << "lib wrapper not initialized.";
    return nullptr;
  }
  return g_symbols.reg_mr(pd, addr, length, access);
}

result_t wrap_ibv_dereg_mr(struct ibv_mr* mr) {
  /*returns 0 on success, or the value of errno
   * on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(dereg_mr, dereg_mr(mr), 0, "ibv_dereg_mr");
}

result_t wrap_ibv_create_cq(
    struct ibv_cq** ret,
    struct ibv_context* context,
    int cqe,
    void* cq_context,
    struct ibv_comp_channel* channel,
    int comp_vector) {
  IBV_PTR_CHECK_ERRNO(
      create_cq, create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

result_t wrap_ibv_destroy_cq(struct ibv_cq* cq) {
  IBV_INT_CHECK_RET_ERRNO(destroy_cq, destroy_cq(cq), 0, "ibv_destroy_cq");
}

result_t wrap_ibv_destroy_qp(struct ibv_qp* qp) {
  IBV_INT_CHECK_RET_ERRNO(destroy_qp, destroy_qp(qp), 0, "ibv_destroy_qp");
}

result_t wrap_ibv_create_qp(struct ibv_qp** ret, struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
  IBV_PTR_CHECK_ERRNO(create_qp, create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp");
}

result_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  /*returns 0 on success, or the value of errno on
   * failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(modify_qp, modify_qp(qp, attr, attr_mask), 0, "ibv_modify_qp");
}

} // namespace tensorcast::communicator
