// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_PROTOCOL_H_
#define CORE_COMMUNICATOR_ENGINE_PROTOCOL_H_

#include <cstdint>

#include "core/communicator/misc/ibv_wrap.h"
#include "core/communicator/transport/rdma_transport.h"

namespace tensorcast::communicator {

constexpr static uint32_t kMaxDevName = 32;
constexpr static uint32_t kMaxTensorNameLen = 512;
constexpr static uint32_t kHeaderPrefix = 0xAABBAABB;

enum {
  ENGINE_OP_INVALID = 0,
  ENGINE_OP_READ_REQUEST,
  ENGINE_OP_READ_RESPONSE,
  ENGINE_OP_READ_RESPONSE_EX,
  ENGINE_OP_READ_FAILED,
  ENGINE_OP_RDMA_CONNECT_REQUEST,
  ENGINE_OP_RDMA_CONNECT_RESPONSE,
  ENGINE_OP_RDMA_CONNECT_FAILED,
  ENGINE_OP_MTCP_CONNECT_REQUEST,
  ENGINE_OP_MTCP_CONNECT_RESPONSE,
  ENGINE_OP_MTCP_CONNECT_FAILED,
  ENGINE_OP_RDMA_READ_DONE,
  ENGINE_OP_RDMA_READ_DONE_EX,
  ENGINE_OP_CLOSE,
};

enum {
  ENGINE_TRANSPORT_MTCP = 0,
  ENGINE_TRANSPORT_RDMA = 1,
};

enum {
  TENSORCAST_READ_FAILED_NO_TENSOR = 1,
  TENSORCAST_READ_FAILED_OVERFLOW = 2,
  TENSORCAST_READ_FAILED_MEM_MISMATCH = 3,
};

struct ProtoHeader {
  uint32_t prefix = kHeaderPrefix;
  uint32_t version = 0;
  uint32_t op = ENGINE_OP_INVALID;
  uint32_t size = 0;
};
#define PROTO_HEADER_SIZE sizeof(ProtoHeader)

struct ProtoRdmaConnectRequest {
  RdmaTransportInfo qp_info = {};
  char dst_dev_name[kMaxDevName] = {};
  char src_dev_name[kMaxDevName] = {};
};

struct ProtoRdmaConnectResponse {
  RdmaTransportInfo qp_info = {};
  char dst_dev_name[kMaxDevName] = {};
  char src_dev_name[kMaxDevName] = {};
};

struct ProtoRdmaConnectFailed {
  char dst_dev_name[kMaxDevName] = {};
  char src_dev_name[kMaxDevName] = {};
};

struct ProtoMtcpConnectRequest {
  int conn_count = 1;
};

struct ProtoMtcpConnectResponse {
  int conn_count = 1;
  uint32_t ip;
  uint32_t port;
};

struct ProtoMtcpConnectFailed {
  uint32_t ip;
};

struct ProtoReadRequest {
  char tensor_key[kMaxTensorNameLen];
  uint8_t transport_type;
  uint64_t offset;
  uint64_t bytes;
};

struct ProtoReadResponse {
  char tensor_key[kMaxTensorNameLen];
  uint8_t transport_type;
  uint8_t staged; // 1 if response advertises a staged MR on server side
  char nic_name[kMaxDevName];
  uint64_t addr;
  uint64_t offset;
  uint32_t bytes;
  uint32_t rkey;
};

// Variable-length response supporting multiple segments.
// Layout: [ProtoReadResponseExHeader][ProtoReadResponseExSeg[num_segments]]
struct ProtoReadResponseExHeader {
  char tensor_key[kMaxTensorNameLen];
  uint8_t transport_type;
  uint8_t staged; // 1 if segments are staged on server
  char nic_name[kMaxDevName];
  uint32_t num_segments;
};

struct ProtoReadResponseExSeg {
  uint64_t addr;
  uint64_t offset;
  uint32_t bytes;
  uint32_t rkey;
};

struct ProtoReadFailed {
  char tensor_key[kMaxTensorNameLen];
  uint64_t offset;
  uint32_t reason;
};

struct ProtoRdmaReadDone {
  char tensor_key[kMaxTensorNameLen];
  uint64_t offset;
};

// Variable-length batched ACK.
// Layout: [ProtoRdmaReadDoneExHeader][ProtoRdmaReadDoneExSeg[num_segments]]
struct ProtoRdmaReadDoneExHeader {
  char tensor_key[kMaxTensorNameLen];
  uint32_t num_segments;
};
struct ProtoRdmaReadDoneExSeg {
  uint64_t offset;
};

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_PROTOCOL_H_
