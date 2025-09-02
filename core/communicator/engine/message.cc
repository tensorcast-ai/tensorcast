
// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/engine/message.h"
#include "core/communicator/engine/protocol.h"

namespace tensorcast::communicator::engine {

EngineMessage::EngineMessage(ProtoHeader* header) : transport::TransportMessage(PROTO_HEADER_SIZE, header->size) {
  memcpy(header_buf_, header, PROTO_HEADER_SIZE);
}

EngineMessage::EngineMessage(uint32_t op, uint32_t payload_size)
    : transport::TransportMessage(PROTO_HEADER_SIZE, payload_size) {
  auto* header = get_header<ProtoHeader>();
  header->prefix = kHeaderPrefix;
  header->size = payload_size;
  header->version = 0;
  header->op = op;
}

uint32_t EngineMessage::get_op() {
  auto* header = get_header<ProtoHeader>();
  return header->op;
}

} // namespace tensorcast::communicator::engine
