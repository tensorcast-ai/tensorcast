// Copyright (c) 2025, TensorCast Team.

#include "core/communicator/transport/transport_message.h"
#include "core/communicator/misc/utils.h"

namespace tensorcast::communicator {

TransportMessage::TransportMessage(uint32_t header_size, uint32_t payload_size)
    : header_size_(header_size), payload_size_(payload_size) {
  ASSERT(payload_size_ > 0, "illegal payload size");
  ASSERT(header_size_ > 0, "illegal header size");
  ALLOC(&payload_buf_, payload_size_);
  bzero(payload_buf_, payload_size_);
}

TransportMessage::~TransportMessage() {
  FREE_PTR(payload_buf_);
}

uint32_t TransportMessage::get_header_size() const {
  return header_size_;
}

uint32_t TransportMessage::get_payload_size() const {
  return payload_size_;
}

} // namespace tensorcast::communicator
