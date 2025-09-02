// Copyright (c) 2025, TensorCast Team.

#ifndef COMMUNICATOR_TRANSPORT_TRANSPORT_MESSAGE_H_
#define COMMUNICATOR_TRANSPORT_TRANSPORT_MESSAGE_H_

#include <cstdint>
#include <memory>

#include "core/communicator/misc/utils.h"

namespace tensorcast::communicator::transport {

class TransportMessage {
 public:
  TransportMessage(uint32_t header_size, uint32_t payload_size);
  virtual ~TransportMessage();

  uint32_t get_header_size() const;
  uint32_t get_payload_size() const;

  template <class T>
  T* get_header() {
    return reinterpret_cast<T*>(header_buf_);
  }

  template <class T>
  T* get_payload() {
    misc::ASSERT(sizeof(T) <= payload_size_, "failed to get payload due to illegal size");
    return reinterpret_cast<T*>(payload_buf_);
  }

 protected:
  uint8_t header_buf_[1024];
  uint32_t header_size_;
  uint8_t* payload_buf_;
  uint32_t payload_size_;
};
typedef std::shared_ptr<TransportMessage> transport_message_t;

} // namespace tensorcast::communicator::transport

#endif // COMMUNICATOR_TRANSPORT_TRANSPORT_MESSAGE_H_
