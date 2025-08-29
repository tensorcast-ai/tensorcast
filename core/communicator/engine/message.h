// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_MESSAGE_H_
#define CORE_COMMUNICATOR_ENGINE_MESSAGE_H_

#include <cstdint>
#include <memory>

#include "core/communicator/engine/protocol.h"
#include "core/communicator/transport/transport_message.h"

namespace tensorcast::communicator {

class EngineMessage;
typedef std::shared_ptr<EngineMessage> engine_message_t;

class EngineMessage : public TransportMessage {
 public:
  explicit EngineMessage(ProtoHeader* header);
  EngineMessage(uint32_t op, uint32_t payload_size);

  ~EngineMessage() = default;

  uint32_t get_op();

 public:
  template <class T>
  static engine_message_t make_message(uint32_t op) {
    return std::make_shared<EngineMessage>(op, sizeof(T));
  }
};

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_MESSAGE_H_
