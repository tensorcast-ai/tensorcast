// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "core/common/memory/memory_location.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/runtime/ingestion_events.h"

namespace tensorcast::store::runtime {

enum class RuntimeEventType {
  kReplicaLoaded,
  kReplicaEvicted,
  kRemoteAccessToggled,
  kIngestionStarted,
  kIngestionCompleted,
  kIngestionFailed,
  kRegistrationCommitted,
  kRegistrationAborted,
  kKeyMappingChanged,
};

struct ReplicaLifecycleEvent {
  loading::ReplicaKey key;
  size_t size_bytes{0};
};

struct RegistrationEvent {
  std::string registration_id;
  std::string artifact_id;
  DeviceKey device;
  uint64_t size_bytes{0};
  std::optional<std::string> view_id;
  bool existed{false};
  bool committed{false};
  absl::Status status;
};

struct KeyMappingEvent {
  std::string key;
  std::string artifact_id;
};

struct RemoteAccessEvent {
  loading::ReplicaKey key;
  common::memory::MemoryLocation location;
  bool enabled{false};
};

using RuntimeEventPayload = std::variant<
    std::monostate,
    ReplicaLifecycleEvent,
    RegistrationEvent,
    IngestionResultEvent,
    KeyMappingEvent,
    RemoteAccessEvent>;

struct RuntimeEvent {
  RuntimeEventType type{RuntimeEventType::kIngestionCompleted};
  RuntimeEventPayload payload;
};

class RuntimeContextEvents {
 public:
  class Subscription;
  class Publisher;

  using Callback = absl::AnyInvocable<void(const RuntimeEvent&)>;

  RuntimeContextEvents();
  ~RuntimeContextEvents();

  RuntimeContextEvents(const RuntimeContextEvents&) = delete;
  RuntimeContextEvents& operator=(const RuntimeContextEvents&) = delete;

  Publisher publisher();
  std::unique_ptr<Subscription> subscribe(Callback callback);
  void drain();

 private:
  struct DispatcherState;
  std::shared_ptr<DispatcherState> state_;
};

class RuntimeContextEvents::Subscription {
 public:
  Subscription();
  Subscription(std::shared_ptr<DispatcherState> state, int64_t id);
  ~Subscription();

  Subscription(const Subscription&) = delete;
  Subscription& operator=(const Subscription&) = delete;

  Subscription(Subscription&& other) noexcept;
  Subscription& operator=(Subscription&& other) noexcept;

  void release();

 private:
  std::shared_ptr<DispatcherState> state_;
  int64_t id_{0};
};

class RuntimeContextEvents::Publisher {
 public:
  Publisher();
  explicit Publisher(std::shared_ptr<DispatcherState> state);

  Publisher(const Publisher&) = default;
  Publisher& operator=(const Publisher&) = default;
  Publisher(Publisher&&) noexcept = default;
  Publisher& operator=(Publisher&&) noexcept = default;

  void publish(RuntimeEvent event) const;

  explicit operator bool() const {
    return static_cast<bool>(state_);
  }

 private:
  std::shared_ptr<DispatcherState> state_;
};

} // namespace tensorcast::store::runtime
