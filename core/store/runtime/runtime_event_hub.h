// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "core/store/device_types.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/runtime/ingestion_events.h"

namespace tensorcast::store::runtime {

enum class RuntimeEventType {
  kReplicaLoaded,
  kReplicaEvicted,
  kIngressCompleted,
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

using RuntimeEventPayload =
    std::variant<std::monostate, ReplicaLifecycleEvent, RegistrationEvent, IngestionResultEvent, KeyMappingEvent>;

struct RuntimeEvent {
  RuntimeEventType type{RuntimeEventType::kIngressCompleted};
  RuntimeEventPayload payload;
};

class RuntimeEventHub {
 public:
  class Subscription {
   public:
    Subscription(RuntimeEventHub* hub, int64_t id);
    ~Subscription();
    Subscription(const Subscription&) = delete;
    Subscription& operator=(const Subscription&) = delete;
    Subscription(Subscription&& other) noexcept;
    Subscription& operator=(Subscription&& other) noexcept;
    void release();

   private:
    RuntimeEventHub* hub_{nullptr};
    int64_t id_{0};
  };

  using Callback = absl::AnyInvocable<void(const RuntimeEvent&)>;

  std::unique_ptr<Subscription> subscribe(Callback callback);
  void publish(const RuntimeEvent& event);
  void drain();

 private:
  struct Subscriber {
    int64_t id;
    std::shared_ptr<Callback> callback;
  };

  void unsubscribe(int64_t id);

  absl::Mutex mu_;
  int64_t next_id_ ABSL_GUARDED_BY(mu_) = 1;
  std::vector<Subscriber> subscribers_ ABSL_GUARDED_BY(mu_);
  int active_publishers_ ABSL_GUARDED_BY(mu_) = 0;
  absl::CondVar drain_cv_;
};

} // namespace tensorcast::store::runtime
