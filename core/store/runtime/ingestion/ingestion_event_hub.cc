// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/ingestion/ingestion_event_hub.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"

namespace tensorcast::store::runtime::ingestion {

namespace {

template <typename Payload>
const Payload* GetPayload(const RuntimeEvent& event, RuntimeEventType expected_type) {
  if (event.type != expected_type) {
    return nullptr;
  }
  return std::get_if<Payload>(&event.payload);
}

} // namespace

IngestionEventHub::IngestionEventHub(RuntimeContextEvents* events)
    : events_(events), publisher_(events != nullptr ? events->publisher() : RuntimeContextEvents::Publisher()) {}

void IngestionEventHub::publish_started(IngestionStartedEvent event) const {
  if (!publisher_) {
    return;
  }
  RuntimeEvent runtime_event;
  runtime_event.type = RuntimeEventType::kIngestionStarted;
  runtime_event.payload = std::move(event);
  publisher_.publish(std::move(runtime_event));
}

void IngestionEventHub::publish_completed(IngestionCompletedEvent event) const {
  if (!publisher_) {
    return;
  }
  RuntimeEvent runtime_event;
  runtime_event.type = event.status.ok() ? RuntimeEventType::kIngestionCompleted : RuntimeEventType::kIngestionFailed;
  runtime_event.payload = event;
  publisher_.publish(std::move(runtime_event));
}

std::unique_ptr<RuntimeContextEvents::Subscription> IngestionEventHub::subscribe_started(StartedCallback callback) {
  if (events_ == nullptr) {
    return nullptr;
  }
  auto shared_cb = std::make_shared<StartedCallback>(std::move(callback));
  return events_->subscribe([shared_cb](const RuntimeEvent& runtime_event) {
    const auto* payload = GetPayload<IngestionStartedEvent>(runtime_event, RuntimeEventType::kIngestionStarted);
    if (payload == nullptr) {
      return;
    }
    (*shared_cb)(*payload);
  });
}

std::unique_ptr<RuntimeContextEvents::Subscription> IngestionEventHub::subscribe_completed(CompletedCallback callback) {
  if (events_ == nullptr) {
    return nullptr;
  }
  auto shared_cb = std::make_shared<CompletedCallback>(std::move(callback));
  return events_->subscribe([shared_cb](const RuntimeEvent& runtime_event) {
    if (runtime_event.type != RuntimeEventType::kIngestionCompleted &&
        runtime_event.type != RuntimeEventType::kIngestionFailed) {
      return;
    }
    const auto* payload = std::get_if<IngestionCompletedEvent>(&runtime_event.payload);
    if (payload == nullptr) {
      return;
    }
    (*shared_cb)(*payload);
  });
}

} // namespace tensorcast::store::runtime::ingestion
