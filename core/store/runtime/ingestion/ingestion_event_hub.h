// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>

#include "absl/functional/any_invocable.h"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion_events.h"

namespace tensorcast::store::runtime::ingestion {

class IngestionEventHub {
 public:
  using StartedCallback = absl::AnyInvocable<void(const IngestionStartedEvent&)>;
  using CompletedCallback = absl::AnyInvocable<void(const IngestionCompletedEvent&)>;

  explicit IngestionEventHub(RuntimeContextEvents* events);

  void publish_started(IngestionStartedEvent event) const;
  void publish_completed(IngestionCompletedEvent event) const;

  std::unique_ptr<RuntimeContextEvents::Subscription> subscribe_started(StartedCallback callback);
  std::unique_ptr<RuntimeContextEvents::Subscription> subscribe_completed(CompletedCallback callback);

 private:
  RuntimeContextEvents* events_;
  RuntimeContextEvents::Publisher publisher_;
};

} // namespace tensorcast::store::runtime::ingestion
