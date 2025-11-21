// Copyright (c) 2025, TensorCast Team.

#include "core/store/runtime/ingestion/ingestion_event_hub.h"

#include <vector>

#include "absl/status/status.h"
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/context/runtime_context_events.h"
#include "core/store/runtime/ingestion_events.h"

namespace tensorcast::store::runtime::ingestion {

using tensorcast::store::runtime::IngestionCompletedEvent;
using tensorcast::store::runtime::IngestionStartedEvent;
using tensorcast::store::runtime::RuntimeContextEvents;

TEST_CASE("IngestionEventHub publishes started and completed events", "[ingestion_event_hub]") {
  RuntimeContextEvents events;
  IngestionEventHub hub(&events);

  std::vector<std::string> started_ids;
  std::vector<absl::Status> completion_statuses;

  auto started_subscription =
      hub.subscribe_started([&](const IngestionStartedEvent& event) { started_ids.push_back(event.request_id); });
  auto completed_subscription = hub.subscribe_completed(
      [&](const IngestionCompletedEvent& event) { completion_statuses.push_back(event.status); });

  IngestionStartedEvent started;
  started.request_id = "req-1";
  started.artifact_id = "artifact-A";
  started.source = IngestionSource::kDisk;
  started.publish_context_id = "ctx-1";
  started.publish_to_global_store = true;
  hub.publish_started(started);

  IngestionCompletedEvent success_event;
  success_event.request_id = "req-1";
  success_event.artifact_id = "artifact-A";
  success_event.status = absl::OkStatus();
  success_event.publish_context_id = "ctx-1";
  hub.publish_completed(success_event);

  IngestionCompletedEvent failure_event = success_event;
  failure_event.request_id = "req-2";
  failure_event.status = absl::InternalError("fail");
  hub.publish_completed(failure_event);
  events.drain();

  REQUIRE(started_ids.size() == 1);
  CHECK(started_ids.front() == "req-1");
  REQUIRE(completion_statuses.size() == 2);
  CHECK(completion_statuses.front().ok());
  CHECK(!completion_statuses.back().ok());
}

TEST_CASE("IngestionEventHub handles null RuntimeContextEvents", "[ingestion_event_hub]") {
  IngestionEventHub hub(nullptr);
  IngestionStartedEvent started;
  started.request_id = "noop";
  hub.publish_started(started);
  auto subscription = hub.subscribe_started([](const IngestionStartedEvent&) {});
  CHECK(subscription == nullptr);
}

} // namespace tensorcast::store::runtime::ingestion
