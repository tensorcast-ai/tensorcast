// Copyright (c) 2025, TensorCast Team.

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "catch2/catch_test_macros.hpp"
#include "core/store/runtime/context/runtime_context.h"
#include "core/store/store_engine_options.h"

using tensorcast::store::StoreEngineOptions;
using tensorcast::store::runtime::RegistrationEvent;
using tensorcast::store::runtime::RuntimeContext;
using tensorcast::store::runtime::RuntimeEvent;
using tensorcast::store::runtime::RuntimeEventType;

namespace {

StoreEngineOptions MakeContextOptions() {
  StoreEngineOptions opts;
  opts.storage_path = "";
  opts.memory_pool_size = 16ull * 1024 * 1024;
  opts.tx_slice_bytes = 512 * 1024;
  opts.artifact_chunk_bytes = opts.tx_slice_bytes * 2;
  opts.num_thread = 1;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  return opts;
}

} // namespace

TEST_CASE("RuntimeContext publishes events and drains", "[runtime][context]") {
  RuntimeContext context(MakeContextOptions());
  REQUIRE(context.start().ok());

  std::vector<RuntimeEventType> observed;
  absl::Mutex mu;
  auto subscription = context.subscribe_to_events([&](const RuntimeEvent& event) {
    absl::MutexLock lock(&mu);
    observed.push_back(event.type);
  });
  REQUIRE(subscription != nullptr);

  auto publisher = context.event_publisher();
  REQUIRE(static_cast<bool>(publisher));

  RuntimeEvent registration_event;
  registration_event.type = RuntimeEventType::kRegistrationCommitted;
  RegistrationEvent payload;
  payload.registration_id = "test-reg";
  payload.committed = true;
  registration_event.payload = payload;
  publisher.publish(registration_event);

  RuntimeEvent abort_event;
  abort_event.type = RuntimeEventType::kRegistrationAborted;
  RegistrationEvent abort_payload;
  abort_payload.registration_id = "test-reg";
  abort_event.payload = abort_payload;
  publisher.publish(abort_event);

  context.drain_events();
  {
    absl::MutexLock lock(&mu);
    REQUIRE(observed.size() == 2);
    CHECK(observed.front() == RuntimeEventType::kRegistrationCommitted);
    CHECK(observed.back() == RuntimeEventType::kRegistrationAborted);
  }
  context.shutdown();
}

TEST_CASE("RuntimeContext handles concurrent publishers", "[runtime][context][concurrency]") {
  constexpr int kThreads = 4;
  constexpr int kEventsPerThread = 200;

  RuntimeContext context(MakeContextOptions());
  REQUIRE(context.start().ok());

  std::vector<std::string> registrations;
  absl::Mutex mu;
  auto subscription = context.subscribe_to_events([&](const RuntimeEvent& event) {
    if (event.type != RuntimeEventType::kRegistrationCommitted) {
      return;
    }
    const auto* reg = std::get_if<RegistrationEvent>(&event.payload);
    if (reg == nullptr) {
      return;
    }
    absl::MutexLock lock(&mu);
    registrations.push_back(reg->registration_id);
  });
  REQUIRE(subscription != nullptr);

  auto publisher = context.event_publisher();
  REQUIRE(static_cast<bool>(publisher));

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([t, publisher]() mutable {
      for (int i = 0; i < kEventsPerThread; ++i) {
        RuntimeEvent event;
        event.type = RuntimeEventType::kRegistrationCommitted;
        RegistrationEvent payload;
        payload.registration_id = "thread-" + std::to_string(t) + "-" + std::to_string(i);
        payload.committed = true;
        event.payload = payload;
        publisher.publish(event);
      }
    });
  }

  for (auto& worker : workers) {
    worker.join();
  }
  context.drain_events();

  {
    absl::MutexLock lock(&mu);
    REQUIRE(registrations.size() == static_cast<size_t>(kThreads * kEventsPerThread));
  }
  context.shutdown();
}
