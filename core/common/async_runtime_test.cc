// Copyright (c) 2025, TensorCast Team.

#include <memory>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/common/async_runtime.h"
#include "core/common/trace/trace_manager.h"
#include "core/common/trace/trace_request_data.h"
#include "folly/io/async/Request.h"

namespace tensorcast::common {

TEST_CASE("AsyncRuntime propagates RequestContext into executors", "[async_runtime]") {
  auto runtime = std::make_shared<AsyncRuntime>(AsyncRuntime::Options{
      .cpu_threads = 2,
      .blocking_threads = 2,
      .thread_name_prefix = "async_runtime_test",
  });

  auto trace_ids = std::make_shared<trace::TraceIds>("req-1", "artifact-1");
  auto request_ctx = std::make_shared<folly::RequestContext>();
  request_ctx->setContextData(
      trace::kRpcMethodToken,
      std::make_unique<folly::ImmutableRequestData<std::string>>(std::string("AsyncRuntimeTest")));
  request_ctx->setContextData(trace::kTraceIdsToken, std::make_unique<trace::TraceIdsRequestData>(trace_ids));
  request_ctx->setContextData(trace::kTraceRequestDataToken, std::make_unique<trace::TraceRequestData>(trace_ids));

  folly::RequestContextScopeGuard request_ctx_guard(request_ctx);

  absl::Notification done;
  std::string got_request_id;
  std::string got_artifact_id;
  runtime->cpu_executor()->add([&]() {
    got_request_id = trace::TraceManager::current_request_id();
    got_artifact_id = trace::TraceManager::current_artifact_id();
    done.Notify();
  });

  REQUIRE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  REQUIRE(got_request_id == "req-1");
  REQUIRE(got_artifact_id == "artifact-1");

  runtime->shutdown();
  REQUIRE(runtime->drain(absl::Now() + absl::Seconds(5)).ok());
}

} // namespace tensorcast::common
