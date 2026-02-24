// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/engine/mtcp_transfer_completion_tracker.h"

#include <optional>
#include <string>

#include "catch2/catch_test_macros.hpp"

namespace tensorcast::communicator::engine {

TEST_CASE("MtcpTransferCompletionTracker waits all segments before success", "[communicator][mtcp]") {
  std::optional<absl::Status> status;
  int callback_count = 0;
  MtcpTransferCompletionTracker tracker([&](const absl::Status& s) {
    status = s;
    callback_count += 1;
  });

  tracker.add_pending_segments(5);
  tracker.mark_final_window_enqueued();
  CHECK(callback_count == 0);

  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);
  CHECK(callback_count == 0);

  tracker.mark_segment_finished(true);
  REQUIRE(callback_count == 1);
  REQUIRE(status.has_value());
  CHECK(status->ok());
}

TEST_CASE("MtcpTransferCompletionTracker handles final window finishing before old windows", "[communicator][mtcp]") {
  std::optional<absl::Status> status;
  int callback_count = 0;
  MtcpTransferCompletionTracker tracker([&](const absl::Status& s) {
    status = s;
    callback_count += 1;
  });

  tracker.add_pending_segments(4); // early windows
  tracker.add_pending_segments(1); // final window
  tracker.mark_final_window_enqueued();

  // Final-window segment finishes first.
  tracker.mark_segment_finished(true);
  CHECK(callback_count == 0);

  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);
  CHECK(callback_count == 0);

  tracker.mark_segment_finished(true);
  REQUIRE(callback_count == 1);
  REQUIRE(status.has_value());
  CHECK(status->ok());
}

TEST_CASE("MtcpTransferCompletionTracker surfaces failed segment status", "[communicator][mtcp]") {
  std::optional<absl::Status> status;
  int callback_count = 0;
  MtcpTransferCompletionTracker tracker([&](const absl::Status& s) {
    status = s;
    callback_count += 1;
  });

  tracker.add_pending_segments(3);
  tracker.mark_final_window_enqueued();

  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(false);
  tracker.mark_segment_finished(true);

  REQUIRE(callback_count == 1);
  REQUIRE(status.has_value());
  CHECK_FALSE(status->ok());
  CHECK(status->code() == absl::StatusCode::kUnavailable);
}

TEST_CASE("MtcpTransferCompletionTracker fail_fast completes only once", "[communicator][mtcp]") {
  std::optional<absl::Status> status;
  int callback_count = 0;
  MtcpTransferCompletionTracker tracker([&](const absl::Status& s) {
    status = s;
    callback_count += 1;
  });

  tracker.add_pending_segments(2);
  tracker.fail_fast(absl::DeadlineExceededError("forced"));
  tracker.mark_final_window_enqueued();
  tracker.mark_segment_finished(true);
  tracker.mark_segment_finished(true);

  REQUIRE(callback_count == 1);
  REQUIRE(status.has_value());
  CHECK(status->code() == absl::StatusCode::kDeadlineExceeded);
}

} // namespace tensorcast::communicator::engine
