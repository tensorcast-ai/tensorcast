// Copyright (c) 2025, StepCast Team. All rights reserved.

// Unit test for status mapping helper
#include "daemon/status_utils.h"

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"

using stepcast::daemon::status_utils::to_grpc_status;

TEST_CASE("Status mapping", "[daemon]") {
  auto s1 = to_grpc_status(absl::InvalidArgumentError("bad arg"));
  REQUIRE(s1.error_code() == grpc::StatusCode::INVALID_ARGUMENT);
  auto s2 = to_grpc_status(absl::NotFoundError("missing"));
  REQUIRE(s2.error_code() == grpc::StatusCode::NOT_FOUND);
  auto s3 = to_grpc_status(absl::InternalError("boom"));
  REQUIRE(s3.error_code() == grpc::StatusCode::INTERNAL);
}
