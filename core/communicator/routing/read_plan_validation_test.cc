// Copyright (c) 2026, TensorCast Team.

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "absl/status/status.h"
#include "core/communicator/base/constants.h"
#include "core/communicator/routing/types.h"

namespace {

using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_CPU;
using tensorcast::communicator::base::COMMUNICATE_ENGINE_DEV_GPU;
using tensorcast::communicator::routing::ConnectionProtocol;
using tensorcast::communicator::routing::LocalRegion;
using tensorcast::communicator::routing::ReadPlan;
using tensorcast::communicator::routing::ReadPlanSlice;
using tensorcast::communicator::routing::ReadRouteContext;
using tensorcast::communicator::routing::SourceSlice;
using tensorcast::communicator::routing::validate_read_plan;

ReadRouteContext make_route_context() {
  return ReadRouteContext{
      .local_endpoint_id = "node_a/dev/cpu/0",
      .remote_endpoint_id = "node_b/dev/gpu/0",
      .protocol = ConnectionProtocol::kRdma,
      .rail_id = 0,
  };
}

ReadPlan make_valid_read_plan() {
  ReadPlan plan;
  plan.local_regions = {
      LocalRegion{
          .addr = 0x1000,
          .bytes = 128,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
      LocalRegion{
          .addr = 0x2000,
          .bytes = 128,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };

  const ReadRouteContext route = make_route_context();
  plan.source_slices = {
      SourceSlice{
          .authority_id = "authority-a",
          .route = route,
          .tensor_key = "tensor-0",
          .remote_offset = 0,
          .bytes = 32,
      },
      SourceSlice{
          .authority_id = "authority-a",
          .route = route,
          .tensor_key = "tensor-1",
          .remote_offset = 64,
          .bytes = 24,
      },
  };

  plan.slices = {
      ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .local_region_offset = 0,
          .bytes = 32,
      },
      ReadPlanSlice{
          .source_slice_index = 1,
          .local_region_index = 1,
          .local_region_offset = 16,
          .bytes = 24,
      },
  };
  return plan;
}

ReadPlan make_split_source_read_plan() {
  ReadPlan plan;
  plan.local_regions = {
      LocalRegion{
          .addr = 0x3000,
          .bytes = 128,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
      LocalRegion{
          .addr = 0x4000,
          .bytes = 128,
          .dev_type = COMMUNICATE_ENGINE_DEV_CPU,
          .dev_id = 0,
      },
  };

  plan.source_slices = {
      SourceSlice{
          .authority_id = "authority-a",
          .route = make_route_context(),
          .tensor_key = "tensor-split",
          .remote_offset = 0,
          .bytes = 48,
      },
  };

  plan.slices = {
      ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 0,
          .source_slice_offset = 0,
          .local_region_offset = 0,
          .bytes = 16,
      },
      ReadPlanSlice{
          .source_slice_index = 0,
          .local_region_index = 1,
          .source_slice_offset = 16,
          .local_region_offset = 32,
          .bytes = 32,
      },
  };
  return plan;
}

} // namespace

TEST_CASE("ReadPlan validation accepts a CPU-only single-authority plan", "[communicator][routing][read_plan]") {
  const ReadPlan plan = make_valid_read_plan();
  CHECK(validate_read_plan(plan).ok());
}

TEST_CASE("ReadPlan validation rejects mixed authority", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_valid_read_plan();
  plan.source_slices[1].authority_id = "authority-b";

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(std::string(status.message()).find("authority") != std::string::npos);
}

TEST_CASE("ReadPlan validation rejects mixed route context", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_valid_read_plan();
  plan.source_slices[1].route.protocol = ConnectionProtocol::kTcp;

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(std::string(status.message()).find("route context") != std::string::npos);
}

TEST_CASE("ReadPlan validation rejects overlapping destination spans", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_valid_read_plan();
  plan.slices[1].local_region_index = 0;
  plan.slices[1].local_region_offset = 16;

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(std::string(status.message()).find("overlap") != std::string::npos);
}

TEST_CASE("ReadPlan validation rejects non-CPU local regions in the first cut", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_valid_read_plan();
  plan.local_regions[1].dev_type = COMMUNICATE_ENGINE_DEV_GPU;

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kFailedPrecondition);
  CHECK(std::string(status.message()).find("CPU local regions only") != std::string::npos);
}

TEST_CASE(
    "ReadPlan validation accepts split source-slice coverage across local regions",
    "[communicator][routing][read_plan]") {
  const ReadPlan plan = make_split_source_read_plan();
  CHECK(validate_read_plan(plan).ok());
}

TEST_CASE("ReadPlan validation rejects source-slice coverage gaps", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_split_source_read_plan();
  plan.slices[1].source_slice_offset = 20;
  plan.slices[1].bytes = 28;

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(std::string(status.message()).find("coverage") != std::string::npos);
}

TEST_CASE("ReadPlan validation rejects source-slice coverage overlap", "[communicator][routing][read_plan]") {
  ReadPlan plan = make_split_source_read_plan();
  plan.slices[1].source_slice_offset = 8;

  const absl::Status status = validate_read_plan(plan);
  REQUIRE(!status.ok());
  CHECK(status.code() == absl::StatusCode::kInvalidArgument);
  CHECK(std::string(status.message()).find("coverage") != std::string::npos);
}
