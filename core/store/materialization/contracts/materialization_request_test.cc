// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/contracts/materialization_request.h"

#include <catch2/catch_test_macros.hpp>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/components/device_manager.h"

using tensorcast::DeviceType;
using tensorcast::store::DeviceKey;
using tensorcast::store::components::DeviceManager;
using tensorcast::store::loading::MaterializationRequest;
using tensorcast::store::loading::MaterializeHints;
using tensorcast::store::loading::MaterializeMode;

namespace {

DeviceManager make_device_manager() {
  // Default-constructed manager reports zero GPUs, which is sufficient for validation tests.
  return DeviceManager();
}

DeviceKey make_gpu_key(int ordinal) {
  return DeviceKey{.type = DeviceType::GPU, .ordinal = ordinal, .uuid = ""};
}

DeviceKey make_cpu_key() {
  return DeviceKey{.type = DeviceType::CPU, .ordinal = -1, .uuid = ""};
}

} // namespace

TEST_CASE("MaterializationRequest validates canonical artifact id for LOAD modes", "[materialization_request]") {
  auto device_manager = make_device_manager();
  MaterializeHints hints;
  hints.artifact_id.clear();

  auto result = MaterializationRequest::Create(make_cpu_key(), MaterializeMode::LOAD_ONLY, hints, device_manager);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("MaterializationRequest rejects COPY_ONLY without canonical identifier", "[materialization_request]") {
  auto device_manager = make_device_manager();
  MaterializeHints hints;
  hints.artifact_id.clear();

  auto result = MaterializationRequest::Create(make_gpu_key(0), MaterializeMode::COPY_ONLY, hints, device_manager);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("MaterializationRequest rejects invalid GPU ordinals", "[materialization_request]") {
  auto device_manager = make_device_manager();
  MaterializeHints hints;
  hints.artifact_id = "cgid:artifact-A";

  auto result = MaterializationRequest::Create(make_gpu_key(0), MaterializeMode::AUTO, hints, device_manager);
  REQUIRE_FALSE(result.ok());
  REQUIRE(result.status().code() == absl::StatusCode::kInvalidArgument);
}

TEST_CASE("MaterializationRequest captures canonical ids and replica key", "[materialization_request]") {
  auto device_manager = make_device_manager();
  MaterializeHints hints;
  hints.artifact_id = "cgid:artifact-A";

  auto result = MaterializationRequest::Create(make_cpu_key(), MaterializeMode::LOAD_ONLY, hints, device_manager);
  REQUIRE(result.ok());
  const auto& request = result.value();
  REQUIRE(request.replica_key().artifact_id == "cgid:artifact-A");
  REQUIRE(request.canonical_artifact_id() == "cgid:artifact-A");
  REQUIRE(request.requested_view_id() == std::nullopt);
  REQUIRE(request.target_device().type == DeviceType::CPU);
}
