// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/persistence_manager.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"
#include "daemon/state/lip_manager.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using tensorcast::daemon::PersistenceManager;
using tensorcast::daemon::PersistenceTaskState;
using tensorcast::daemon::RequirementLevel;
using tensorcast::daemon::resolve_store_policy;
using tensorcast::daemon::ResolvedStorePolicy;
using tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED;
using tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED;
using tensorcast::daemon::v2::PERSISTENCE_STATE_PENDING;
using tensorcast::daemon::v2::PERSISTENCE_STATE_RUNNING;
using tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS;
using tensorcast::daemon::v2::PLACEMENT_POLICY_LOCAL_ONLY;
using tensorcast::daemon::v2::PLACEMENT_POLICY_REPLICATED;
using tensorcast::daemon::v2::POLICY_LAYOUT_UNSHARDED;

namespace {

static std::filesystem::path persistence_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env) / "tensorcast_persistence_manager";
  }
  return std::filesystem::temp_directory_path() / "tensorcast_persistence_manager";
}

static std::filesystem::path make_test_storage_root(std::string_view name) {
  auto root = persistence_tmpdir() / std::string(name);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  return root;
}

tensorcast::store::StoreEngineOptions make_engine_opts(const std::filesystem::path& storage_root) {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = storage_root;
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47101;
  opts.memory_pool_size = 64ull << 20; // 64 MiB
  opts.tx_slice_bytes = 1ull << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

tensorcast::store::StoreEngineOptions make_engine_opts() {
  return make_engine_opts(persistence_tmpdir());
}

std::string register_stable_artifact(
    tensorcast::store::StoreEngine& engine,
    std::string artifact_id,
    uint64_t total_size_bytes) {
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = std::move(artifact_id);
  reg.client_artifact_id = reg.artifact_id;
  reg.device_id = 0;
  reg.total_size_bytes = total_size_bytes;
  reg.plan = tensorcast::store::runtime::metadata::RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;
  reg.tensor_index_data = std::string("{}");
  reg.schema_version = "v3";
  reg.encoding = "json";

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  std::vector<std::byte> payload(static_cast<size_t>(total_size_bytes), std::byte{0});
  auto ingest_status =
      engine.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(payload));
  REQUIRE(ingest_status.ok());
  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  return commit_or->artifact_id;
}

PersistenceTaskState advance_task_to_terminal(
    PersistenceManager& mgr,
    std::string_view task_id,
    int max_iterations = 60) {
  PersistenceTaskState task;
  for (int i = 0; i < max_iterations; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task_id);
    REQUIRE(maybe.has_value());
    task = *maybe;
    if (task.state == PERSISTENCE_STATE_SUCCESS || task.state == PERSISTENCE_STATE_FAILED ||
        task.state == PERSISTENCE_STATE_DEGRADED) {
      return task;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL("persistence task did not reach terminal state");
  return task;
}

} // namespace

TEST_CASE("PersistenceManager advances state machine", "[daemon][persistence]") {
  const auto storage_root = make_test_storage_root("advance_state_machine");
  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 8ULL * 1024 * 1024;
  const auto artifact_id = register_stable_artifact(engine, "mi2:indexhash:datahash", total_size);
  REQUIRE(artifact_id.rfind("mi2:", 0) == 0);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());

  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(storage_root);
  mgr.set_local_node_id("node-local");

  const PersistenceTaskState first =
      mgr.start_task_for_test(artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);

  REQUIRE(first.state == PERSISTENCE_STATE_PENDING);
  REQUIRE(first.progress == 0.0);
  REQUIRE(first.shards.size() == 1);
  REQUIRE(first.shards[0].state == PERSISTENCE_STATE_PENDING);
  REQUIRE(first.shards[0].targets.size() == 2); // local + shared disk

  PersistenceTaskState final;
  bool done = false;
  for (int i = 0; i < 60; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(first.task_id);
    if (maybe.has_value()) {
      final = *maybe;
      if (final.state == PERSISTENCE_STATE_SUCCESS || final.state == PERSISTENCE_STATE_FAILED ||
          final.state == PERSISTENCE_STATE_DEGRADED) {
        done = true;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(done);
  REQUIRE(final.state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final.progress == Approx(1.0));
  REQUIRE(final.shards[0].state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final.disk_location_registered);
  REQUIRE_FALSE(final.disk_relative_path.empty());

  const std::filesystem::path object_dir = storage_root / final.disk_relative_path;
  REQUIRE(std::filesystem::exists(object_dir / "artifact_descriptor.json"));
  REQUIRE(std::filesystem::exists(object_dir / "tensor_index.json"));
  REQUIRE(std::filesystem::exists(object_dir / "tensor.data_0"));

  bool location_seen = false;
  for (const auto& entry : client_ptr->disk_locations) {
    if (entry.artifact_id == artifact_id && entry.relative_path == final.disk_relative_path) {
      location_seen = true;
      break;
    }
  }
  REQUIRE(location_seen);
}

TEST_CASE(
    "PersistenceManager persists routed byte-artifact content via external backing source",
    "[daemon][persistence][byte_artifact]") {
  const auto storage_root = make_test_storage_root("byte_artifact_external_source");
  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 8ULL * 1024 * 1024;
  const std::string physical_artifact_id = "__tc_byte_body__:persist_source";
  (void)register_stable_artifact(engine, physical_artifact_id, total_size);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());

  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(storage_root);
  mgr.set_local_node_id("node-local");
  mgr.set_external_source_resolver(
      [&](std::string_view artifact_id) -> absl::StatusOr<PersistenceManager::PersistenceSource> {
        if (artifact_id != "cgid:byte_artifact~tenant~engine~b64u.cGVyc2lzdA~layout_v1~b64u.azE") {
          return absl::NotFoundError("unknown byte-artifact persistence source");
        }
        tensorcast::store::runtime::ingestion::VerifiedContentDescriptor verified;
        verified.content_identity.semantic_layout_identity.kind =
            tensorcast::store::runtime::ingestion::SemanticLayoutKind::kNamedLayoutId;
        verified.content_identity.semantic_layout_identity.value = "layout_v1";
        verified.content_identity.logical_size_bytes = total_size;
        verified.content_identity.digest_alg = "sha256";
        verified.content_identity.digest_bytes.assign(32, '\x01');
        return PersistenceManager::PersistenceSource{
            .artifact_id = std::string(artifact_id),
            .source_artifact_id = physical_artifact_id,
            .total_size_bytes = total_size,
            .verified_content_descriptor = verified,
        };
      });

  ResolvedStorePolicy policy;
  policy.shared_disk_requirement = RequirementLevel::kMust;
  auto task_or = mgr.start_task("cgid:byte_artifact~tenant~engine~b64u.cGVyc2lzdA~layout_v1~b64u.azE", policy);
  REQUIRE(task_or.ok());
  const auto task = *task_or;

  PersistenceTaskState final;
  bool done = false;
  for (int i = 0; i < 60; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    REQUIRE(maybe.has_value());
    final = *maybe;
    if (final.state == PERSISTENCE_STATE_SUCCESS || final.state == PERSISTENCE_STATE_FAILED ||
        final.state == PERSISTENCE_STATE_DEGRADED) {
      done = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  REQUIRE(done);
  REQUIRE(final.state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final.source_artifact_id == physical_artifact_id);
  REQUIRE(final.verified_content_descriptor.has_value());

  auto shared_disk_source = mgr.resolve_policy_source(final.artifact_id);
  REQUIRE(shared_disk_source.has_value());
  REQUIRE(shared_disk_source->path_kind == PersistenceManager::PolicySourceKind::kSharedDisk);
  REQUIRE(shared_disk_source->verified_content_descriptor == final.verified_content_descriptor);
  REQUIRE(std::filesystem::exists(shared_disk_source->local_path / "artifact_descriptor.json"));
  REQUIRE(std::filesystem::exists(shared_disk_source->local_path / "tensor_index.json"));
}

TEST_CASE(
    "PersistenceManager drops shared-disk source once backing path disappears",
    "[daemon][persistence][source_truth]") {
  const auto storage_root = make_test_storage_root("shared_disk_source_missing");
  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 8ULL * 1024 * 1024;
  const auto artifact_id = register_stable_artifact(engine, "mi2:source_missing:index", total_size);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());

  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(storage_root);
  mgr.set_local_node_id("node-local");

  const auto task = mgr.start_task_for_test(artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);
  const auto final = advance_task_to_terminal(mgr, task.task_id);
  REQUIRE(final.state == PERSISTENCE_STATE_SUCCESS);

  auto source = mgr.resolve_policy_source(artifact_id);
  REQUIRE(source.has_value());
  std::filesystem::remove_all(source->local_path);

  REQUIRE_FALSE(mgr.resolve_policy_source(artifact_id).has_value());
  REQUIRE_FALSE(mgr.resolve_policy_source(artifact_id, task.task_id).has_value());
  REQUIRE(mgr.list_policy_sources(artifact_id).empty());
}

TEST_CASE(
    "PersistenceManager keeps older actionable shared-disk source after a later failed task",
    "[daemon][persistence][source_truth]") {
  const auto storage_root = make_test_storage_root("shared_disk_source_truth");
  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 8ULL * 1024 * 1024;
  const auto artifact_id = register_stable_artifact(engine, "mi2:source_truth:index", total_size);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());

  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(storage_root);
  mgr.set_local_node_id("node-local");

  const auto first = mgr.start_task_for_test(artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);
  const auto first_final = advance_task_to_terminal(mgr, first.task_id);
  REQUIRE(first_final.state == PERSISTENCE_STATE_SUCCESS);

  mgr.set_fail_shared_disk_for_test(true);
  const auto second = mgr.start_task_for_test(artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);
  const auto second_final = advance_task_to_terminal(mgr, second.task_id);
  REQUIRE(second_final.state == PERSISTENCE_STATE_FAILED);

  auto latest = mgr.get_latest_for_artifact(artifact_id);
  REQUIRE(latest.has_value());
  REQUIRE(latest->task_id == second.task_id);

  auto source = mgr.resolve_policy_source(artifact_id);
  REQUIRE(source.has_value());
  REQUIRE(source->control_ref == first.task_id);
  REQUIRE(source->workflow_task_id == first.task_id);
  REQUIRE(std::filesystem::exists(source->local_path / "artifact_descriptor.json"));
  const auto listed_sources = mgr.list_policy_sources(artifact_id);
  REQUIRE(listed_sources.size() == 1);
  REQUIRE(listed_sources.front().control_ref == first.task_id);
  REQUIRE(mgr.resolve_policy_source(artifact_id, first.task_id).has_value());
  REQUIRE_FALSE(mgr.resolve_policy_source(artifact_id, second.task_id).has_value());
}

TEST_CASE("PersistenceManager tracks latest task per artifact", "[daemon][persistence]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);

  auto first = mgr.start_task_for_test("artifact-2", PLACEMENT_POLICY_LOCAL_ONLY, false);
  mgr.advance_once_for_test();

  auto latest_before = mgr.get_latest_for_artifact("artifact-2");
  REQUIRE(latest_before.has_value());
  REQUIRE(latest_before->task_id == first.task_id);

  auto second = mgr.start_task_for_test("artifact-2", PLACEMENT_POLICY_REPLICATED, false);
  auto latest_after = mgr.get_latest_for_artifact("artifact-2");
  REQUIRE(latest_after.has_value());
  REQUIRE(latest_after->task_id == second.task_id);
  REQUIRE(latest_after->state == PERSISTENCE_STATE_PENDING);
}

TEST_CASE("PersistenceManager requests placement plans", "[daemon][persistence][plan]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-A");

  auto task = mgr.start_task_for_test("artifact-plan", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  REQUIRE(task.plan_id == "plan-0");
  REQUIRE(task.shards.size() == 1);
  REQUIRE(task.shards[0].targets.size() == 2); // local + remote
  const auto node_ids = std::vector<std::string>{
      task.shards[0].targets[0].node_id,
      task.shards[0].targets[1].node_id,
  };
  REQUIRE(std::find(node_ids.begin(), node_ids.end(), "node-A") != node_ids.end());
  REQUIRE(std::find(node_ids.begin(), node_ids.end(), client_ptr->remote_node_id) != node_ids.end());

  mgr.advance_once_for_test(); // running
  mgr.advance_once_for_test(); // targets copying
  mgr.advance_once_for_test(); // complete
  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final->shards[0].targets.size() == 2);
  for (const auto& tgt : final->shards[0].targets) {
    REQUIRE(tgt.target_state == tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_COMPLETE);
  }
  REQUIRE_FALSE(client_ptr->persistence_reports.empty());
}

TEST_CASE("PersistenceManager throttles unchanged status reports", "[daemon][persistence][report]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;

  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(1));
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task =
      mgr.start_task_for_test("artifact-report-throttle", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  for (int i = 0; i < 20; ++i) {
    mgr.advance_once_for_test();
  }

  REQUIRE_FALSE(client_ptr->persistence_reports.empty());
  REQUIRE(client_ptr->persistence_reports.size() <= 8);
  auto latest = mgr.get_by_task_id(task.task_id);
  REQUIRE(latest.has_value());
}

TEST_CASE("PersistenceManager degrades when plan fails", "[daemon][persistence][plan][degraded]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->allow_plan_placement = false;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task =
      mgr.start_task_for_test("artifact-plan-degraded", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->degraded_reason.empty());
  REQUIRE(final->shards[0].targets.size() == 1); // local only
  REQUIRE(final->shards[0].targets[0].node_id == "node-local");
}

TEST_CASE(
    "PersistenceManager fails when remote placement is required and plan degrades",
    "[daemon][persistence][plan]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->plan_degraded = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  ResolvedStorePolicy policy;
  policy.remote_requirement = RequirementLevel::kMust;
  policy.layout = POLICY_LAYOUT_UNSHARDED;
  auto task = mgr.start_task_for_test_with_policy("artifact-plan-required", policy, 140ULL * 1024 * 1024);

  REQUIRE(task.state == PERSISTENCE_STATE_FAILED);
  REQUIRE_FALSE(task.last_error.empty());
}

TEST_CASE(
    "PersistenceManager degrades when remote placement is preferred and plan degrades",
    "[daemon][persistence][plan]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->plan_degraded = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  ResolvedStorePolicy policy;
  policy.remote_requirement = RequirementLevel::kShould;
  policy.layout = POLICY_LAYOUT_UNSHARDED;
  auto task = mgr.start_task_for_test_with_policy("artifact-plan-optional", policy, 140ULL * 1024 * 1024);

  mgr.advance_once_for_test();
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();
  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->degraded_reason.empty());
}

TEST_CASE("PersistenceManager honors layout overrides for placement policy", "[daemon][persistence][plan]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);

  ResolvedStorePolicy auto_layout;
  auto_layout.remote_requirement = RequirementLevel::kShould;
  auto_layout.layout = tensorcast::daemon::v2::POLICY_LAYOUT_AUTO;
  auto auto_small = mgr.start_task_for_test_with_policy("artifact-auto-small", auto_layout, 64ULL * 1024 * 1024);
  REQUIRE(auto_small.placement_policy == PLACEMENT_POLICY_REPLICATED);
  auto auto_large = mgr.start_task_for_test_with_policy("artifact-auto-large", auto_layout, 200ULL * 1024 * 1024);
  REQUIRE(auto_large.placement_policy == tensorcast::daemon::v2::PLACEMENT_POLICY_SHARDED);

  ResolvedStorePolicy unsharded;
  unsharded.remote_requirement = RequirementLevel::kShould;
  unsharded.layout = POLICY_LAYOUT_UNSHARDED;
  auto unsharded_task = mgr.start_task_for_test_with_policy("artifact-unsharded", unsharded, 200ULL * 1024 * 1024);
  REQUIRE(unsharded_task.placement_policy == PLACEMENT_POLICY_REPLICATED);

  ResolvedStorePolicy sharded;
  sharded.remote_requirement = RequirementLevel::kShould;
  sharded.layout = tensorcast::daemon::v2::POLICY_LAYOUT_SHARDED;
  auto sharded_task = mgr.start_task_for_test_with_policy("artifact-sharded", sharded, 64ULL * 1024 * 1024);
  REQUIRE(sharded_task.placement_policy == tensorcast::daemon::v2::PLACEMENT_POLICY_SHARDED);
}

TEST_CASE("PersistenceManager lowers durable profile to shared disk persistence", "[daemon][persistence][policy]") {
  tensorcast::daemon::v2::StorePolicy policy;
  policy.set_profile(tensorcast::daemon::v2::POLICY_PROFILE_DURABLE);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE(resolved_or.ok());

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(make_test_storage_root("durable_policy"));
  auto task = mgr.start_task_for_test_with_policy("artifact-durable", *resolved_or, 64ULL * 1024 * 1024);
  REQUIRE(task.persist_to_shared_disk);
  REQUIRE(task.placement_policy == PLACEMENT_POLICY_LOCAL_ONLY);
}

TEST_CASE("PersistenceManager degrades when leases are denied", "[daemon][persistence][lease]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-lease-fail", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->shards.empty());
  REQUIRE(final->shards[0].state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->shards[0].degraded_reason.empty());
}

TEST_CASE(
    "PersistenceManager fails when remote placement is required and leases are denied",
    "[daemon][persistence][lease]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  ResolvedStorePolicy policy;
  policy.remote_requirement = RequirementLevel::kMust;
  policy.layout = POLICY_LAYOUT_UNSHARDED;
  auto task = mgr.start_task_for_test_with_policy("artifact-lease-must", policy, 140ULL * 1024 * 1024);
  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() && maybe->state == PERSISTENCE_STATE_FAILED) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == PERSISTENCE_STATE_FAILED);
  REQUIRE_FALSE(final->last_error.empty());
}

TEST_CASE("PersistenceManager shards large artifacts within bounds", "[daemon][persistence][shard_planner]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  const uint64_t total_size = 400ULL * 1024 * 1024; // Forces 2 shards (256MB + 144MB)
  auto task = mgr.start_task_for_test("artifact-large", PLACEMENT_POLICY_LOCAL_ONLY, false, total_size);
  REQUIRE(task.shards.size() == 2);
  uint64_t summed = 0;
  uint32_t expected_chunk = 0;
  for (const auto& shard : task.shards) {
    summed += shard.size_bytes;
    REQUIRE(shard.size_bytes >= 64ULL * 1024 * 1024);
    REQUIRE(shard.size_bytes <= 256ULL * 1024 * 1024);
    for (const auto cid : shard.chunk_ids) {
      REQUIRE(cid == expected_chunk++);
    }
  }
  REQUIRE(summed == total_size);
}

TEST_CASE("PersistenceManager updates durability index for spill gating", "[daemon][persistence][durability]") {
  const auto storage_root = make_test_storage_root("spill_durability");
  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 96ULL * 1024 * 1024;
  const auto disk_artifact_id = register_stable_artifact(engine, "mi2:spillindex:spilldata", total_size);
  REQUIRE(disk_artifact_id.rfind("mi2:", 0) == 0);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(storage_root);
  mgr.set_local_node_id("node-local");

  auto disk_task = mgr.start_task_for_test(disk_artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);
  auto remote_task =
      mgr.start_task_for_test("artifact-spill-remote", PLACEMENT_POLICY_REPLICATED, false, 96ULL * 1024 * 1024);

  REQUIRE_FALSE(mgr.is_spill_evictable(disk_artifact_id, true, false));
  REQUIRE_FALSE(mgr.is_spill_evictable("artifact-spill-remote", false, true));

  for (int i = 0; i < 60; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(disk_task.task_id);
    if (maybe.has_value() && maybe->state == PERSISTENCE_STATE_SUCCESS) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  auto disk_done = mgr.get_by_task_id(disk_task.task_id);
  REQUIRE(disk_done.has_value());
  REQUIRE(disk_done->state == PERSISTENCE_STATE_SUCCESS);
  auto remote_done = mgr.get_by_task_id(remote_task.task_id);
  REQUIRE(remote_done.has_value());
  REQUIRE(remote_done->state == PERSISTENCE_STATE_SUCCESS);

  REQUIRE(mgr.is_spill_evictable(disk_artifact_id, true, false));
  REQUIRE_FALSE(mgr.is_spill_evictable(disk_artifact_id, false, true));
  REQUIRE(mgr.is_spill_evictable("artifact-spill-remote", false, true));
}

TEST_CASE("PersistenceManager reloads task log snapshots", "[daemon][persistence][log]") {
  const auto log_path = std::filesystem::temp_directory_path() / "tensorcast_persist_log.jsonl";
  std::filesystem::remove(log_path);

  {
    PersistenceManager mgr(
        nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(5), log_path);
    auto task = mgr.start_task_for_test("artifact-log", PLACEMENT_POLICY_LOCAL_ONLY, false, 64ULL * 1024 * 1024);
    mgr.advance_once_for_test();
    mgr.advance_once_for_test();
    mgr.advance_once_for_test();
    auto final = mgr.get_by_task_id(task.task_id);
    REQUIRE(final.has_value());
    REQUIRE(final->state == PERSISTENCE_STATE_SUCCESS);
  }

  PersistenceManager reloaded(
      nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(5), log_path);
  auto recovered = reloaded.get_latest_for_artifact("artifact-log");
  REQUIRE(recovered.has_value());
  REQUIRE(recovered->state == PERSISTENCE_STATE_SUCCESS);

  std::filesystem::remove(log_path);
}

TEST_CASE(
    "PersistenceManager reloads actionable shared-disk sources from task log",
    "[daemon][persistence][log][shared_disk]") {
  const auto log_path = std::filesystem::temp_directory_path() / "tensorcast_persist_shared_disk_log.jsonl";
  const auto storage_root = make_test_storage_root("reload_shared_disk_source");
  std::filesystem::remove(log_path);

  tensorcast::store::StoreEngine engine(make_engine_opts(storage_root));
  const uint64_t total_size = 8ULL * 1024 * 1024;
  const auto artifact_id = register_stable_artifact(engine, "mi2:reload_shared_disk:index", total_size);

  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());

  std::string task_id;
  {
    PersistenceManager mgr(
        nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes(), std::chrono::milliseconds(5), log_path);
    mgr.set_global_store_client(client_ptr);
    mgr.set_storage_path(storage_root);
    mgr.set_local_node_id("node-local");

    auto task = mgr.start_task_for_test(artifact_id, PLACEMENT_POLICY_LOCAL_ONLY, true, total_size);
    task_id = task.task_id;
    const auto final = advance_task_to_terminal(mgr, task.task_id);
    REQUIRE(final.state == PERSISTENCE_STATE_SUCCESS);
    REQUIRE(final.disk_location_registered);
    REQUIRE(mgr.is_spill_evictable(artifact_id, true, false));
  }

  PersistenceManager reloaded(
      nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes(), std::chrono::milliseconds(5), log_path);
  reloaded.set_storage_path(storage_root);

  auto recovered = reloaded.get_latest_for_artifact(artifact_id);
  REQUIRE(recovered.has_value());
  REQUIRE(recovered->task_id == task_id);
  REQUIRE(recovered->state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(recovered->disk_location_registered);
  REQUIRE_FALSE(recovered->disk_relative_path.empty());

  auto source = reloaded.resolve_policy_source(artifact_id);
  REQUIRE(source.has_value());
  REQUIRE(source->control_ref == task_id);
  REQUIRE(source->workflow_task_id == task_id);
  REQUIRE(std::filesystem::exists(source->local_path / "artifact_descriptor.json"));
  const auto listed_sources = reloaded.list_policy_sources(artifact_id);
  REQUIRE(listed_sources.size() == 1);
  REQUIRE(listed_sources.front().control_ref == task_id);
  REQUIRE(reloaded.is_spill_evictable(artifact_id, true, false));

  std::filesystem::remove(log_path);
}

TEST_CASE("PersistenceManager retries leases before succeeding", "[daemon][persistence][lease][retry]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(1));
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-lease-retry", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  mgr.advance_once_for_test(); // pending -> running
  mgr.advance_once_for_test(); // first lease attempt fails
  client_ptr->deny_leases = false;
  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == PERSISTENCE_STATE_SUCCESS || maybe->state == PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(client_ptr->memory_tier_leases.size() >= 1); // request + optional ack entries
}

TEST_CASE("PersistenceManager registers remote replicas and acks leases", "[daemon][persistence][register]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-register", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  for (int i = 0; i < 8; ++i) {
    mgr.advance_once_for_test();
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final->shards.size() == 1);
  bool remote_registered = false;
  bool ack_seen = false;
  for (const auto& lease : client_ptr->memory_tier_leases) {
    if (lease.state == tensorcast::store::components::MemoryTierLeaseState::kActive) {
      ack_seen = true;
    }
  }
  for (const auto& replica : client_ptr->registered_replicas) {
    if (replica == "artifact-register") {
      remote_registered = true;
    }
  }
  REQUIRE(ack_seen);
  REQUIRE(remote_registered);
}

TEST_CASE("PersistenceManager degrades when remote registration fails", "[daemon][persistence][register][failure]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->fail_register_replica = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task =
      mgr.start_task_for_test("artifact-register-fail", PLACEMENT_POLICY_REPLICATED, false, 140ULL * 1024 * 1024);
  for (int i = 0; i < 24; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->degraded_reason.empty());
  bool remote_failed = false;
  for (const auto& target : final->shards[0].targets) {
    if (!target.is_shared_disk && target.node_id != "node-local") {
      remote_failed = target.target_state == tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_FAILED;
      REQUIRE_FALSE(target.degraded_reason.empty());
    }
  }
  REQUIRE(remote_failed);
}

TEST_CASE("PersistenceManager degrades when shared disk is optional and write fails", "[daemon][persistence][disk]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(make_test_storage_root("disk_optional"));
  mgr.set_fail_shared_disk_for_test(true);

  ResolvedStorePolicy policy;
  policy.shared_disk_requirement = RequirementLevel::kShould;
  policy.layout = POLICY_LAYOUT_UNSHARDED;
  auto task = mgr.start_task_for_test_with_policy("artifact-disk-optional", policy, 96ULL * 1024 * 1024);
  for (int i = 0; i < 6; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == PERSISTENCE_STATE_DEGRADED || maybe->state == PERSISTENCE_STATE_FAILED ||
         maybe->state == PERSISTENCE_STATE_SUCCESS)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->degraded_reason.empty());
}

TEST_CASE("PersistenceManager fails when shared disk write fails", "[daemon][persistence][disk][failure]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_storage_path(make_test_storage_root("disk_fail"));
  mgr.set_fail_shared_disk_for_test(true);

  auto task = mgr.start_task_for_test("artifact-disk-fail", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);
  for (int i = 0; i < 6; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED ||
         maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v2::PERSISTENCE_STATE_SUCCESS)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v2::PERSISTENCE_STATE_FAILED);
  bool disk_failed = false;
  for (const auto& target : final->shards[0].targets) {
    if (target.is_shared_disk) {
      disk_failed = target.target_state == tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_FAILED;
    }
  }
  REQUIRE(disk_failed);
  REQUIRE_FALSE(final->shards[0].last_error.empty());
}

TEST_CASE("PersistenceManager starts from stable DRAM replica without LIP", "[daemon][persistence][source]") {
  tensorcast::store::StoreEngine engine(make_engine_opts());
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "cgid:stable-source";
  reg.client_artifact_id = reg.artifact_id;
  reg.device_id = 0;
  reg.total_size_bytes = 64ull << 10;
  reg.plan = tensorcast::store::runtime::metadata::RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;
  reg.tensor_index_data = std::string("{}");
  reg.schema_version = "v3";
  reg.encoding = "json";

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  std::vector<std::byte> payload(static_cast<size_t>(reg.total_size_bytes), std::byte{0});
  auto ingest_status =
      engine.ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(payload));
  REQUIRE(ingest_status.ok());
  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  ResolvedStorePolicy policy;
  auto task_or = mgr.start_task(commit_or->artifact_id, policy);
  REQUIRE(task_or.ok());
}

TEST_CASE(
    "PersistenceManager prefers stable DRAM source when both stable and LIP exist",
    "[daemon][persistence][source]") {
  const uint64_t stable_size_bytes = 64ull << 10;
  const uint64_t lip_size_bytes = 96ull << 10;

  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_engine_opts());
  tensorcast::store::StoreEngine::ArtifactRegistration reg;
  reg.artifact_id = "cgid:stable-preferred";
  reg.client_artifact_id = reg.artifact_id;
  reg.device_id = 0;
  reg.total_size_bytes = stable_size_bytes;
  reg.plan = tensorcast::store::runtime::metadata::RegistrationPlan::kStableDram;
  reg.stable_dram.stage_on_gpu = false;
  reg.stable_dram.release_gpu_on_commit = false;
  reg.tensor_index_data = std::string("{}");
  reg.schema_version = "v3";
  reg.encoding = "json";

  auto begin_or = engine->begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  std::vector<std::byte> payload(static_cast<size_t>(reg.total_size_bytes), std::byte{0});
  auto ingest_status =
      engine->ingest_registration_chunk(begin_or->registration_id, /*offset=*/0, absl::MakeConstSpan(payload));
  REQUIRE(ingest_status.ok());
  auto commit_or = engine->commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  tensorcast::daemon::LipManager lip_mgr(engine, /*regions=*/nullptr);
  tensorcast::daemon::ArtifactDeviceKey lease_key{.artifact_id = commit_or->artifact_id, .device_id = 0};
  tensorcast::daemon::LipLeaseEntry lease_entry;
  lease_entry.registration_id = "reg:lip-source";
  lease_entry.artifact_id = commit_or->artifact_id;
  lease_entry.device_id = lease_key.device_id;
  lease_entry.total_size = lip_size_bytes;
  lip_mgr.put_lease(lease_entry.registration_id, lease_key, std::move(lease_entry));

  PersistenceManager mgr(nullptr, &lip_mgr, engine.get(), nullptr, engine->get_artifact_chunk_bytes());
  ResolvedStorePolicy policy;
  auto task_or = mgr.start_task(commit_or->artifact_id, policy);
  REQUIRE(task_or.ok());
  REQUIRE(task_or->shards.size() == 1);
  REQUIRE(task_or->shards[0].byte_range_length == stable_size_bytes);
}

TEST_CASE("PersistenceManager fails when no persistence source is available", "[daemon][persistence][source]") {
  tensorcast::store::StoreEngine engine(make_engine_opts());
  PersistenceManager mgr(nullptr, nullptr, &engine, nullptr, engine.get_artifact_chunk_bytes());
  ResolvedStorePolicy policy;
  auto task_or = mgr.start_task("missing-artifact", policy);
  REQUIRE_FALSE(task_or.ok());
  REQUIRE(absl::IsFailedPrecondition(task_or.status()));
}
