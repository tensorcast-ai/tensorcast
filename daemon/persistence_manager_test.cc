// Copyright (c) 2025, TensorCast Team.

#include "daemon/persistence_manager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "core/store/testing/recording_global_store_client.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using tensorcast::daemon::PersistenceManager;
using tensorcast::daemon::PersistenceTaskState;
using tensorcast::daemon::RequirementLevel;
using tensorcast::daemon::resolve_store_policy;
using tensorcast::daemon::ResolvedStorePolicy;
using tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED;
using tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED;
using tensorcast::daemon::v1::PERSISTENCE_STATE_PENDING;
using tensorcast::daemon::v1::PERSISTENCE_STATE_RUNNING;
using tensorcast::daemon::v1::PERSISTENCE_STATE_SUCCESS;
using tensorcast::daemon::v1::PLACEMENT_POLICY_LOCAL_ONLY;
using tensorcast::daemon::v1::PLACEMENT_POLICY_REPLICATED;
using tensorcast::daemon::v1::POLICY_LAYOUT_UNSHARDED;

namespace {

static std::filesystem::path persistence_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env) / "tensorcast_persistence_manager";
  }
  return std::filesystem::temp_directory_path() / "tensorcast_persistence_manager";
}

tensorcast::store::StoreEngineOptions make_engine_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = persistence_tmpdir();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47101;
  opts.memory_pool_size = 64ull << 20; // 64 MiB
  opts.tx_slice_bytes = 1ull << 20; // 1 MiB
  opts.num_thread = 2;
  return opts;
}

} // namespace

TEST_CASE("PersistenceManager advances state machine", "[daemon][persistence]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  const PersistenceTaskState first =
      mgr.start_task_for_test("artifact-1", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);

  REQUIRE(first.state == PERSISTENCE_STATE_PENDING);
  REQUIRE(first.progress == 0.0);
  REQUIRE(first.shards.size() == 1);
  REQUIRE(first.shards[0].state == PERSISTENCE_STATE_PENDING);
  REQUIRE(first.shards[0].targets.size() == 2); // local + shared disk

  mgr.advance_once_for_test();
  auto running = mgr.get_by_task_id(first.task_id);
  REQUIRE(running.has_value());
  REQUIRE(running->state == PERSISTENCE_STATE_RUNNING);
  REQUIRE(running->shards.size() == 1);
  REQUIRE(running->shards[0].state == PERSISTENCE_STATE_RUNNING);

  mgr.advance_once_for_test();
  auto copying_disk = mgr.get_by_task_id(first.task_id);
  REQUIRE(copying_disk.has_value());
  REQUIRE(copying_disk->shards[0].progress == Approx(0.5)); // local done, disk copying

  mgr.advance_once_for_test();
  auto success = mgr.get_by_task_id(first.task_id);
  REQUIRE(success.has_value());
  REQUIRE(success->state == PERSISTENCE_STATE_SUCCESS);
  REQUIRE(success->progress == Approx(1.0));
  REQUIRE(success->shards[0].state == PERSISTENCE_STATE_SUCCESS);
}

TEST_CASE("PersistenceManager tracks latest task per artifact", "[daemon][persistence]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);

  auto first = mgr.start_task_for_test("artifact-2", PLACEMENT_POLICY_LOCAL_ONLY, false);
  mgr.advance_once_for_test();

  auto latest_before = mgr.get_latest_for_artifact("artifact-2");
  REQUIRE(latest_before.has_value());
  REQUIRE(latest_before->task_id == first.task_id);

  auto second = mgr.start_task_for_test("artifact-2", PLACEMENT_POLICY_REPLICATED, true);
  auto latest_after = mgr.get_latest_for_artifact("artifact-2");
  REQUIRE(latest_after.has_value());
  REQUIRE(latest_after->task_id == second.task_id);
  REQUIRE(latest_after->state == PERSISTENCE_STATE_PENDING);
}

TEST_CASE("PersistenceManager requests placement plans", "[daemon][persistence][plan]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-A");

  auto task = mgr.start_task_for_test("artifact-plan", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  REQUIRE(task.plan_id == "plan-0");
  REQUIRE(task.shards.size() == 1);
  REQUIRE(task.shards[0].targets.size() == 3); // local + remote + shared disk
  const auto node_ids = std::vector<std::string>{
      task.shards[0].targets[0].node_id,
      task.shards[0].targets[1].node_id,
      task.shards[0].targets[2].node_id,
  };
  REQUIRE(std::find(node_ids.begin(), node_ids.end(), "node-A") != node_ids.end());
  REQUIRE(std::find(node_ids.begin(), node_ids.end(), client_ptr->remote_node_id) != node_ids.end());

  mgr.advance_once_for_test(); // running
  mgr.advance_once_for_test(); // targets copying
  mgr.advance_once_for_test(); // complete
  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_SUCCESS);
  REQUIRE(final->shards[0].targets.size() == 3);
  for (const auto& tgt : final->shards[0].targets) {
    REQUIRE(tgt.target_state == tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_COMPLETE);
  }
  REQUIRE_FALSE(client_ptr->persistence_reports.empty());
}

TEST_CASE("PersistenceManager degrades when plan fails", "[daemon][persistence][plan][degraded]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->allow_plan_placement = false;
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task =
      mgr.start_task_for_test("artifact-plan-degraded", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->degraded_reason.empty());
  REQUIRE(final->shards[0].targets.size() == 2); // local + disk
  REQUIRE(final->shards[0].targets[0].node_id == "node-local");
}

TEST_CASE(
    "PersistenceManager fails when remote placement is required and plan degrades",
    "[daemon][persistence][plan]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->plan_degraded = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);

  ResolvedStorePolicy auto_layout;
  auto_layout.remote_requirement = RequirementLevel::kShould;
  auto_layout.layout = tensorcast::daemon::v1::POLICY_LAYOUT_AUTO;
  auto auto_small = mgr.start_task_for_test_with_policy("artifact-auto-small", auto_layout, 64ULL * 1024 * 1024);
  REQUIRE(auto_small.placement_policy == PLACEMENT_POLICY_REPLICATED);
  auto auto_large = mgr.start_task_for_test_with_policy("artifact-auto-large", auto_layout, 200ULL * 1024 * 1024);
  REQUIRE(auto_large.placement_policy == tensorcast::daemon::v1::PLACEMENT_POLICY_SHARDED);

  ResolvedStorePolicy unsharded;
  unsharded.remote_requirement = RequirementLevel::kShould;
  unsharded.layout = POLICY_LAYOUT_UNSHARDED;
  auto unsharded_task = mgr.start_task_for_test_with_policy("artifact-unsharded", unsharded, 200ULL * 1024 * 1024);
  REQUIRE(unsharded_task.placement_policy == PLACEMENT_POLICY_REPLICATED);

  ResolvedStorePolicy sharded;
  sharded.remote_requirement = RequirementLevel::kShould;
  sharded.layout = tensorcast::daemon::v1::POLICY_LAYOUT_SHARDED;
  auto sharded_task = mgr.start_task_for_test_with_policy("artifact-sharded", sharded, 64ULL * 1024 * 1024);
  REQUIRE(sharded_task.placement_policy == tensorcast::daemon::v1::PLACEMENT_POLICY_SHARDED);
}

TEST_CASE("PersistenceManager lowers durable profile to shared disk persistence", "[daemon][persistence][policy]") {
  tensorcast::daemon::v1::StorePolicy policy;
  policy.set_profile(tensorcast::daemon::v1::POLICY_PROFILE_DURABLE);
  auto resolved_or = resolve_store_policy(&policy);
  REQUIRE(resolved_or.ok());

  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  auto task = mgr.start_task_for_test_with_policy("artifact-durable", *resolved_or, 64ULL * 1024 * 1024);
  REQUIRE(task.persist_to_shared_disk);
  REQUIRE(task.placement_policy == PLACEMENT_POLICY_LOCAL_ONLY);
}

TEST_CASE("PersistenceManager degrades when leases are denied", "[daemon][persistence][lease]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-lease-fail", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->shards.empty());
  REQUIRE(final->shards[0].state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED);
  REQUIRE_FALSE(final->shards[0].degraded_reason.empty());
}

TEST_CASE(
    "PersistenceManager fails when remote placement is required and leases are denied",
    "[daemon][persistence][lease]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
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

TEST_CASE("PersistenceManager deduplicates shared disk targets", "[daemon][persistence][disk]") {
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  auto first = mgr.start_task_for_test("artifact-dedup", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);
  mgr.advance_once_for_test();
  mgr.advance_once_for_test();
  mgr.advance_once_for_test(); // disk completes
  auto done = mgr.get_by_task_id(first.task_id);
  REQUIRE(done.has_value());
  REQUIRE(done->state == PERSISTENCE_STATE_SUCCESS);

  auto second = mgr.start_task_for_test("artifact-dedup", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);
  mgr.advance_once_for_test(); // running
  mgr.advance_once_for_test(); // dedup applies immediately
  auto deduped = mgr.get_by_task_id(second.task_id);
  REQUIRE(deduped.has_value());
  REQUIRE(deduped->progress == Approx(1.0));
  bool skipped_disk = false;
  for (const auto& target : deduped->shards[0].targets) {
    if (target.is_shared_disk) {
      skipped_disk = target.target_state == tensorcast::global_store::v1::PLACEMENT_TARGET_STATE_SKIPPED;
    }
  }
  REQUIRE(skipped_disk);
}

TEST_CASE("PersistenceManager updates durability index for spill gating", "[daemon][persistence][durability]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto disk_task =
      mgr.start_task_for_test("artifact-spill-disk", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);
  auto remote_task =
      mgr.start_task_for_test("artifact-spill-remote", PLACEMENT_POLICY_REPLICATED, false, 96ULL * 1024 * 1024);

  REQUIRE_FALSE(mgr.is_spill_evictable("artifact-spill-disk", true, false));
  REQUIRE_FALSE(mgr.is_spill_evictable("artifact-spill-remote", false, true));

  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
  }

  auto disk_done = mgr.get_by_task_id(disk_task.task_id);
  REQUIRE(disk_done.has_value());
  REQUIRE(disk_done->state == PERSISTENCE_STATE_SUCCESS);
  auto remote_done = mgr.get_by_task_id(remote_task.task_id);
  REQUIRE(remote_done.has_value());
  REQUIRE(remote_done->state == PERSISTENCE_STATE_SUCCESS);

  REQUIRE(mgr.is_spill_evictable("artifact-spill-disk", true, false));
  REQUIRE_FALSE(mgr.is_spill_evictable("artifact-spill-disk", false, true));
  REQUIRE(mgr.is_spill_evictable("artifact-spill-remote", false, true));
}

TEST_CASE("PersistenceManager reloads task log snapshots", "[daemon][persistence][log]") {
  const auto log_path = std::filesystem::temp_directory_path() / "tensorcast_persist_log.jsonl";
  std::filesystem::remove(log_path);

  {
    PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(5), log_path);
    auto task = mgr.start_task_for_test("artifact-log", PLACEMENT_POLICY_LOCAL_ONLY, true, 64ULL * 1024 * 1024);
    mgr.advance_once_for_test();
    mgr.advance_once_for_test();
    mgr.advance_once_for_test();
    auto final = mgr.get_by_task_id(task.task_id);
    REQUIRE(final.has_value());
    REQUIRE(final->state == PERSISTENCE_STATE_SUCCESS);
  }

  PersistenceManager reloaded(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(5), log_path);
  auto recovered = reloaded.get_latest_for_artifact("artifact-log");
  REQUIRE(recovered.has_value());
  REQUIRE(recovered->state == PERSISTENCE_STATE_SUCCESS);

  std::filesystem::remove(log_path);
}

TEST_CASE("PersistenceManager retries leases before succeeding", "[daemon][persistence][lease][retry]") {
  auto client = tensorcast::store::testing::MakeRecordingGlobalStoreClient();
  auto* client_ptr = static_cast<tensorcast::store::testing::RecordingGlobalStoreClient*>(client.get());
  client_ptr->deny_leases = true;
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024, std::chrono::milliseconds(1));
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-lease-retry", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  mgr.advance_once_for_test(); // pending -> running
  mgr.advance_once_for_test(); // first lease attempt fails
  client_ptr->deny_leases = false;
  for (int i = 0; i < 12; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == PERSISTENCE_STATE_SUCCESS || maybe->state == PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED)) {
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task = mgr.start_task_for_test("artifact-register", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  for (int i = 0; i < 8; ++i) {
    mgr.advance_once_for_test();
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_SUCCESS);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_global_store_client(client_ptr);
  mgr.set_local_node_id("node-local");

  auto task =
      mgr.start_task_for_test("artifact-register-fail", PLACEMENT_POLICY_REPLICATED, true, 140ULL * 1024 * 1024);
  for (int i = 0; i < 24; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
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
  PersistenceManager mgr(nullptr, nullptr, nullptr, 4ULL * 1024 * 1024);
  mgr.set_fail_shared_disk_for_test(true);

  auto task = mgr.start_task_for_test("artifact-disk-fail", PLACEMENT_POLICY_LOCAL_ONLY, true, 96ULL * 1024 * 1024);
  for (int i = 0; i < 6; ++i) {
    mgr.advance_once_for_test();
    auto maybe = mgr.get_by_task_id(task.task_id);
    if (maybe.has_value() &&
        (maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED ||
         maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_DEGRADED ||
         maybe->state == tensorcast::daemon::v1::PERSISTENCE_STATE_SUCCESS)) {
      break;
    }
  }

  auto final = mgr.get_by_task_id(task.task_id);
  REQUIRE(final.has_value());
  REQUIRE(final->state == tensorcast::daemon::v1::PERSISTENCE_STATE_FAILED);
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
  reg.stable_dram.stage_on_gpu = true;
  reg.stable_dram.release_gpu_on_commit = true;
  reg.tensor_index_data = std::string("{}");
  reg.schema_version = "v3";
  reg.encoding = "json";

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());

  PersistenceManager mgr(nullptr, nullptr, &engine, engine.get_artifact_chunk_bytes());
  ResolvedStorePolicy policy;
  auto task_or = mgr.start_task(commit_or->artifact_id, policy);
  REQUIRE(task_or.ok());
}

TEST_CASE("PersistenceManager fails when no persistence source is available", "[daemon][persistence][source]") {
  tensorcast::store::StoreEngine engine(make_engine_opts());
  PersistenceManager mgr(nullptr, nullptr, &engine, engine.get_artifact_chunk_bytes());
  ResolvedStorePolicy policy;
  auto task_or = mgr.start_task("missing-artifact", policy);
  REQUIRE_FALSE(task_or.ok());
  REQUIRE(absl::IsFailedPrecondition(task_or.status()));
}
