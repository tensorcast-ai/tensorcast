// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "core/store/device_registry.h"
#include "core/store/materialization/dataplane/view/view_identity.h"
#include "core/store/store_engine.h"
#include "core/store/testing/global_store_client_stub.h"
#include "daemon/state/derived_view_export_manager.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "daemon/state/session_lifecycle.h"

namespace {

using tensorcast::daemon::DerivedViewExportManager;
using tensorcast::daemon::RefTracker;
using tensorcast::daemon::ReplicaSessionManager;
using tensorcast::daemon::SessionLifecycleManager;
using tensorcast::store::DeviceRegistry;
using tensorcast::store::StoreEngine;
using tensorcast::store::StoreEngineOptions;
using tensorcast::store::components::ReplicaDrainStatus;
using tensorcast::store::loader::ViewSpec;
using tensorcast::store::loading::ReplicaKey;
using tensorcast::store::materialization::view::NarrowOp;
using tensorcast::store::materialization::view::TensorViewOps;
using tensorcast::store::materialization::view::ViewOp;
using tensorcast::store::runtime::metadata::ViewPlacement;
using tensorcast::store::runtime::metadata::ViewRegistration;
using tensorcast::store::runtime::metadata::ViewRegistrationKind;
using tensorcast::store::testing::GlobalStoreClientStub;

class NoopGlobalStoreClient final : public GlobalStoreClientStub {
 public:
  absl::Status unregister_replica(std::string_view, std::string_view) override {
    return absl::OkStatus();
  }

  absl::StatusOr<bool> mark_replica_unavailable(
      std::string_view,
      std::string_view,
      std::optional<std::string_view>,
      std::optional<std::string_view>) override {
    return true;
  }

  absl::StatusOr<ReplicaDrainStatus> wait_replica_drain(std::string_view, uint32_t, std::optional<std::string_view>)
      override {
    return ReplicaDrainStatus{.drained = true};
  }
};

StoreEngineOptions make_engine_opts(const std::filesystem::path& temp_root) {
  StoreEngineOptions opts;
  opts.storage_path = temp_root.string();
  opts.memory_pool_size = 32ULL * 1024 * 1024;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.pinned_memory_timeout = std::chrono::milliseconds(0);
  tensorcast::store::MemoryTierConfig tiers;
  tiers.stable_bytes = 1ULL << 20;
  opts.memory_tier_config = tiers;
  return opts;
}

struct ManagerHarness {
  explicit ManagerHarness(const DerivedViewExportManager::Options& options)
      : temp_root(
            std::filesystem::temp_directory_path() /
            absl::StrCat("derived-view-manager-", absl::ToUnixMicros(absl::Now()))),
        engine(std::make_shared<StoreEngine>(make_engine_opts(temp_root))),
        sessions(std::chrono::seconds(60)),
        lip(engine, nullptr),
        lifecycle(sessions, refs, lip),
        manager(*engine, lifecycle, options) {
    std::filesystem::create_directories(temp_root);
    manager.set_global_store_client(std::make_shared<NoopGlobalStoreClient>());
  }

  ~ManagerHarness() {
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
  }

  std::filesystem::path temp_root;
  std::shared_ptr<StoreEngine> engine;
  ReplicaSessionManager sessions;
  RefTracker refs;
  tensorcast::daemon::LipManager lip;
  SessionLifecycleManager lifecycle;
  DerivedViewExportManager manager;
};

ReplicaKey make_view_key(std::string artifact_id, std::string view_id, int device_ordinal = 0, uint32_t replica = 0) {
  return ReplicaKey{
      .artifact_id = std::move(artifact_id),
      .view_id = std::optional<std::string>(std::move(view_id)),
      .device = DeviceRegistry::instance().gpu_key(device_ordinal),
      .replica = replica,
  };
}

std::string make_index_json(int64_t numel) {
  const uint64_t size_bytes = static_cast<uint64_t>(numel) * sizeof(float);
  return absl::StrCat("{\"x\":[0,", size_bytes, ",[", numel, "],[1],\"torch.float32\",0]}");
}

ViewSpec make_full_view_spec(int64_t numel) {
  ViewSpec spec;
  TensorViewOps ops;
  ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 0, .start = 0, .length = static_cast<uint64_t>(numel)}));
  spec.tensors.emplace("x", std::move(ops));
  return spec;
}

ReplicaKey register_view_replica(
    StoreEngine& engine,
    std::string artifact_id,
    std::string view_id,
    int device_ordinal) {
  auto reg = StoreEngine::ArtifactRegistration{};
  reg.artifact_id = artifact_id;
  reg.tensor_index_data = make_index_json(/*numel=*/2);
  reg.schema_version = "v3";
  reg.encoding = "json";
  reg.device_id = device_ordinal;
  reg.total_size_bytes = 2 * sizeof(float);

  ViewRegistration view;
  view.view_id = view_id;
  view.spec = make_full_view_spec(/*numel=*/2);
  view.placement = ViewPlacement::kClient;
  view.canonical_size_bytes = reg.total_size_bytes;
  view.registration_kind = ViewRegistrationKind::kCanonical;
  reg.view = std::move(view);

  auto begin_or = engine.begin_register_artifact(reg);
  REQUIRE(begin_or.ok());
  auto commit_or = engine.commit_registered_artifact(begin_or->registration_id);
  REQUIRE(commit_or.ok());
  REQUIRE(commit_or->view_id.has_value());
  REQUIRE(*commit_or->view_id == view_id);
  return make_view_key(commit_or->artifact_id, std::move(view_id), device_ordinal);
}

DerivedViewExportManager::EntrySnapshot require_entry(DerivedViewExportManager& manager, const ReplicaKey& key) {
  auto snapshot = manager.find_entry(key);
  REQUIRE(snapshot.has_value());
  return *snapshot;
}

} // namespace

TEST_CASE("DerivedViewExportManager reuses identical cache keys", "[daemon][derived_view_export_manager][reuse]") {
  DerivedViewExportManager::Options options;
  options.ttl = absl::Seconds(2);
  options.budget_override_bytes = 1ULL << 20;
  options.headroom_override_bytes = 0;
  ManagerHarness harness(options);

  const ReplicaKey key = register_view_replica(*harness.engine, "artifact-a", "view-a", 0);
  REQUIRE(harness.manager.retain_or_refresh(key).ok());
  const auto first = require_entry(harness.manager, key);

  REQUIRE(harness.manager.retain_or_refresh(key).ok());
  const auto reused = require_entry(harness.manager, key);
  REQUIRE(reused.generation == first.generation);
  REQUIRE(reused.use_lease_id == first.use_lease_id);
  REQUIRE(reused.retention_lease_id == first.retention_lease_id);
  REQUIRE(reused.expiry_time >= first.expiry_time);

  const ReplicaKey different_view = register_view_replica(*harness.engine, "artifact-a", "view-b", 0);
  REQUIRE(harness.manager.retain_or_refresh(different_view).ok());
  const auto different_view_entry = require_entry(harness.manager, different_view);
  REQUIRE(different_view_entry.generation != first.generation);

  const ReplicaKey different_device = register_view_replica(*harness.engine, "artifact-a", "view-a", 1);
  REQUIRE(harness.manager.retain_or_refresh(different_device).ok());
  const auto different_device_entry = require_entry(harness.manager, different_device);
  REQUIRE(different_device_entry.generation != first.generation);
}

TEST_CASE(
    "DerivedViewExportManager refreshes TTL only on data-plane use",
    "[daemon][derived_view_export_manager][ttl]") {
  DerivedViewExportManager::Options options;
  options.ttl = absl::Milliseconds(60);
  options.budget_override_bytes = 1ULL << 20;
  options.headroom_override_bytes = 0;
  ManagerHarness harness(options);

  const ReplicaKey key = register_view_replica(*harness.engine, "artifact-ttl", "view-ttl", 0);
  REQUIRE(harness.manager.retain_or_refresh(key).ok());
  const auto initial = require_entry(harness.manager, key);

  absl::SleepFor(absl::Milliseconds(20));
  const auto after_lookup = require_entry(harness.manager, key);
  REQUIRE(after_lookup.expiry_time == initial.expiry_time);

  absl::SleepFor(absl::Milliseconds(20));
  REQUIRE(harness.manager.begin_fetch(key, "fetch-1").ok());
  const auto during_fetch = require_entry(harness.manager, key);
  REQUIRE(during_fetch.active_fetches == 1);
  REQUIRE(during_fetch.expiry_time > initial.expiry_time);

  harness.manager.end_fetch("fetch-1", "test_complete");
  const auto after_fetch = require_entry(harness.manager, key);
  REQUIRE(after_fetch.active_fetches == 0);
  REQUIRE(after_fetch.expiry_time == during_fetch.expiry_time);

  absl::SleepFor(absl::Milliseconds(30));
  harness.lifecycle.expire_due(absl::Now());
  REQUIRE(harness.manager.find_entry(key).has_value());

  absl::SleepFor(absl::Milliseconds(40));
  harness.lifecycle.expire_due(absl::Now());
  REQUIRE_FALSE(harness.manager.find_entry(key).has_value());
}
