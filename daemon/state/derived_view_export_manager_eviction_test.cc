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
using tensorcast::store::runtime::metadata::ArtifactRegistration;
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
            absl::StrCat("derived-view-eviction-", absl::ToUnixMicros(absl::Now()))),
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

ReplicaKey register_view_replica(StoreEngine& engine, std::string artifact_id, std::string view_id, int64_t numel = 2) {
  ArtifactRegistration reg;
  reg.artifact_id = artifact_id;
  reg.tensor_index_data = make_index_json(numel);
  reg.schema_version = "v3";
  reg.encoding = "json";
  reg.device_id = 0;
  reg.total_size_bytes = static_cast<uint64_t>(numel) * sizeof(float);

  ViewRegistration view;
  view.view_id = view_id;
  view.spec = make_full_view_spec(numel);
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
  return ReplicaKey{
      .artifact_id = commit_or->artifact_id,
      .view_id = std::optional<std::string>(std::move(view_id)),
      .device = DeviceRegistry::instance().gpu_key(0),
      .replica = 0,
  };
}

DerivedViewExportManager::EntrySnapshot require_entry(DerivedViewExportManager& manager, const ReplicaKey& key) {
  auto snapshot = manager.find_entry(key);
  REQUIRE(snapshot.has_value());
  return *snapshot;
}

} // namespace

TEST_CASE(
    "DerivedViewExportManager evicts expired and oldest idle entries before active or pending entries",
    "[daemon][derived_view_export_manager][eviction]") {
  DerivedViewExportManager::Options options;
  options.ttl = absl::Milliseconds(60);
  options.budget_override_bytes = 40;
  options.headroom_override_bytes = 0;
  ManagerHarness harness(options);

  const ReplicaKey expired_idle = register_view_replica(*harness.engine, "artifact-expired", "view-expired");
  REQUIRE(harness.manager.retain_or_refresh(expired_idle).ok());
  absl::SleepFor(absl::Milliseconds(70));

  const ReplicaKey older_idle = register_view_replica(*harness.engine, "artifact-older", "view-older");
  REQUIRE(harness.manager.retain_or_refresh(older_idle).ok());
  absl::SleepFor(absl::Milliseconds(5));

  const ReplicaKey newer_idle = register_view_replica(*harness.engine, "artifact-newer", "view-newer");
  REQUIRE(harness.manager.retain_or_refresh(newer_idle).ok());
  absl::SleepFor(absl::Milliseconds(5));

  const ReplicaKey active_entry = register_view_replica(*harness.engine, "artifact-active", "view-active");
  REQUIRE(harness.manager.retain_or_refresh(active_entry).ok());
  REQUIRE(harness.manager.begin_fetch(active_entry, "active-fetch").ok());

  const ReplicaKey pending_entry{
      .artifact_id = "artifact-pending",
      .view_id = std::optional<std::string>("view-pending"),
      .device = DeviceRegistry::instance().gpu_key(0),
      .replica = 0,
  };
  REQUIRE(harness.manager.reserve(pending_entry, /*reserved_bytes=*/8).ok());
  const auto pending_snapshot = require_entry(harness.manager, pending_entry);
  REQUIRE(pending_snapshot.state == DerivedViewExportManager::EntryState::kPending);

  const ReplicaKey first_trigger = register_view_replica(*harness.engine, "artifact-trigger-1", "view-trigger-1");
  REQUIRE(harness.manager.retain_or_refresh(first_trigger).ok());
  REQUIRE_FALSE(harness.manager.find_entry(expired_idle).has_value());
  REQUIRE(harness.manager.find_entry(older_idle).has_value());
  REQUIRE(harness.manager.find_entry(newer_idle).has_value());
  REQUIRE(harness.manager.find_entry(active_entry).has_value());
  REQUIRE(harness.manager.find_entry(pending_entry).has_value());

  const ReplicaKey second_trigger = register_view_replica(*harness.engine, "artifact-trigger-2", "view-trigger-2");
  REQUIRE(harness.manager.retain_or_refresh(second_trigger).ok());
  REQUIRE_FALSE(harness.manager.find_entry(older_idle).has_value());
  REQUIRE(harness.manager.find_entry(newer_idle).has_value());
  const auto active_snapshot = require_entry(harness.manager, active_entry);
  REQUIRE(active_snapshot.active_fetches == 1);
  REQUIRE(harness.manager.find_entry(pending_entry).has_value());

  harness.manager.end_fetch("active-fetch", "test_complete");
}
