// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/retention_registry.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/device_types.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"
#include "daemon/state/ipc_region_registry.h"
#include "daemon/state/lip_manager.h"
#include "daemon/state/ref_tracker.h"
#include "daemon/state/replica_session_manager.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

namespace store = tensorcast::store;

std::filesystem::path test_tmpdir() {
  const char* env = std::getenv("TEST_TMPDIR");
  if (env && *env) {
    return std::filesystem::path(env);
  }
  return std::filesystem::temp_directory_path() / "tensorcast_retention_registry_test";
}

tensorcast::store::StoreEngineOptions make_opts() {
  tensorcast::store::StoreEngineOptions opts;
  opts.storage_path = (test_tmpdir() / "engine").string();
  std::filesystem::create_directories(opts.storage_path);
  opts.p2p_port = 47022;
  opts.memory_pool_size = 32ULL << 20;
  opts.tx_slice_bytes = 1ULL << 20;
  opts.num_thread = 2;
  opts.global_store_address.clear();
  return opts;
}

class FakeRetentionBackend final : public tensorcast::daemon::RetentionBackend {
 public:
  Target target;
  store::components::StableDramCachePolicy last_policy{};
  std::optional<absl::Time> last_deadline{};
  std::string last_target_artifact_id;
  int update_calls{0};

  absl::StatusOr<Target> resolve_target(const tensorcast::common::v1::ArtifactSelection& selection) override {
    Target resolved = target;
    if (resolved.key.artifact_id.empty()) {
      resolved.key.artifact_id = selection.artifact_id();
    }
    return resolved;
  }

  absl::StatusOr<AdmissionResult> admit(const Target&, const store::components::StableDramCachePolicy&) override {
    return AdmissionResult{.admitted = true, .skipped = false};
  }

  absl::Status update_policy(
      const Target& target,
      const store::components::StableDramCachePolicy& policy,
      std::optional<absl::Time> retention_deadline) override {
    last_target_artifact_id = target.key.artifact_id;
    last_policy = policy;
    last_deadline = retention_deadline;
    ++update_calls;
    return absl::OkStatus();
  }
};

} // namespace

TEST_CASE("RetentionRegistry downgrades policy on last release", "[daemon][retention]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::ReplicaSessionManager sessions(std::chrono::seconds(60));
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry region_registry(
      tensorcast::daemon::IpcRegionRegistry::Options{.capacity = 16, .max_ttl = absl::Minutes(10)});
  tensorcast::daemon::LipManager lip_manager(engine, &region_registry);
  tensorcast::daemon::SessionLifecycleManager lifecycle(sessions, refs, lip_manager, *engine);
  tensorcast::daemon::LifecycleKernel lifecycle_kernel("daemon-test");

  auto backend = std::make_unique<FakeRetentionBackend>();
  backend->target.key = tensorcast::store::loading::ReplicaKey{
      .artifact_id = "mi2:retention:artifact",
      .device = tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1},
      .replica = 0,
  };
  backend->target.charged_bytes = 4096;
  FakeRetentionBackend* backend_ptr = backend.get();

  tensorcast::common::CapabilityTokenManager token_mgr(
      tensorcast::common::CapabilityTokenConfig{
          .active = tensorcast::common::CapabilityTokenKey{.version = 1, .secret = "retention_secret"},
          .previous = {},
      });

  tensorcast::daemon::RetentionRegistry::Options opts;
  opts.enabled = true;
  opts.default_ttl = absl::Seconds(5);
  opts.max_ttl = absl::Minutes(5);

  tensorcast::daemon::RetentionRegistry registry(
      opts, std::move(backend), lifecycle, lifecycle_kernel, &token_mgr, "daemon-test");

  tensorcast::common::v1::ArtifactSelection selection;
  selection.set_artifact_id("mi2:retention:artifact");
  selection.set_logical_layout_hash("logical_hash");
  selection.set_selection_hash("selection_hash");

  tensorcast::daemon::v2::StorePolicy pinned_policy;
  pinned_policy.set_profile(tensorcast::daemon::v2::POLICY_PROFILE_PINNED);

  auto best_effort_or = registry.acquire(selection, nullptr, 1000);
  REQUIRE(best_effort_or.ok());
  CHECK(best_effort_or->charged_bytes == 4096);

  auto pinned_or = registry.acquire(selection, &pinned_policy, 2000);
  REQUIRE(pinned_or.ok());
  CHECK(backend_ptr->last_policy.retention_policy == tensorcast::store::components::StableRetentionPolicy::kPinned);

  auto release_pinned = registry.release(pinned_or->capability_token);
  REQUIRE(release_pinned.ok());
  CHECK(*release_pinned);
  CHECK(backend_ptr->last_policy.retention_policy == tensorcast::store::components::StableRetentionPolicy::kBestEffort);

  auto release_best_effort = registry.release(best_effort_or->capability_token);
  REQUIRE(release_best_effort.ok());
  CHECK(*release_best_effort);
  CHECK(backend_ptr->last_policy.retention_policy == tensorcast::store::components::StableRetentionPolicy::kBestEffort);
}

TEST_CASE("RetentionRegistry classifies invalid tokens", "[daemon][retention]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::ReplicaSessionManager sessions(std::chrono::seconds(60));
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry region_registry(
      tensorcast::daemon::IpcRegionRegistry::Options{.capacity = 16, .max_ttl = absl::Minutes(10)});
  tensorcast::daemon::LipManager lip_manager(engine, &region_registry);
  tensorcast::daemon::SessionLifecycleManager lifecycle(sessions, refs, lip_manager, *engine);
  tensorcast::daemon::LifecycleKernel lifecycle_kernel("daemon-test");

  auto backend = std::make_unique<FakeRetentionBackend>();
  backend->target.key = tensorcast::store::loading::ReplicaKey{
      .artifact_id = "mi2:retention:invalid-token",
      .device = tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1},
      .replica = 0,
  };
  backend->target.charged_bytes = 1024;

  tensorcast::common::CapabilityTokenManager token_mgr(
      tensorcast::common::CapabilityTokenConfig{
          .active = tensorcast::common::CapabilityTokenKey{.version = 1, .secret = "retention_secret"},
          .previous = {},
      });

  tensorcast::daemon::RetentionRegistry::Options opts;
  opts.enabled = true;
  opts.default_ttl = absl::Seconds(5);
  opts.max_ttl = absl::Minutes(5);

  tensorcast::daemon::RetentionRegistry registry(
      opts, std::move(backend), lifecycle, lifecycle_kernel, &token_mgr, "daemon-test");

  auto invalid = registry.renew("not-a-token", 1000);
  REQUIRE_FALSE(invalid.ok());
  CHECK((absl::IsInvalidArgument(invalid.status()) || absl::IsFailedPrecondition(invalid.status())));

  tensorcast::common::v1::RetentionHandleScope scope;
  scope.set_handle_id("missing-handle");
  auto scope_or = tensorcast::common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  REQUIRE(scope_or.ok());
  const uint64_t expired_ms = static_cast<uint64_t>(absl::ToUnixMillis(absl::Now() - absl::Seconds(1)));
  auto token_or = token_mgr.mint(
      "daemon-test", tensorcast::common::v1::CAPABILITY_AUDIENCE_RETENTION_HANDLE, *scope_or, expired_ms);
  REQUIRE(token_or.ok());

  auto expired = registry.renew(*token_or, 1000);
  REQUIRE_FALSE(expired.ok());
  CHECK(absl::IsPermissionDenied(expired.status()));
}

TEST_CASE("RetentionRegistry keys selections by typed SelectionIdentity", "[daemon][retention]") {
  auto engine = std::make_shared<tensorcast::store::StoreEngine>(make_opts());
  tensorcast::daemon::ReplicaSessionManager sessions(std::chrono::seconds(60));
  tensorcast::daemon::RefTracker refs;
  tensorcast::daemon::IpcRegionRegistry region_registry(
      tensorcast::daemon::IpcRegionRegistry::Options{.capacity = 16, .max_ttl = absl::Minutes(10)});
  tensorcast::daemon::LipManager lip_manager(engine, &region_registry);
  tensorcast::daemon::SessionLifecycleManager lifecycle(sessions, refs, lip_manager, *engine);
  tensorcast::daemon::LifecycleKernel lifecycle_kernel("daemon-test");

  auto backend = std::make_unique<FakeRetentionBackend>();
  backend->target.key = tensorcast::store::loading::ReplicaKey{
      .artifact_id = "",
      .device = tensorcast::store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1},
      .replica = 0,
  };
  backend->target.charged_bytes = 2048;
  FakeRetentionBackend* backend_ptr = backend.get();

  tensorcast::common::CapabilityTokenManager token_mgr(
      tensorcast::common::CapabilityTokenConfig{
          .active = tensorcast::common::CapabilityTokenKey{.version = 1, .secret = "retention_secret"},
          .previous = {},
      });

  tensorcast::daemon::RetentionRegistry::Options opts;
  opts.enabled = true;
  opts.default_ttl = absl::Seconds(5);
  opts.max_ttl = absl::Minutes(5);

  tensorcast::daemon::RetentionRegistry registry(
      opts, std::move(backend), lifecycle, lifecycle_kernel, &token_mgr, "daemon-test");

  tensorcast::common::v1::ArtifactSelection selection_a;
  selection_a.set_artifact_id("mi2:retention:artifact_a");
  selection_a.set_logical_layout_hash("logical_hash");
  selection_a.set_selection_hash("selection_hash");

  tensorcast::common::v1::ArtifactSelection selection_b;
  selection_b.set_artifact_id("mi2:retention:artifact_b");
  selection_b.set_logical_layout_hash("logical_hash");
  selection_b.set_selection_hash("selection_hash");

  tensorcast::daemon::v2::StorePolicy pinned_policy;
  pinned_policy.set_profile(tensorcast::daemon::v2::POLICY_PROFILE_PINNED);

  auto handle_a_or = registry.acquire(selection_a, nullptr, 1000);
  REQUIRE(handle_a_or.ok());
  CHECK(backend_ptr->last_target_artifact_id == "mi2:retention:artifact_a");

  auto handle_b_or = registry.acquire(selection_b, &pinned_policy, 1000);
  REQUIRE(handle_b_or.ok());
  CHECK(backend_ptr->last_target_artifact_id == "mi2:retention:artifact_b");

  auto release_b_or = registry.release(handle_b_or->capability_token);
  REQUIRE(release_b_or.ok());
  CHECK(*release_b_or);
  CHECK(backend_ptr->last_target_artifact_id == "mi2:retention:artifact_b");
}
