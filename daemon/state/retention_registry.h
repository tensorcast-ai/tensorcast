// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/capability_token.h"
#include "core/store/components/stable_dram_cache_policy.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "daemon/state/session_lifecycle.h"

namespace tensorcast::common::v1 {
class ArtifactSelection;
} // namespace tensorcast::common::v1

namespace tensorcast::store {
class StoreEngine;
} // namespace tensorcast::store

namespace tensorcast::daemon::v2 {
class StorePolicy;
} // namespace tensorcast::daemon::v2

namespace tensorcast::daemon {

class RetentionBackend {
 public:
  struct Target {
    store::loading::ReplicaKey key;
    uint64_t charged_bytes{0};
  };

  struct AdmissionResult {
    bool admitted{false};
    bool skipped{false};
  };

  virtual ~RetentionBackend() = default;

  [[nodiscard]] virtual absl::StatusOr<Target> resolve_target(
      const tensorcast::common::v1::ArtifactSelection& selection) = 0;

  [[nodiscard]] virtual absl::StatusOr<AdmissionResult> admit(
      const Target& target,
      const store::components::StableDramCachePolicy& policy) = 0;

  [[nodiscard]] virtual absl::Status update_policy(
      const Target& target,
      const store::components::StableDramCachePolicy& policy,
      std::optional<absl::Time> retention_deadline) = 0;
};

class RetentionRegistry {
 public:
  struct Options {
    bool enabled{false};
    absl::Duration default_ttl{absl::Minutes(10)};
    absl::Duration max_ttl{absl::Hours(24)};
  };

  struct Handle {
    std::string handle_id;
    uint64_t expires_at_ms{0};
    std::string capability_token;
    uint64_t charged_bytes{0};
    std::string diagnostics;
  };

  RetentionRegistry(
      Options opts,
      std::unique_ptr<RetentionBackend> backend,
      SessionLifecycleManager& lifecycle,
      common::CapabilityTokenManager* capability_tokens,
      std::string daemon_id);

  [[nodiscard]] absl::StatusOr<Handle> acquire(
      const tensorcast::common::v1::ArtifactSelection& selection,
      const v2::StorePolicy* policy,
      uint64_t ttl_ms);

  [[nodiscard]] absl::StatusOr<Handle> renew(std::string_view handle_token, uint64_t extend_ttl_ms);

  [[nodiscard]] absl::StatusOr<bool> release(std::string_view handle_token);

  [[nodiscard]] size_t size() const;

 private:
  struct SelectionKey {
    std::string logical_layout_hash;
    std::string selection_hash;

    bool operator==(const SelectionKey& other) const {
      return logical_layout_hash == other.logical_layout_hash && selection_hash == other.selection_hash;
    }
  };

  struct SelectionKeyHash {
    size_t operator()(const SelectionKey& key) const {
      return absl::HashOf(key.logical_layout_hash, key.selection_hash);
    }
  };

  struct HandleRecord {
    std::string handle_id;
    SelectionKey selection_key;
    store::components::StableDramCachePolicy policy;
    absl::Time expires_at{absl::InfinitePast()};
    uint64_t charged_bytes{0};
    SessionLifecycleManager::LeaseId lease_id{0};
  };

  struct SelectionRecord {
    RetentionBackend::Target target;
    absl::flat_hash_set<std::string> handles;
  };

  struct EffectivePolicy {
    store::components::StableDramCachePolicy policy;
    std::optional<absl::Time> retention_deadline;
  };

  [[nodiscard]] absl::StatusOr<std::string> mint_handle_id_() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  [[nodiscard]] absl::StatusOr<std::string> mint_token_(const std::string& handle_id, absl::Time expires_at) const;

  [[nodiscard]] absl::StatusOr<std::string> handle_id_from_token_(
      std::string_view handle_token,
      bool require_not_expired) const;

  [[nodiscard]] absl::Status expire_handle_(const std::string& handle_id);

  [[nodiscard]] absl::StatusOr<EffectivePolicy> compute_effective_policy_locked_(const SelectionRecord& selection) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  [[nodiscard]] absl::Status apply_effective_policy_(const SelectionRecord& selection, const EffectivePolicy& policy)
      const;

  void record_handle_delta_(int64_t delta) const;
  void record_bytes_delta_(int64_t delta) const;
  void record_expiration_() const;

  Options opts_;
  std::unique_ptr<RetentionBackend> backend_;
  SessionLifecycleManager* lifecycle_;
  common::CapabilityTokenManager* capability_tokens_;
  std::string daemon_id_;

  mutable absl::Mutex mu_;
  absl::BitGen bitgen_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, HandleRecord> handles_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<SelectionKey, SelectionRecord, SelectionKeyHash> selections_ ABSL_GUARDED_BY(mu_);
};

std::unique_ptr<RetentionBackend> make_store_engine_retention_backend(store::StoreEngine& engine);

} // namespace tensorcast::daemon
