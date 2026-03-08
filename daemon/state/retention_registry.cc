// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/retention_registry.h"

#include <algorithm>
#include <cstring>
#include <format>

#include "absl/log/log.h"
#include "absl/strings/escaping.h"
#include "absl/time/clock.h"
#include "core/store/device_types.h"
#include "core/store/store_engine.h"
#include "daemon/state/store_policy_resolver.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"
#include "tensorcast/common/v1/capability_token.pb.h"
#include "tensorcast/common/v1/common.pb.h"

namespace tensorcast::daemon {
namespace {

constexpr absl::string_view kMetricsScope = "tensorcast.daemon";

uint64_t to_unix_ms(absl::Time t) {
  return static_cast<uint64_t>(absl::ToUnixMillis(t));
}

int retention_rank(store::components::StableRetentionPolicy policy) {
  return static_cast<int>(policy);
}

int overflow_rank(store::components::StableOverflowPolicy policy) {
  return static_cast<int>(policy);
}

store::components::StableDramCachePolicy best_effort_policy() {
  store::components::StableDramCachePolicy policy;
  policy.retention_policy = store::components::StableRetentionPolicy::kBestEffort;
  policy.overflow_policy = store::components::StableOverflowPolicy::kEvict;
  policy.required = false;
  policy.require_shared_disk_for_spill = false;
  policy.require_remote_stable_for_spill = false;
  policy.retention_ttl = std::nullopt;
  return policy;
}

class StoreEngineRetentionBackend final : public RetentionBackend {
 public:
  explicit StoreEngineRetentionBackend(store::StoreEngine& engine) : engine_(&engine) {}

  absl::StatusOr<Target> resolve_target(const tensorcast::common::v1::ArtifactSelection& selection) override {
    if (selection.artifact_id().empty()) {
      return absl::InvalidArgumentError("selection.artifact_id is required");
    }
    store::loading::ReplicaKey key;
    key.artifact_id = selection.artifact_id();
    if (!selection.view_id().empty()) {
      key.view_id = selection.view_id();
    }
    key.device = store::DeviceKey{.type = tensorcast::DeviceType::CPU, .ordinal = -1};
    key.replica = 0;
    auto size_or = engine_->get_replica_size(key);
    if (!size_or.ok()) {
      return size_or.status();
    }
    return Target{.key = key, .charged_bytes = *size_or};
  }

  absl::StatusOr<AdmissionResult> admit(const Target& target, const store::components::StableDramCachePolicy& policy)
      override {
    auto admit_or = engine_->admit_stable_cache_policy(target.key, policy);
    if (!admit_or.ok()) {
      return admit_or.status();
    }
    return AdmissionResult{.admitted = admit_or->admitted, .skipped = admit_or->skipped};
  }

  absl::Status update_policy(
      const Target& target,
      const store::components::StableDramCachePolicy& policy,
      std::optional<absl::Time> retention_deadline) override {
    return engine_->update_stable_cache_policy(target.key, policy, retention_deadline);
  }

 private:
  store::StoreEngine* engine_;
};

} // namespace

RetentionRegistry::RetentionRegistry(
    Options opts,
    std::unique_ptr<RetentionBackend> backend,
    SessionLifecycleManager& lifecycle,
    common::CapabilityTokenManager* capability_tokens,
    std::string daemon_id)
    : opts_(std::move(opts)),
      backend_(std::move(backend)),
      lifecycle_(&lifecycle),
      capability_tokens_(capability_tokens),
      daemon_id_(std::move(daemon_id)) {}

size_t RetentionRegistry::size() const {
  absl::MutexLock lock(&mu_);
  return handles_.size();
}

absl::StatusOr<std::string> RetentionRegistry::mint_handle_id_() {
  std::string token;
  token.resize(32);
  for (size_t i = 0; i < token.size(); ++i) {
    token[i] = static_cast<char>(absl::Uniform<uint32_t>(bitgen_, 0u, 256u));
  }
  return absl::BytesToHexString(token);
}

absl::StatusOr<std::string> RetentionRegistry::mint_token_(const std::string& handle_id, absl::Time expires_at) const {
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are not configured");
  }
  tensorcast::common::v1::RetentionHandleScope scope;
  scope.set_handle_id(handle_id);
  auto scope_or = common::CapabilityTokenManager::serialize_scope_deterministic(scope);
  if (!scope_or.ok()) {
    return scope_or.status();
  }
  const uint64_t expires_at_ms = to_unix_ms(expires_at);
  return capability_tokens_->mint(
      daemon_id_, tensorcast::common::v1::CAPABILITY_AUDIENCE_RETENTION_HANDLE, *scope_or, expires_at_ms);
}

absl::StatusOr<std::string> RetentionRegistry::handle_id_from_token_(
    std::string_view handle_token,
    bool require_not_expired) const {
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are not configured");
  }
  auto env_or = capability_tokens_->verify(
      handle_token,
      tensorcast::common::v1::CAPABILITY_AUDIENCE_RETENTION_HANDLE,
      daemon_id_,
      absl::Now(),
      require_not_expired);
  if (!env_or.ok()) {
    return env_or.status();
  }
  tensorcast::common::v1::RetentionHandleScope scope;
  if (!scope.ParseFromString(env_or->scope())) {
    return absl::InvalidArgumentError("invalid retention handle scope");
  }
  if (scope.handle_id().empty()) {
    return absl::InvalidArgumentError("retention handle scope missing handle_id");
  }
  return scope.handle_id();
}

absl::StatusOr<RetentionRegistry::EffectivePolicy> RetentionRegistry::compute_effective_policy_locked_(
    const SelectionRecord& selection) const {
  EffectivePolicy effective;
  effective.policy = best_effort_policy();
  effective.retention_deadline = std::nullopt;
  if (selection.handles.empty()) {
    return effective;
  }

  std::optional<std::chrono::milliseconds> max_ttl;
  absl::Time max_expiry = absl::InfinitePast();

  for (const auto& handle_id : selection.handles) {
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      continue;
    }
    const auto& rec = it->second;
    if (retention_rank(rec.policy.retention_policy) > retention_rank(effective.policy.retention_policy)) {
      effective.policy.retention_policy = rec.policy.retention_policy;
    }
    if (overflow_rank(rec.policy.overflow_policy) > overflow_rank(effective.policy.overflow_policy)) {
      effective.policy.overflow_policy = rec.policy.overflow_policy;
    }
    effective.policy.required = effective.policy.required || rec.policy.required;
    effective.policy.require_shared_disk_for_spill =
        effective.policy.require_shared_disk_for_spill || rec.policy.require_shared_disk_for_spill;
    effective.policy.require_remote_stable_for_spill =
        effective.policy.require_remote_stable_for_spill || rec.policy.require_remote_stable_for_spill;

    if (rec.policy.retention_policy == store::components::StableRetentionPolicy::kTtl) {
      if (rec.policy.retention_ttl.has_value()) {
        if (!max_ttl.has_value() || *rec.policy.retention_ttl > *max_ttl) {
          max_ttl = rec.policy.retention_ttl;
        }
      }
      if (rec.expires_at > max_expiry) {
        max_expiry = rec.expires_at;
      }
    }
  }

  if (effective.policy.retention_policy == store::components::StableRetentionPolicy::kTtl) {
    effective.policy.retention_ttl = max_ttl;
    if (max_expiry != absl::InfinitePast()) {
      effective.retention_deadline = max_expiry;
    }
  } else {
    effective.policy.retention_ttl = std::nullopt;
    effective.retention_deadline = std::nullopt;
  }

  return effective;
}

absl::Status RetentionRegistry::apply_effective_policy_(const SelectionRecord& selection, const EffectivePolicy& policy)
    const {
  if (!backend_) {
    return absl::FailedPreconditionError("retention backend unavailable");
  }
  return backend_->update_policy(selection.target, policy.policy, policy.retention_deadline);
}

absl::Status RetentionRegistry::expire_handle_(const std::string& handle_id) {
  std::optional<SelectionKey> selection_key;
  std::optional<SelectionRecord> selection_record;
  std::optional<EffectivePolicy> effective_policy;
  absl::Status effective_status = absl::OkStatus();
  bool removed = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return absl::OkStatus();
    }
    selection_key = it->second.selection_key;
    auto sel_it = selections_.find(*selection_key);
    if (sel_it != selections_.end()) {
      sel_it->second.handles.erase(handle_id);
      selection_record = sel_it->second;
      if (sel_it->second.handles.empty()) {
        selections_.erase(sel_it);
      }
    }
    record_handle_delta_(-1);
    record_bytes_delta_(-static_cast<int64_t>(it->second.charged_bytes));
    handles_.erase(it);
    if (selection_record.has_value()) {
      auto effective_or = compute_effective_policy_locked_(*selection_record);
      if (effective_or.ok()) {
        effective_policy = *effective_or;
      } else {
        effective_status = effective_or.status();
      }
    }
    removed = true;
  }

  if (removed) {
    record_expiration_();
  }

  if (selection_record.has_value()) {
    if (effective_policy.has_value()) {
      auto st = apply_effective_policy_(*selection_record, *effective_policy);
      if (!st.ok()) {
        LOG(WARNING) << "retention: failed to apply downgraded policy after expiry: " << st;
      }
    } else if (!effective_status.ok()) {
      LOG(WARNING) << "retention: failed to compute downgraded policy after expiry: " << effective_status;
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<RetentionRegistry::Handle> RetentionRegistry::acquire(
    const tensorcast::common::v1::ArtifactSelection& selection,
    const v2::StorePolicy* policy,
    uint64_t ttl_ms) {
  if (!opts_.enabled) {
    return absl::FailedPreconditionError("retention handles are disabled");
  }
  if (!backend_) {
    return absl::FailedPreconditionError("retention backend unavailable");
  }
  if (capability_tokens_ == nullptr || !capability_tokens_->configured()) {
    return absl::FailedPreconditionError("capability tokens are not configured");
  }
  if (selection.artifact_id().empty()) {
    return absl::InvalidArgumentError("selection.artifact_id is required");
  }
  if (selection.logical_layout_hash().empty() || selection.selection_hash().empty()) {
    return absl::InvalidArgumentError("selection logical_layout_hash and selection_hash are required");
  }
  auto selection_identity_or = common::build_selection_identity(selection);
  if (!selection_identity_or.ok()) {
    return selection_identity_or.status();
  }
  if (ttl_ms == 0 && opts_.default_ttl <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("ttl_ms is required");
  }
  absl::Duration ttl = ttl_ms > 0 ? absl::Milliseconds(static_cast<int64_t>(ttl_ms)) : opts_.default_ttl;
  if (ttl <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("ttl_ms must be > 0");
  }
  if (opts_.max_ttl > absl::ZeroDuration() && ttl > opts_.max_ttl) {
    ttl = opts_.max_ttl;
  }

  auto resolved_or = resolve_store_policy(policy);
  if (!resolved_or.ok()) {
    return resolved_or.status();
  }
  auto stable_policy_opt = stable_cache_policy_from_resolved(*resolved_or);
  if (!stable_policy_opt.has_value()) {
    return absl::FailedPreconditionError("retention requires local stable_dram policy");
  }
  auto stable_policy = *stable_policy_opt;
  if (stable_policy.retention_policy == store::components::StableRetentionPolicy::kTtl &&
      stable_policy.retention_ttl.has_value()) {
    const absl::Duration policy_ttl = absl::Milliseconds(stable_policy.retention_ttl->count());
    if (policy_ttl > absl::ZeroDuration() && ttl > policy_ttl) {
      ttl = policy_ttl;
    }
  }
  if (ttl <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("effective ttl_ms must be > 0");
  }

  const absl::Time expires_at = absl::Now() + ttl;

  auto target_or = backend_->resolve_target(selection);
  if (!target_or.ok()) {
    return target_or.status();
  }
  const auto target = *target_or;

  std::string handle_id;
  {
    absl::MutexLock lock(&mu_);
    auto id_or = mint_handle_id_();
    if (!id_or.ok()) {
      return absl::Status(
          id_or.status().code(), std::format("retention: failed to mint handle id: {}", id_or.status().message()));
    }
    handle_id = *id_or;
  }

  auto lease_or = lifecycle_->create_retention_lease(
      ttl,
      std::vector<std::function<absl::Status()>>{
          [this, handle_id]() -> absl::Status { return this->expire_handle_(handle_id); },
      });
  if (!lease_or.ok()) {
    return absl::Status(
        lease_or.status().code(),
        std::format("retention: failed to create retention lease: {}", lease_or.status().message()));
  }

  auto token_or = mint_token_(handle_id, expires_at);
  if (!token_or.ok()) {
    lifecycle_->release_lease(*lease_or);
    return absl::Status(
        token_or.status().code(),
        std::format("retention: failed to mint capability token: {}", token_or.status().message()));
  }

  auto admit_or = backend_->admit(target, stable_policy);
  if (!admit_or.ok()) {
    lifecycle_->release_lease(*lease_or);
    return absl::Status(
        admit_or.status().code(),
        std::format("retention: stable cache admission failed: {}", admit_or.status().message()));
  }

  SelectionKey selection_key{
      .selection_identity = *selection_identity_or,
  };

  SelectionRecord selection_record;
  std::optional<EffectivePolicy> effective_policy;
  absl::Status effective_status = absl::OkStatus();
  {
    absl::MutexLock lock(&mu_);
    auto& record = selections_[selection_key];
    if (record.handles.empty()) {
      record.target = target;
    }
    record.handles.insert(handle_id);
    handles_[handle_id] = HandleRecord{
        .handle_id = handle_id,
        .selection_key = selection_key,
        .policy = stable_policy,
        .expires_at = expires_at,
        .charged_bytes = target.charged_bytes,
        .lease_id = *lease_or,
    };
    selection_record = record;
    record_handle_delta_(1);
    record_bytes_delta_(static_cast<int64_t>(target.charged_bytes));
    auto effective_or = compute_effective_policy_locked_(selection_record);
    if (effective_or.ok()) {
      effective_policy = *effective_or;
    } else {
      effective_status = effective_or.status();
    }
  }

  if (effective_policy.has_value()) {
    auto st = apply_effective_policy_(selection_record, *effective_policy);
    if (!st.ok()) {
      LOG(WARNING) << "retention: failed to apply effective policy: " << st;
    }
  } else if (!effective_status.ok()) {
    LOG(WARNING) << "retention: failed to compute effective policy: " << effective_status;
  }

  Handle out;
  out.handle_id = handle_id;
  out.expires_at_ms = to_unix_ms(expires_at);
  out.capability_token = *token_or;
  out.charged_bytes = target.charged_bytes;
  if (!admit_or->admitted && admit_or->skipped) {
    out.diagnostics = "stable_cache_skipped";
  }
  return out;
}

absl::StatusOr<RetentionRegistry::Handle> RetentionRegistry::renew(
    std::string_view handle_token,
    uint64_t extend_ttl_ms) {
  if (!opts_.enabled) {
    return absl::FailedPreconditionError("retention handles are disabled");
  }
  if (extend_ttl_ms == 0) {
    return absl::InvalidArgumentError("extend_ttl_ms is required");
  }

  auto handle_id_or = handle_id_from_token_(handle_token, /*require_not_expired=*/true);
  if (!handle_id_or.ok()) {
    return handle_id_or.status();
  }
  const std::string handle_id = *handle_id_or;

  HandleRecord record;
  SelectionRecord selection_record;
  {
    absl::MutexLock lock(&mu_);
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return absl::FailedPreconditionError("retention handle not found");
    }
    record = it->second;
    auto sel_it = selections_.find(record.selection_key);
    if (sel_it == selections_.end()) {
      return absl::FailedPreconditionError("retention handle selection missing");
    }
    selection_record = sel_it->second;
  }

  absl::Duration ttl = absl::Milliseconds(static_cast<int64_t>(extend_ttl_ms));
  if (ttl <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("extend_ttl_ms must be > 0");
  }
  if (opts_.max_ttl > absl::ZeroDuration() && ttl > opts_.max_ttl) {
    ttl = opts_.max_ttl;
  }
  if (record.policy.retention_policy == store::components::StableRetentionPolicy::kTtl &&
      record.policy.retention_ttl.has_value()) {
    const absl::Duration policy_ttl = absl::Milliseconds(record.policy.retention_ttl->count());
    if (policy_ttl > absl::ZeroDuration() && ttl > policy_ttl) {
      ttl = policy_ttl;
    }
  }
  if (ttl <= absl::ZeroDuration()) {
    return absl::InvalidArgumentError("effective ttl_ms must be > 0");
  }

  auto admit_or = backend_->admit(selection_record.target, record.policy);
  if (!admit_or.ok()) {
    return admit_or.status();
  }

  auto st = lifecycle_->renew_retention(record.lease_id, ttl);
  if (!st.ok()) {
    return st;
  }

  const absl::Time expires_at = absl::Now() + ttl;
  auto token_or = mint_token_(handle_id, expires_at);
  if (!token_or.ok()) {
    return token_or.status();
  }

  SelectionRecord updated_selection;
  std::optional<EffectivePolicy> effective_policy;
  absl::Status effective_status = absl::OkStatus();
  bool selection_found = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return absl::FailedPreconditionError("retention handle not found");
    }
    it->second.expires_at = expires_at;
    auto sel_it = selections_.find(it->second.selection_key);
    if (sel_it != selections_.end()) {
      updated_selection = sel_it->second;
      selection_found = true;
    }
    if (selection_found) {
      auto effective_or = compute_effective_policy_locked_(updated_selection);
      if (effective_or.ok()) {
        effective_policy = *effective_or;
      } else {
        effective_status = effective_or.status();
      }
    }
  }

  if (selection_found) {
    if (effective_policy.has_value()) {
      auto policy_st = apply_effective_policy_(updated_selection, *effective_policy);
      if (!policy_st.ok()) {
        LOG(WARNING) << "retention: failed to apply renewed policy: " << policy_st;
      }
    } else if (!effective_status.ok()) {
      LOG(WARNING) << "retention: failed to compute renewed policy: " << effective_status;
    }
  }

  Handle out;
  out.handle_id = handle_id;
  out.expires_at_ms = to_unix_ms(expires_at);
  out.capability_token = *token_or;
  out.charged_bytes = record.charged_bytes;
  return out;
}

absl::StatusOr<bool> RetentionRegistry::release(std::string_view handle_token) {
  if (!opts_.enabled) {
    return absl::FailedPreconditionError("retention handles are disabled");
  }

  auto handle_id_or = handle_id_from_token_(handle_token, /*require_not_expired=*/false);
  if (!handle_id_or.ok()) {
    return handle_id_or.status();
  }
  const std::string handle_id = *handle_id_or;

  std::optional<SelectionRecord> selection_record;
  std::optional<EffectivePolicy> effective_policy;
  absl::Status effective_status = absl::OkStatus();
  SessionLifecycleManager::LeaseId lease_id = 0;
  bool removed = false;
  {
    absl::MutexLock lock(&mu_);
    auto it = handles_.find(handle_id);
    if (it == handles_.end()) {
      return false;
    }
    lease_id = it->second.lease_id;
    auto sel_it = selections_.find(it->second.selection_key);
    if (sel_it != selections_.end()) {
      sel_it->second.handles.erase(handle_id);
      selection_record = sel_it->second;
      if (sel_it->second.handles.empty()) {
        selections_.erase(sel_it);
      }
    }
    record_handle_delta_(-1);
    record_bytes_delta_(-static_cast<int64_t>(it->second.charged_bytes));
    handles_.erase(it);
    removed = true;
    if (selection_record.has_value()) {
      auto effective_or = compute_effective_policy_locked_(*selection_record);
      if (effective_or.ok()) {
        effective_policy = *effective_or;
      } else {
        effective_status = effective_or.status();
      }
    }
  }

  if (lease_id != 0) {
    lifecycle_->release_lease(lease_id);
  }

  if (selection_record.has_value()) {
    if (effective_policy.has_value()) {
      auto st = apply_effective_policy_(*selection_record, *effective_policy);
      if (!st.ok()) {
        LOG(WARNING) << "retention: failed to apply downgraded policy: " << st;
      }
    } else if (!effective_status.ok()) {
      LOG(WARNING) << "retention: failed to compute downgraded policy: " << effective_status;
    }
  }

  return removed;
}

void RetentionRegistry::record_handle_delta_(int64_t delta) const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateInt64UpDownCounter("tc_retention_handles_active");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(delta, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void RetentionRegistry::record_bytes_delta_(int64_t delta) const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateInt64UpDownCounter("tc_retention_bytes_charged");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(delta, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

void RetentionRegistry::record_expiration_() const {
  try {
    auto meter = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter(std::string(kMetricsScope));
    static auto counter = meter->CreateUInt64Counter("tc_retention_handles_expired_total");
    std::map<std::string, opentelemetry::common::AttributeValue> attrs;
    counter->Add(1, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
  } catch (...) {
  }
}

std::unique_ptr<RetentionBackend> make_store_engine_retention_backend(store::StoreEngine& engine) {
  return std::make_unique<StoreEngineRetentionBackend>(engine);
}

} // namespace tensorcast::daemon
