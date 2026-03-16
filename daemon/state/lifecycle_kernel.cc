// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/lifecycle_kernel.h"

#include <utility>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {

namespace {

bool epochs_match(const LifecycleEpochs& lhs, const LifecycleEpochs& rhs) {
  return lhs == rhs;
}

bool route_principal_is_local_issuer(
    const LifecycleRoutePrincipal& route_principal,
    std::string_view issuer_daemon_id) {
  return route_principal.principal_kind == LifecycleRoutePrincipalKind::kIssuerDaemon &&
      route_principal.principal_id == issuer_daemon_id;
}

} // namespace

LifecycleKernel::LifecycleKernel(std::string issuer_daemon_id) : issuer_daemon_id_(std::move(issuer_daemon_id)) {}

std::string LifecycleKernel::address_index_key(const CapabilityBindingAddress& address) {
  return absl::StrCat(
      static_cast<int>(address.route_principal.principal_kind),
      "|",
      address.route_principal.principal_id,
      "|",
      static_cast<int>(address.family),
      "|",
      static_cast<int>(address.binding_space),
      "|",
      static_cast<int>(address.binding_key_kind),
      "|",
      address.binding_key);
}

absl::Status LifecycleKernel::validate_mint_request_locked_(const MintCapabilityRequest& request) const {
  if (request.capability_id.empty()) {
    return absl::InvalidArgumentError("capability_id is required");
  }
  if (request.subject.subject_id.empty()) {
    return absl::InvalidArgumentError("subject.subject_id is required");
  }
  if (request.address.route_principal.principal_id.empty()) {
    return absl::InvalidArgumentError("address.route_principal.principal_id is required");
  }
  if (!route_principal_is_local_issuer(request.address.route_principal, issuer_daemon_id_)) {
    return absl::FailedPreconditionError("lifecycle kernel only mints issuer-local capabilities");
  }
  if (request.address.binding_key.empty()) {
    return absl::InvalidArgumentError("address.binding_key is required");
  }
  if (!epochs_match(request.subject.epochs, request.address.epochs)) {
    return absl::InvalidArgumentError("subject epochs must match binding address epochs");
  }
  if (request.capability_expires_at == absl::InfinitePast()) {
    return absl::InvalidArgumentError("capability_expires_at is required");
  }
  if (request.binding_mode == LifecycleBindingMode::kBindingRecord) {
    const std::string binding_id =
        request.binding_id.has_value() ? *request.binding_id : request.address.binding_id.value_or(std::string{});
    if (binding_id.empty()) {
      return absl::InvalidArgumentError("binding_record mode requires binding_id");
    }
    if (!request.credential_expires_at.has_value()) {
      return absl::InvalidArgumentError("binding_record mode requires credential_expires_at");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<LifecycleCapabilityRecord> LifecycleKernel::mint_capability(const MintCapabilityRequest& request) {
  absl::MutexLock lock(&mu_);
  auto st = validate_mint_request_locked_(request);
  if (!st.ok()) {
    return st;
  }
  if (capabilities_by_id_.contains(request.capability_id)) {
    return absl::AlreadyExistsError("capability_id already exists");
  }
  const std::string address_key = address_index_key(request.address);
  if (request.binding_mode == LifecycleBindingMode::kAddressDerived &&
      address_to_capability_id_.contains(address_key)) {
    return absl::AlreadyExistsError("binding address already exists");
  }

  CapabilityEntry entry;
  entry.subject = request.subject;
  if (entry.subject.created_at == absl::InfinitePast()) {
    entry.subject.created_at = absl::Now();
  }
  entry.subject.last_observed_at = entry.subject.created_at;
  entry.address = request.address;
  entry.carriage_kind = request.carriage_kind;
  entry.binding_mode = request.binding_mode;
  entry.binding_id = request.binding_id.has_value() ? request.binding_id : request.address.binding_id;

  entry.capability.capability_id = request.capability_id;
  entry.capability.family = request.address.family;
  entry.capability.front_door_kind = request.front_door_kind;
  entry.capability.subject_id = entry.subject.subject_id;
  entry.capability.epochs = request.address.epochs;
  entry.capability.lease_id = request.lease_id;
  entry.capability.lease_generation_or_equivalent = request.lease_generation_or_equivalent;
  entry.capability.issuer_daemon_id = issuer_daemon_id_;
  entry.capability.issued_at = absl::Now();
  entry.capability.capability_expires_at = request.capability_expires_at;
  entry.capability.state = LifecycleRecordState::kActive;
  entry.capability.workflow_companion = request.workflow_companion;
  entry.capability.holder_scope = request.holder_scope;
  entry.capability.resolution_hints = request.resolution_hints;
  entry.capability.direction = request.direction;
  entry.capability.local_only = request.local_only;
  entry.capability.workflow_gate = request.workflow_gate;
  entry.capability.constraint_claims = request.constraint_claims;

  capabilities_by_id_.emplace(request.capability_id, entry);
  if (request.binding_mode == LifecycleBindingMode::kAddressDerived) {
    address_to_capability_id_[address_key] = request.capability_id;
  } else {
    LifecycleBindingRecord binding;
    binding.binding_id = entry.binding_id.value_or("");
    binding.capability_id = request.capability_id;
    binding.address = request.address;
    binding.address.binding_id = binding.binding_id;
    binding.issued_at = entry.capability.issued_at;
    binding.credential_expires_at = *request.credential_expires_at;
    binding.state = LifecycleBindingState::kActive;
    binding.binding_fencing_context = request.binding_fencing_context;
    bindings_by_id_[binding.binding_id] = binding;
  }

  return capabilities_by_id_.at(request.capability_id).capability;
}

absl::StatusOr<LifecycleCapabilityRecord> LifecycleKernel::renew_capability(const RenewCapabilityRequest& request) {
  absl::MutexLock lock(&mu_);
  auto it = capabilities_by_id_.find(request.capability_id);
  if (it == capabilities_by_id_.end()) {
    return absl::NotFoundError("capability_id not found");
  }
  it->second.capability.capability_expires_at = request.capability_expires_at;
  if (it->second.binding_mode == LifecycleBindingMode::kBindingRecord && request.credential_expires_at.has_value()) {
    const std::string binding_id =
        request.binding_id.has_value() ? *request.binding_id : it->second.binding_id.value_or(std::string{});
    auto bit = bindings_by_id_.find(binding_id);
    if (bit == bindings_by_id_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    bit->second.credential_expires_at = *request.credential_expires_at;
    bit->second.state = LifecycleBindingState::kActive;
  }
  if (it->second.capability.state == LifecycleRecordState::kExpired) {
    it->second.capability.state = LifecycleRecordState::kActive;
  }
  return it->second.capability;
}

absl::Status LifecycleKernel::validate_constraint_claims_(
    const ConstraintClaims& expected,
    const ConstraintClaims& observed) const {
  if (!expected.artifact_id.empty() && !observed.artifact_id.empty() && expected.artifact_id != observed.artifact_id) {
    return absl::FailedPreconditionError("artifact_id constraint mismatch");
  }
  if (!expected.digest_alg.empty() && !observed.digest_alg.empty() && expected.digest_alg != observed.digest_alg) {
    return absl::FailedPreconditionError("digest_alg constraint mismatch");
  }
  if (!expected.digest_hex.empty() && !observed.digest_hex.empty() && expected.digest_hex != observed.digest_hex) {
    return absl::FailedPreconditionError("digest_hex constraint mismatch");
  }
  if (!expected.direction.empty() && !observed.direction.empty() && expected.direction != observed.direction) {
    return absl::FailedPreconditionError("direction constraint mismatch");
  }
  if (!expected.operation_id.empty() && !observed.operation_id.empty() &&
      expected.operation_id != observed.operation_id) {
    return absl::FailedPreconditionError("operation_id constraint mismatch");
  }
  if (!expected.holder_scope.empty() && !observed.holder_scope.empty() &&
      expected.holder_scope != observed.holder_scope) {
    return absl::PermissionDeniedError("holder_scope constraint mismatch");
  }
  if (expected.local_only && !observed.local_only) {
    return absl::FailedPreconditionError("local_only constraint mismatch");
  }
  return absl::OkStatus();
}

absl::StatusOr<LifecycleKernel::CapabilityEntry*> LifecycleKernel::resolve_entry_for_credential_locked_(
    const ParsedCredential& credential) {
  if (!route_principal_is_local_issuer(credential.address.route_principal, issuer_daemon_id_)) {
    return absl::FailedPreconditionError("credential issuer is not local to this lifecycle kernel");
  }
  if (credential.binding_mode == LifecycleBindingMode::kBindingRecord) {
    const std::string binding_id = credential.address.binding_id.value_or(std::string{});
    if (binding_id.empty()) {
      return absl::InvalidArgumentError("binding_record credential missing binding_id");
    }
    auto binding_it = bindings_by_id_.find(binding_id);
    if (binding_it == bindings_by_id_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    if (binding_it->second.address.family != credential.address.family ||
        binding_it->second.address.binding_space != credential.address.binding_space ||
        binding_it->second.address.binding_key_kind != credential.address.binding_key_kind ||
        binding_it->second.address.binding_key != credential.address.binding_key) {
      return absl::FailedPreconditionError("binding_id address mismatch");
    }
    if (binding_it->second.state == LifecycleBindingState::kRevoked) {
      return absl::PermissionDeniedError("binding_id revoked");
    }
    if (binding_it->second.credential_expires_at <= absl::Now()) {
      binding_it->second.state = LifecycleBindingState::kExpired;
      return absl::PermissionDeniedError("binding_id expired");
    }
    auto entry_it = capabilities_by_id_.find(binding_it->second.capability_id);
    if (entry_it == capabilities_by_id_.end()) {
      return absl::NotFoundError("binding capability not found");
    }
    return &entry_it->second;
  }

  const std::string address_key = address_index_key(credential.address);
  auto address_it = address_to_capability_id_.find(address_key);
  if (address_it == address_to_capability_id_.end()) {
    return absl::NotFoundError("binding address not found");
  }
  auto entry_it = capabilities_by_id_.find(address_it->second);
  if (entry_it == capabilities_by_id_.end()) {
    return absl::NotFoundError("binding capability not found");
  }
  return &entry_it->second;
}

void LifecycleKernel::maybe_expire_entry_locked_(CapabilityEntry& entry, absl::Time now) {
  if (entry.capability.capability_expires_at <= now) {
    entry.capability.state = LifecycleRecordState::kExpired;
    if (entry.binding_id.has_value()) {
      auto it = bindings_by_id_.find(*entry.binding_id);
      if (it != bindings_by_id_.end()) {
        it->second.state = LifecycleBindingState::kExpired;
      }
    }
  }
}

absl::StatusOr<AdmittedCapabilityUse> LifecycleKernel::admit_redemption(const ParsedCredential& credential) {
  absl::MutexLock lock(&mu_);
  auto entry_or = resolve_entry_for_credential_locked_(credential);
  if (!entry_or.ok()) {
    return entry_or.status();
  }
  CapabilityEntry& entry = **entry_or;
  maybe_expire_entry_locked_(entry, absl::Now());
  if (entry.capability.state != LifecycleRecordState::kActive) {
    return absl::FailedPreconditionError("capability is not active");
  }
  if (!epochs_match(entry.capability.epochs, credential.address.epochs)) {
    return absl::FailedPreconditionError("credential epoch mismatch");
  }
  auto constraint_status =
      validate_constraint_claims_(entry.capability.constraint_claims, credential.constraint_claims);
  if (!constraint_status.ok()) {
    return constraint_status;
  }

  const std::string guard_id = absl::StrCat("lkg-", next_guard_id_++);
  use_guard_to_capability_id_[guard_id] = entry.capability.capability_id;
  entry.active_use_count += 1;
  entry.subject.last_observed_at = absl::Now();

  std::optional<LifecycleBindingRecord> binding_record;
  if (entry.binding_id.has_value()) {
    auto bit = bindings_by_id_.find(*entry.binding_id);
    if (bit != bindings_by_id_.end()) {
      binding_record = bit->second;
    }
  }

  return AdmittedCapabilityUse{
      .capability = entry.capability,
      .subject = entry.subject,
      .binding_record = std::move(binding_record),
      .use_guard =
          LifecycleUseGuard{
              .guard_id = guard_id,
              .capability_id = entry.capability.capability_id,
          },
  };
}

absl::Status LifecycleKernel::release_use_guard(const LifecycleUseGuard& use_guard) {
  absl::MutexLock lock(&mu_);
  auto guard_it = use_guard_to_capability_id_.find(use_guard.guard_id);
  if (guard_it == use_guard_to_capability_id_.end()) {
    return absl::NotFoundError("use_guard not found");
  }
  auto entry_it = capabilities_by_id_.find(guard_it->second);
  if (entry_it != capabilities_by_id_.end() && entry_it->second.active_use_count > 0) {
    entry_it->second.active_use_count -= 1;
    if (entry_it->second.active_use_count == 0 &&
        (entry_it->second.capability.state == LifecycleRecordState::kDraining ||
         entry_it->second.capability.state == LifecycleRecordState::kExpired ||
         entry_it->second.capability.state == LifecycleRecordState::kReleased)) {
      erase_entry_locked_(entry_it->second.capability.capability_id);
    }
  }
  use_guard_to_capability_id_.erase(guard_it);
  return absl::OkStatus();
}

absl::Status LifecycleKernel::revoke_binding_record(std::string_view binding_id) {
  absl::MutexLock lock(&mu_);
  auto it = bindings_by_id_.find(std::string(binding_id));
  if (it == bindings_by_id_.end()) {
    return absl::NotFoundError("binding_id not found");
  }
  it->second.state = LifecycleBindingState::kRevoked;
  return absl::OkStatus();
}

void LifecycleKernel::erase_entry_locked_(std::string_view capability_id) {
  auto entry_it = capabilities_by_id_.find(std::string(capability_id));
  if (entry_it == capabilities_by_id_.end()) {
    return;
  }
  address_to_capability_id_.erase(address_index_key(entry_it->second.address));
  if (entry_it->second.binding_id.has_value()) {
    bindings_by_id_.erase(*entry_it->second.binding_id);
  }
  entry_it->second.capability.state = LifecycleRecordState::kReleased;
  capabilities_by_id_.erase(entry_it);
}

absl::Status LifecycleKernel::release_capability(std::string_view capability_id) {
  absl::MutexLock lock(&mu_);
  auto entry_it = capabilities_by_id_.find(std::string(capability_id));
  if (entry_it == capabilities_by_id_.end()) {
    return absl::OkStatus();
  }
  entry_it->second.capability.state = LifecycleRecordState::kDraining;
  if (entry_it->second.binding_id.has_value()) {
    auto bit = bindings_by_id_.find(*entry_it->second.binding_id);
    if (bit != bindings_by_id_.end()) {
      bit->second.state = LifecycleBindingState::kRevoked;
    }
  }
  address_to_capability_id_.erase(address_index_key(entry_it->second.address));
  if (entry_it->second.active_use_count == 0) {
    erase_entry_locked_(capability_id);
  }
  return absl::OkStatus();
}

absl::StatusOr<LifecycleCapabilityRecord> LifecycleKernel::inspect_capability(std::string_view capability_id) const {
  absl::MutexLock lock(&mu_);
  auto it = capabilities_by_id_.find(std::string(capability_id));
  if (it == capabilities_by_id_.end()) {
    return absl::NotFoundError("capability_id not found");
  }
  return it->second.capability;
}

absl::StatusOr<LifecycleBindingRecord> LifecycleKernel::inspect_binding(std::string_view binding_id) const {
  absl::MutexLock lock(&mu_);
  auto it = bindings_by_id_.find(std::string(binding_id));
  if (it == bindings_by_id_.end()) {
    return absl::NotFoundError("binding_id not found");
  }
  return it->second;
}

} // namespace tensorcast::daemon
