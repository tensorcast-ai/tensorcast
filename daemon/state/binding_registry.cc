// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/binding_registry.h"

#include <utility>

namespace tensorcast::daemon {

namespace {

bool is_ready_state(v2::BindingState state) {
  return state == v2::BINDING_STATE_READY_ARTIFACT || state == v2::BINDING_STATE_READY_LOCAL;
}

bool finite_deadline_due(absl::Time deadline, absl::Time now) {
  return deadline != absl::InfiniteFuture() && now >= deadline;
}

bool same_binding_value_ref(
    const tensorcast::publication::v1::BindingValueRef& lhs,
    const tensorcast::publication::v1::BindingValueRef& rhs) {
  return lhs.binding_id() == rhs.binding_id() && lhs.binding_layout_id() == rhs.binding_layout_id() &&
      lhs.binding_value_id() == rhs.binding_value_id() && lhs.seal_generation() == rhs.seal_generation();
}

bool same_member_ref(
    const tensorcast::operation::v1::ServingBindingMemberRef& lhs,
    const tensorcast::operation::v1::ServingBindingMemberRef& rhs) {
  return lhs.member_id() == rhs.member_id() && lhs.member_index() == rhs.member_index() &&
      lhs.member_count() == rhs.member_count() && lhs.group_id() == rhs.group_id();
}

absl::Status validate_nonempty(std::string_view value, std::string_view field_name) {
  if (value.empty()) {
    return absl::InvalidArgumentError(std::string(field_name) + " is required");
  }
  return absl::OkStatus();
}

} // namespace

absl::Status BindingRegistry::insert(std::shared_ptr<Record> record) {
  if (!record) {
    return absl::InvalidArgumentError("binding record is required");
  }
  if (record->binding_id.empty()) {
    return absl::InvalidArgumentError("binding_id is required");
  }
  absl::MutexLock lock(&mu_);
  auto [it, inserted] = records_.emplace(record->binding_id, std::move(record));
  if (!inserted) {
    return absl::AlreadyExistsError("binding_id already exists");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<BindingRegistry::Record>> BindingRegistry::get(std::string_view binding_id) const {
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(binding_id));
  if (it == records_.end()) {
    return absl::NotFoundError("binding_id not found");
  }
  return it->second;
}

bool BindingRegistry::close_control(std::string_view binding_id) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return false;
    }
    record = it->second;
  }
  {
    absl::MutexLock lock(&record->mu);
    record->closed = true;
    if (record->control_lifetime == ControlLifetime::kPidBound) {
      record->retained_ref = false;
    }
  }
  erase_if_reclaimable_(binding_id, record);
  return true;
}

absl::Status BindingRegistry::retire_retained(std::string_view binding_id, std::string_view reason) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    record = it->second;
  }
  {
    absl::MutexLock lock(&record->mu);
    record->retired = true;
    record->retained_ref = false;
    record->retired_reason = std::string(reason);
  }
  erase_if_reclaimable_(binding_id, record);
  return absl::OkStatus();
}

absl::Status BindingRegistry::acquire_attachment_ref(std::string_view binding_id, absl::Time now) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    record = it->second;
  }
  absl::MutexLock lock(&record->mu);
  if (record->retired) {
    return absl::FailedPreconditionError("binding is retired");
  }
  if (!record->retained_ref) {
    return absl::FailedPreconditionError("binding is not retained");
  }
  if (!is_ready_state(record->state)) {
    return absl::FailedPreconditionError("binding is not ready");
  }
  if (record->active_attachment_refs == 0 && record->first_acquired_at == absl::InfinitePast()) {
    record->first_acquired_at = now;
  }
  record->last_acquired_at = now;
  record->idle_deadline = absl::InfiniteFuture();
  record->active_attachment_refs++;
  return absl::OkStatus();
}

absl::Status BindingRegistry::validate_acquire_request(const v2::AcquireBindingValueRequest& request, absl::Time now)
    const {
  const auto& ref = request.binding_value_ref();
  if (auto status = validate_nonempty(ref.binding_id(), "binding_value_ref.binding_id"); !status.ok()) {
    if (!request.local_serving_ref().empty()) {
      return absl::InvalidArgumentError("local_serving_ref is diagnostic only; binding_value_ref is required");
    }
    return status;
  }
  if (auto status = validate_nonempty(ref.binding_layout_id(), "binding_value_ref.binding_layout_id"); !status.ok()) {
    return status;
  }
  if (auto status = validate_nonempty(ref.binding_value_id(), "binding_value_ref.binding_value_id"); !status.ok()) {
    return status;
  }
  if (ref.seal_generation() == 0) {
    return absl::InvalidArgumentError("binding_value_ref.seal_generation is required");
  }

  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(ref.binding_id());
    if (it == records_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    record = it->second;
  }

  absl::MutexLock lock(&record->mu);
  if (record->retired) {
    return absl::FailedPreconditionError("binding is retired");
  }
  if (!record->retained_ref) {
    return absl::FailedPreconditionError("binding is not retained");
  }
  if (!is_ready_state(record->state)) {
    return absl::FailedPreconditionError("binding is not ready");
  }
  if (record->binding_layout_id != ref.binding_layout_id()) {
    return absl::FailedPreconditionError("binding_layout_id mismatch");
  }
  if (record->current_binding_value_id != ref.binding_value_id()) {
    return absl::FailedPreconditionError("binding_value_id mismatch");
  }
  if (record->seal_generation != ref.seal_generation()) {
    return absl::FailedPreconditionError("seal_generation mismatch");
  }
  if (!request.local_serving_ref().empty() && record->local_serving_ref != request.local_serving_ref()) {
    return absl::FailedPreconditionError("local_serving_ref mismatch");
  }
  if (record->daemon_id != request.expected_daemon_id()) {
    return absl::FailedPreconditionError("daemon_id mismatch");
  }
  if (record->daemon_session_id != request.expected_daemon_session_id()) {
    return absl::FailedPreconditionError("daemon_session_id mismatch");
  }
  if (record->device_uuid != request.expected_device_uuid()) {
    return absl::FailedPreconditionError("device_uuid mismatch");
  }
  if (!same_member_ref(record->serving_member, request.expected_member())) {
    return absl::FailedPreconditionError("member mismatch");
  }
  if (record->target_layout_hash != request.expected_target_layout_hash()) {
    return absl::FailedPreconditionError("target_layout_hash mismatch");
  }
  if (record->tensor_schema_hash != request.expected_tensor_schema_hash()) {
    return absl::FailedPreconditionError("tensor_schema_hash mismatch");
  }
  if (record->serving_build_digest != request.expected_serving_build_digest()) {
    return absl::FailedPreconditionError("serving_build_digest mismatch");
  }
  if (record->allowed_caller_pid.has_value()) {
    if (!request.has_caller_pid()) {
      return absl::UnauthenticatedError("caller_pid is required");
    }
    if (*record->allowed_caller_pid != request.caller_pid()) {
      return absl::PermissionDeniedError("caller_pid mismatch");
    }
  }

  const auto& capability = request.reservation_capability();
  if (auto status = validate_nonempty(capability.capability_id(), "reservation_capability.capability_id");
      !status.ok()) {
    return status;
  }
  if (!record->reservation_capability_id.empty() && record->reservation_capability_id != capability.capability_id()) {
    return absl::PermissionDeniedError("reservation capability mismatch");
  }
  if (!same_binding_value_ref(capability.binding_value_ref(), ref)) {
    return absl::PermissionDeniedError("reservation capability binding value mismatch");
  }
  if (capability.daemon_id() != request.expected_daemon_id()) {
    return absl::PermissionDeniedError("reservation capability daemon_id mismatch");
  }
  if (capability.daemon_session_id() != request.expected_daemon_session_id()) {
    return absl::PermissionDeniedError("reservation capability daemon_session_id mismatch");
  }
  if (capability.device_uuid() != request.expected_device_uuid()) {
    return absl::PermissionDeniedError("reservation capability device_uuid mismatch");
  }
  if (!same_member_ref(capability.member(), request.expected_member())) {
    return absl::PermissionDeniedError("reservation capability member mismatch");
  }
  if (capability.has_expires_at_ms()) {
    const absl::Time expires_at = absl::UnixEpoch() + absl::Milliseconds(capability.expires_at_ms());
    if (now >= expires_at) {
      return absl::UnauthenticatedError("reservation capability expired");
    }
  }
  if (finite_deadline_due(record->reservation_expires_at, now)) {
    return absl::UnauthenticatedError("reservation expired");
  }
  return absl::OkStatus();
}

absl::Status BindingRegistry::validate_and_acquire_attachment_ref(
    const v2::AcquireBindingValueRequest& request,
    absl::Time now) {
  if (auto status = validate_acquire_request(request, now); !status.ok()) {
    return status;
  }
  return acquire_attachment_ref(request.binding_value_ref().binding_id(), now);
}

absl::Status BindingRegistry::keepalive_attachment_ref(std::string_view binding_id, absl::Time now) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return absl::NotFoundError("binding_id not found");
    }
    record = it->second;
  }
  absl::MutexLock lock(&record->mu);
  if (record->retired) {
    return absl::FailedPreconditionError("binding is retired");
  }
  if (record->active_attachment_refs <= 0) {
    return absl::FailedPreconditionError("binding has no active attachment refs");
  }
  record->last_acquired_at = now;
  record->idle_deadline = absl::InfiniteFuture();
  return absl::OkStatus();
}

void BindingRegistry::release_attachment_ref(std::string_view binding_id, absl::Time now) {
  release_attachment_ref(binding_id, now, absl::InfiniteDuration());
}

void BindingRegistry::release_attachment_ref(std::string_view binding_id, absl::Time now, absl::Duration idle_ttl) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return;
    }
    record = it->second;
  }
  {
    absl::MutexLock lock(&record->mu);
    if (record->active_attachment_refs > 0) {
      record->active_attachment_refs--;
    }
    if (record->active_attachment_refs == 0) {
      record->last_released_at = now;
      if (idle_ttl != absl::InfiniteDuration()) {
        record->idle_deadline = now + idle_ttl;
      }
    }
  }
  erase_if_reclaimable_(binding_id, record);
}

size_t BindingRegistry::sweep_retention(absl::Time now) {
  std::vector<std::pair<std::string, std::shared_ptr<Record>>> candidates;
  {
    absl::MutexLock lock(&mu_);
    candidates.reserve(records_.size());
    for (const auto& [binding_id, record] : records_) {
      if (record != nullptr) {
        candidates.emplace_back(binding_id, record);
      }
    }
  }

  size_t retired_count = 0;
  for (const auto& [binding_id, record] : candidates) {
    bool should_retire = false;
    std::string reason;
    {
      absl::MutexLock lock(&record->mu);
      if (record->retired || !record->retained_ref) {
        continue;
      }
      if (!is_ready_state(record->state) && finite_deadline_due(record->materialization_deadline, now)) {
        should_retire = true;
        reason = "materialization_timeout";
      } else if (
          record->first_acquired_at == absl::InfinitePast() && finite_deadline_due(record->unacquired_deadline, now)) {
        should_retire = true;
        reason = "unacquired_ttl";
      } else if (record->active_attachment_refs == 0 && finite_deadline_due(record->idle_deadline, now)) {
        should_retire = true;
        reason = "idle_ttl";
      }
      if (should_retire) {
        record->retired = true;
        record->retained_ref = false;
        record->retired_reason = std::move(reason);
        retired_count++;
      }
    }
    if (should_retire) {
      erase_if_reclaimable_(binding_id, record);
    }
  }
  return retired_count;
}

void BindingRegistry::release_export_ref(std::string_view binding_id) {
  std::shared_ptr<Record> record;
  {
    absl::MutexLock lock(&mu_);
    auto it = records_.find(std::string(binding_id));
    if (it == records_.end()) {
      return;
    }
    record = it->second;
  }
  {
    absl::MutexLock lock(&record->mu);
    if (record->export_refs > 0) {
      record->export_refs--;
    }
  }
  erase_if_reclaimable_(binding_id, record);
}

void BindingRegistry::handle_pid_exit(int owner_pid) {
  if (owner_pid <= 0) {
    return;
  }
  std::vector<std::pair<std::string, std::shared_ptr<Record>>> matches;
  {
    absl::MutexLock lock(&mu_);
    matches.reserve(records_.size());
    for (const auto& [binding_id, record] : records_) {
      if (record && record->owner_pid == owner_pid) {
        matches.emplace_back(binding_id, record);
      }
    }
  }
  for (const auto& [binding_id, record] : matches) {
    {
      absl::MutexLock lock(&record->mu);
      if (record->control_lifetime == ControlLifetime::kDaemonRetained && record->retained_ref) {
        record->owner_pid = 0;
        if (record->creator_pid == owner_pid) {
          record->creator_pid = 0;
        }
      } else {
        record->closed = true;
        record->retained_ref = false;
        record->export_refs = 0;
      }
    }
    erase_if_reclaimable_(binding_id, record);
  }
}

size_t BindingRegistry::size() const {
  absl::MutexLock lock(&mu_);
  return records_.size();
}

void BindingRegistry::erase_if_reclaimable_(std::string_view binding_id, const std::shared_ptr<Record>& record) {
  bool reclaimable = false;
  {
    absl::MutexLock lock(&record->mu);
    reclaimable = (record->closed || record->retired) && !record->retained_ref && record->export_refs <= 0 &&
        record->active_attachment_refs <= 0;
  }
  if (!reclaimable) {
    return;
  }
  absl::MutexLock lock(&mu_);
  auto it = records_.find(std::string(binding_id));
  if (it == records_.end()) {
    return;
  }
  if (it->second.get() != record.get()) {
    return;
  }
  records_.erase(it);
}

} // namespace tensorcast::daemon
