// Copyright (c) 2026, TensorCast Team.

#include "daemon/state/binding_registry.h"

#include <utility>

namespace tensorcast::daemon {

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
  }
  erase_if_reclaimable_(binding_id, record);
  return true;
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
      record->closed = true;
      record->export_refs = 0;
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
    reclaimable = record->closed && record->export_refs <= 0;
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
