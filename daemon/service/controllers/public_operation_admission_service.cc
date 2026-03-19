// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/public_operation_admission_service.h"

namespace tensorcast::daemon {

void PublicOperationAdmissionService::register_handler(std::string operation_kind, AdmitFn admit_fn) {
  if (operation_kind.empty() || admit_fn == nullptr) {
    return;
  }
  handlers_[std::move(operation_kind)] = std::move(admit_fn);
}

absl::Status PublicOperationAdmissionService::admit(
    const tensorcast::operation::v1::OperationRef& operation_ref,
    absl::Time now) const {
  const auto it = handlers_.find(operation_ref.kind());
  if (it == handlers_.end()) {
    return absl::OkStatus();
  }
  return it->second(operation_ref, now);
}

} // namespace tensorcast::daemon
