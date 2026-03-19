// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::daemon {

// Routing-only admission seam for public operation observation.
// Child owners remain the semantic owners of their own continuation scope and
// fail-closed checks; this service only dispatches by public-safe metadata.
class PublicOperationAdmissionService {
 public:
  using AdmitFn = std::function<absl::Status(const tensorcast::operation::v1::OperationRef&, absl::Time)>;

  void register_handler(std::string operation_kind, AdmitFn admit_fn);

  [[nodiscard]] absl::Status admit(const tensorcast::operation::v1::OperationRef& operation_ref, absl::Time now) const;

 private:
  absl::flat_hash_map<std::string, AdmitFn> handlers_;
};

} // namespace tensorcast::daemon
