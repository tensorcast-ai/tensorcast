// Copyright (c) 2025, TensorCast Team.

#include "core/store/materialization/contracts/materialization_request.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "core/common/artifact_identity.h"
#include "core/store/components/device_manager.h"

namespace tensorcast::store::loading {

absl::StatusOr<MaterializationRequest> MaterializationRequest::Create(
    DeviceKey target_device,
    MaterializeMode mode,
    const MaterializeHints& hints,
    const components::DeviceManager& device_manager) {
  MaterializationRequest request;
  request.mode_ = mode;
  request.target_device_ = target_device;
  request.hints_ = hints;

  const std::optional<VariantIdentity>& variant = hints.variant;
  if (variant.has_value()) {
    if (variant->canonical_artifact_id.empty()) {
      return absl::InvalidArgumentError("VariantIdentity must provide canonical_artifact_id when present");
    }
    if (!hints.artifact_id.empty() && hints.artifact_id != variant->canonical_artifact_id) {
      return absl::InvalidArgumentError(
          "MaterializeHints.artifact_id must match VariantIdentity.canonical_artifact_id");
    }
    request.canonical_artifact_id_ = variant->canonical_artifact_id;
    request.requested_view_id_ = variant->view_id;

    auto id_kind_or = tensorcast::common::validate_and_get_artifact_id_kind(request.canonical_artifact_id_);
    if (!id_kind_or.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(
              "VariantIdentity.canonical_artifact_id must be a valid mi2: or cgid: identifier: ",
              id_kind_or.status().message()));
    }
  } else {
    request.canonical_artifact_id_ = !hints.artifact_id.empty() ? hints.artifact_id : hints.disk_path;
    request.requested_view_id_.reset();

    if (!hints.artifact_id.empty()) {
      const auto id_kind = tensorcast::common::infer_artifact_id_kind(request.canonical_artifact_id_);
      if (id_kind == tensorcast::common::ArtifactIdKind::kMi2 || id_kind == tensorcast::common::ArtifactIdKind::kCgid) {
        auto id_kind_or = tensorcast::common::validate_and_get_artifact_id_kind(request.canonical_artifact_id_);
        if (!id_kind_or.ok()) {
          return absl::InvalidArgumentError(
              absl::StrCat(
                  "MaterializeHints.artifact_id must be a valid mi2: or cgid: identifier: ",
                  id_kind_or.status().message()));
        }
      }
    }
  }

  if (request.canonical_artifact_id_.empty()) {
    if (mode == MaterializeMode::COPY_ONLY) {
      return absl::InvalidArgumentError(
          "COPY_ONLY requires hints.artifact_id or VariantIdentity.canonical_artifact_id");
    }
    return absl::InvalidArgumentError("MaterializeHints must provide artifact_id or disk_path for LOAD modes");
  }

  switch (target_device.type) {
    case DeviceType::GPU: {
      const int num_gpus = device_manager.get_num_gpus();
      if (target_device.ordinal < 0 || target_device.ordinal >= num_gpus) {
        return absl::InvalidArgumentError(absl::StrCat("Invalid GPU device ordinal: ", target_device.ordinal));
      }
      break;
    }
    case DeviceType::CPU:
      break;
    default:
      return absl::InvalidArgumentError("Unsupported target device type for materialize_replica()");
  }

  request.target_location_ = (target_device.type == DeviceType::GPU) ? common::memory::MemoryLocation::GPU
                                                                     : common::memory::MemoryLocation::CPU;

  request.replica_key_ = ReplicaKey{
      .artifact_id = request.canonical_artifact_id_,
      .view_id = request.requested_view_id_,
      .device = target_device,
      .replica = 0};

  return request;
}

} // namespace tensorcast::store::loading
