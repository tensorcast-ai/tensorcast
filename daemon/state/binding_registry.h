// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/memory/cuda_memory.h"
#include "core/cuda/cuda_ipc.h"
#include "daemon/state/types.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"
#include "tensorcast/operation/v1/operation.pb.h"

namespace tensorcast::daemon {

class BindingRegistry {
 public:
  enum class ControlLifetime {
    kPidBound,
    kDaemonRetained,
  };

  struct StagedBindingValue {
    std::string transaction_id;
    std::string version_set_id;
    std::string part_id;
    std::string binding_value_id;
    std::string staging_token;
    uint64_t staging_epoch{0};
    std::string daemon_id;
    std::string daemon_session_id;
    std::string materialization_attempt_id;
    std::shared_ptr<common::memory::GpuDeviceMemory> allocation;
    cuda::IpcHandleBytes handle_bytes;
    tensorcast::common::v1::ArtifactSelection selection;
    std::string artifact_id;
    std::string canonical_index_json;
    v2::TargetLayout target_layout;
    std::string target_index_json;
    std::string target_layout_hash;
    std::string tensor_schema_hash;
    std::vector<v2::TensorPayloadDescriptor> payloads;
    uint64_t logical_total_size{0};
    uint64_t expected_previous_seal_generation{0};
    std::string source_replica_id;
    uint64_t source_export_generation{0};
    std::string child_transport_request_id;
    v2::BindingValueVerificationState verification_state{v2::BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED};
    std::string verification_failure_reason;
    absl::Time created_at{absl::InfinitePast()};
    absl::Time expires_at{absl::InfiniteFuture()};
  };

  struct Record {
    struct PublishedCurrentValue {
      std::string binding_id;
      std::string binding_layout_id;
      std::string binding_value_id;
      uint64_t seal_generation{0};
      std::string publication_id;
      std::string lease_id;
      std::string replica_id;
      std::string operation_id;
    };

    mutable absl::Mutex mu;
    std::string binding_id;
    std::string binding_layout_id;
    int owner_pid{0};
    int creator_pid{0};
    int device_id{-1};
    std::string device_uuid;
    v2::BindingOwnership ownership{v2::BINDING_OWNERSHIP_UNSPECIFIED};
    v2::BindingState state{v2::BINDING_STATE_UNSPECIFIED};
    ControlLifetime control_lifetime{ControlLifetime::kPidBound};
    bool mapped{false};
    bool closed{false};
    bool retained_ref{false};
    int export_refs{0};
    int active_attachment_refs{0};
    std::unique_ptr<common::memory::GpuDeviceMemory> allocation;
    cuda::IpcHandleBytes handle_bytes;
    tensorcast::common::v1::ArtifactSelection source_selection;
    tensorcast::common::v1::ArtifactSelection current_selection;
    v2::TargetLayout target_layout;
    std::string target_index_json;
    std::string target_layout_hash;
    std::string tensor_schema_hash;
    std::string current_artifact_id;
    std::string current_artifact_canonical_index_json;
    std::string binding_current_value_publication_token;
    std::string current_binding_value_id;
    uint64_t seal_generation{0};
    std::optional<PublishedCurrentValue> active_published_current;
    std::optional<CommitLeaseResult> sealed_commit_result;
    v2::BindingValueVerificationState verification_state{v2::BINDING_VALUE_VERIFICATION_STATE_UNSPECIFIED};
    std::string verification_job_id;
    std::string source_artifact_ref;
    std::string local_serving_ref;
    std::string serving_artifact_id;
    std::string verification_failure_reason;
    std::string daemon_id;
    std::string daemon_session_id;
    tensorcast::operation::v1::ServingBindingMemberRef serving_member;
    std::string serving_build_digest;
    std::string reservation_capability_id;
    absl::Time reservation_expires_at{absl::InfiniteFuture()};
    std::optional<int> allowed_caller_pid;
    uint64_t update_epoch_counter{0};
    std::string active_update_epoch;
    v2::CopyPlan copy_plan;
    std::vector<v2::MappedTensorSpec> dst_tensors;
    std::vector<v2::TensorPayloadDescriptor> payloads;
    absl::Time created_at{absl::InfinitePast()};
    absl::Time reserved_at{absl::InfinitePast()};
    absl::Time ready_at{absl::InfinitePast()};
    absl::Time first_acquired_at{absl::InfinitePast()};
    absl::Time last_acquired_at{absl::InfinitePast()};
    absl::Time last_released_at{absl::InfinitePast()};
    absl::Time unacquired_deadline{absl::InfiniteFuture()};
    absl::Time idle_deadline{absl::InfiniteFuture()};
    absl::Time materialization_deadline{absl::InfiniteFuture()};
    bool retired{false};
    std::string retired_reason;
    absl::flat_hash_map<std::string, StagedBindingValue> staged_values_by_id;
  };

  [[nodiscard]] absl::Status insert(std::shared_ptr<Record> record);

  [[nodiscard]] absl::StatusOr<std::shared_ptr<Record>> get(std::string_view binding_id) const;

  [[nodiscard]] absl::StatusOr<std::shared_ptr<Record>> get_by_local_serving_ref(
      std::string_view local_serving_ref) const;

  [[nodiscard]] bool close_control(std::string_view binding_id);

  [[nodiscard]] absl::Status retire_retained(std::string_view binding_id, std::string_view reason);

  [[nodiscard]] absl::Status insert_staged_value(
      std::string_view binding_id,
      StagedBindingValue staged_value,
      absl::Time now);

  [[nodiscard]] absl::StatusOr<StagedBindingValue> get_staged_value(
      std::string_view binding_id,
      std::string_view binding_value_id) const;

  [[nodiscard]] absl::Status remove_staged_value(
      std::string_view binding_id,
      std::string_view binding_value_id,
      std::string_view reason);

  [[nodiscard]] size_t remove_staged_values_for_transaction(
      std::string_view transaction_id,
      std::string_view reason,
      size_t max_to_remove = 0);

  [[nodiscard]] std::vector<std::string> staged_transaction_ids(size_t max_ids = 0) const;

  [[nodiscard]] size_t sweep_staged_values(absl::Time now, size_t max_to_remove = 0);

  [[nodiscard]] size_t clear_staged_values(std::string_view reason);

  [[nodiscard]] absl::Status acquire_attachment_ref(std::string_view binding_id, absl::Time now);

  [[nodiscard]] absl::Status validate_acquire_request(const v2::AcquireBindingValueRequest& request, absl::Time now)
      const;

  [[nodiscard]] absl::Status validate_and_acquire_attachment_ref(
      const v2::AcquireBindingValueRequest& request,
      absl::Time now);

  [[nodiscard]] absl::Status validate_group_staged_acquire_request(
      const v2::AcquireBindingValueRequest& request,
      absl::Time now) const;

  [[nodiscard]] absl::Status validate_and_acquire_group_staged_attachment_ref(
      const v2::AcquireBindingValueRequest& request,
      absl::Time now);

  [[nodiscard]] absl::Status keepalive_attachment_ref(std::string_view binding_id, absl::Time now);

  void release_attachment_ref(std::string_view binding_id, absl::Time now);
  void release_attachment_ref(std::string_view binding_id, absl::Time now, absl::Duration idle_ttl);

  [[nodiscard]] size_t sweep_retention(absl::Time now);

  void release_export_ref(std::string_view binding_id);

  void handle_pid_exit(int owner_pid);

  [[nodiscard]] size_t size() const;

 private:
  void erase_if_reclaimable_(std::string_view binding_id, const std::shared_ptr<Record>& record);

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<Record>> records_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
