// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "core/common/memory/cuda_memory.h"
#include "core/cuda/cuda_ipc.h"
#include "tensorcast/common/v1/common.pb.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class BindingRegistry {
 public:
  struct Record {
    mutable absl::Mutex mu;
    std::string binding_id;
    std::string binding_layout_id;
    int owner_pid{0};
    int device_id{-1};
    std::string device_uuid;
    v2::BindingOwnership ownership{v2::BINDING_OWNERSHIP_UNSPECIFIED};
    v2::BindingState state{v2::BINDING_STATE_UNSPECIFIED};
    bool mapped{false};
    bool closed{false};
    int export_refs{0};
    std::unique_ptr<common::memory::GpuDeviceMemory> allocation;
    cuda::IpcHandleBytes handle_bytes;
    tensorcast::common::v1::ArtifactSelection source_selection;
    tensorcast::common::v1::ArtifactSelection current_selection;
    v2::TargetLayout target_layout;
    std::string target_index_json;
    std::string target_layout_hash;
    std::string current_artifact_id;
    std::string current_binding_value_id;
    uint64_t seal_generation{0};
    uint64_t update_epoch_counter{0};
    std::string active_update_epoch;
    std::string target_write_token;
    v2::CopyPlan copy_plan;
    std::vector<v2::MappedTensorSpec> dst_tensors;
  };

  [[nodiscard]] absl::Status insert(std::shared_ptr<Record> record);

  [[nodiscard]] absl::StatusOr<std::shared_ptr<Record>> get(std::string_view binding_id) const;

  [[nodiscard]] bool close_control(std::string_view binding_id);

  void release_export_ref(std::string_view binding_id);

  void handle_pid_exit(int owner_pid);

  [[nodiscard]] size_t size() const;

 private:
  void erase_if_reclaimable_(std::string_view binding_id, const std::shared_ptr<Record>& record);

  mutable absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::shared_ptr<Record>> records_ ABSL_GUARDED_BY(mu_);
};

} // namespace tensorcast::daemon
