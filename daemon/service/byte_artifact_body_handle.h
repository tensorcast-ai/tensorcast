// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "core/common/memory/memory_location.h"
#include "core/store/communication_types.h"
#include "core/store/runtime/ingestion/artifact_truth.h"
#include "core/store/store_engine.h"

namespace tensorcast::daemon {

struct BodyExportRequest {
  common::memory::MemoryLocation preferred_location{common::memory::MemoryLocation::CPU};
  bool require_remote_source{false};
  bool allow_segmented_export{true};
};

struct BodyExportCapability {
  common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::NONE};
  bool local_loader_available{false};
  bool remote_source_eligible{false};
  bool supports_segmented_export{false};
};

struct BodyExportView {
  store::runtime::ingestion::BackingIdentity backing_identity;
  std::uint64_t binding_generation{0};
  common::memory::MemoryLocation memory_location{common::memory::MemoryLocation::NONE};
  std::optional<store::ExportRegistration> communicator_export;
  std::shared_ptr<void> keepalive;
};

class BodyHandle {
 public:
  enum class Kind {
    kCoreReplica = 0,
  };

  BodyHandle() = default;

  [[nodiscard]] static absl::StatusOr<BodyHandle> create(
      store::StoreEngine& engine,
      store::loading::ReplicaHandle replica_handle);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] Kind kind() const;
  [[nodiscard]] std::uint64_t size_bytes() const;
  [[nodiscard]] std::uint64_t binding_generation() const;
  [[nodiscard]] const store::loading::ReplicaHandle& replica_handle() const;
  [[nodiscard]] common::memory::MemoryLocation location() const;
  [[nodiscard]] bool unique_owner() const;
  [[nodiscard]] absl::StatusOr<BodyExportCapability> inspect_export_capability() const;
  [[nodiscard]] absl::StatusOr<BodyExportView> acquire_export_view(const BodyExportRequest& request) const;
  [[nodiscard]] absl::Status pin_export_keepalive(
      common::memory::MemoryLocation location,
      std::shared_ptr<void> keepalive,
      absl::Time expires_at) const;

  [[nodiscard]] absl::StatusOr<std::unique_ptr<store::IArtifactLoader>> make_loader() const;
  [[nodiscard]] absl::Status read_into_range(std::uint64_t offset, void* dst, std::size_t bytes) const;
  [[nodiscard]] absl::StatusOr<std::string> read_range(std::uint64_t offset, std::size_t bytes) const;
  [[nodiscard]] absl::StatusOr<std::string> read_all_bytes() const;
  [[nodiscard]] absl::StatusOr<std::string> compute_sha256_hex() const;
  [[nodiscard]] absl::Status retire() const;

 private:
  struct CoreBacking {
    store::StoreEngine* engine{nullptr};
    store::loading::ReplicaHandle replica_handle;
    std::uint64_t size_bytes{0};
    std::uint64_t binding_generation{0};
    std::atomic<bool> retired{false};

    struct ExportLease {
      store::StoreEngine* engine{nullptr};
      store::loading::ReplicaKey replica_key;
      common::memory::MemoryLocation location{common::memory::MemoryLocation::NONE};
      store::ExportRegistration registration;

      ~ExportLease();
    };

    mutable absl::Mutex export_mu;
    std::weak_ptr<ExportLease> cpu_export ABSL_GUARDED_BY(export_mu);
    std::weak_ptr<ExportLease> gpu_export ABSL_GUARDED_BY(export_mu);
    std::shared_ptr<void> cpu_publish_prereg_pin ABSL_GUARDED_BY(export_mu);
    absl::Time cpu_publish_prereg_pin_expires_at ABSL_GUARDED_BY(export_mu){absl::InfinitePast()};
  };

  explicit BodyHandle(std::shared_ptr<CoreBacking> backing);

  std::shared_ptr<CoreBacking> backing_;
};

} // namespace tensorcast::daemon
