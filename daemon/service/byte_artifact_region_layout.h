// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/store/components/communication_manager.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/contracts/source.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"
#include "daemon/state/ipc_region_registry.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon {

class ByteArtifactRegionLayout {
 public:
  struct SlotToken {
    std::optional<std::uint64_t> slot_index;
    std::optional<std::uint64_t> slot_generation;
  };

  struct HostSharedSourceSpan {
    const void* data{nullptr};
    std::uint64_t length{0};
    std::string region_id;
    IpcRegionRegistry::HostRegionClass host_region_class{IpcRegionRegistry::HostRegionClass::kNone};
    bool daemon_managed{false};
    std::optional<SlotToken> slot_token;
    std::optional<store::StableLocalBackingRef> stable_backing;
    std::shared_ptr<void> stable_backing_keepalive;
    std::shared_ptr<void> keepalive;
  };

  ByteArtifactRegionLayout() = default;
  ByteArtifactRegionLayout(const ByteArtifactRegionLayout&) = delete;
  ByteArtifactRegionLayout& operator=(const ByteArtifactRegionLayout&) = delete;
  ByteArtifactRegionLayout(ByteArtifactRegionLayout&&) noexcept = default;
  ByteArtifactRegionLayout& operator=(ByteArtifactRegionLayout&&) noexcept = default;
  ~ByteArtifactRegionLayout() = default;

  [[nodiscard]] static absl::StatusOr<ByteArtifactRegionLayout> acquire(
      IpcRegionRegistry& registry,
      const v2::TargetLayout& layout,
      int owner_pid,
      std::string_view device_uuid,
      const absl::flat_hash_map<std::string, std::uint64_t>& expected_lengths);

  [[nodiscard]] bool contains(std::string_view artifact_id) const;
  [[nodiscard]] std::uint64_t expected_length(std::string_view artifact_id) const;
  [[nodiscard]] int device_id() const;
  [[nodiscard]] std::optional<SlotToken> slot_token(std::string_view artifact_id) const;
  [[nodiscard]] absl::StatusOr<store::loading::IntoTargetLayout> build_item_target_layout(
      std::string_view artifact_id) const;
  [[nodiscard]] absl::StatusOr<std::shared_ptr<store::loader::SeekableSource>> open_item_source(
      std::string_view artifact_id) const;
  [[nodiscard]] absl::StatusOr<HostSharedSourceSpan> open_host_shared_source_span(std::string_view artifact_id) const;
  [[nodiscard]] absl::Status activate_stable_local_backings(
      store::components::CommunicationManager& comm_manager) const;

 private:
  struct StorageRange {
    std::string storage_id;
    std::string region_id;
    std::uint64_t logical_base{0};
    std::uint64_t length{0};
    IpcRegionRegistry::MemoryKind memory_kind{IpcRegionRegistry::MemoryKind::kVram};
    IpcRegionRegistry::HostRegionClass host_region_class{IpcRegionRegistry::HostRegionClass::kNone};
    bool daemon_managed{false};
    void* base_ptr{nullptr};
    int device_id{-1};
    std::optional<store::StableLocalBackingRef> stable_backing;
    std::shared_ptr<void> stable_backing_keepalive;
    std::shared_ptr<void> keepalive;
  };

  struct ItemRange {
    std::size_t storage_index{0};
    std::uint64_t logical_offset{0};
    std::uint64_t logical_length{0};
    std::uint64_t storage_local_offset{0};
    std::optional<SlotToken> slot_token;
  };

  materialization_target_storage::TargetStorageLease storage_lease_;
  std::vector<StorageRange> storages_;
  absl::flat_hash_map<std::string, ItemRange> items_;
  int device_id_{-1};
};

} // namespace tensorcast::daemon
