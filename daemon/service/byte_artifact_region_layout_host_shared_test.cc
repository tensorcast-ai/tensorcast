// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/external_target_access_service.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string>

#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "core/store/device_registry.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

constexpr std::uint64_t kStorageBytes = 8192;
constexpr std::uint64_t kItemBytes = 4096;

TEST_CASE(
    "ExternalTargetAccessService accepts daemon-managed HOST_SHARED source layouts",
    "[daemon][byte_artifact][host_shared]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kStorageBytes;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());

  auto mapping_or = registry.acquire_host_shared_local_mapping(desc_or->region_id, params.owner_pid);
  REQUIRE(mapping_or.ok());
  std::memset(mapping_or->base_ptr, 'A', static_cast<size_t>(kItemBytes));
  REQUIRE(registry.release(desc_or->region_id).ok());

  tensorcast::daemon::v2::TargetLayout layout;
  layout.set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout.set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout.set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout.add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(-1);
  storage->set_storage_length(kStorageBytes);
  storage->set_mapping_base_offset(0);
  auto* region_ref = storage->mutable_region_ref();
  region_ref->set_region_id(desc_or->region_id);
  region_ref->set_memory_kind(tensorcast::daemon::v2::REGION_MEMORY_KIND_HOST_SHARED);
  region_ref->set_device_id(-1);
  region_ref->set_size_bytes(kStorageBytes);

  auto* offset = layout.add_offsets();
  offset->set_name("artifact-a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(kItemBytes);

  absl::flat_hash_map<std::string, std::uint64_t> expected_lengths;
  expected_lengths.emplace("artifact-a", kItemBytes);

  tensorcast::daemon::DeviceResolver devices(tensorcast::store::DeviceRegistry::instance());
  tensorcast::daemon::ExternalTargetAccessService svc(
      tensorcast::daemon::ExternalTargetAccessService::Dep{
          .devices = devices,
          .regions = registry,
      });

  auto validated_or = svc.validate_local_source_layout(
      "", "BatchPutIfAbsentFromRegion", layout, params.owner_pid, "", expected_lengths);
  REQUIRE(validated_or.ok());
  REQUIRE(validated_or->layout.device_id() == -1);

  auto source_or = validated_or->layout.open_item_source("artifact-a");
  REQUIRE(source_or.ok());
  char buffer[kItemBytes];
  std::memset(buffer, 0, sizeof(buffer));
  auto read_or = (*source_or)->read_at(0, buffer, sizeof(buffer));
  REQUIRE(read_or.ok());
  REQUIRE(*read_or == sizeof(buffer));
  for (char byte : buffer) {
    REQUIRE(byte == 'A');
  }

  auto item_layout_or = validated_or->layout.build_item_target_layout("artifact-a");
  REQUIRE(item_layout_or.ok());
  REQUIRE(item_layout_or->total_size == kItemBytes);
  REQUIRE(item_layout_or->storages.size() == 1);
  auto* region_bytes = static_cast<const char*>(item_layout_or->storages[0].base_ptr.get());
  REQUIRE(std::memcmp(region_bytes, buffer, sizeof(buffer)) == 0);
}

TEST_CASE(
    "ExternalTargetAccessService accepts pure HOST_SHARED target layouts without device_uuid",
    "[daemon][byte_artifact][host_shared][target]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kStorageBytes;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kScratch;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());

  tensorcast::daemon::v2::TargetLayout layout;
  layout.set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout.set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout.set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout.add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(-1);
  storage->set_storage_length(kStorageBytes);
  storage->set_mapping_base_offset(0);
  auto* region_ref = storage->mutable_region_ref();
  region_ref->set_region_id(desc_or->region_id);
  region_ref->set_memory_kind(tensorcast::daemon::v2::REGION_MEMORY_KIND_HOST_SHARED);
  region_ref->set_device_id(-1);
  region_ref->set_size_bytes(kStorageBytes);

  tensorcast::daemon::DeviceResolver devices(tensorcast::store::DeviceRegistry::instance());
  tensorcast::daemon::ExternalTargetAccessService svc(
      tensorcast::daemon::ExternalTargetAccessService::Dep{
          .devices = devices,
          .regions = registry,
      });

  auto validated_or =
      svc.validate_local_target_layout("", "BatchGetIntoRegion", layout, params.owner_pid, /*device_uuid=*/"");
  REQUIRE(validated_or.ok());
  REQUIRE(validated_or->device.type == tensorcast::DeviceType::CPU);
  REQUIRE(validated_or->device.ordinal == -1);
  REQUIRE(validated_or->storage_lease.storages().size() == 1);

  auto rejected_or = svc.validate_local_target_layout("", "BatchGetIntoRegion", layout, params.owner_pid, "gpu-uuid");
  REQUIRE_FALSE(rejected_or.ok());
  REQUIRE(rejected_or.status().code() == absl::StatusCode::kInvalidArgument);

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

} // namespace
