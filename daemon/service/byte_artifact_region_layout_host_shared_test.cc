// Copyright (c) 2026, TensorCast Team.

#include "daemon/service/controllers/external_target_access_service.h"
#include "daemon/service/controllers/materialization_target_storage_utils.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string>

#include <unistd.h>

#include "absl/container/flat_hash_map.h"
#include "absl/time/time.h"
#include "core/store/components/communication_manager.h"
#include "core/store/device_registry.h"
#include "core/testing/test_helpers.h"
#include "daemon/state/device_resolver.h"
#include "daemon/state/ipc_region_registry.h"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace {

constexpr std::uint64_t kStorageBytes = 8192;
constexpr std::uint64_t kItemBytes = 4096;

TEST_CASE(
    "ExternalTargetAccessService accepts allocator-backed HOST_SHARED source layouts with slot tokens",
    "[daemon][byte_artifact][host_shared][allocator]") {
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
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator;

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
  offset->set_slot_index(7);
  offset->set_slot_generation(11);

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
  auto slot_token = validated_or->layout.slot_token("artifact-a");
  REQUIRE(slot_token.has_value());
  REQUIRE(slot_token->slot_index.has_value());
  REQUIRE(slot_token->slot_generation.has_value());
  REQUIRE(*slot_token->slot_index == 7);
  REQUIRE(*slot_token->slot_generation == 11);

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

  auto source_span_or = validated_or->layout.open_host_shared_source_span("artifact-a");
  REQUIRE(source_span_or.ok());
  REQUIRE(source_span_or->data != nullptr);
  REQUIRE(source_span_or->length == kItemBytes);
  REQUIRE(source_span_or->region_id == desc_or->region_id);
  REQUIRE(source_span_or->host_region_class == tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator);
  REQUIRE(source_span_or->daemon_managed);
  REQUIRE(source_span_or->slot_token.has_value());
  REQUIRE(source_span_or->slot_token->slot_index.has_value());
  REQUIRE(source_span_or->slot_token->slot_generation.has_value());
  REQUIRE(*source_span_or->slot_token->slot_index == 7);
  REQUIRE(*source_span_or->slot_token->slot_generation == 11);
  REQUIRE(source_span_or->keepalive != nullptr);
  REQUIRE(std::memcmp(source_span_or->data, buffer, sizeof(buffer)) == 0);
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

TEST_CASE(
    "ExternalTargetAccessService rejects allocator-backed HOST_SHARED target layouts on the generic target path",
    "[daemon][byte_artifact][host_shared][target][allocator]") {
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
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator;

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
  REQUIRE_FALSE(validated_or.ok());
  REQUIRE(validated_or.status().code() == absl::StatusCode::kInvalidArgument);

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

TEST_CASE(
    "TargetStorageLease keeps stable backing alive independently from request lease",
    "[daemon][byte_artifact][host_shared][target][stable_backing_keepalive]") {
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

  tensorcast::store::components::CommunicationManager comm_manager;
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true, /*buffers_per_flow=*/2);
  auto init_status = comm_manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, cfg);
  REQUIRE(init_status.ok());
  REQUIRE(comm_manager.stable_local_backing_supported_for_test());

  tensorcast::daemon::v2::TargetLayout layout;
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

  auto lease_or = tensorcast::daemon::materialization_target_storage::TargetStorageLease::acquire(
      registry, layout.storages(), params.owner_pid, /*error=*/nullptr);
  REQUIRE(lease_or.ok());
  REQUIRE(lease_or->storages().size() == 1);
  REQUIRE(lease_or->storages()[0].stable_backing.has_value());
  REQUIRE(lease_or->storages()[0].keepalive != nullptr);
  REQUIRE(lease_or->storages()[0].stable_backing_keepalive != nullptr);
  CHECK(lease_or->storages()[0].stable_backing_keepalive != lease_or->storages()[0].keepalive);
  auto lease = std::move(*lease_or);
  auto activate_status = comm_manager.activate_stable_local_backing(
      *lease.storages()[0].stable_backing, lease.storages()[0].stable_backing_keepalive);
  REQUIRE(activate_status.ok());
  lease = {};
  REQUIRE(comm_manager.stable_local_backing_active_for_test(desc_or->region_id));

  REQUIRE(comm_manager.deactivate_stable_local_backing(desc_or->region_id).ok());
  CHECK_FALSE(comm_manager.stable_local_backing_active_for_test(desc_or->region_id));
  auto unregister_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unregister_or.ok());
  REQUIRE(*unregister_or);
}

TEST_CASE(
    "ExternalTargetAccessService rejects allocator-backed HOST_SHARED target layouts before communicator activation",
    "[daemon][byte_artifact][host_shared][target][allocator][activation_order]") {
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
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());

  tensorcast::store::components::CommunicationManager comm_manager;
  auto cfg = tensorcast::testing::make_tcp_communicator_config(/*enable_rdma=*/true, /*buffers_per_flow=*/2);
  auto init_status = comm_manager.initialize_with_config("127.0.0.1", /*listen_port=*/0, cfg);
  REQUIRE(init_status.ok());
  REQUIRE(comm_manager.stable_local_backing_supported_for_test());

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
          .comm_manager = &comm_manager,
      });

  auto validated_or =
      svc.validate_local_target_layout("", "BatchGetIntoRegion", layout, params.owner_pid, /*device_uuid=*/"");
  REQUIRE_FALSE(validated_or.ok());
  REQUIRE(validated_or.status().code() == absl::StatusCode::kInvalidArgument);
  CHECK_FALSE(comm_manager.stable_local_backing_active_for_test(desc_or->region_id));

  auto unregister_or = registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/false);
  REQUIRE(unregister_or.ok());
  REQUIRE(*unregister_or);
}

TEST_CASE(
    "ExternalTargetAccessService rejects non-local peers for HOST_SHARED target validation",
    "[daemon][byte_artifact][host_shared][trust_boundary][target]") {
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

  auto rejected_or = svc.validate_local_target_layout(
      "ipv4:10.0.0.8:12345", "BatchGetIntoRegion", layout, params.owner_pid, /*device_uuid=*/"");
  REQUIRE_FALSE(rejected_or.ok());
  REQUIRE(rejected_or.status().code() == absl::StatusCode::kPermissionDenied);

  auto accepted_or =
      svc.validate_local_target_layout("unix:/tmp/tensorcast.sock", "BatchGetIntoRegion", layout, params.owner_pid, "");
  REQUIRE(accepted_or.ok());

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

TEST_CASE(
    "ExternalTargetAccessService rejects non-local peers for HOST_SHARED source validation",
    "[daemon][byte_artifact][host_shared][trust_boundary][source]") {
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
  std::memset(mapping_or->base_ptr, 'B', static_cast<size_t>(kItemBytes));
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

  auto rejected_or = svc.validate_local_source_layout(
      "ipv4:10.0.0.8:12345", "BatchPutIfAbsentFromRegion", layout, params.owner_pid, "", expected_lengths);
  REQUIRE_FALSE(rejected_or.ok());
  REQUIRE(rejected_or.status().code() == absl::StatusCode::kPermissionDenied);

  auto accepted_or = svc.validate_local_source_layout(
      "ipv4:127.0.0.1:12345", "BatchPutIfAbsentFromRegion", layout, params.owner_pid, "", expected_lengths);
  REQUIRE(accepted_or.ok());

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

TEST_CASE(
    "ExternalTargetAccessService rejects HOST_SHARED byte-artifact offsets with partial slot tokens",
    "[daemon][byte_artifact][host_shared][slot_token]") {
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
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator;

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

  auto* offset = layout.add_offsets();
  offset->set_name("artifact-a");
  offset->set_storage_id("storage-0");
  offset->set_storage_offset(0);
  offset->set_logical_length(kItemBytes);
  offset->set_slot_index(7);

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
  REQUIRE_FALSE(validated_or.ok());
  REQUIRE(validated_or.status().code() == absl::StatusCode::kInvalidArgument);

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

TEST_CASE(
    "ExternalTargetAccessService requires explicit offsets for allocator-backed HOST_SHARED byte-artifact layouts",
    "[daemon][byte_artifact][host_shared][slot_token][allocator]") {
  tensorcast::daemon::IpcRegionRegistry registry(
      tensorcast::daemon::IpcRegionRegistry::Options{
          .capacity = 16,
          .max_ttl = absl::Milliseconds(5000),
      });

  tensorcast::daemon::IpcRegionRegistry::RegisterParams params;
  params.memory_kind = tensorcast::daemon::IpcRegionRegistry::MemoryKind::kHostShared;
  params.device_id = -1;
  params.owner_pid = getpid();
  params.size_bytes = kItemBytes;
  params.ttl_ms = 1000;
  params.daemon_managed = true;
  params.host_region_class = tensorcast::daemon::IpcRegionRegistry::HostRegionClass::kAllocator;

  auto desc_or = registry.register_region(params);
  REQUIRE(desc_or.ok());

  tensorcast::daemon::v2::TargetLayout layout;
  layout.set_layout_kind(tensorcast::daemon::v2::TargetLayout::LAYOUT_KIND_COALESCED_UNSPECIFIED);
  layout.set_index_kind(tensorcast::daemon::v2::TargetLayout::INDEX_KIND_CANONICAL_UNSPECIFIED);
  layout.set_tensor_spec_kind(tensorcast::daemon::v2::TargetLayout::TENSOR_SPEC_KIND_OFFSETS);

  auto* storage = layout.add_storages();
  storage->set_storage_id("storage-0");
  storage->set_device_id(-1);
  storage->set_storage_length(kItemBytes);
  storage->set_mapping_base_offset(0);
  auto* region_ref = storage->mutable_region_ref();
  region_ref->set_region_id(desc_or->region_id);
  region_ref->set_memory_kind(tensorcast::daemon::v2::REGION_MEMORY_KIND_HOST_SHARED);
  region_ref->set_device_id(-1);
  region_ref->set_size_bytes(kItemBytes);

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
  REQUIRE_FALSE(validated_or.ok());
  REQUIRE(validated_or.status().code() == absl::StatusCode::kInvalidArgument);

  REQUIRE(registry.unregister_region(desc_or->region_id, params.owner_pid, /*force=*/true).ok());
}

} // namespace
