// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/state/lip_manager.h"

#include <cstdint>
#include <optional>

#include <unistd.h>
#include <algorithm>
#include <utility>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "core/common/artifact_hash.h"
#include "core/cuda/cuda_ipc.h"
#include "core/cuda/device_guard.h"
#include "core/store/materialization/dataplane/metadata/canonical_index.h"
#include "core/store/materialization/dataplane/metadata/source_hash.h"
#include "core/store/materialization/dataplane/sources/byte_range_map_builder.h"
#include "daemon/state/lip_metadata_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

namespace {

std::string normalize_view_id(std::optional<std::string_view> view_id) {
  if (!view_id.has_value() || view_id->empty()) {
    return "";
  }
  return std::string(*view_id);
}

std::string format_tensor_key(std::string_view artifact_id, std::string_view view_id, std::string_view suffix) {
  if (view_id.empty()) {
    return absl::StrCat(artifact_id, "_", suffix);
  }
  return absl::StrCat(artifact_id, "_view_", view_id, "_", suffix);
}

class RegionAcquireGuard {
 public:
  explicit RegionAcquireGuard(IpcRegionRegistry* registry) : registry_(registry) {}

  absl::StatusOr<IpcRegionRegistry::RegionDescriptor> acquire(const std::string& region_id, int owner_pid) {
    if (registry_ == nullptr) {
      return absl::FailedPreconditionError("region registry unavailable");
    }
    auto desc_or = registry_->acquire(region_id, owner_pid);
    if (!desc_or.ok()) {
      return desc_or.status();
    }
    ++refs_[region_id];
    return desc_or;
  }

  ~RegionAcquireGuard() {
    if (registry_ == nullptr)
      return;
    for (const auto& [region_id, count] : refs_) {
      for (uint32_t i = 0; i < count; ++i) {
        absl::Status st = registry_->release(region_id);
        if (!st.ok()) {
          LOG(WARNING) << "RegionAcquireGuard: release failed for region_id=" << region_id << ": " << st;
        }
      }
    }
  }

 private:
  IpcRegionRegistry* registry_;
  absl::flat_hash_map<std::string, uint32_t> refs_;
};

absl::StatusOr<cuda::IpcMapping*> GetOrOpenMappingForStorage(
    const RegisterStorageMeta& storage,
    absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>>& cache,
    RegionAcquireGuard& guard,
    IpcRegionRegistry* registry,
    int owner_pid) {
  const std::string cache_key =
      storage.has_handle() ? absl::StrCat("h:", storage.handle_bytes) : absl::StrCat("r:", storage.region_id);
  auto it = cache.find(cache_key);
  if (it != cache.end()) {
    return it->second.get();
  }

  std::unique_ptr<cuda::IpcMapping> mapping;
  if (storage.has_handle()) {
    auto map_or =
        cuda::IpcMapping::open(storage.handle_bytes, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok())
      return map_or.status();
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else if (storage.has_region()) {
    auto desc_or = guard.acquire(storage.region_id, owner_pid);
    if (!desc_or.ok())
      return desc_or.status();
    if (registry == nullptr) {
      return absl::FailedPreconditionError("region registry unavailable");
    }
    auto handle_or = registry->get_handle_bytes(storage.region_id);
    if (!handle_or.ok())
      return handle_or.status();
    auto map_or = cuda::IpcMapping::open(*handle_or, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
    if (!map_or.ok())
      return map_or.status();
    mapping = std::make_unique<cuda::IpcMapping>(std::move(*map_or));
  } else {
    return absl::InvalidArgumentError("storage entry missing source handle or region");
  }

  auto [insert_it, _] = cache.emplace(cache_key, std::move(mapping));
  return insert_it->second.get();
}

} // namespace

namespace {

void release_region_refs(IpcRegionRegistry* registry, const absl::flat_hash_map<std::string, uint32_t>& refs) {
  if (registry == nullptr) {
    return;
  }
  for (const auto& [region_id, count] : refs) {
    for (uint32_t i = 0; i < count; ++i) {
      absl::Status st = registry->release(region_id);
      if (!st.ok()) {
        LOG(WARNING) << "LipManager: region release failed for region_id=" << region_id << ": " << st;
      }
    }
  }
}

} // namespace

absl::StatusOr<std::vector<uint8_t>> LipManager::copy_to_new_coalesced(
    int target_device_id,
    const std::string& canonical_index_json,
    uint64_t total_size,
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages,
    int owner_pid) {
  if (storages.empty()) {
    return absl::InvalidArgumentError("lip copy requires storage metadata");
  }
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  for (const auto& storage : storages) {
    storage_by_id.emplace(storage.storage_id, &storage);
  }
  for (const auto& seg : segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("segment references unknown storage_id");
    }
    const RegisterStorageMeta* storage = it->second;
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
              static_cast<uint64_t>(seg.length),
              static_cast<uint64_t>(storage->storage_length),
              storage->storage_id));
    }
    if (seg.storage_offset != 0) {
      return absl::InvalidArgumentError("segment storage_offset must be 0 for full-storage registrations");
    }
  }

  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  mapping_cache.reserve(storages.size());
  // Begin destination allocation on target device
  store::StoreEngine::ArtifactRegistration areg;
  areg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()));
  areg.tensor_index_data = canonical_index_json;
  areg.schema_version = "v3";
  areg.encoding = "json";
  areg.device_id = target_device_id;
  areg.total_size_bytes = total_size;
  areg.enable_p2p = true;
  auto begin_or = engine_->begin_register_artifact(areg);
  if (!begin_or.ok())
    return begin_or.status();
  const auto& out = *begin_or;

  struct RegAbortGuard {
    store::StoreEngine* engine;
    std::string id;
    bool active{true};

    ~RegAbortGuard() {
      if (active && engine) {
        absl::Status _st = engine->abort_registered_artifact(id);
        if (!_st.ok()) {
          LOG(WARNING) << "LipManager RegAbortGuard: abort_registered_artifact failed for id=" << id << ": " << _st;
        }
      }
    }

    void release() {
      active = false;
    }
  } abort_guard{.engine = engine_.get(), .id = out.registration_id};

  // Build plan and zero PAD regions on destination
  auto map_or = store::loader::build_byte_range_map_from_canonical_index_json(canonical_index_json, total_size);
  if (!map_or.ok()) {
    return map_or.status();
  }
  const auto& map = *map_or;

  auto dst_map_or =
      cuda::IpcMapping::open(out.cuda_ipc_handle_bytes, cuda::OpenOptions{.flags = cudaIpcMemLazyEnablePeerAccess});
  if (!dst_map_or.ok())
    return dst_map_or.status();
  auto dst_map = std::move(*dst_map_or);
  void* dst_dev = dst_map.get();
  cuda::CudaDeviceGuard dst_guard(target_device_id);
  if (!dst_guard.status().ok()) {
    return dst_guard.status();
  }
  for (const auto& seg : map.segments) {
    if (seg.kind != store::loader::ByteRangeSegment::Kind::kPad || seg.length == 0)
      continue;
    auto st = cuda::memset(static_cast<uint8_t*>(dst_dev) + seg.dst_offset, 0, static_cast<size_t>(seg.length));
    if (!st.ok())
      return st;
  }

  struct Opened {
    int device_id;
    cuda::IpcMapping* map;
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  std::vector<Opened> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("segment references unknown storage_id");
    }
    const RegisterStorageMeta* storage = it->second;
    auto map_or = GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, owner_pid);
    if (!map_or.ok()) {
      return map_or.status();
    }
    const uint64_t source_base = storage->mapping_base_offset + seg.storage_offset;
    opened.push_back(
        Opened{
            .device_id = storage->device_id,
            .map = *map_or,
            .base = source_base,
            .len = seg.length,
            .dst = seg.artifact_offset});
  }
  const size_t chunk_size = engine_->get_artifact_chunk_bytes();
  if (chunk_size == 0) {
    return absl::FailedPreconditionError("invalid artifact_chunk_bytes (0)");
  }
  auto is_aligned = [&](uint64_t v) { return (v % chunk_size) == 0; };
  for (const auto& o : opened) {
    const bool is_tail = (o.dst + o.len == total_size);
    if (!is_aligned(o.dst) || (!is_aligned(o.len) && !is_tail)) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "LIP segment not aligned to artifact_chunk_bytes: artifact_offset=",
              o.dst,
              ", length=",
              o.len,
              ", chunk=",
              chunk_size));
    }
    if (o.dst > total_size || o.len > total_size || o.dst + o.len > total_size) {
      return absl::OutOfRangeError("LIP segment dst range out of bounds");
    }
    auto st = cuda::memcpy(
        static_cast<uint8_t*>(dst_dev) + o.dst,
        static_cast<uint8_t*>(o.map->get()) + o.base,
        static_cast<size_t>(o.len),
        cudaMemcpyDeviceToDevice);
    if (!st.ok())
      return st;
  }
  if (auto sync = cuda::device_synchronize(); !sync.ok()) {
    return sync;
  }
  // Commit and return destination IPC handle bytes
  auto commit_or = engine_->commit_registered_artifact(out.registration_id);
  if (!commit_or.ok())
    return commit_or.status();
  abort_guard.release();
  auto handle_span = out.cuda_ipc_handle_bytes.as_bytes();
  std::vector<uint8_t> bytes(handle_span.begin(), handle_span.end());
  return bytes;
}

absl::StatusOr<std::string> LipManager::create_staged_export(
    const LipLeaseEntry& lip,
    absl::Span<const uint32_t> chunk_indices,
    store::StoreEngine& engine) {
  {
    absl::MutexLock l(&mu_);
    ArtifactDeviceKey key{.artifact_id = lip.artifact_id, .view_id = lip.view_id, .device_id = lip.device_id};
    if (quiesced_.contains(key)) {
      return absl::FailedPreconditionError("artifact is quiesced; staging new exports is blocked");
    }
  }
  const auto now = std::chrono::steady_clock::now();
  if (lip.expiry.time_since_epoch().count() > 0 && now > lip.expiry) {
    return absl::DeadlineExceededError("lease expired (TTL)");
  }
  auto comm_mgr = engine.get_shared_comm_manager();
  if (!comm_mgr->is_enabled()) {
    return absl::UnavailableError("communication engine not enabled");
  }
  auto& comm_engine = comm_mgr->get_engine();
  const size_t chunk_size = engine.get_artifact_chunk_bytes();
  if (chunk_size == 0) {
    return absl::FailedPreconditionError("invalid chunk_size (0)");
  }

  if (lip.storages.empty()) {
    return absl::FailedPreconditionError("staged export requires storage metadata");
  }

  struct OpenedSeg {
    int device_id;
    cuda::IpcMapping* map; // non-owning; lifetime held by mapping_cache/export record
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  std::vector<OpenedSeg> opened;
  opened.reserve(lip.segments.size());
  // When storage metadata is present, prefer region-backed or cached mappings via registry.
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(lip.storages.size());
  for (const auto& s : lip.storages) {
    storage_by_id.emplace(s.storage_id, &s);
  }
  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  absl::flat_hash_map<std::string, uint32_t> region_hold_counts; // region_id -> refcount to hold beyond this scope
  mapping_cache.reserve(lip.storages.size());
  for (const auto& seg : lip.segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("staged export: unknown storage_id in segment");
    }
    const RegisterStorageMeta* storage = it->second;
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
              static_cast<uint64_t>(seg.length),
              static_cast<uint64_t>(storage->storage_length),
              storage->storage_id));
    }
    if (seg.storage_offset != 0) {
      return absl::InvalidArgumentError("segment storage_offset must be 0 for full-storage registrations");
    }
    // Track first-time region-backed mapping per storage to hold registry refs.
    if (storage->has_region()) {
      const std::string region_key = absl::StrCat("r:", storage->region_id);
      if (mapping_cache.find(region_key) == mapping_cache.end()) {
        // Hold exactly one ref per region used by this export.
        ++region_hold_counts[storage->region_id];
      }
    }
    auto map_or = GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, lip.owner_pid);
    if (!map_or.ok()) {
      return map_or.status();
    }
    const uint64_t source_base = storage->mapping_base_offset + seg.storage_offset;
    opened.push_back(
        OpenedSeg{
            .device_id = storage->device_id,
            .map = *map_or,
            .base = source_base,
            .len = seg.length,
            .dst = seg.artifact_offset,
        });
  }
  // Enforce alignment of LIP segments to artifact_chunk_bytes
  auto is_aligned = [&](uint64_t v) { return (v % chunk_size) == 0; };
  for (const auto& s : opened) {
    const bool is_tail = (s.dst + s.len == lip.total_size);
    if (!is_aligned(s.dst) || (!is_aligned(s.len) && !is_tail)) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "LIP segment not aligned to artifact_chunk_bytes: artifact_offset=",
              s.dst,
              ", length=",
              s.len,
              ", chunk=",
              chunk_size));
    }
  }
  // Reacquire region refs we need to hold for the lifetime of this export.
  if (!region_hold_counts.empty()) {
    if (region_registry_ == nullptr) {
      return absl::FailedPreconditionError("region registry unavailable");
    }
    for (const auto& kv : region_hold_counts) {
      const std::string& region_id = kv.first;
      const uint32_t count = kv.second;
      for (uint32_t i = 0; i < count; ++i) {
        auto desc_or = region_registry_->acquire(region_id, lip.owner_pid);
        if (!desc_or.ok()) {
          return desc_or.status();
        }
      }
    }
  }
  auto find_covering = [&](uint64_t off, uint64_t len) -> const OpenedSeg* {
    for (const auto& s : opened) {
      if (off >= s.dst && (off + len) <= (s.dst + s.len))
        return &s;
    }
    return nullptr;
  };
  std::vector<std::string> tensor_keys;
  tensor_keys.reserve(chunk_indices.size());

  // Guard to ensure partial registrations are cleaned up on failure
  struct KeysGuard {
    communicator::engine::Communicator* comm_engine{nullptr};
    std::vector<std::string>* keys{nullptr};

    ~KeysGuard() {
      if (comm_engine && keys) {
        for (const auto& k : *keys) {
          auto st = comm_engine->unregister_tensor(k);
          if (!st.ok()) {
            LOG(WARNING) << "LIP KeysGuard: unregister_tensor failed for key=" << k << ": " << st;
          }
        }
      }
    }

    void release() {
      comm_engine = nullptr;
      keys = nullptr;
    }
  } guard;

  guard.comm_engine = &comm_engine;
  guard.keys = &tensor_keys;
  for (uint32_t idx : chunk_indices) {
    uint64_t off = static_cast<uint64_t>(idx) * static_cast<uint64_t>(chunk_size);
    if (off >= lip.total_size) {
      return absl::OutOfRangeError("chunk offset exceeds artifact size");
    }
    uint64_t len = std::min<uint64_t>(chunk_size, lip.total_size - off);
    const OpenedSeg* seg = find_covering(off, len);
    if (!seg) {
      return absl::UnimplementedError("LIP chunk crosses segment boundary (unsupported)");
    }
    uint64_t addr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(seg->map->get()) + (seg->base + (off - seg->dst)));
    std::string tkey = format_tensor_key(lip.artifact_id, lip.view_id, absl::StrCat("GPU_chunk_", idx));
    communicator::engine::Communicator::RegisterTensorOptions ro;
    ro.register_mr = comm_engine.is_rdma_enabled();
    ro.needs_staging = (!comm_engine.is_rdma_enabled());
    ro.async = false;
    ro.direct_rdma_enabled = comm_engine.is_rdma_enabled();
    auto st = comm_engine.register_tensor_ex(
        tkey, addr, static_cast<size_t>(len), communicator::base::COMMUNICATE_ENGINE_DEV_GPU, seg->device_id, ro);
    if (!st.ok())
      return st;
    tensor_keys.push_back(std::move(tkey));
  }
  std::string token = absl::StrCat("lipx_", absl::ToUnixNanos(absl::Now()));
  {
    absl::MutexLock lk(&exp_mu_);
    LipExportRecord rec;
    rec.artifact_id = lip.artifact_id;
    rec.view_id = lip.view_id;
    rec.device_id = lip.device_id;
    rec.region_registry = region_registry_;
    rec.held_region_refs = std::move(region_hold_counts);
    // Transfer ownership of cached mappings so they outlive this function
    for (auto& kv : mapping_cache) {
      rec.opened_maps.push_back(std::move(*kv.second));
    }
    rec.tensor_keys = std::move(tensor_keys);
    exports_.emplace(token, std::move(rec));
  }
  // Success: prevent cleanup of registered keys
  guard.release();
  return token;
}

absl::Status LipManager::release_staged_export(const std::string& token, store::StoreEngine& engine) {
  absl::MutexLock lk(&exp_mu_);
  auto it = exports_.find(token);
  if (it == exports_.end())
    return absl::NotFoundError("unknown lock token");
  auto comm_mgr = engine.get_shared_comm_manager();
  if (comm_mgr->is_enabled()) {
    auto& comm_engine = comm_mgr->get_engine();
    for (const auto& k : it->second.tensor_keys) {
      auto st = comm_engine.unregister_tensor(k);
      if (!st.ok()) {
        LOG(WARNING) << "release_staged_export: unregister_tensor failed for key=" << k << ": " << st;
      }
    }
  }
  // Preserve region lease info, then erase to release CUDA mappings before releasing region refs
  IpcRegionRegistry* registry = it->second.region_registry;
  absl::flat_hash_map<std::string, uint32_t> held = std::move(it->second.held_region_refs);
  exports_.erase(it);
  if (registry != nullptr) {
    for (const auto& kv : held) {
      const std::string& region_id = kv.first;
      const uint32_t count = kv.second;
      for (uint32_t i = 0; i < count; ++i) {
        absl::Status st = registry->release(region_id);
        if (!st.ok()) {
          LOG(WARNING) << "release_staged_export: region release failed for region_id=" << region_id << ": " << st;
        }
      }
    }
  }
  return absl::OkStatus();
}

void LipManager::put_lease(const std::string& registration_id, const ArtifactDeviceKey& key, LipLeaseEntry entry) {
  absl::MutexLock l(&mu_);
  entry.view_id = key.view_id;
  reg_to_key_[registration_id] = key;
  leases_[key] = std::move(entry);
}

absl::Status LipManager::keepalive_lease(
    const std::string& registration_id,
    int owner_pid,
    uint64_t epoch,
    uint32_t ttl_ms) {
  std::vector<RegisterStorageMeta> storages;
  uint32_t extend = 0;
  {
    absl::MutexLock l(&mu_);
    auto itk = reg_to_key_.find(registration_id);
    if (itk == reg_to_key_.end())
      return absl::NotFoundError("registration_id not found");
    auto it = leases_.find(itk->second);
    if (it == leases_.end())
      return absl::NotFoundError("lease not found");
    if (it->second.owner_pid != owner_pid)
      return absl::PermissionDeniedError("owner_pid mismatch");
    it->second.epoch = epoch;
    extend = ttl_ms;
    it->second.ttl_ms = extend;
    if (extend == 0) {
      it->second.expiry = std::chrono::steady_clock::time_point{};
    } else {
      it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(extend);
    }
    storages = it->second.storages;
  }
  if (region_registry_ != nullptr && extend > 0) {
    for (const auto& storage : storages) {
      if (storage.has_region()) {
        (void)region_registry_->refresh_ttl(storage.region_id, extend);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LipManager::revoke_by_registration_id(const std::string& registration_id) {
  std::optional<ArtifactDeviceKey> key;
  {
    absl::MutexLock l(&mu_);
    auto itk = reg_to_key_.find(registration_id);
    if (itk == reg_to_key_.end()) {
      return absl::NotFoundError("registration_id not found");
    }
    key = itk->second;
    leases_.erase(*key);
    reg_to_key_.erase(itk);
    quiesced_.erase(*key);
  }

  std::optional<LipExportRecord> export_record;
  std::string replica_id;
  {
    absl::MutexLock lk(&routable_mu_);
    auto it = routable_exports_.find(*key);
    if (it != routable_exports_.end()) {
      export_record = std::move(it->second);
      routable_exports_.erase(it);
    }
    auto rid = routable_replica_ids_.find(*key);
    if (rid != routable_replica_ids_.end()) {
      replica_id = std::move(rid->second);
      routable_replica_ids_.erase(rid);
    }
  }

  if (global_store_client_ && global_store_client_->is_connected() && !replica_id.empty()) {
    absl::Status st = global_store_client_->unregister_replica(key->artifact_id, replica_id);
    if (!st.ok()) {
      LOG(WARNING) << "LipManager revoke: unregister_replica failed for artifact_id=" << key->artifact_id
                   << " replica_id=" << replica_id << ": " << st;
    }
  }

  if (export_record.has_value()) {
    auto comm_mgr = engine_->get_shared_comm_manager();
    if (comm_mgr->is_enabled()) {
      auto& comm_engine = comm_mgr->get_engine();
      for (const auto& k : export_record->tensor_keys) {
        auto st = comm_engine.unregister_tensor(k);
        if (!st.ok()) {
          LOG(WARNING) << "LipManager revoke: unregister_tensor failed for key=" << k << ": " << st;
        }
      }
    }
    release_region_refs(export_record->region_registry, export_record->held_region_refs);
  }

  return absl::OkStatus();
}

std::optional<LipLeaseEntry> LipManager::find_active_by_artifact_id(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  const std::string view_key = normalize_view_id(view_id);
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  for (const auto& kv : leases_) {
    const auto& key = kv.first;
    const auto& e = kv.second;
    if (key.artifact_id != artifact_id)
      continue;
    if (key.view_id != view_key)
      continue;
    if (e.expiry.time_since_epoch().count() > 0 && now > e.expiry)
      continue;
    return e;
  }
  return std::nullopt;
}

std::optional<LipLeaseEntry> LipManager::find_active_by_key(const ArtifactDeviceKey& key) const {
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  auto it = leases_.find(key);
  if (it == leases_.end()) {
    return std::nullopt;
  }
  if (it->second.expiry.time_since_epoch().count() > 0 && now > it->second.expiry) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<LipLeaseEntry> LipManager::list_active_by_artifact_id(
    std::string_view artifact_id,
    std::optional<std::string_view> view_id) const {
  const std::string view_key = normalize_view_id(view_id);
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  std::vector<LipLeaseEntry> matches;
  for (const auto& kv : leases_) {
    const auto& key = kv.first;
    const auto& e = kv.second;
    if (key.artifact_id != artifact_id) {
      continue;
    }
    if (key.view_id != view_key) {
      continue;
    }
    if (e.expiry.time_since_epoch().count() > 0 && now > e.expiry) {
      continue;
    }
    matches.push_back(e);
  }
  return matches;
}

std::optional<ArtifactDeviceKey> LipManager::find_key_by_registration_id(std::string_view registration_id) const {
  absl::MutexLock l(&mu_);
  auto it = reg_to_key_.find(std::string(registration_id));
  if (it == reg_to_key_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<std::string> LipManager::find_replica_id(const ArtifactDeviceKey& key) const {
  absl::MutexLock lk(&routable_mu_);
  auto it = routable_replica_ids_.find(key);
  if (it == routable_replica_ids_.end()) {
    return std::nullopt;
  }
  return it->second;
}

bool LipManager::has_active_on_device(
    std::string_view artifact_id,
    int device_id,
    std::optional<std::string_view> view_id) const {
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  ArtifactDeviceKey k{
      .artifact_id = std::string(artifact_id), .view_id = normalize_view_id(view_id), .device_id = device_id};
  auto it = leases_.find(k);
  if (it == leases_.end())
    return false;
  if (it->second.expiry.time_since_epoch().count() > 0 && now > it->second.expiry)
    return false;
  return true;
}

void LipManager::sweep_expired_and_dead_pids() {
  std::vector<ArtifactDeviceKey> to_erase;
  {
    absl::MutexLock l(&mu_);
    const auto now = std::chrono::steady_clock::now();
    to_erase.reserve(leases_.size());
    for (const auto& kv : leases_) {
      const auto& key = kv.first;
      const auto& e = kv.second;
      if (e.expiry.time_since_epoch().count() > 0 && now > e.expiry) {
        to_erase.push_back(key);
        continue;
      }
      std::string proc_path = absl::StrCat("/proc/", e.owner_pid);
      if (::access(proc_path.c_str(), F_OK) != 0) {
        to_erase.push_back(key);
      }
    }
    if (!to_erase.empty()) {
      for (const auto& key : to_erase) {
        // erase reg->key mappings that match
        std::vector<std::string> regs;
        regs.reserve(reg_to_key_.size());
        for (const auto& rk : reg_to_key_) {
          if (rk.second == key)
            regs.push_back(rk.first);
        }
        for (const auto& rid : regs)
          reg_to_key_.erase(rid);
        leases_.erase(key);
        quiesced_.erase(key);
      }
    }
  }

  for (const auto& key : to_erase) {
    std::optional<LipExportRecord> export_record;
    std::string replica_id;
    {
      absl::MutexLock lk(&routable_mu_);
      auto it = routable_exports_.find(key);
      if (it != routable_exports_.end()) {
        export_record = std::move(it->second);
        routable_exports_.erase(it);
      }
      auto rid = routable_replica_ids_.find(key);
      if (rid != routable_replica_ids_.end()) {
        replica_id = std::move(rid->second);
        routable_replica_ids_.erase(rid);
      }
    }

    if (global_store_client_ && global_store_client_->is_connected() && !replica_id.empty()) {
      absl::Status st = global_store_client_->unregister_replica(key.artifact_id, replica_id);
      if (!st.ok()) {
        LOG(WARNING) << "LipManager sweep: unregister_replica failed for artifact_id=" << key.artifact_id
                     << " replica_id=" << replica_id << ": " << st;
      }
    }

    if (export_record.has_value()) {
      auto comm_mgr = engine_->get_shared_comm_manager();
      if (comm_mgr->is_enabled()) {
        auto& comm_engine = comm_mgr->get_engine();
        for (const auto& k : export_record->tensor_keys) {
          auto st = comm_engine.unregister_tensor(k);
          if (!st.ok()) {
            LOG(WARNING) << "LipManager sweep: unregister_tensor failed for key=" << k << ": " << st;
          }
        }
      }
      release_region_refs(export_record->region_registry, export_record->held_region_refs);
    }
  }
}

bool LipManager::revoke_commit_lease_if_owner_matches(
    std::string_view artifact_id,
    int device_id,
    int owner_pid,
    std::optional<std::string_view> view_id) {
  absl::MutexLock l(&mu_);
  ArtifactDeviceKey key{
      .artifact_id = std::string(artifact_id), .view_id = normalize_view_id(view_id), .device_id = device_id};
  auto it = leases_.find(key);
  if (it == leases_.end())
    return false;
  if (it->second.owner_pid != owner_pid)
    return false;
  // Erase any reg->key mappings that point to this key
  std::vector<std::string> regs;
  regs.reserve(reg_to_key_.size());
  for (const auto& rk : reg_to_key_) {
    if (rk.second == key)
      regs.push_back(rk.first);
  }
  for (const auto& rid : regs)
    reg_to_key_.erase(rid);
  leases_.erase(it);
  quiesced_.erase(key);
  VLOG(2) << "revoke_commit_lease_if_owner_matches: erased lease for artifact_id=" << artifact_id
          << " device_id=" << device_id;
  return true;
}

absl::Status LipManager::extend_ttl_for_artifact(
    std::string_view artifact_id,
    uint32_t extend_ttl_ms,
    std::optional<std::string_view> view_id) {
  if (extend_ttl_ms == 0)
    return absl::OkStatus();
  const std::string view_key = normalize_view_id(view_id);
  std::optional<LipLeaseEntry> found;
  {
    absl::MutexLock l(&mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto& kv : leases_) {
      if (kv.first.artifact_id != artifact_id)
        continue;
      if (kv.first.view_id != view_key)
        continue;
      auto& e = kv.second;
      // Update TTL/expiry
      e.ttl_ms = extend_ttl_ms;
      e.expiry = now + std::chrono::milliseconds(extend_ttl_ms);
      found = e; // copy for region bump below
      break;
    }
  }
  if (!found.has_value())
    return absl::NotFoundError("no active lease for artifact");
  // Opportunistically bump region TTLs referenced by the lease
  if (region_registry_ != nullptr && extend_ttl_ms > 0) {
    for (const auto& s : found->storages) {
      if (s.has_region()) {
        (void)region_registry_->refresh_ttl(s.region_id, extend_ttl_ms);
      }
    }
  }
  return absl::OkStatus();
}

void LipManager::quiesce_lease(const ArtifactDeviceKey& key) {
  absl::MutexLock l(&mu_);
  quiesced_.insert(key);
}

bool LipManager::wait_exports_drained(const ArtifactDeviceKey& key, absl::Time deadline) {
  // Simple polling with backoff; exports_ rarely large
  while (absl::Now() < deadline) {
    size_t active = 0;
    {
      absl::MutexLock lk(&exp_mu_);
      for (const auto& kv : exports_) {
        if (kv.second.artifact_id == key.artifact_id && kv.second.view_id == key.view_id &&
            kv.second.device_id == key.device_id)
          ++active;
      }
    }
    if (active == 0)
      return true;
    absl::SleepFor(absl::Milliseconds(5));
  }
  return false;
}

absl::StatusOr<CommitLeaseResult> LipManager::commit_lease_in_place(
    const std::string& registration_id,
    int device_id,
    int owner_pid,
    uint32_t ttl_ms,
    uint64_t epoch,
    uint64_t total_size,
    tensorcast::common::ArtifactIdKind id_kind,
    const std::string& client_artifact_id,
    const std::string& index_data,
    const std::string& index_key_hex,
    std::vector<LeaseSegMeta>&& segments,
    std::vector<RegisterStorageMeta>&& storages,
    std::vector<RegisterTensorAliasMeta>&& aliases) {
  std::string canonical_index_json = index_data;
  if (!storages.empty() && !aliases.empty()) {
    auto rebuilt_or = build_canonical_index_from_metadata(
        absl::MakeSpan(segments), absl::MakeSpan(storages), absl::MakeSpan(aliases), device_id);
    if (!rebuilt_or.ok()) {
      return rebuilt_or.status();
    }
    const std::string& rebuilt = *rebuilt_or;
    if (canonical_index_json.empty()) {
      canonical_index_json = rebuilt;
    } else {
      auto stable_or = store::loader::rebuild_stable_canonical_index(canonical_index_json, device_id);
      if (stable_or.ok() && *stable_or != rebuilt) {
        LOG(WARNING) << "Rebuilt canonical index differs from client-provided data; "
                     << "preferring rebuilt version for registration_id=" << registration_id;
      }
      canonical_index_json = rebuilt;
    }
  }

  if (canonical_index_json.empty() && index_key_hex.empty()) {
    return absl::FailedPreconditionError("canonical index is required for LIP commit");
  }

  // Build seekable source over LIP segments using RAII CUDA IPC mappings.
  struct OpenedSeg {
    int device_id;
    cuda::IpcMapping* map; // non-owning; lifetime held by mapping_cache
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  for (const auto& s : storages) {
    storage_by_id.emplace(s.storage_id, &s);
  }

  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  mapping_cache.reserve(storages.size());

  std::vector<OpenedSeg> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("segment references unknown storage_id");
    }
    const RegisterStorageMeta* storage = it->second;
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
              static_cast<uint64_t>(seg.length),
              static_cast<uint64_t>(storage->storage_length),
              storage->storage_id));
    }
    if (seg.storage_offset != 0) {
      return absl::InvalidArgumentError("segment storage_offset must be 0 for full-storage registrations");
    }
    auto map_or = GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, owner_pid);
    if (!map_or.ok()) {
      return map_or.status();
    }
    const uint64_t source_base = storage->mapping_base_offset + seg.storage_offset;
    opened.push_back(
        OpenedSeg{
            .device_id = storage->device_id,
            .map = *map_or,
            .base = source_base,
            .len = seg.length,
            .dst = seg.artifact_offset,
        });
  }

  class LipSeekableSource final : public store::loader::SeekableSource {
   public:
    explicit LipSeekableSource(std::vector<OpenedSeg> segs, uint64_t total) : segs_(std::move(segs)), total_(total) {
      std::sort(segs_.begin(), segs_.end(), [](const OpenedSeg& a, const OpenedSeg& b) { return a.dst < b.dst; });
    }

    ~LipSeekableSource() override = default;

    [[nodiscard]] uint64_t total_bytes() const override {
      return total_;
    }

    absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
      auto read_or = read_at(cur_, dst, max_bytes);
      if (!read_or.ok()) {
        return read_or;
      }
      cur_ += *read_or;
      return read_or;
    }

    absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
      if (offset >= total_)
        return static_cast<size_t>(0);
      size_t remaining = static_cast<size_t>(std::min<uint64_t>(bytes, total_ - offset));
      uint8_t* out = static_cast<uint8_t*>(dst);
      while (remaining > 0) {
        // Find segment that covers offset
        const OpenedSeg* seg = nullptr;
        for (const auto& s : segs_) {
          if (offset >= s.dst && offset < s.dst + s.len) {
            seg = &s;
            break;
          }
          if (offset < s.dst)
            break;
        }
        if (seg == nullptr) {
          // Fill until next segment or end
          uint64_t next = total_;
          for (const auto& s : segs_) {
            if (s.dst > offset) {
              next = s.dst;
              break;
            }
          }
          size_t take = static_cast<size_t>(std::min<uint64_t>(remaining, next - offset));
          std::memset(out, 0, take);
          out += take;
          offset += take;
          remaining -= take;
          continue;
        }
        const uint64_t local = offset - seg->dst;
        const size_t avail = static_cast<size_t>(seg->len - local);
        const size_t take = std::min(remaining, avail);
        cuda::CudaDeviceGuard seg_guard(seg->device_id);
        if (!seg_guard.status().ok()) {
          return seg_guard.status();
        }
        auto st = cuda::memcpy(
            out, static_cast<uint8_t*>(seg->map->get()) + (seg->base + local), take, cudaMemcpyDeviceToHost);
        if (!st.ok())
          return st;
        if (auto sync = cuda::device_synchronize(); !sync.ok())
          return sync;
        out += take;
        offset += take;
        remaining -= take;
      }
      return static_cast<size_t>(out - static_cast<uint8_t*>(dst));
    }

    [[nodiscard]] bool supports_direct_write_at() const override {
      return false;
    }

    absl::StatusOr<size_t> read_into_at(uint64_t, uint64_t, size_t, const store::DirectWriteGrant&) override {
      return absl::UnimplementedError("direct write not supported for LIP seekable source");
    }

   private:
    std::vector<OpenedSeg> segs_;
    uint64_t total_;
    uint64_t cur_{0};
  } src(std::move(opened), total_size);

  std::string index_multihash;
  std::string data_multihash;
  CommitLeaseResult out;

  auto index_mh_or = common::compute_index_multihash(
      canonical_index_json.empty() ? std::optional<std::string>() : std::optional<std::string>(canonical_index_json),
      index_key_hex);
  if (!index_mh_or.ok()) {
    return index_mh_or.status();
  }
  index_multihash = *index_mh_or;

  if (id_kind == common::ArtifactIdKind::kMi2 || client_artifact_id.empty()) {
    auto data_mh_or = store::loader::compute_data_multihash_from_seekable_source(src, total_size);
    if (!data_mh_or.ok())
      return data_mh_or.status();
    data_multihash = *data_mh_or;

    out.index_multihash = index_multihash;
    out.data_multihash = data_multihash;
    out.artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);
    out.id_kind = common::ArtifactIdKind::kMi2;
  } else {
    out.index_multihash = index_multihash;
    out.data_multihash.clear();
    out.artifact_id = client_artifact_id;
    out.id_kind = common::ArtifactIdKind::kCgid;
  }

  out.schema_version = "v3";
  out.encoding = "json";
  out.total_size = total_size;

  // Enforce device-unique commit for VRAM_LEASED: (artifact_id, device_id)
  {
    absl::MutexLock l(&mu_);
    ArtifactDeviceKey k{.artifact_id = out.artifact_id, .view_id = "", .device_id = device_id};
    auto it = leases_.find(k);
    if (it != leases_.end()) {
      const auto now = std::chrono::steady_clock::now();
      const bool active = it->second.expiry.time_since_epoch().count() <= 0 || !(now > it->second.expiry);
      if (active) {
        return absl::AlreadyExistsError(
            absl::StrCat("lease already exists for artifact on device (pid=", it->second.owner_pid, ")"));
      }
    }
  }

  // Verification JSON: read three 8-byte values (start/middle/end) if possible
  auto read_u64 = [&](uint64_t off) -> std::optional<uint64_t> {
    uint64_t v = 0;
    auto st = src.read_at(off, &v, sizeof(v));
    if (!st.ok())
      return std::nullopt;
    if (*st != sizeof(v))
      return std::nullopt;
    return v;
  };
  if (total_size >= 8) {
    auto v0 = read_u64(0);
    auto v1 = read_u64(total_size / 2);
    auto v2 = read_u64(total_size - 8);
    if (v0 && v1 && v2) {
      nlohmann::json jv;
      jv["artifact_size"] = total_size;
      jv["key_values"] = {*v0, *v1, *v2};
      out.verification_json = jv.dump();
    }
  }

  // Register LIP entry for TTL/keepalive
  LipLeaseEntry lease;
  lease.registration_id = registration_id;
  lease.artifact_id = out.artifact_id;
  lease.view_id = "";
  lease.client_artifact_id = client_artifact_id;
  lease.id_kind = out.id_kind;
  lease.device_id = device_id;
  lease.owner_pid = owner_pid;
  lease.ttl_ms = ttl_ms;
  if (ttl_ms == 0) {
    lease.expiry = std::chrono::steady_clock::time_point{};
  } else {
    lease.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  }
  lease.epoch = epoch;
  lease.total_size = total_size;
  lease.index_data = canonical_index_json;
  lease.segments = std::move(segments);
  lease.storages = std::move(storages);
  lease.aliases = std::move(aliases);
  lease.verification_json = out.verification_json;

  {
    ArtifactDeviceKey key{.artifact_id = lease.artifact_id, .view_id = lease.view_id, .device_id = lease.device_id};
    put_lease(registration_id, key, std::move(lease));
  }

  return out;
}

absl::StatusOr<LipManager::RoutableLeaseResult> LipManager::commit_routable_view_lease_in_place(
    const std::string& registration_id,
    std::string_view artifact_id,
    std::string_view view_id,
    int device_id,
    int owner_pid,
    uint32_t ttl_ms,
    uint64_t epoch,
    uint64_t total_size,
    std::vector<LeaseSegMeta>&& segments,
    std::vector<RegisterStorageMeta>&& storages) {
  if (artifact_id.empty()) {
    return absl::InvalidArgumentError("artifact_id is required for routable LIP lease");
  }
  if (storages.empty()) {
    return absl::InvalidArgumentError("routable LIP lease requires storage metadata");
  }
  if (segments.empty()) {
    return absl::InvalidArgumentError("routable LIP lease requires lease segments");
  }
  {
    absl::MutexLock l(&mu_);
    ArtifactDeviceKey key{
        .artifact_id = std::string(artifact_id), .view_id = std::string(view_id), .device_id = device_id};
    if (quiesced_.contains(key)) {
      return absl::FailedPreconditionError("artifact is quiesced; new routable leases are blocked");
    }
  }

  // Enforce device+view unique lease for VRAM_LEASED pieces: (artifact_id, view_id, device_id)
  {
    absl::MutexLock l(&mu_);
    ArtifactDeviceKey k{
        .artifact_id = std::string(artifact_id), .view_id = std::string(view_id), .device_id = device_id};
    auto it = leases_.find(k);
    if (it != leases_.end()) {
      const auto now = std::chrono::steady_clock::now();
      const bool active = it->second.expiry.time_since_epoch().count() <= 0 || !(now > it->second.expiry);
      if (active) {
        return absl::AlreadyExistsError(
            absl::StrCat("lease already exists for view on device (pid=", it->second.owner_pid, ")"));
      }
    }
  }

  auto comm_mgr = engine_->get_shared_comm_manager();
  if (!comm_mgr->is_enabled()) {
    return absl::UnavailableError("communication engine not enabled");
  }
  auto& comm_engine = comm_mgr->get_engine();

  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  for (const auto& s : storages) {
    storage_by_id.emplace(s.storage_id, &s);
  }

  struct OpenedSeg {
    int device_id;
    cuda::IpcMapping* map; // non-owning; lifetime held by mapping_cache/export record
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<cuda::IpcMapping>> mapping_cache;
  mapping_cache.reserve(storages.size());
  absl::flat_hash_map<std::string, uint32_t> region_hold_counts; // region_id -> refcount to hold beyond this scope

  std::vector<OpenedSeg> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    auto it = storage_by_id.find(seg.storage_id);
    if (it == storage_by_id.end()) {
      return absl::InvalidArgumentError("segment references unknown storage_id");
    }
    const RegisterStorageMeta* storage = it->second;
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
              static_cast<uint64_t>(seg.length),
              static_cast<uint64_t>(storage->storage_length),
              storage->storage_id));
    }
    if (seg.storage_offset != 0) {
      return absl::InvalidArgumentError("segment storage_offset must be 0 for full-storage registrations");
    }
    if (seg.artifact_offset > total_size || seg.length > total_size || seg.artifact_offset + seg.length > total_size) {
      return absl::OutOfRangeError("segment dst range out of bounds");
    }
    if (storage->device_id != device_id) {
      return absl::FailedPreconditionError("storage device_id mismatch for routable view lease");
    }
    if (storage->has_region()) {
      const std::string region_key = absl::StrCat("r:", storage->region_id);
      if (mapping_cache.find(region_key) == mapping_cache.end()) {
        ++region_hold_counts[storage->region_id];
      }
    }
    auto map_or = GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, owner_pid);
    if (!map_or.ok()) {
      return map_or.status();
    }
    const uint64_t source_base = storage->mapping_base_offset + seg.storage_offset;
    opened.push_back(
        OpenedSeg{
            .device_id = storage->device_id,
            .map = *map_or,
            .base = source_base,
            .len = seg.length,
            .dst = seg.artifact_offset,
        });
  }

  std::sort(opened.begin(), opened.end(), [](const OpenedSeg& a, const OpenedSeg& b) { return a.dst < b.dst; });

  // Require dense coverage [0,total_size) without gaps so remote keys can be sequenced.
  uint64_t cursor = 0;
  for (const auto& seg : opened) {
    if (seg.len == 0) {
      continue;
    }
    if (seg.dst != cursor) {
      return absl::FailedPreconditionError("routable view lease requires dense segments (gap or overlap)");
    }
    cursor = seg.dst + seg.len;
  }
  if (cursor != total_size) {
    return absl::FailedPreconditionError("routable view lease requires dense segments (tail gap)");
  }

  // Reacquire region refs we need to hold for the lifetime of this export.
  if (!region_hold_counts.empty()) {
    if (region_registry_ == nullptr) {
      return absl::FailedPreconditionError("region registry unavailable");
    }
    for (const auto& kv : region_hold_counts) {
      const std::string& region_id = kv.first;
      const uint32_t count = kv.second;
      for (uint32_t i = 0; i < count; ++i) {
        auto desc_or = region_registry_->acquire(region_id, owner_pid);
        if (!desc_or.ok()) {
          return desc_or.status();
        }
      }
    }
  }

  std::vector<std::string> tensor_keys;
  std::vector<std::string> remote_memory_keys;
  std::vector<uint64_t> buffer_sizes;
  tensor_keys.reserve(opened.size());
  remote_memory_keys.reserve(opened.size());
  buffer_sizes.reserve(opened.size());

  // Guard to ensure partial registrations are cleaned up on failure.
  struct KeysGuard {
    communicator::engine::Communicator* comm_engine{nullptr};
    std::vector<std::string>* keys{nullptr};

    ~KeysGuard() {
      if (comm_engine && keys) {
        for (const auto& k : *keys) {
          auto st = comm_engine->unregister_tensor(k);
          if (!st.ok()) {
            LOG(WARNING) << "LIP KeysGuard: unregister_tensor failed for key=" << k << ": " << st;
          }
        }
      }
    }

    void release() {
      comm_engine = nullptr;
      keys = nullptr;
    }
  } guard;

  guard.comm_engine = &comm_engine;
  guard.keys = &tensor_keys;

  for (size_t i = 0; i < opened.size(); ++i) {
    const auto& seg = opened[i];
    const uint64_t addr = reinterpret_cast<uint64_t>(static_cast<uint8_t*>(seg.map->get()) + seg.base);
    std::string tkey = format_tensor_key(artifact_id, view_id, absl::StrCat("GPU_view_seg_", i));
    communicator::engine::Communicator::RegisterTensorOptions ro;
    ro.register_mr = comm_engine.is_rdma_enabled();
    ro.needs_staging = (!comm_engine.is_rdma_enabled());
    ro.async = false;
    ro.direct_rdma_enabled = comm_engine.is_rdma_enabled();
    auto st = comm_engine.register_tensor_ex(
        tkey, addr, static_cast<size_t>(seg.len), communicator::base::COMMUNICATE_ENGINE_DEV_GPU, seg.device_id, ro);
    if (!st.ok()) {
      release_region_refs(region_registry_, region_hold_counts);
      return st;
    }
    tensor_keys.push_back(tkey);
    remote_memory_keys.push_back(std::move(tkey));
    buffer_sizes.push_back(seg.len);
  }

  // Persist the lease for keepalive/revoke.
  LipLeaseEntry lease;
  lease.registration_id = registration_id;
  lease.artifact_id = std::string(artifact_id);
  lease.view_id = std::string(view_id);
  lease.client_artifact_id = std::string(artifact_id);
  lease.id_kind = common::ArtifactIdKind::kCgid;
  lease.device_id = device_id;
  lease.owner_pid = owner_pid;
  lease.ttl_ms = ttl_ms;
  if (ttl_ms == 0) {
    lease.expiry = std::chrono::steady_clock::time_point{};
  } else {
    lease.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(ttl_ms);
  }
  lease.epoch = epoch;
  lease.total_size = total_size;
  lease.index_data.clear();
  lease.segments = std::move(segments);
  lease.storages = std::move(storages);
  lease.aliases.clear();

  ArtifactDeviceKey key{.artifact_id = lease.artifact_id, .view_id = lease.view_id, .device_id = lease.device_id};
  put_lease(registration_id, key, std::move(lease));

  {
    absl::MutexLock lk(&routable_mu_);
    LipExportRecord rec;
    rec.artifact_id = std::string(artifact_id);
    rec.view_id = std::string(view_id);
    rec.device_id = device_id;
    rec.region_registry = region_registry_;
    rec.held_region_refs = std::move(region_hold_counts);
    for (auto& kv : mapping_cache) {
      rec.opened_maps.push_back(std::move(*kv.second));
    }
    rec.tensor_keys = std::move(tensor_keys);
    routable_exports_[key] = std::move(rec);
  }

  // Success: prevent cleanup of registered keys.
  guard.release();

  RoutableLeaseResult result;
  result.key = key;
  result.remote_memory_keys = std::move(remote_memory_keys);
  result.buffer_sizes = std::move(buffer_sizes);
  return result;
}

void LipManager::attach_replica_id(const std::string& registration_id, std::string replica_id) {
  if (replica_id.empty()) {
    return;
  }
  std::optional<ArtifactDeviceKey> key;
  {
    absl::MutexLock l(&mu_);
    auto it = reg_to_key_.find(registration_id);
    if (it == reg_to_key_.end()) {
      return;
    }
    key = it->second;
  }
  absl::MutexLock lk(&routable_mu_);
  routable_replica_ids_[*key] = std::move(replica_id);
}

} // namespace tensorcast::daemon
