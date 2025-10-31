// Copyright (c) 2025, TensorCast Team.

#include "daemon/lip_manager.h"

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
#include "core/store/loader/canonical_index.h"
#include "core/store/loader/segment_plan_source.h"
#include "core/store/loader/source_hash.h"
#include "daemon/cuda_ipc_raii.h"
#include "daemon/lip_metadata_utils.h"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

namespace {

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

absl::StatusOr<CudaIpcMapping*> GetOrOpenMappingForStorage(
    const RegisterStorageMeta& storage,
    absl::flat_hash_map<std::string, std::unique_ptr<CudaIpcMapping>>& cache,
    RegionAcquireGuard& guard,
    IpcRegionRegistry* registry,
    int owner_pid) {
  auto it = cache.find(storage.storage_id);
  if (it != cache.end()) {
    return it->second.get();
  }

  std::unique_ptr<CudaIpcMapping> mapping;
  if (storage.has_handle()) {
    auto map_or = CudaIpcMapping::open(storage.handle_bytes, cudaIpcMemLazyEnablePeerAccess);
    if (!map_or.ok())
      return map_or.status();
    mapping = std::make_unique<CudaIpcMapping>(std::move(*map_or));
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
    auto map_or = CudaIpcMapping::open(*handle_or, cudaIpcMemLazyEnablePeerAccess);
    if (!map_or.ok())
      return map_or.status();
    mapping = std::make_unique<CudaIpcMapping>(std::move(*map_or));
  } else {
    return absl::InvalidArgumentError("storage entry missing source handle or region");
  }

  auto [insert_it, _] = cache.emplace(storage.storage_id, std::move(mapping));
  return insert_it->second.get();
}

} // namespace

absl::StatusOr<std::vector<uint8_t>> LipManager::copy_to_new_coalesced(
    int target_device_id,
    const std::string& canonical_index_json,
    uint64_t total_size,
    absl::Span<const LeaseSegMeta> segments,
    absl::Span<const RegisterStorageMeta> storages) {
  if (storages.empty()) {
    return absl::InvalidArgumentError("lip copy requires storage metadata");
  }
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  storage_by_id.reserve(storages.size());
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_handle;
  storage_by_handle.reserve(storages.size());
  for (const auto& storage : storages) {
    storage_by_id.emplace(storage.storage_id, &storage);
    if (storage.has_handle()) {
      storage_by_handle.emplace(storage.handle_bytes, &storage);
    }
  }
  for (const auto& seg : segments) {
    const RegisterStorageMeta* storage = nullptr;
    if (!seg.storage_id.empty()) {
      auto it = storage_by_id.find(seg.storage_id);
      if (it == storage_by_id.end()) {
        return absl::InvalidArgumentError("segment references unknown storage_id");
      }
      storage = it->second;
    } else {
      auto it = storage_by_handle.find(seg.handle_bytes);
      if (it == storage_by_handle.end()) {
        return absl::InvalidArgumentError("segment handle missing matching RegisterStorage entry");
      }
      storage = it->second;
    }
    if (seg.length != storage->storage_length) {
      return absl::InvalidArgumentError(
          absl::StrFormat(
              "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
              static_cast<uint64_t>(seg.length),
              static_cast<uint64_t>(storage->storage_length),
              storage->storage_id));
    }
  }

  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<CudaIpcMapping>> mapping_cache;
  mapping_cache.reserve(storages.size());
  // Begin destination allocation on target device
  store::StoreEngine::ArtifactRegistration areg;
  areg.artifact_id = absl::StrCat("mem_reg:", absl::ToUnixNanos(absl::Now()));
  areg.tensor_index_data = canonical_index_json;
  areg.schema_version = "v2";
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
  auto plan_or = store::loader::build_segment_plan_from_canonical_index_json(canonical_index_json, total_size, 8);
  if (!plan_or.ok())
    return plan_or.status();
  auto plan = *plan_or;

  auto dst_map_or = CudaIpcMapping::open(out.cuda_ipc_handle_bytes, cudaIpcMemLazyEnablePeerAccess);
  if (!dst_map_or.ok())
    return dst_map_or.status();
  auto dst_map = std::move(*dst_map_or);
  void* dst_dev = dst_map.get();
  if (auto st_set = cuda::set_device(target_device_id); !st_set.ok()) {
    return st_set;
  }
  for (const auto& p : plan) {
    if (p.kind != store::loader::SegmentPiece::PAD || p.length == 0)
      continue;
    auto st = cuda::memset(static_cast<uint8_t*>(dst_dev) + p.dst_offset, 0, static_cast<size_t>(p.length));
    if (!st.ok())
      return st;
  }

  struct Opened {
    int device_id;
    CudaIpcMapping* map;
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  std::vector<Opened> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    const RegisterStorageMeta* storage = nullptr;
    if (!seg.storage_id.empty()) {
      auto it = storage_by_id.find(seg.storage_id);
      if (it == storage_by_id.end()) {
        return absl::InvalidArgumentError("segment references unknown storage_id");
      }
      storage = it->second;
    } else {
      auto it = storage_by_handle.find(seg.handle_bytes);
      if (it == storage_by_handle.end()) {
        return absl::InvalidArgumentError("segment handle missing matching RegisterStorage entry");
      }
      storage = it->second;
    }
    auto map_or = GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, 0);
    if (!map_or.ok()) {
      return map_or.status();
    }
    opened.push_back(
        Opened{
            .device_id = seg.device_id,
            .map = *map_or,
            .base = seg.base_offset,
            .len = seg.length,
            .dst = seg.dst_offset});
  }
  const size_t chunk_size = engine_->get_artifact_chunk_bytes();
  if (chunk_size == 0) {
    return absl::FailedPreconditionError("invalid artifact_chunk_bytes (0)");
  }
  auto is_aligned = [&](uint64_t v) { return (v % chunk_size) == 0; };
  for (const auto& o : opened) {
    const bool is_tail = (o.dst + o.len == total_size);
    if (!is_aligned(o.dst) || !is_aligned(o.base) || (!is_aligned(o.len) && !is_tail)) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "LIP segment not aligned to artifact_chunk_bytes: base_offset=",
              o.base,
              ", dst_offset=",
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
  std::vector<uint8_t> bytes(out.cuda_ipc_handle_bytes.size());
  std::memcpy(bytes.data(), out.cuda_ipc_handle_bytes.data(), out.cuda_ipc_handle_bytes.size());
  return bytes;
}

absl::StatusOr<std::string> LipManager::create_staged_export(
    const LipLeaseEntry& lip,
    absl::Span<const uint32_t> chunk_indices,
    store::StoreEngine& engine) {
  {
    absl::MutexLock l(&mu_);
    if (quiesced_.contains(lip.artifact_id)) {
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

  if (!lip.storages.empty()) {
    absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_handle;
    storage_by_handle.reserve(lip.storages.size());
    for (const auto& storage : lip.storages) {
      storage_by_handle.emplace(storage.handle_bytes, &storage);
    }
    for (const auto& seg : lip.segments) {
      auto it = storage_by_handle.find(seg.handle_bytes);
      if (it == storage_by_handle.end()) {
        return absl::InvalidArgumentError("staged export missing storage metadata for a segment handle");
      }
      if (seg.length != it->second->storage_length) {
        return absl::InvalidArgumentError(
            absl::StrFormat(
                "segment length (%llu) does not match storage_length (%llu) for storage_id=%s",
                static_cast<uint64_t>(seg.length),
                static_cast<uint64_t>(it->second->storage_length),
                it->second->storage_id));
      }
    }
  }

  struct OpenedSeg {
    int device_id;
    CudaIpcMapping* map; // non-owning; lifetime held by owned_maps or mapping_cache/export record
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  std::vector<OpenedSeg> opened;
  opened.reserve(lip.segments.size());
  // When storage metadata is present, prefer region-backed or cached mappings via registry.
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_id;
  absl::flat_hash_map<std::string, const RegisterStorageMeta*> storage_by_handle;
  storage_by_id.reserve(lip.storages.size());
  storage_by_handle.reserve(lip.storages.size());
  for (const auto& s : lip.storages) {
    storage_by_id.emplace(s.storage_id, &s);
    if (s.has_handle()) {
      storage_by_handle.emplace(s.handle_bytes, &s);
    }
  }
  RegionAcquireGuard region_guard(region_registry_);
  absl::flat_hash_map<std::string, std::unique_ptr<CudaIpcMapping>> mapping_cache;
  std::vector<std::unique_ptr<CudaIpcMapping>> owned_maps;
  mapping_cache.reserve(lip.storages.size());
  for (const auto& seg : lip.segments) {
    std::optional<OpenedSeg> opened_seg;
    if (!lip.storages.empty()) {
      const RegisterStorageMeta* storage = nullptr;
      if (!seg.storage_id.empty()) {
        auto it = storage_by_id.find(seg.storage_id);
        if (it == storage_by_id.end())
          return absl::InvalidArgumentError("staged export: unknown storage_id in segment");
        storage = it->second;
      } else if (!seg.handle_bytes.empty()) {
        auto it = storage_by_handle.find(seg.handle_bytes);
        if (it == storage_by_handle.end())
          return absl::InvalidArgumentError("staged export: segment handle missing matching storage entry");
        storage = it->second;
      }
      if (storage != nullptr) {
        auto map_or =
            GetOrOpenMappingForStorage(*storage, mapping_cache, region_guard, region_registry_, lip.owner_pid);
        if (!map_or.ok())
          return map_or.status();
        opened_seg = OpenedSeg{
            .device_id = seg.device_id,
            .map = *map_or,
            .base = seg.base_offset,
            .len = seg.length,
            .dst = seg.dst_offset};
      }
    }
    if (!opened_seg.has_value()) {
      auto map_or = CudaIpcMapping::open(seg.handle_bytes, cudaIpcMemLazyEnablePeerAccess);
      if (!map_or.ok())
        return map_or.status();
      owned_maps.push_back(std::make_unique<CudaIpcMapping>(std::move(*map_or)));
      opened_seg = OpenedSeg{
          .device_id = seg.device_id,
          .map = owned_maps.back().get(),
          .base = seg.base_offset,
          .len = seg.length,
          .dst = seg.dst_offset};
    }
    opened.push_back(std::move(*opened_seg));
  }
  // Enforce alignment of LIP segments to artifact_chunk_bytes
  auto is_aligned = [&](uint64_t v) { return (v % chunk_size) == 0; };
  for (const auto& s : opened) {
    const bool is_tail = (s.dst + s.len == lip.total_size);
    if (!is_aligned(s.dst) || !is_aligned(s.base) || (!is_aligned(s.len) && !is_tail)) {
      return absl::FailedPreconditionError(
          absl::StrCat(
              "LIP segment not aligned to artifact_chunk_bytes: base_offset=",
              s.base,
              ", dst_offset=",
              s.dst,
              ", length=",
              s.len,
              ", chunk=",
              chunk_size));
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
    std::string tkey = absl::StrCat(lip.artifact_id, "_GPU_chunk_", idx);
    communicator::engine::Communicator::RegisterTensorOptions ro;
    ro.register_mr = comm_engine.is_rdma_enabled();
    ro.needs_staging = (!comm_engine.is_rdma_enabled());
    ro.async = false;
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
    rec.device_id = lip.device_id;
    // Transfer ownership of new opens
    for (auto& up : owned_maps) {
      rec.opened_maps.push_back(std::move(*up));
    }
    owned_maps.clear();
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
  exports_.erase(it);
  return absl::OkStatus();
}

void LipManager::put_lease(const std::string& registration_id, const ArtifactDeviceKey& key, LipLeaseEntry entry) {
  absl::MutexLock l(&mu_);
  reg_to_key_[registration_id] = key;
  leases_[key] = std::move(entry);
}

absl::Status LipManager::keepalive_lease(
    const std::string& registration_id,
    int owner_pid,
    uint64_t epoch,
    uint32_t ttl_ms) {
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
  const uint32_t extend = ttl_ms > 0 ? ttl_ms : (it->second.ttl_ms > 0 ? it->second.ttl_ms : 600000U);
  it->second.ttl_ms = extend;
  it->second.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(extend);
  return absl::OkStatus();
}

absl::Status LipManager::revoke_by_registration_id(const std::string& registration_id) {
  absl::MutexLock l(&mu_);
  auto itk = reg_to_key_.find(registration_id);
  if (itk == reg_to_key_.end())
    return absl::NotFoundError("registration_id not found");
  leases_.erase(itk->second);
  reg_to_key_.erase(itk);
  return absl::OkStatus();
}

std::optional<LipLeaseEntry> LipManager::find_active_by_artifact_id(const std::string& artifact_id) const {
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  for (const auto& kv : leases_) {
    const auto& key = kv.first;
    const auto& e = kv.second;
    if (key.artifact_id != artifact_id)
      continue;
    if (e.expiry.time_since_epoch().count() > 0 && now > e.expiry)
      continue;
    return e;
  }
  return std::nullopt;
}

bool LipManager::has_active_on_device(const std::string& artifact_id, int device_id) const {
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  ArtifactDeviceKey k{.artifact_id = artifact_id, .device_id = device_id};
  auto it = leases_.find(k);
  if (it == leases_.end())
    return false;
  if (it->second.expiry.time_since_epoch().count() > 0 && now > it->second.expiry)
    return false;
  return true;
}

void LipManager::sweep_expired_and_dead_pids() {
  absl::MutexLock l(&mu_);
  const auto now = std::chrono::steady_clock::now();
  std::vector<ArtifactDeviceKey> to_erase;
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
    }
  }
}

bool LipManager::revoke_commit_lease_if_owner_matches(const std::string& artifact_id, int device_id, int owner_pid) {
  absl::MutexLock l(&mu_);
  ArtifactDeviceKey key{.artifact_id = artifact_id, .device_id = device_id};
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
  VLOG(2) << "revoke_commit_lease_if_owner_matches: erased lease for artifact_id=" << artifact_id
          << " device_id=" << device_id;
  return true;
}

absl::Status LipManager::extend_ttl_for_artifact(const std::string& artifact_id, uint32_t extend_ttl_ms) {
  if (extend_ttl_ms == 0)
    return absl::OkStatus();
  std::optional<LipLeaseEntry> found;
  {
    absl::MutexLock l(&mu_);
    const auto now = std::chrono::steady_clock::now();
    for (auto& kv : leases_) {
      if (kv.first.artifact_id != artifact_id)
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

void LipManager::quiesce_artifact(const std::string& artifact_id) {
  absl::MutexLock l(&mu_);
  quiesced_.insert(artifact_id);
}

bool LipManager::wait_exports_drained(const std::string& artifact_id, absl::Time deadline) {
  // Simple polling with backoff; exports_ rarely large
  while (absl::Now() < deadline) {
    size_t active = 0;
    {
      absl::MutexLock lk(&exp_mu_);
      for (const auto& kv : exports_) {
        if (kv.second.artifact_id == artifact_id)
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

  // Build seekable source over LIP segments using RAII CUDA IPC mappings
  struct OpenedSeg {
    int device_id;
    CudaIpcMapping map;
    uint64_t base;
    uint64_t len;
    uint64_t dst;
  };

  std::vector<OpenedSeg> opened;
  opened.reserve(segments.size());
  for (const auto& seg : segments) {
    auto map_or = CudaIpcMapping::open(seg.handle_bytes, cudaIpcMemLazyEnablePeerAccess);
    if (!map_or.ok())
      return map_or.status();
    opened.push_back(
        OpenedSeg{
            .device_id = seg.device_id,
            .map = std::move(*map_or),
            .base = seg.base_offset,
            .len = seg.length,
            .dst = seg.dst_offset});
  }

  class LipSeekableSource final : public store::loader::SeekableSource {
   public:
    explicit LipSeekableSource(std::vector<OpenedSeg> segs, uint64_t total) : segs_(std::move(segs)), total_(total) {
      std::sort(segs_.begin(), segs_.end(), [](const OpenedSeg& a, const OpenedSeg& b) { return a.dst < b.dst; });
    }

    ~LipSeekableSource() override = default;

    absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
      return read_at(cur_, dst, max_bytes);
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
        if (auto st = cuda::set_device(seg->device_id); !st.ok())
          return st;
        auto st = cuda::memcpy(
            out, static_cast<uint8_t*>(seg->map.get()) + (seg->base + local), take, cudaMemcpyDeviceToHost);
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

   private:
    std::vector<OpenedSeg> segs_;
    uint64_t total_;
    uint64_t cur_{0};
  } src(std::move(opened), total_size);

  std::string index_multihash;
  std::string data_multihash;
  CommitLeaseResult out;

  if (id_kind == common::ArtifactIdKind::kMi2 || client_artifact_id.empty()) {
    auto index_mh_or = common::compute_index_multihash(
        canonical_index_json.empty() ? std::optional<std::string>() : std::optional<std::string>(canonical_index_json),
        index_key_hex);
    if (!index_mh_or.ok())
      return index_mh_or.status();
    index_multihash = *index_mh_or;

    auto data_mh_or = store::loader::compute_data_multihash_from_seekable_source(src, total_size);
    if (!data_mh_or.ok())
      return data_mh_or.status();
    data_multihash = *data_mh_or;

    out.index_multihash = index_multihash;
    out.data_multihash = data_multihash;
    out.artifact_id = absl::StrCat("mi2:", index_multihash, ":", data_multihash);
    out.id_kind = common::ArtifactIdKind::kMi2;
  } else {
    out.index_multihash.clear();
    out.data_multihash.clear();
    out.artifact_id = client_artifact_id;
    out.id_kind = common::ArtifactIdKind::kCgid;
  }

  out.schema_version = "v2";
  out.encoding = "json";
  out.total_size = total_size;

  // Enforce device-unique commit for VRAM_LEASED: (artifact_id, device_id)
  {
    absl::MutexLock l(&mu_);
    ArtifactDeviceKey k{.artifact_id = out.artifact_id, .device_id = device_id};
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
  lease.client_artifact_id = client_artifact_id;
  lease.id_kind = out.id_kind;
  lease.device_id = device_id;
  lease.owner_pid = owner_pid;
  lease.ttl_ms = ttl_ms > 0 ? ttl_ms : 600000U; // default 10 minutes
  lease.expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(lease.ttl_ms);
  lease.epoch = epoch;
  lease.total_size = total_size;
  lease.index_data = canonical_index_json;
  lease.segments = std::move(segments);
  lease.storages = std::move(storages);
  lease.aliases = std::move(aliases);
  lease.verification_json = out.verification_json;

  {
    ArtifactDeviceKey key{.artifact_id = lease.artifact_id, .device_id = lease.device_id};
    put_lease(registration_id, key, std::move(lease));
  }

  return out;
}

} // namespace tensorcast::daemon
