### 0020-Region‑Backed Registration

— Provide first‑class, low‑overhead support for paged‑attention KVCache by: 1) pre‑registering large VRAM regions for zero‑copy reuse, 2) registering KV pages as CGID artifacts via in‑place leases that reference those regions (no per‑page IPC export), and 3) adding explicit deregister with safe quiesce semantics so replicas are discoverable/serving between register→deregister, but never torn down while transfers are active.

## Summary
- Introduce `register_vram_region` so frameworks can pre‑register one or more large VRAM slabs (per device) using a single CUDA IPC export. KV pages within the slab reuse that export with relative offsets.
- Register KV pages as CGID artifacts (`cgid:kv:<block_hash>`) via LIP; storage/segment metadata references the region + offset/length rather than exporting a new IPC handle per page.
- Add `deregister_artifact` with “quiesce + wait for active transfers to drain” semantics, then drop the replica and de‑advertise it from Global Store (GS).
- Keep the API surface simple by extending 0014’s Store verbs with small, optional parameters; reuse existing lock + staged‑export for transfer safety; leverage 0017’s CGID path to avoid hashing in hot paths.

## Goals / Non‑Goals
- Goals
  - Minimize registration overhead for many short‑lived KV pages (no per‑page IPC export, no hashing).
  - Preserve simple caller contract: per‑page `register`/`get/get_into` + explicit `deregister`.
  - Ensure correctness: source stability during transfer, no partial teardown while reads are active, fast local alias when possible.
- Non‑Goals
  - No new transport protocols; reuse current P2P/TCP/RDMA pipeline.
  - No change to view/variant system (0016) for KV; KV remains canonical byte‑space.
  - No requirement to persist leaves/digests for CGID.

## Architecture & Interfaces

### Identity policy (0017)
- KV pages use CGID: `artifact_id = "cgid:kv:<block_hash_hex>"`.
- Both `artifact_id` and `key` are supported; keys should simply mirror CGID where convenient.
- GS supports `id_kind=CGID` with nullable digests (0017).

### SDK: New/Extended APIs (0014‑compatible)
- New
  - `store.register_vram_region(name, *, base: torch.Tensor|int, size_bytes: int, device: str|torch.device|int, ttl_ms: int|None=None) -> VramRegion`
  - `store.unregister_vram_region(region_id: str) -> None`
  - `store.deregister_artifact(*, artifact_id: str|None=None, key: str|None=None, wait: bool=True, drain_timeout_s: float=30.0) -> None`
- Extended (optional params)
  - `store.register(tensors, *, artifact_id: str|None=None, key: str|None=None, ttl_ms: int|None=None) -> RegisteredArtifact`
    - When `artifact_id.startswith("cgid:")`, use CGID path (0017); skip hashing.
    - On LIP path, if tensors’ storages fall inside a registered region, reuse region reference + offset instead of per‑storage IPC export.
  - `store.get/get_into(..., *, transport_hold_ms: int|None=None)` forwards a “hold/extend suggestion” to the lock RPC (see Daemon), lowering TTL race risk on long transfers.

### Daemon: New/Extended RPCs
- New
  - `RegisterVramRegion(name, device_id, cuda_ipc_handle, size_bytes, ttl_ms?) -> region_id`
  - `UnregisterVramRegion(region_id) -> ok`
  - `DeregisterArtifact(artifact_id, wait, drain_timeout_ms) -> ok`
- Extended
  - `BeginRegisterArtifact(client_artifact_id?)` (already in 0017): CGID when present.
  - `FeedRegisterArtifactLeaseSegments(...)` to accept region‑based storage/segment metadata (see Proto below).
  - `LockTransportChunks(artifact_id, chunk_indices, device_id?, extend_ttl_ms?) -> lock_token`:
    - `extend_ttl_ms` is optional; when provided and the artifact is LIP, perform a one‑shot keepalive to reduce TTL expiry races.

### Proto (additions; minimal diff)
- Vram Region
  - `RegisterVramRegionRequest { string name; int32 device_id; bytes cuda_ipc_handle; uint64 size_bytes; uint32 ttl_ms; }`
  - `RegisterVramRegionResponse { string region_id; }`
  - `UnregisterVramRegionRequest { string region_id; }`
  - `UnregisterVramRegionResponse { bool ok; }`
- LIP metadata (region reuse; storage precision)
  - In `RegisterStorageMeta` add:
    - `oneof storage_source { bytes cuda_ipc_handle = 10; string vram_region_id = 11; }`
    - `uint64 region_base_offset = 12;` // storage begins at this offset within region
  - In `LeaseSegMeta` add:
    - `string storage_id = 1;` // switch matching from handle‑bytes to storage_id
    - existing `base_offset/length/dst_offset` remain; `handle_bytes` becomes optional/legacy
- Transport lock
  - `LockTransportChunksRequest { ...; uint32 extend_ttl_ms = 6; }`
- Deregister
  - `DeregisterArtifactRequest { string artifact_id; bool wait; uint32 drain_timeout_ms; }`
  - `DeregisterArtifactResponse { bool ok; }`

### Server‑side components
- `IpcRegionRegistry` (daemon‑local):
  - `region_id -> { device_id, size_bytes, handle_bytes, opened_map(optional) }`.
  - Open map lazily; reuse across segments; guard lifetimes with refcounts.
- `LipManager` updates:
  - Accept region‑backed storages; open map from `IpcRegionRegistry` when segment references a region; compute `src = region.map + region_base_offset + segment.local_offset`.
  - Switch segment validation from “handle→storage_length” to precise `storage_id` matching; keep chunk alignment enforcement.
- Deregistration flow:
  - Mark artifact as “quiescing” (denies new locks).
  - Wait for active transfers to drain:
    - No tokens in `TransportLockManager` for this artifact.
    - No staged exports in `LipManager` for this artifact.
  - Revoke leases/free coalesced replicas.
  - Inform GS to drop replica rows / key mapping (if any).
- Lock flow:
  - Reject locks for quiescing artifacts.
  - If `extend_ttl_ms` set and LIP lease found, use existing keepalive to extend expiry once.

### Client execution model (hot path)
- Pre‑allocation: framework allocates large `torch.empty(size, device="cuda:X", dtype=torch.uint8")` and calls `register_vram_region(...)`.
- Per‑page registration:
  - Build `TensorStorageGraph` → detect that each K/V storage base_ptr ∈ region range → compute `region_base_offset`.
  - Register via LIP with CGID `cgid:kv:<block_hash>` and storages referencing `region_id` + relative offsets; alias list indexes into these storages.
- Retrieval:
  - `get_into` on target device; source daemon locks; LIP path either IPC alias (same GPU) or staged P2P; unlock on completion.
- Deregistration:
  - `deregister_artifact(cgid)` marks quiescing → waits for active transfers → drops lease/replica → GS update. Errors if owner mismatch.

## Schema Changes (GS)
- 0017 already extended `artifacts` with `id_kind` and `replicas` with `expires_at`. For KV:
  - No further mandatory changes; deregister removes the replica row(s) for the (artifact_id, device), and optional key mapping rows.
  - Optional: add `replicas.state = {READY|QUIESCING|REMOVED}` if external observability of quiesce is desired (not required for correctness).

## Invariants & Error Model
- Invariants
  - Region map must precede any registration that references it; unregister region only after all referencing artifacts are deregistered or expired.
  - KV blocks are immutable by contract (CGID implies caller‑managed integrity).
  - Lock is required for any P2P read; quiescing denies new locks; teardown waits for locks to drain.
- Errors
  - `INVALID_ARGUMENT`: region not found/size mismatch, CGID grammar invalid.
  - `FAILED_PRECONDITION`: lock denied (quiescing), region unregistered while in use, LIP segment misaligned.
  - `NOT_FOUND`: artifact or region missing.
  - `UNAVAILABLE`: no resident replica; fallbacks disabled.
  - `PERMISSION_DENIED`: deregister by non‑owner when policy requires ownership.

## Compatibility
- 0014 verbs remain; new APIs are additive.
- 0017 CGID support is leveraged; MI2 unchanged.
- Existing LIP and transport code paths stay intact; region reuse is an optimization and a new capability, not a breaking change.

## Alternatives considered
- Per‑page IPC export (status quo): too expensive for high‑churn KV.
- Global Store‑managed page table for KV: out of scope; page table stays engine‑local; TensorCast provides per‑page replica serviceability only.
- Implicit deregistration by TTL only: insufficient control for cache eviction under load; explicit `deregister` required.

## Risks & mitigations
- Region lifetime leaks: tie `register_vram_region` to process session; auto‑unregister on exit; warn on lingering references.
- TTL races on long transfers: `extend_ttl_ms` on lock plus reasonable default TTL on register alleviate.
- Cardinality explosion (CGID space): allow deployment policy to namespace (`cgid:kv:<tenant>:<hash>`) and impose TTL/cardinality limits.

## Acceptance Criteria
- Register KV page using CGID with LIP referencing a pre‑registered region returns quickly (no hashing, no per‑page IPC export).
- `get_into` across nodes succeeds; source uses staged export under lock; unlock frees staged resources; performance on par or better than current LIP.
- `deregister_artifact(wait=True)` blocks new locks, waits for active locks to drain, then removes daemon replica and GS entry; concurrent gets either complete or fail cleanly if started after quiesce.
- Unregistering a region while any referencing artifact exists fails with clear `FAILED_PRECONDITION`.

## Key code references (current capabilities that this design builds on)
```118:132:daemon/lip_manager.cc
std::vector<Opened> opened;
opened.reserve(segments.size());
for (const auto& seg : segments) {
  auto map_or = CudaIpcMapping::open(seg.handle_bytes, cudaIpcMemLazyEnablePeerAccess);
  if (!map_or.ok())
    return map_or.status();
  opened.push_back(Opened{.device_id = seg.device_id, .map = std::move(*map_or),
                          .base = seg.base_offset, .len = seg.length, .dst = seg.dst_offset});
}
```

```19:43:daemon/service/controllers/transport_controller.cc
if (auto lip_opt = d_.lip.find_active_by_artifact_id(req.artifact_id()); lip_opt.has_value()) {
  std::vector<uint32_t> indices(req.chunk_indices().begin(), req.chunk_indices().end());
  auto tok_or = d_.lip.create_staged_export(lip, absl::MakeSpan(indices), d_.engine);
  if (!tok_or.ok()) return to_grpc_status(tok_or.status());
  resp.set_lock_token(*tok_or);
  ...
  return Status::OK;
}
```

```869:907:tensorcast/api/_register.py
for storage_id in sorted(graph.storages.keys()):
  entry = graph.storages[storage_id]
  ...
  handle_bytes = _export_cuda_ipc_handle(int(entry.base_ptr))
  segments.append(LeaseSegment(device_id=int(ctx.device_id),
                               cuda_ipc_handle=handle_bytes,
                               base_addr=0, length=length_bytes,
                               dst_offset=int(dst_offset)))
```

## Rollout Plan (high‑level)
- Phase 1: Proto + daemon
  - Add region RPCs and registry; extend LIP to accept region‑backed storages; quiesce+deregister flow; lock’s `extend_ttl_ms`.
- Phase 2: SDK
  - Expose `register_vram_region`/`unregister_vram_region`/`deregister_artifact`; modify LIP uploader to prefer region reuse; add `transport_hold_ms`.
  - Provide `register_kv_block({"k","v"}, block_hash, ttl_ms=...)` convenience wrapper that sets `artifact_id=f"cgid:kv:{block_hash}"`.
- Phase 3: GS
  - Ensure `id_kind=CGID` flows are fully supported (0017); add simple “drop replica” path used by deregister; optional quiesce state if desired.
- Phase 4: Tests & perf
  - Unit + integration for region reuse, lock/extend, deregister drains; KV microbenchmarks.
