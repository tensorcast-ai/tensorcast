---
slug: 0018-artifact-view-registration
title: Draft — Variant View Registration (v1.5)
areas: ["core","daemon","global_store","sdk"]
related_code:
  - core/store/**
  - daemon/**
  - tensorcast/api/**
  - tensorcast/global_store/**
status: drafting
---

# Summary

We already ship view-aware retrieval: the daemon, core engine, and SDK can narrow or transpose canonical artifacts without materialising the full tensor set, using `ViewPlanner`, `ViewPlanSource`, and the variant metadata plumbed through Global Store (`variants`, `leaves`, `UpdateArtifactViewState`). Registration, however, still requires clients to upload canonical byte layouts. This draft spreads the same ergonomics to **registration** so trainers can register partial slices or layout-permuted tensors while the system computes canonical bytes, hashes, and variant metadata on their behalf.

Key ingredients we can reuse today:

- Canonical index v3 enforcement (`StoreEngine`, SDK) and the normalized JSON rebuild.
- `ViewPlanner`/`TransformPlan` infrastructure that already knows how to map canonical → view.
- Global Store tables + RPCs that persist variant specs, sizes, hashes, and leaf digests.
- Variant-aware residency keys and orchestration path in the daemon.

The registration design therefore focuses on the inverse data path (view → canonical), lifecycle wiring, and SDK ergonomics, rather than brand-new primitives.

# Goals

1. Allow Python SDK users to call `Store.register_view(...)` with either a `ViewSpec` or `view_id` and upload **only** the bytes required for that view (slices or transpose).
2. Ensure the daemon/core transform uploaded view bytes back into canonical ByteSpace, compute canonical + variant tree hashes, and persist both via Global Store.
3. Support partial canonical coverage: multiple view registrations (e.g., TP shards) can collectively compose the canonical artifact while each shard only transfers its slice.
4. Maintain parity with retrieval placement semantics (SERVER vs CLIENT) and reuse `ViewPlanner` so semantics stay symmetrical.
5. Preserve existing registration metrics/telemetry and extend them with view identifiers without regressing canonical flows.

# Non-Goals

- No changes to canonical chunk-directory routing or replica discovery (still anchored on canonical ByteSpace).
- Do not introduce new transform kinds beyond narrow + transpose.
- Do not reintroduce schema_version downgrade paths; everything stays v3-only.
- Out of scope: orchestrating automatic view merges across multiple tenants; we just enforce consistent metadata and integrity.

# Current Gaps

| Layer | Retrieval status | Registration gap |
|-------|------------------|------------------|
| SDK   | `get_view`, `get_view_into` shipped; `register_view` missing | Need user-facing API, argument validation, placement defaults |
| Daemon Proto | `MaterializeReplica` carries view spec/id | `RegisterArtifact` fields exist in proto but daemon ignores them |
| Core Engine | View planning/execution for load path | No ingest path that maps view bytes back into canonical storage |
| Global Store | Variant metadata + leaf persistence wired for loads | Registration path never calls `UpdateArtifactViewState`; canonical leaves not updated during partial writes |
| Tests | Load path tests & Python view API unit suite | Lack end-to-end coverage for register flows (slice + transpose) |

# Proposed Design

## 1. Bidirectional View Plans in Core

### 1.1 Extend `ViewPlanner`

`ViewPlanner::compute_view_plan` currently emits:

- `SelectionPlan` mapping canonical offsets → view offsets (`src_offset`, `dst_offset`, len).
- `TransformPlan` describing forward permutations (canonical → view).
- Metadata booleans (alias eligibility, requires_materialization, view size bytes, etc.).

For registration we need the inverse:

1. **WritePlan** — describes how to write view bytes into canonical storage.
   - For every `kData` range in `SelectionPlan`, create a `ViewWritePlan::Chunk` with `view_offset`, `canonical_offset`, `length`, and `is_segment_aligned`.
   - Preserve PAD semantics: skips canonical gaps unless the view provided zero-fill (should not happen for narrow).
2. **InverseTransformPlan** — reuse the exact tensor metadata but invert permutations.
   - Precompute inverse permutation vectors (e.g., `[1,2,0]` -> `[2,0,1]`) so ingestion can call `permute` once on the view tensor to reconstruct canonical layout.
   - Flag tensors that require materialisation (transpose) so ingestion can allocate staging buffers or use in-place operations depending on placement.

Implementation sketch:

```cpp
struct ViewWritePlan {
  struct Chunk {
    uint64_t canonical_offset;
    uint64_t view_offset;
    uint64_t length;
    bool segment_aligned;
  };
  std::vector<Chunk> chunks;
};

struct BidirectionalViewPlan {
  SelectionPlan forward;
  TransformPlan forward_transform;
  ViewWritePlan inverse;
  TransformPlan inverse_transform; // permutations inverted, sizes identical
};
```

`ViewPlanner` produces `BidirectionalViewPlan`. Existing retrieval code uses `forward`; registration code consumes `inverse`.

### 1.2 Ingest Executor

Add `core/store/loader/view_ingest_executor.{h,cc}`:

- Mirrors `view_plan_source` but performs writes.
- Input: `BidirectionalViewPlan`, `SeekableSink` (wrapper over canonical backing buffer or DMA writer), pointer to view staging buffer.
- For narrow (no transform): iterate `chunks`, memcpy from view buffer into canonical sink. When `segment_aligned == true`, use alias path via UMA to avoid extra copy.
- For transpose: apply `inverse_transform` using LibTorch. Implementation mirrors `execute_transform` but runs permutations first (view → canonical) and writes to canonical buffer.
- Expose status metrics: bytes written, number of staging copies, GPU vs CPU path, etc.

Reuse the newly added executor inside `StoreEngine::register_artifact_impl`.

## 2. Daemon & Engine Integration

### 2.1 Request Flow

1. SDK sends `RegisterArtifactRequest` with `view` or `view_id` + `placement`.
2. Daemon’s `RegistrationController` (or equivalent template) normalises:
   - If only `view_id`, fetch `ViewSpec` from GS via `GetArtifactInfoById(include_view_meta=true)` to keep server canonical.
   - Populate `MaterializeHints::variant` (already exists) with view identity + placement.
3. Pass hints to `StoreEngine::register_artifact`.

### 2.2 Engine Flow

Inside registration path (currently canonical-only):

1. Fetch canonical index JSON (already mandatory v3).
2. Invoke `ViewPlanner::compute_view_plan` if a view is present.
3. Allocate canonical replica buffer as usual (VRAM lease, UMA, etc.).
4. Materialise view bytes:
   - If placement `SERVER`: the incoming buffers (e.g., gRPC stream, LIP) contain view layout. Use `ViewIngestExecutor` to write into canonical buffer.
   - If placement `CLIENT`: assert that view is identity after normalisation; accept canonical data (defer to existing path).
5. Compute hashes:
   - Canonical `TreeHash` over full ByteSpace (existing path).
   - Variant `TreeHash` using `ViewPlanSource` (replay forward plan over canonical buffer). Reuses retrieval helper so we don’t duplicate hashing logic.
6. Collect leaf digests for variant ByteSpace. If the view fully covers canonical ranges exactly aligned with chunk boundaries, optionally update canonical leaves (existing pipeline already knows how to write canonical leaves; we can reuse by converting `ViewWritePlan::Chunk` into chunk ranges).
7. Update verification metadata on disk (`verification.view_<id>.json`) using existing code.

### 2.3 Global Store Mutations

After successful registration + verification:

1. Call `UpdateArtifactViewState` with:
   - `VariantUpsert { view_id, view_spec_json, view_size, view_data_hash, mark_verified=true }`.
   - `LeafWrite` entries for variant ByteSpace. Optionally include canonical leaf writes for ranges that were fully populated.
2. Continue to call `register_replica_with_global_store` (already view-aware) so the canonical replica is visible.

Error handling:

- If validation fails (unknown tensor, invalid narrow length, etc.), return `INVALID_ARGUMENT` early.
- On partial coverage (e.g., this registration doesn’t provide bytes for a canonical range yet): include `PartialCoverageDetail` with canonical ByteSpace to help orchestrators schedule remaining shards.

## 3. SDK API & Ergonomics

### 3.1 Surface

```python
store.register_view(
    tensors=my_tp_shard,
    *,
    key="model:v2",
    slices={"wte.weight": (slice(tp_off, tp_off + tp_len),)},
    transpose=None,
    placement="SERVER",
    ttl_ms=3600,
) -> RegisteredArtifact
```

Semantics:

- `tensors` follows the same contract as `Store.register`: in-memory torch tensors matching the view output layout.
- `_resolve_view_inputs` (already used for retrieval) normalises `ViewSpec`, ensures per-tensor exclusivity, folds identities.
- Placement default:
  - If any transpose op present → default `CLIENT` (upload canonical bytes, transform locally).
  - For narrow-only → default `SERVER` (min-byte transfer).
- When `view_id` supplied, skip normalisation and treat request as referencing an existing spec (still verify after fetch from GS).

### 3.2 Registration Loop

`Store._perform_registration` gains a fast path:

1. Build view spec and placement.
2. Annotate registration options (proto) with view fields.
3. During stream upload, send view tensors only (not canonical) when placement==SERVER.
4. Receive `RegisterArtifactResponse` containing `view_index_json` + `view_data_hash`; store inside `RegisteredArtifact` (new fields).
5. On completion, call Global Store helper to persist variant metadata/leaves (mirrors load path but for registration we call immediately).

Error surfacing:

- If daemon responds `FAILED_PRECONDITION` (e.g., server lacks GPU for transpose SERVER placement), rethrow `ArtifactError` with suggestion to retry `placement="CLIENT"`.
- If `PartialCoverageDetail` indicates missing canonical ranges, capture in exception to allow orchestrators to schedule remaining shards.

## 4. Global Store Enhancements

The infrastructure already exists; we need minor additions:

- `ViewStateService` (new) exposes helper `record_variant_registration(...)` to wrap `UpdateArtifactViewState` with retries + telemetry.
- CLI or admin tooling to inspect `variants`/`leaves` now includes registration timestamp, view size, view hash.
- Telemetry: counters for number of variant writes, bytes per view.

# Implementation Plan (High Level)

| Phase | Scope | Notes |
|-------|-------|-------|
| **P0 — API plumbing** | SDK arg validation + proto fields plumbed end-to-end (no-op on daemon) | Canonical registrations continue to flow unchanged when `view` is omitted |
| **P1 — Core ingest plan** | `BidirectionalViewPlan`, `ViewIngestExecutor`, engine integration for in-memory registration | Feature-gated until P2 ready |
| **P2 — Global Store updates** | Hook registration completion → `UpdateArtifactViewState`, ensure canonical leaf writes optional | End-to-end slice coverage tests |
| **P3 — Transpose support** | GPU/CPU transpose ingestion, placement fallback handling, LibTorch kernels | Perf tests on CUDA + fake CUDA |
| **P4 — SDK+Tests** | Python API, unit tests, integration tests with fake daemon, doc updates | Includes docs/design sync |

# Validation Strategy

- **Unit tests:** new Catch2 suite `//core/store/loader:view_ingest_executor_test` verifying narrow + transpose ingestion (CPU + fake CUDA).
- **Integration tests:** extend `//core/store:store_engine_test` and Python `tests/python/test_store_view_api.py` with end-to-end `register_view` flows (slice + transpose).
- **Global Store tests:** add coverage in `tests/python/global_store/test_services.py` for variant upsert during registration.
- **Perf smoke:** measure registration throughput vs canonical baseline; ensure slice registration reduces bytes transferred and CPU time remains within +5%.

# Residual Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| View ingestion bug corrupts canonical replica | Symmetric test suite + `ABSL_CHECK` invariants comparing number of written bytes vs plan total |
| Transpose server placement requires large staging buffer | Apply streaming kernels (tile-based permute) with bounded scratch; fallback to CLIENT placement when GPU mem insufficient |
| Variants table grows rapidly with per-shard registrations | Introduce TTL/retention policy per view_id; add metrics to track growth |
| Canonical leaf alignment ambiguous for partial slices | Extend `ViewPlanner` to flag chunk-aligned ranges; only write canonical leaves when aligned, otherwise defer |

# Open Questions

1. **Registration dedupe:** Should we allow multiple registrations for the same view_id to skip re-upload when view already verified? Proposed answer: yes—daemon short-circuits if view replica already canonical + verified (similar to retrieval fast-path).
2. **Conflict detection:** How do we handle overlapping narrow registrations uploading different bytes? Suggest: same as today—latest write wins; add optional checksum guard (future work).
3. **Client tensor management:** For transpose SERVER placement, do we expect clients to pre-permute? Proposed: no; server handles but we warn about GPU requirement.

# Next Steps

- Circulate this draft with core + daemon owners for feedback (focus on bidirectional plan shape and GS update timing).
- Once approved, spin out execution plan `docs/plans/0018-artifact-view-registration.md` referencing the phases above.
