---
slug: 0016-artifact-view-v1
title: Variant-Aware Artifact Views (Narrow & Transpose)
areas: ["core","daemon","global_store","sdk"]
related_code:
  - core/store/materialization/dataplane/**
  - daemon/**
  - tensorcast/api/**
links:
  plan_core: ../plans/0016-a-artifact-view-v1-core.md
  plan_daemon_proto: ../plans/0016-b-artifact-view-v1-daemon-proto.md
  plan_global_store: ../plans/0016-c--artifact-view-v1-global-store.md
  plan_sdk: ../plans/0016-d--artifact-view-v1-sdk.md
  schema: ../../schema.sql
---

# Summary

This design introduces **Variant-Aware Views** for TensorCast artifacts.  A *view* is a reproducible, verifiable logical transformation over a canonical artifact that allows:

* partial registration & retrieval (tensor slicing / TP shards)
* optional layout permutation (channel–dimension transpose)
* bandwidth-reduced P2P transport by transferring only required bytes

Version 1 supports two PyTorch-compatible transforms:

* `torch.narrow(dim, start, length)` — contiguous slicing (step = 1)
* `torch.transpose(dim0, dim1)` — dimension permutation

All metadata, identity, and integrity rules are enforced in C++ core; Python SDK and Daemon merely call into the core helpers.

The v1 release prioritises retrieval (`get_view` / `get_view_into`). Registration semantics are defined here but roll out with follow-up plan `0018-artifact-view-registration`; until then, canonical registration remains the only write path.

# Goals / Non-Goals

## Goals

1. Define the architecture so clients can **retrieve** and, in a follow-up release, **register** canonical *slices* or *transposed* views without materialising the whole artifact.
2. Maintain a single canonical identity (`mi2_core`) while allowing multiple variant identities (`view_id`).
3. Reduce network and IO by server-side byte range selection when possible.
4. Preserve end-to-end verifiability via one-time strong validation and fast leaf validation thereafter.
5. Provide a database schema extension that represents variants and variant leaves unambiguously.

## Non-Goals

* Support for `reshape`, arbitrary stride slicing, channels-last conversion, or quantisation (future work).
* Runtime on-the-fly graph rewrites; only storage & transport concerns are addressed.

# Architecture & Interfaces

## 1. Coordinate Systems

| Term | ID | Byte range | Source |
|------|----|------------|--------|
| **Canonical ByteSpace** | `index_multihash` | `[0,total_size)` | strict canonical index v3 |
| **Variant ByteSpace**   | `view_id`         | `[0,view_size)`  | canonical index ⊕ ViewSpec |

All coverage ranges must be expressed in exactly one ByteSpace.

```
flowchart LR
  A[Canonical
  ByteSpace (indexᵐ)] -- ViewSpec --> B[Variant
  ByteSpace (view_id)]
```

### 1.1 Identity

```
indexᵐ          = MH(canonical_index_v3_json)
dataᵐ           = TreeHash( AVBS[indexᵐ] )
mi2_core        = "mi2:" + indexᵐ + ":" + dataᵐ

// Identity is defined by deterministic Proto serialization of ViewSpec (transform-only; no placement)
view_id         = MH( DETERMINISTIC_PROTO(ViewSpec) ⊕ indexᵐ )
view_data_hash  = TreeHash( Transform(ViewSpec, AVBS[indexᵐ]) )
```

Identity folding (no‑op views)
- During NORMALIZE, eliminate operations equivalent to identity (e.g., `narrow(dim,0,shape[dim])`, `transpose(dim,dim)`, and compositions that cancel).
- If the entire ViewSpec becomes identity after folding, treat it as “no view”: omit the `view` field, do not compute a `view_id`, and route via the canonical path/ByteSpace.

### 1.2 ViewSpec (Proto; canonical JSON for storage)

```json
{
  "tensors": {
    "<tensor_name>": {
      "ops": [
        {"type": "narrow", "dim": 0, "start": 0, "length": 128},
        {"type": "transpose", "dim0": 1, "dim1": 2}
      ]
    }
  }
}
```

Normalization rules:
* Map keys (`tensor_name`) are ASCII-sorted.
* `ops` order is preserved; adjacent compatible ops MAY be canonically folded (full‑length `narrow` → no‑op; `transpose(dim,dim)` → no‑op; consecutive `transpose` composition).
* JSON shown here is a canonical storage form for readability/auditing only and is NOT used for identity hashing; identity uses deterministic Proto serialization of `ViewSpec`.
* For `narrow`, `start < 0` is canonicalised to `start' = shape[dim] + start`.
* Placement is request-level and not part of `ViewSpec` or identity.

#### 1.2.1 Field Definitions

`tensors: map<string, TensorViewOps>`
* Scope: transforms per tensor. Tensors not present are implicitly identity (unmodified) and must appear in `view_index_json` unaltered.
* Key: `tensor_name` must exist in canonical index; keys are ASCII-sorted during normalisation.
* Value: `TensorViewOps` is an ordered list `ops` (see below) applied sequentially.

`TensorViewOps.ops: list<Op>`
* Evaluation order: left → right, each op transforms the logical view for that tensor.
* Allowed ops (v1):
  - `narrow` — contiguous slice. Semantics: `torch.narrow(input, dim, start, length)`
    - `dim: u32` in `[0, ndim)`.
    - `start: i64` allowed negative; canonicalised to non-negative.
    - `length: u64 > 0`, with `start + length ≤ shape[dim]`.
  - `transpose` — swap two dimensions. Semantics: `torch.transpose(input, dim0, dim1)`
    - `dim0, dim1: u32` in `[0, ndim)`, `dim0 != dim1`.
* Invalid mixes: empty `ops` list is allowed (no-op) but discouraged; unsupported op kinds are rejected `INVALID_ARGUMENT`.
* v1 exclusivity (per tensor): A given tensor may use either `narrow` or `transpose`, never both. Mixing kinds across different tensors within the same `ViewSpec` is allowed.
* v1 single-dimension narrow: For any single tensor, at most one `narrow` op is allowed (one dimension only). Supplying multiple `narrow` ops for the same tensor (including across different dims) is `INVALID_ARGUMENT`.

Placement (request-level; AUTO removed)
* Purpose: decide where compute-heavy transforms occur. Placement never changes mathematical output.
* Policy (v1):
  - `SERVER`:
    - For `narrow`: server performs byte-range selection and sends minimal bytes; no GPU requirement.
    - For `transpose`: server-side materialization required; if capability unavailable, return `FAILED_PRECONDITION` (or optional CPU fallback if explicitly enabled).
  - `CLIENT`: server applies byte-range selection for slices; transpose is executed on client.

Server behaviour for slices
* The server ALWAYS applies byte-range selection for all `narrow` operations and sends minimal bytes.

Validation & Errors
* `INVALID_ARGUMENT`: unknown tensor name, out-of-bound dims/indices, negative or zero `length`, or unsupported op type.
* `FAILED_PRECONDITION`: `SERVER` placement requested but source replicas do not allow server-side transform (e.g., lack of GPU residency or permissions).
* `UNAVAILABLE` with detail `PARTIAL_COVERAGE`: P2P cannot assemble requested coverage within the chosen ByteSpace. The error MUST include a `PartialCoverageDetail` with `(space_kind, space_id, missing_ranges)`.

### 1.3 TreeHash and leaf digest (normative)
- Hash: SHA‑256 for leaves and internal nodes (current policy v1).
- Leaf chunking policy (current): `determine_leaf_chunk_size(total_size, 4MiB)` per `core/common/artifact_hash_*` with constants `kMaxChunkSize=64MiB`, `kMinLeafChunkBytes=512KiB`, `kTargetLeafCount=4096`, 64‑byte alignment. This is a pure function of `total_size` and identical for canonical and variant ByteSpaces.
- Leaves: contiguous left→right over the normalized byte stream of the target ByteSpace (canonical for `dataᵐ`, variant for `view_data_hash`). Final leaf may be shorter; at least one leaf always exists.
- Leaf digest storage: raw 32‑byte `SHA256(leaf_bytes)` is stored in `leaves.digest` (BLOB). No per‑leaf base32/multihash; root uses multihash sha2‑256 and multibase base32 (lowercase, no padding) per mi2 rules.
- Internal nodes: pairwise left→right reduction hashing `SHA256(L || R)`; odd node promoted unchanged.
- Compatibility: Any future hashing policy/version MUST be recorded alongside variant rows and validated end‑to‑end before adoption.

## 2. Core Components

```
core/store/materialization/dataplane/
  canonical_index.{h,cc}        // v3 stabiliser (existing, bumped)
  segment_plan_source.{h,cc}    // DATA & PAD plan (existing)
  view_planner.{h,cc}           // NEW  – View → {TargetLayout,SelectionPlan,TransformPlan}
  view_plan_source.{h,cc}       // NEW  – streaming / GPU transform executor
```

### 2.1 ViewPlanner

Inputs
* canonical_index_json (**v3 mandatory**)
* ViewSpec

Outputs
* `TargetLayout` → `view_index_json` (same field order/typing as v3)
* `SelectionPlan`  → minimal source ranges within canonical ByteSpace
* `TransformPlan`  → steps for transpose when needed (`narrow` uses stride alias)

### 2.2 Execution Rules

| Transform pattern | Placement | Transport bytes | Notes |
|-------------------|-----------|-----------------|-------|
| Slice (`narrow`)  | SERVER    | min-ranges      | 0‑copy alias only when segment‑aligned; else scatter/gather |
| Transpose only    | CLIENT    | full bytes      | avoids double compute |

Server-side transpose materialises a *derived coalesced replica* on GPU memory (ByteSpace = `view_id`).

v1 combination rule: a single tensor may not combine `narrow` and `transpose`; mixing across different tensors within the same view is allowed. `narrow` is limited to a single dimension per tensor.

SelectionPlan augmentation (normative)
- `is_contiguous: bool`
- `num_ranges: u32`
- `total_bytes: u64`
- `is_segment_aligned: bool` (eligible for 0‑copy alias)
- `requires_materialization: bool`

### 2.3 Replica Residency & Identity

Variant replicas must coexist with canonical replicas on the same daemon without clobbering registry state or altering the canonical identity used for routing.

- **ReplicaKey** extends to `(canonical_artifact_id, view_id?, device, replica_idx)`. When `view_id` is absent the key represents the canonical ByteSpace. Variant replicas always retain the canonical id so that downstream metrics and GS bookkeeping remain anchored to the original artifact entry.
- `ReplicaRegistry`, LIP caches, eviction heuristics, and lease tracking operate on the extended key. Fast-path lookups must match both canonical id and optional `view_id`; this prevents treating a variant materialisation as an existing canonical instance.
- `register_replica_with_global_store` continues to publish replicas under the canonical id. Variant-specific metadata (view hash, leaves) is written through the dedicated GS RPC (§3) so that chunk directory semantics remain canonical in v1.
- Canonical index **v3** is authoritative throughout the pipeline. Older schema variants (v2) are no longer accepted; any loader or repository code that previously inspected `schema_version` must be updated to expect `"v3"` exclusively.

### 2.4 Variant-Aware Materialisation Flow

`MaterializeHints` carries an optional `VariantIdentity` struct:

```cpp
struct VariantIdentity {
  std::string canonical_artifact_id;
  std::optional<std::string> view_id;
  std::optional<ViewSpec> view;
  TransformPlacement placement;
};
```

- The daemon populates `VariantIdentity` whenever the request contains a view. `canonical_artifact_id` is always set (either resolved via key mapping or provided directly).
- `StoreEngine::materialize_replica` first checks the registry using the extended key. On cache hit it reuses the resident replica that already matches the requested ByteSpace.
- On miss, the engine invokes `ViewPlanner` (for slice) to produce a `SelectionPlan`.
- `MaterializeOrchestrator` (AUTO mode) now becomes view-aware:
  - When `VariantIdentity.view_id` is set it queries Global Store for view metadata/leaves. If GS has no variant records the orchestrator falls back to canonical transport.
  - P2P transport requests propagate the variant identity so the sender can either stream minimal byte ranges (slice) or decline with `PARTIAL_COVERAGE`.
  - Disk fallback and registration reuse the canonical id; variant metadata is flushed through `UpdateArtifactViewState`.
- The resulting replica handle records the `view_id` so subsequent lookups resolve the correct ByteSpace.
- Prior to invoking `ViewPlanner`, the engine obtains canonical index bytes either from on-disk `tensor_index.json`/descriptor or, when loading via AUTO, through `get_artifact_index_by_id` on the Global Store client. This keeps planning authoritative and avoids re-parsing JSON in multiple layers.

### 2.5 Transpose Execution

- `ViewPlanner` produces a `TransformPlan` for `transpose` ops that lists dimension swaps and expected output strides.
- When placement is `SERVER`, the daemon executes transpose on GPU memory when available; a CPU fallback path materialises into pinned memory and re-uploads if GPU execution is not possible. Placement `CLIENT` skips server transforms and mirrors canonical transport.
- The transform executor integrates with `ViewPlanSource`: for transpose, the SelectionPlan streams canonical bytes into a staging buffer before the transform kernel runs.
- TreeHash computation uses the transformed byte stream so `view_data_hash` represents the post-transpose ByteSpace.
- All TreeHash and metadata emission leverage v3 ordering and typing rules. Persisted descriptors, indices, and GS mutations set `schema_version="v3"`; down-level schema branches are removed from the codebase once this design lands.

## 3. gRPC / Proto Additions

```proto
message NarrowOp   { int32 dim = 1; int64 start = 2; uint64 length = 3; }
message TransposeOp{ int32 dim0 = 1; int32 dim1 = 2; }
message Op { oneof kind { NarrowOp narrow = 1; TransposeOp transpose = 2; } }
message TensorViewOps { repeated Op ops = 1; }
enum TransformPlacement { TP_SERVER = 1; TP_CLIENT = 2; } // no AUTO
message ViewSpec { map<string, TensorViewOps> tensors = 1; } // transform-only; placement not included

message MaterializeReplicaRequest  {
  // ... existing fields
  oneof view_identity {
    ViewSpec view = 1001;   // transform-only spec; placement is request-level
    string   view_id = 1002; // direct identity; skip JSON normalization
  }
  TransformPlacement placement = 1003; // request-level placement
}
message MaterializeReplicaResponse { ...; bytes view_index_json = 1001; string view_data_hash = 1002; }

// Error detail for partial coverage (attached via google.rpc.Status.details)
enum ByteSpaceKind { BS_CANONICAL = 1; BS_VARIANT = 2; }
message Range { uint64 off = 1; uint64 len = 2; }
message PartialCoverageDetail {
  ByteSpaceKind space_kind = 1; // canonical or variant
  string        space_id   = 2; // index_multihash or view_id
  repeated Range missing_ranges = 3; // compact set of uncovered byte ranges
}
```

Registration path:

```proto
message RegisterArtifactRequest {
  // ... existing registration fields (storages, aliases, LIP/coalesced options)
  oneof view_identity {
    ViewSpec view = 1001; // when present, registers a variant view (partial/full)
    string   view_id = 1002; // direct identity path
  }
  TransformPlacement placement = 1003; // request-level placement
}

message RegisterArtifactResponse {
  // ... existing fields (descriptor, plan, lease)
  bytes  view_index_json = 1001; // view output layout (if view present)
  string view_data_hash  = 1002; // set when the view is verified on first completion
  // dims: int32; offsets: int64; lengths: uint64 in proto.
}
```

## Global Store Integration (v1)

Role & boundaries
- Global Store (GS) does not perform view planning, routing, or transforms. Those remain in the C++ core/daemon.
- GS provides durable anchors for variant identity and fast verification (leaves), and keeps replica discovery on canonical `chunk_directory` unchanged.

Minimal RPC surface (reuse existing service; fewest new methods)
- Extend `GetArtifactInfoById` to optionally return leaves and view metadata for a selected ByteSpace:
  - Request additions:
    - `include_replicas: bool` (default true; preserves existing behavior)
    - `include_leaves: bool` (default false)
    - `oneof space { bool canonical; string view_id; }` (required when `include_leaves` or `include_view_meta`)
    - `leaf_idxs: repeated uint64` (when `include_leaves`)
    - `include_view_meta: bool` (default false; only valid if `space=view_id`)
  - Response additions:
    - `repeated Leaf { uint64 leaf_idx; string digest; } leaves`
    - `ViewMeta { bytes view_spec_json; uint64 view_size; string view_data_hash; google.protobuf.Timestamp verified_at; } view_meta`
- Add `UpdateArtifactViewState` for variant upsert & leaf writes as described in the plan; daemon callers always pass the canonical artifact id plus the view identity.
- v1 does **not** change `chunk_directory` routing. Canonical coverage continues to determine replica discovery, while variants rely on the leaves table for integrity checks.

- Add a single write RPC `UpdateArtifactViewState` to upsert variant metadata and/or batch write leaves in one call:
  - `artifact_id: string`
  - Optional `VariantUpsert { view_id, view_spec_json (bytes), view_size (u64), optional view_data_hash (string), mark_verified (bool) }`
  - `repeated LeafWrite { enum SpaceKind { CANONICAL=0, VARIANT=1 }; SpaceKind space_kind; string space_id; repeated Leaf leaves; }`
  - Response: `Status`

Read/Write flows (daemon ↔ GS)
- First verification: daemon computes `view_id` and completes a strong verification, then calls `UpdateArtifactViewState` with `VariantUpsert` (and `mark_verified=true`) and `LeafWrite` for the variant (and optionally canonical if aligned).
- Retrieval-time fast check: daemon uses `GetArtifactInfoById` with `include_leaves=true`, `space=view_id`, and `leaf_idxs=[...]` to fetch only needed leaf digests; `include_replicas=true` continues to return canonical replicas as today.

Operational notes
- Single `GlobalStoreService` remains; no separate view service in v1.
- P2P selection and transports continue to rely on canonical `chunk_directory`; no ByteSpace-aware routing in GS for v1.

## 4. Database Schema Changes (`schema.sql`)

This design adopts the optimal path for a greenfield system: add new tables and move canonical index to v3 for all new artifacts. No backward-compatibility constraints are considered.

### 4.1 New Tables

```sql
-- Variant registry per artifact/view
CREATE TABLE IF NOT EXISTS variants (
  artifact_id    TEXT    NOT NULL,
  view_id        TEXT    NOT NULL,
  view_spec_json TEXT    NOT NULL,
  view_size      BIGINT  NOT NULL,
  view_data_hash TEXT    NULL,        -- set when first full verification completes
  verified_at    TIMESTAMP WITH TIME ZONE NULL,
  created_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (artifact_id, view_id)
);

CREATE INDEX IF NOT EXISTS idx_variants_artifact ON variants(artifact_id);
CREATE INDEX IF NOT EXISTS idx_variants_verified  ON variants(artifact_id, verified_at);
CREATE INDEX IF NOT EXISTS idx_variants_view_id  ON variants(view_id);

-- Leaf digests anchored to a ByteSpace (canonical or variant)
CREATE TABLE IF NOT EXISTS leaves (
  artifact_id TEXT    NOT NULL,
  space_kind  CHAR(1) NOT NULL, -- 'C' canonical, 'V' variant
  space_id    TEXT    NOT NULL, -- index_multihash or view_id
  leaf_idx    BIGINT  NOT NULL,
  digest      BLOB    NOT NULL, -- raw 32-byte SHA-256 leaf digest
  PRIMARY KEY (artifact_id, space_kind, space_id, leaf_idx)
);

CREATE INDEX IF NOT EXISTS idx_leaves_space ON leaves(artifact_id, space_kind, space_id);
```

Notes:
* `leaves` rows are anchored to a ByteSpace defined by `(space_kind, space_id)`; 'C' denotes canonical, 'V' denotes variant.
* `variants.view_spec_json` stores a canonicalized JSON form for auditing and is not used for identity hashing.
* `leaves.digest` stores the raw 32-byte SHA-256 digest for each leaf; textual forms (hex/base32) should be produced on demand by clients if needed. No per-leaf multihash/base32 encoding is stored.
* v1 不引入 per-replica ByteSpace 覆盖持久化；GS 仍基于 canonical `chunk_directory` 进行路由。未来若需要 ByteSpace 感知的覆盖/路由能力，将在后续版本单独扩展 schema。

### 4.2 Canonical Index v3 Only

* All artifacts must use canonical index v3 (strict u64 fields and ordering) and record `artifacts.schema_version='v3'`.
* No mixed v2/v3 handling.

### 4.3 Operational Simplicity

* Tables are created with `IF NOT EXISTS`; no backfill is required.
* Variant rows and leaves are written when `register_view`/`get_view` runs; canonical leaves are written only for partial registration flows.

## 5. SDK API

```python
store.get_view(
    key="model:v2",
    slices={"wte.weight": (slice(0, 128),)},
    placement="SERVER",   # explicit; guarantees min-byte path
)

store.put_view(tensors=my_tp_shard_state,
               key="model:v2",
               slices={"wte.weight": (slice(tp_off, tp_off+tp_len),)},
               placement="SERVER")  # implemented with plan 0018
```

> **Scope note:** The v1 rollout focuses on retrieval (`get_view` / `get_view_into`). SDK-side `register_view` will ship alongside the follow-up execution plan (`0018-artifact-view-registration`) so registration flows remain canonical in the initial release.

### 5.1 Additional SDK APIs

```python
def get_view_into(target: dict[str, torch.Tensor], *,
                  artifact_id: str | None = None,
                  key: str | None = None,
                  slices: Mapping[str, Sequence[slice]] | None = None,
                  transpose: Mapping[str, Sequence[tuple[int,int]]] | None = None,
                  placement: Literal["SERVER","CLIENT"],
                  device: torch.device | str | None = None) -> None:
    """Materialize a view and copy into preallocated target tensors on the selected device.
    - Validates shapes/dtypes vs view_index_json returned by daemon.
    - Releases any temporary replica after copy.
    - Per-tensor mixing allowed: some tensors may be in `slices`, others in `transpose`. A single tensor cannot appear in both. For `slices`, each tensor may specify at most one dimension slice.
    """

def register_view(tensors: Mapping[str, torch.Tensor], *,
                  key: str | None = None,
                  slices: Mapping[str, Sequence[slice]] | None = None,
                  transpose: Mapping[str, Sequence[tuple[int,int]]] | None = None,
                  placement: Literal["SERVER","CLIENT"],
                  ttl_ms: int | None = None) -> RegisteredArtifact:
    """Register a view (partial or full) under the artifact key.
    - Persists variant metadata and variant leaves;当切片对齐时可同时写入 canonical leaves（可选）。
    - For transpose-only, default placement=CLIENT to avoid double compute.
    - Per-tensor mixing allowed: some tensors may be in `slices`, others in `transpose`. A single tensor cannot appear in both. For `slices`, each tensor may specify at most one dimension slice.
    """
```

## 6. Compatibility & Acceptance Criteria

* **Backward-compatible**: existing canonical paths untouched (`view` absent ⇒ old behaviour).
* **Integrity**: `view_data_hash` must equal recomputed value on first verification; subsequent replicas require only leaf checks. Variant loads persist per-view hashes under `verification.view_<sanitized_view_id>.json`; each record includes `byte_space_id` so canonical `verification.json` is never reused for a different ByteSpace.
* **Performance**: slice + server-side selection must not exceed 2 µs/MB extra latency on intra-node LIP fast path.
* **P2P**: router returns `UNAVAILABLE` with detail `PARTIAL_COVERAGE` if coverage within the requested ByteSpace cannot be satisfied.
* **Test Cases**
  1. Register two partial TP slices, verify they compose full canonical and individual variant digests.
  2. Retrieve a slice view and verify output tensor equals ground-truth torch.narrow().
  3. Transpose view retrieved with client-side transform equals torch.transpose() result.
  4. get_view_into: preallocate target tensors with expected shapes; API copies and validates against view_index_json.
  5. register_view: variant coverage and leaves recorded; first verification yields view_data_hash; subsequent registrations only check leaves.
  6. ViewSpec normalization folds identity; when identity, `view` omitted and canonical path is taken.
  7. `view_id` is invariant to placement; providing only `view_id` yields identical output without JSON parsing.
  8. TreeHash root matches CPU and GPU implementations given the same bytes; chunk sizing follows the fixed policy.
  9. Providing both `slices` and `transpose` in v1 yields `INVALID_ARGUMENT`.

## Implementation Tracks & Status

- **Core (`docs/plans/0016-a-artifact-view-v1-core.md`)** — Completed. ViewPlanner, ViewPlanSource, transform executor, residency/AUTO wiring, and Canonical Index v3 enforcement ship with full Catch2 coverage.
- **Daemon & Proto (`docs/plans/0016-b-artifact-view-v1-daemon-proto.md`)** — Completed. RPCs carry `ViewSpec`/`view_id`, placement, and view metadata; orchestrator/P2P paths propagate variant identity.
- **Global Store (`docs/plans/0016-c--artifact-view-v1-global-store.md`)** — Completed. Schema adds `variants`/`leaves`, GS RPCs serve variant metadata, and repositories/tests cover partial-coverage detail.
- **SDK (`docs/plans/0016-d--artifact-view-v1-sdk.md`)** — Completed for retrieval scope. `get_view`/`get_view_into` normalise specs, call the daemon with placement, and validate layouts; registration helpers follow plan 0018.

# Trade-offs & Risks

| Risk | Mitigation |
|------|------------|
| Canonical-variant drift (bug) | single C++ ViewPlanner implementation used by SDK & daemon |
| Increased DB writes (variant leaves) | batched leaf writes; only variant ByteSpace persisted |
| GPU cost for server transpose | default placement=CLIENT when bytes unchanged |

# References

* 0015-artifact-view.md – prior proposal
* 0007-content-addressed-artifact-id.md
* docs/internals/canonical-index.md
* Torch operators docs: `torch.narrow`, `torch.transpose`
