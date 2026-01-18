---
slug: 0016-artifact-view-v1
title: Variant-Aware Artifact Views (Retrieval & Registration)
areas: ["core","daemon","global_store","sdk"]
related_code:
  - core/store/materialization/dataplane/**
  - daemon/**
  - tensorcast/api/**
  - tensorcast/global_store/**
links:
  schema: ../../schema.sql
---

# Summary

This document is the canonical description of **Variant-Aware Artifact Views** in TensorCast. A view is a reproducible, verifiable logical transformation over a canonical artifact that allows clients to work on slices or layout-transposed tensors without materialising the entire artifact. The same architecture powers both retrieval (`get_view`, `get_view_into`) and registration (`register_view`).

Variant views enable:

* **Partial transfer** – fetch or upload only the byte ranges that matter (for example tensor-parallel shards).
* **Layout control** – optionally transpose dimension pairs so consumers see the layout they expect.
* **Precise identity** – canonical `mi2_core` identity stays authoritative while each view receives its own `view_id` and `view_data_hash`.
* **Symmetric verification** – TreeHash plus per-leaf digests cover both canonical and variant ByteSpaces with identical policies.
* **Placement-aware execution** – clients choose whether transforms execute on the daemon (`SERVER`) or client (`CLIENT`) side; ingestion mirrors retrieval.

> This document supersedes the earlier registration draft (`0018-artifact-view-registration`). All artifact-view behaviour now lives here as a single source of truth.

# Goals / Non-Goals

## Goals

1. Allow clients to **retrieve** and **register** contiguous slices or per-tensor transposes without materialising the full artifact.
2. Keep the canonical identity (`mi2_core`) authoritative while introducing deterministic `view_id` + `view_data_hash` for each variant.
3. Reduce network, disk, and GPU IO by selecting minimal byte ranges server-side whenever possible.
4. Provide symmetric placement semantics (`SERVER`/`CLIENT`) and bidirectional planning so the same code paths cover loads and ingests.
5. Persist variant metadata and leaves in Global Store with clear schema extensions and SDK ergonomics.
6. Maintain end-to-end verifiability through TreeHash roots and per-leaf digests shared by canonical and variant ByteSpaces.

## Non-Goals

* New transform kinds (`reshape`, arbitrary striding, channels-last conversion, quantisation) remain out of scope.
* Runtime graph rewrites and execution planning beyond storage/transport are not addressed.
* Chunk-directory routing stays canonical-only in v1; replicas are still discovered via canonical coverage.
* No schema downgrades or v2 compatibility paths; canonical index v3 is mandatory everywhere.
* Automatic conflict detection or reconciliation between overlapping registrations is not part of this release.

# 1. Architecture & Interfaces

## 1.1 Supported Transform Surface

Version 1 supports two PyTorch-compatible per-tensor transforms:

* `torch.narrow(dim, start, length)` — contiguous slice (`step = 1`).
* `torch.transpose(dim0, dim1)` — dimension permutation.

Rules:

* A single tensor may participate in **either** `narrow` **or** `transpose`, never both.
* `narrow` is limited to one dimension per tensor; `length > 0` and `start + length ≤ shape[dim]`.
* You may mix transform kinds across different tensors inside the same view.
* Tensors omitted from the `ViewSpec` are treated as identity passes.

## 1.2 Coordinate Systems

| Term | ID | Byte range | Source |
|------|----|------------|--------|
| **Canonical ByteSpace** | `index_multihash` | `[0,total_size)` | canonical index v3 |
| **Variant ByteSpace**   | `view_id`         | `[0,view_size)`  | canonical index ⊕ ViewSpec |

All coverage ranges and leaves are expressed in exactly one ByteSpace.

```
flowchart LR
  A[Canonical
  ByteSpace (indexᵐ)] -- ViewSpec --> B[Variant
  ByteSpace (view_id)]
```

## 1.3 Identity

```
indexᵐ          = MH(canonical_index_v3_json)
dataᵐ           = TreeHash( AVBS[indexᵐ] )
mi2_core        = "mi2:" + indexᵐ + ":" + dataᵐ

// Identity is defined by deterministic Proto serialization of ViewSpec (transform-only; no placement)
view_id         = MH( DETERMINISTIC_PROTO(ViewSpec) ⊕ indexᵐ )
view_data_hash  = TreeHash( Transform(ViewSpec, AVBS[indexᵐ]) )
```

Identity folding (no-op views):

* During NORMALIZE, drop operations equivalent to identity (`narrow(dim,0,shape[dim])`, `transpose(dim,dim)`, cancelling compositions).
* If the normalized `ViewSpec` becomes identity, omit the `view` field, skip `view_id`, and route via the canonical path.

## 1.4 ViewSpec (Proto; canonical JSON for storage/auditing)

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
* `ops` order is preserved; adjacent compatible ops may be folded (full-length `narrow`, redundant `transpose`, transpose composition).
* JSON exists for readability; identity hashing always uses deterministic Proto serialization.
* Negative `start` values are canonicalised to positive offsets via `start' = shape[dim] + start`.
* Placement is request-level and never part of the ViewSpec or identity.

Field definitions:

* `tensors: map<string, TensorViewOps>` — per-tensor transforms; missing tensors imply identity.
* `TensorViewOps.ops: list<Op>` — evaluated left→right.
* Allowed ops: `narrow` (`dim: u32`, `start: i64`, `length: u64`) and `transpose` (`dim0`, `dim1` in `[0, ndim)` and distinct).
* Empty `ops` is allowed (treated as identity); unsupported ops yield `INVALID_ARGUMENT`.

## 1.5 Placement & Validation

Placement (request-level; AUTO removed):

* `SERVER`
  * `narrow`: daemon performs byte-range selection, returning minimal bytes; no GPU requirement.
  * `transpose`: daemon materialises the permutation (GPU preferred, CPU fallback when configured).
* `CLIENT`
  * `narrow`: daemon still performs byte-range selection; clients consume contiguous ranges.
  * `transpose`: daemon ships canonical layout; client permutes locally.

Validation & errors:

* `INVALID_ARGUMENT`: unknown tensor, dimension out of bounds, invalid offsets/lengths, unsupported op mix.
* `FAILED_PRECONDITION`: `SERVER` placement requested but daemon lacks capability (for example GPU transpose disabled).
* `UNAVAILABLE` + `PartialCoverageDetail`: P2P or disk sources cannot satisfy the requested ByteSpace coverage.

## 1.6 TreeHash & Leaf Digests (normative)

* Hash: SHA-256 for leaves and internal nodes (policy v1).
* Leaf chunk sizing: `determine_leaf_chunk_size(total_size, 4MiB)` from `core/common/artifact_hash_*` (`kMaxChunkSize=64MiB`, `kMinLeafChunkBytes=512KiB`, `kTargetLeafCount=4096`, 64-byte alignment).
* Leaves traverse the normalized byte stream of the selected ByteSpace; final leaf may be shorter but at least one leaf always exists.
* Stored leaf digest: raw 32-byte `SHA256(leaf_bytes)` inside `leaves.digest` (BLOB). Tree roots use multihash sha2-256 + multibase base32 (lowercase, no padding).
* Internal nodes: `SHA256(L || R)` with odd-node promotion.
* Any future hashing policy/version must be recorded alongside variant rows and validated end-to-end before adoption.

# 2. Core Components

```
core/store/materialization/dataplane/
  canonical_index.{h,cc}        // v3 stabiliser
  segment_plan_source.{h,cc}    // DATA & PAD plan
  view_planner.{h,cc}           // View → {TargetLayout, SelectionPlan, TransformPlan, ViewWritePlan}
  view_plan_source.{h,cc}       // streaming / GPU transform executor (retrieval)
  view_ingest_executor.{h,cc}   // streaming ingest + inverse transforms (registration)
```

## 2.1 ViewPlanner & Bidirectional Plans

`ViewPlanner::compute_view_plan` consumes canonical index v3 JSON plus a ViewSpec and emits:

* `TargetLayout` → `view_index_json` (strict v3 typing/order).
* `SelectionPlan` → minimal canonical ranges for retrieval.
* `TransformPlan` → permutation steps when needed.
* `ViewWritePlan` → inverse mapping for ingestion.
* Metadata flags (`is_contiguous`, `num_ranges`, `total_bytes`, `is_segment_aligned`, `requires_materialization`).

Bidirectional plan structures:

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
  TransformPlan inverse_transform;  // pre-computed inverse permutations
};
```

Retrieval code consumes `forward` paths; registration code consumes `inverse` paths. The shared planner ensures canonical/variant drift cannot occur.

## 2.2 ViewPlanSource (retrieval executor)

`ViewPlanSource` streams canonical bytes according to the `SelectionPlan`, performing scatter/gather as needed and invoking the forward transform when `requires_materialization` is true.

Execution rules:

| Transform pattern | Placement | Transport bytes | Notes |
|-------------------|-----------|-----------------|-------|
| Slice (`narrow`)  | SERVER    | min ranges      | 0-copy alias only when segment-aligned; otherwise scatter/gather |
| Transpose         | CLIENT    | full bytes      | avoids double compute unless explicitly SERVER |

SelectionPlan metadata:

* `is_contiguous: bool`
* `num_ranges: u32`
* `total_bytes: u64`
* `is_segment_aligned: bool`
* `requires_materialization: bool`

## 2.3 ViewIngestExecutor (registration executor)

`view_ingest_executor` mirrors `view_plan_source` but writes into canonical storage:

* Input: `BidirectionalViewPlan`, canonical sink (`SeekableSink` over UMA/VRAM), pointer to uploaded view buffer(s).
* `narrow`: iterate `ViewWritePlan::Chunk` entries; memcpy from view buffer into canonical offsets. `segment_aligned == true` enables UMA alias paths to avoid extra copies.
* `transpose`: apply `inverse_transform` via LibTorch kernels (GPU preferred). CPU fallback materialises into pinned memory and re-uploads if needed.
* Emits telemetry (bytes written, number of staging copies, GPU vs CPU, alias hit-rate) for observability.
* Integrated inside `StoreEngine::register_artifact_impl` whenever the request includes a view.

## 2.4 Transform Execution

* `TransformPlan` lists dimension swaps and expected output strides, reused by both executors.
* For `SERVER` placement, transpose materialisation occurs on GPU when available; CPU fallback can be enabled explicitly.
* TreeHash computation always uses the final byte stream (post-transform) so `view_data_hash` is consistent regardless of placement.
* All descriptors, indices, and verification artifacts set `schema_version="v3"`; down-level schema branches are removed.

## 2.5 Replica Residency & VariantIdentity

Variant replicas coexist with canonical replicas on the same daemon:

* `ReplicaKey` extends to `(canonical_artifact_id, view_id?, device, replica_idx)`. Absence of `view_id` denotes canonical ByteSpace.
* `ReplicaRegistry`, LIP caches, eviction heuristics, and lease tracking operate on the extended key to avoid mistaking a variant for a canonical replica.
* `register_replica_with_global_store` always publishes replicas under the canonical artifact id; variant metadata (view hash plus leaves) are handled via `UpdateArtifactViewState`.
* Canonical index v3 is the sole accepted schema.

`MaterializeHints` embeds the optional `VariantIdentity`:

```cpp
struct VariantIdentity {
  std::string canonical_artifact_id;
  std::optional<std::string> view_id;
  std::optional<ViewSpec> view;
  TransformPlacement placement;
};
```

The daemon fills `VariantIdentity` when requests carry a view, guaranteeing that both retrieval and registration flows query GS with consistent identities.

**Normative invariant (required)**
- Any non-identity view MUST have a deterministic `view_id`. If a request provides `view` but omits `view_id`, the daemon MUST resolve `view_id` deterministically (per the `view_id = MH(DETERMINISTIC_PROTO(ViewSpec) ⊕ indexᵐ)` rule in this design) and propagate it into `VariantIdentity.view_id` so `ReplicaKey` disambiguation and variant verification apply. Non-identity views that still lack a resolved `view_id` MUST be rejected to avoid canonical/variant `ReplicaKey` collisions.

# 3. Lifecycle Flows

## 3.1 Retrieval (`MaterializeReplica`)

1. SDK normalises `ViewSpec` (or accepts a `view_id`), selects placement, and sends `MaterializeReplicaRequest`.
2. Daemon resolves `view_id` (when only `view` is provided), populates `MaterializeHints::variant`, fetches canonical index v3 (local or GS), and probes the registry with the extended `ReplicaKey`. Identity views fold to the canonical path (no `view_id`).
3. On cache miss, `ViewPlanner` produces the bidirectional plan; retrieval path uses `SelectionPlan` plus `view_plan_source`.
4. P2P and disk transports propagate `(canonical_artifact_id, view_id)` so senders can stream minimal byte ranges or reject with `PARTIAL_COVERAGE`.
5. Daemon persists `view_index_json`, computes/validates `view_data_hash`, and records per-view verification metadata on disk (`verification.view_<id>.json`).
6. If variant metadata/leaves are missing from GS, the daemon upserts them after successful verification.

## 3.2 Registration (`RegisterArtifact`)

1. SDK exposes `register_view`, which normalises inputs, chooses placement defaults (SERVER for slices, CLIENT for transpose-only unless overridden), and streams tensors in view layout when placement is SERVER.
2. Daemon’s registration controller resolves `view_id` (fetching `ViewSpec` from GS when only an ID is provided) and passes `VariantIdentity` hints to `StoreEngine::register_artifact`.
3. The engine fetches canonical index v3, allocates canonical storage, and invokes `ViewPlanner`.
4. For `SERVER` placement, `view_ingest_executor` writes view bytes into canonical offsets (aliasing when possible) and performs inverse transforms. For `CLIENT` placement, only identity views are accepted and ingestion reuses the canonical path.
5. After canonical bytes exist, the engine computes:
   * canonical `TreeHash` (existing path),
   * variant `TreeHash` via `view_plan_source` replay over the canonical buffer,
   * per-leaf digests for the variant ByteSpace (and optionally canonical leaves for chunk-aligned slices).
6. Registration metadata is flushed:
   * local verification artifacts (`verification.json` plus `verification.view_<id>.json`),
   * `UpdateArtifactViewState` with `VariantUpsert {view_id, view_spec_json, view_size, view_data_hash, mark_verified=true}` and `LeafWrite` batches.
7. `register_replica_with_global_store` publishes the canonical replica as usual.
8. Errors:
   * `FAILED_PRECONDITION` instructs clients to retry with `placement="CLIENT"` when server-side transpose is unavailable.
   * `UNAVAILABLE` plus `PartialCoverageDetail` (`ByteSpaceKind=BS_CANONICAL`) surfaces missing canonical coverage so orchestrators can schedule additional shards.
   * `INVALID_ARGUMENT` rejects inconsistent tensor specs before any bytes are ingested.

# 4. gRPC / Proto Surface

```proto
message NarrowOp    { int32 dim = 1; int64 start = 2; uint64 length = 3; }
message TransposeOp { int32 dim0 = 1; int32 dim1 = 2; }
message Op { oneof kind { NarrowOp narrow = 1; TransposeOp transpose = 2; } }
message TensorViewOps { repeated Op ops = 1; }

enum TransformPlacement { TP_SERVER = 1; TP_CLIENT = 2; } // no AUTO
message ViewSpec { map<string, TensorViewOps> tensors = 1; } // transform-only; placement excluded

message MaterializeReplicaRequest {
  // ... existing fields
  oneof view_identity {
    ViewSpec view = 1001;   // normalized by daemon if provided
    string  view_id = 1002; // skip JSON normalization
  }
  TransformPlacement placement = 1003;
}

message MaterializeReplicaResponse {
  // ... existing fields
  bytes  view_index_json = 1001;
  string view_data_hash  = 1002;
}

message RegisterArtifactRequest {
  // ... existing registration fields
  oneof view_identity {
    ViewSpec view = 1001;
    string  view_id = 1002;
  }
  TransformPlacement placement = 1003;
}

message RegisterArtifactResponse {
  // ... existing fields
  bytes  view_index_json = 1001;
  string view_data_hash  = 1002;
}

enum ByteSpaceKind { BS_CANONICAL = 1; BS_VARIANT = 2; }
message Range { uint64 off = 1; uint64 len = 2; }
message PartialCoverageDetail {
  ByteSpaceKind space_kind = 1;
  string        space_id   = 2; // index_multihash or view_id
  repeated Range missing_ranges = 3;
}
```

`PartialCoverageDetail` is attached via `google.rpc.Status.details` for both retrieval and registration errors.

# 5. Global Store Integration

Role & boundaries:

* GS never plans or executes transforms; it stores canonical artifacts, variant metadata, and per-ByteSpace leaves.
* Replica discovery (`chunk_directory`) remains canonical-only; variants rely on the leaves table for integrity.

Read path (`GetArtifactInfoById`):

* Add knobs: `include_replicas`, `include_leaves`, `include_view_meta`, and `oneof space { bool canonical; string view_id; }`.
* When `include_leaves` is true, clients may pass `leaf_idxs` to fetch a subset.
* Response additions: `repeated Leaf { uint64 leaf_idx; string digest; }` and `ViewMeta { bytes view_spec_json; uint64 view_size; string view_data_hash; google.protobuf.Timestamp verified_at; }`.

Write path (`UpdateArtifactViewState`):

```
message VariantUpsert {
  string view_id = 1;
  bytes  view_spec_json = 2;
  uint64 view_size = 3;
  string view_data_hash = 4;
  bool   mark_verified = 5;
}

message LeafWrite {
  enum SpaceKind { CANONICAL = 0; VARIANT = 1; }
  SpaceKind space_kind = 1;
  string    space_id   = 2;
  repeated Leaf leaves = 3;
}
```

* Daemon calls `UpdateArtifactViewState` with `VariantUpsert` plus one or more `LeafWrite` batches after successful retrieval verification (first-time) or registration.
* Canonical leaves are optionally written when chunk-aligned slices fully populate canonical ranges; variant leaves are always written for verified views.
* `ViewStateService` (Python) wraps these RPCs with retries/telemetry and powers CLI tooling for inspection.

Operational notes:

* Tables are created lazily (see §6) and require no backfill.
* A single `GlobalStoreService` serves all RPCs; no dedicated view service is introduced in v1.
* Telemetry counters track variant writes, bytes per view, and verification latency.

# 6. Database Schema (`schema.sql`)

## 6.1 New Tables

```sql
CREATE TABLE IF NOT EXISTS variants (
  artifact_id    TEXT    NOT NULL,
  view_id        TEXT    NOT NULL,
  view_spec_json TEXT    NOT NULL,
  view_size      BIGINT  NOT NULL,
  view_data_hash TEXT    NULL,
  verified_at    TIMESTAMP WITH TIME ZONE NULL,
  created_at     TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (artifact_id, view_id)
);

CREATE INDEX IF NOT EXISTS idx_variants_artifact ON variants(artifact_id);
CREATE INDEX IF NOT EXISTS idx_variants_verified  ON variants(artifact_id, verified_at);
CREATE INDEX IF NOT EXISTS idx_variants_view_id   ON variants(view_id);

CREATE TABLE IF NOT EXISTS leaves (
  artifact_id TEXT    NOT NULL,
  space_kind  CHAR(1) NOT NULL, -- 'C' canonical, 'V' variant
  space_id    TEXT    NOT NULL, -- index_multihash or view_id
  leaf_idx    BIGINT  NOT NULL,
  digest      BLOB    NOT NULL,
  PRIMARY KEY (artifact_id, space_kind, space_id, leaf_idx)
);

CREATE INDEX IF NOT EXISTS idx_leaves_space ON leaves(artifact_id, space_kind, space_id);
```

Notes:

* `leaves` rows are anchored to `(space_kind, space_id)`; canonical and variant digests coexist.
* `variants.view_spec_json` stores the normalized JSON form for auditing and is not used for identity hashing.
* `leaves.digest` stores raw 32-byte SHA-256 digests; textual encodings are produced on demand.

## 6.2 Canonical Index v3 Only

* All artifacts record `artifacts.schema_version='v3'`.
* No mixed v2/v3 handling exists anywhere in the stack.

## 6.3 Operational Simplicity

* Tables use `IF NOT EXISTS`; no migrations/backfill required.
* Variant rows and leaves are written whenever `register_view` or first-time `get_view` verification runs; canonical leaves are written only when partial registrations align with canonical chunking.

# 7. SDK API & Ergonomics

```python
store.get_view(
    key="model:v2",
    slices={"wte.weight": (slice(0, 128),)},
    placement="SERVER",
)

store.get_view_into(
    target={"wte.weight": torch.empty((128, 4096), device="cuda:0")},
    key="model:v2",
    slices={"wte.weight": (slice(0, 128),)},
    placement="SERVER",
)

store.register_view(
    tensors=my_tp_shard,
    key="model:v2",
    slices={"wte.weight": (slice(tp_off, tp_off + tp_len),)},
    placement="SERVER",
    ttl_ms=3600,
)
```

Key behaviours:

* `_resolve_view_inputs` enforces per-tensor exclusivity, canonicalises slices/transposes, folds identities, and produces the deterministic `ViewSpec`.
* Placement defaults: slice-only views default to `SERVER`; transpose-only views default to `CLIENT` to avoid redundant compute, though callers may override.
* When `view_id` is provided, the SDK skips local normalization and asks the daemon/GS for the canonical spec. Outputs remain identical regardless of placement.
* `get_view_into` validates target tensor shapes/dtypes against `view_index_json` and releases temporary replicas after copy.
* `register_view` streams tensors in view layout only when placement is `SERVER`; otherwise it reuses the canonical upload path.
* Errors from `PartialCoverageDetail` are surfaced as `ArtifactError` objects so orchestrators can schedule missing slices.
* Upon completion, `RegisteredArtifact` instances include `view_index_json` plus `view_data_hash` for downstream validation.

# 8. Validation & Acceptance

* **Backward compatibility**: canonical flows remain untouched when `view` is absent.
* **Integrity**: `view_data_hash` must equal the recomputed value on every first verification; repeat loads rely on leaf checks.
* **Performance**: slice plus server-side selection adds ≤2 µs/MB on intra-node LIP fast paths; registration ingestion stays within +5% CPU of canonical uploads for narrow-only views.
* **Test coverage**:
  1. Two partial TP slices register successfully, composing the canonical artifact and producing individual variant digests plus shared canonical leaves.
  2. Slice retrieval equals the output of `torch.narrow`.
  3. Transpose retrieval equals `torch.transpose` when executed client-side and server-side.
  4. `get_view_into` copies into preallocated tensors and validates shapes/dtypes.
  5. `register_view` writes variant metadata, variant leaves, and optional canonical leaves for chunk-aligned slices.
  6. Identity ViewSpec folding results in canonical paths (no `view_id` emission).
  7. `view_id` is invariant to placement; providing only the ID yields identical results.
  8. TreeHash roots match between CPU and GPU implementations for the same bytes; chunk sizing follows policy constants.
  9. `view_ingest_executor` unit tests cover narrow plus transpose ingestion (CPU + fake CUDA) with alias hit/miss cases.
 10. Global Store unit tests verify `UpdateArtifactViewState` and `GetArtifactInfoById` leaf/view metadata semantics, including partial coverage errors.

# 9. Trade-offs & Risks

| Risk | Mitigation |
|------|------------|
| Canonical ⇔ variant drift | Single `ViewPlanner` plus bidirectional plans guarantee identical math for load/ingest. |
| View ingestion bug corrupts canonical replica | Symmetric unit tests plus `ABSL_CHECK` on total written bytes per plan chunk. |
| Increased DB writes (variant leaves) | Batched `LeafWrite` RPCs; canonical leaves only written when chunk-aligned. |
| GPU cost for server-side transpose | Default placement `CLIENT` unless bytes shrink; CPU fallback plus placement override guidance. |
| Transpose SERVER placement may require large staging buffers | Tile-based kernels with bounded scratch; fallback to CLIENT when resources insufficient. |
| Variants table growth due to per-shard registrations | Metrics plus retention tooling via `ViewStateService`. |

# 10. Status & References

Implementation status:

* **Core** — `ViewPlanner`, `view_plan_source`, and `view_ingest_executor` ship with Catch2 coverage and fake-CUDA parity.
* **Daemon & Proto** — gRPC surfaces carry `ViewSpec`/`view_id`, placement, and view metadata across both Materialize and Register RPCs; orchestrators propagate variant identity through P2P.
* **Global Store** — `variants`/`leaves` tables, `GetArtifactInfoById` extensions, and `UpdateArtifactViewState` are live with Python service helpers.
* **SDK** — `get_view`, `get_view_into`, and `register_view` share the same normalization pipeline and expose placement-aware ergonomics.
