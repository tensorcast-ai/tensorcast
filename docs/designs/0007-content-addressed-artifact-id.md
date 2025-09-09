---
id: design-0007-content-addressed-artifact-id
slug: 0007-content-addressed-artifact-id
title: Content-Addressed Artifact ID (mi2)
status: proposed
areas: ["core", "daemon", "global_store", "sdk"]
links:
  plan: ../plans/0007-content-addressed-artifact-id.md
related_code:
  - core/store/loader/**
  - core/store/replica/**
  - daemon/**
  - tensorcast/global_store/**
  - tensorcast/api/**
created: 2025-09-09
last_updated: 2025-09-09
---

# Summary

Adopt a content-addressed identity for artifacts so that identical content—regardless of origin (disk, memory, or P2P)—collapses under a single stable identifier. The new ID format, mi2, combines a structural fingerprint of the Canonical Index with a data fingerprint of the normalized linear byte stream. Global Store keys all replicas by this ID; Local Store and the daemon select optimal sources using it. This enables uniform routing, stronger integrity guarantees, and de-duplication across sources.

Key outcomes
- Stable identity: identical content yields the same `artifact_id` independent of source.
- Two-part fingerprint: `artifact_id = "mi2:" + index_multihash + ":" + data_multihash`.
- Fast paths preserved: runtime uses KEY_POINTS/SEGMENT verification; full digest remains optional.
- Clear authority: Canonical Index is generated in C++ core to avoid cross-language drift.

# Goals / Non‑Goals

Goals
- Converge disk, memory, and P2P replicas under one stable `artifact_id`.
- Keep verification cost acceptable in load and P2P fast paths; compute full digest at save/register/commit.
- Provide schema and API surfaces so Global Store and the daemon can route by `artifact_id`.

Non‑Goals / Constraints
- Do not interpret `mi2:` IDs as filesystem paths.
- No change to client-facing high-level API ergonomics beyond exposing `artifact_id`.
- Preserve 8-byte alignment and v2 index semantics; negative stride/mixed dtype/multi-device remain supported under v2 rules.

# Architecture & Interfaces

## Artifact ID format (mi2)

- Format: `artifact_id = "mi2:" + index_multihash + ":" + data_multihash`.
- Multihash: default `sha2-256`; future evolution via self-describing Multihash.
- Multibase: default base32.

## Canonical Index (input to structural fingerprint)

- Encoding: Canonical CBOR preferred; Strict Canonical JSON permitted during transition (ordered keys, fixed fields).
- Ordering: top-level entries sorted by `tensor_name` (ascending); fixed record field order `offset, size, shape, stride, dtype, storage_offset`.
- Alignment: 8-byte alignment consistent with v2 index rules.
- Authority: generated in C++ core to prevent divergence.

Stable grouping and layout
- Grouping key: `(dtype_code, device_id, group_key)` with `group_key = H(sorted(tensor_names_in_group))`, using the same hash as `index_multihash`.
- Within-group ordering: `tensor_name` ascending; maintain 8-byte alignment.

## Normalized linear data stream (input to data fingerprint)

- Definition: sequential bytes over `[0, total_size)` per the Canonical Index coalesced layout.
- Sources (as `SeekableSource`):
  - Disk: partitions read in filename order (`FilePartitionSource`).
  - P2P: streamed via `RemoteKeySource`.
  - Memory: contiguous buffers allocated during Begin; GPU-first with CPU/PCIe fallback.

## Fingerprint computation

- `index_multihash = MULTIHASH(canonical_index_bytes)`.
- `data_multihash = MULTIHASH(TREE_HASH(stream_bytes(0..total_size)))`.
- Tree hash: 1–16 MiB chunking, leaves/root with `sha2-256`; prefer GPU parallelization; CPU fallback available.
- Runtime verification: fast KEY_POINTS/SEGMENT checks during load/P2P; FULL verification optional or gated by hints.

## System integration

Global Store (keyed by `artifact_id`)
- Tables (logical model):
  - `artifacts(artifact_id PK, index_multihash, data_multihash, schema_version, encoding, hash_params_json, created_at, ...)`.
  - `artifact_index(index_multihash PK, schema_version, encoding, size_bytes, index_data BLOB, created_at, ...)`.
  - `replicas(id PK, artifact_id FK, source_type ENUM('DISK','MEMORY','P2P'), location|disk_path, device_id, created_at, ...)`.
- RPCs: `RegisterReplica(artifact_id, ...)`, `GetArtifactInfoById(artifact_id)`, `GetArtifactIndex(index_key)`.

Local Store / Store Daemon
- Begin: create in-memory artifact, allocate device memory, return `registration_id + cuda_ipc_handle`.
- Commit: compute `index_multihash` and `data_multihash`, build mi2 ID, return `ArtifactDescriptor`; register replica in Global Store.
- Prepare: single entry `materialize_replica(DeviceKey, mode, hints)`; supports `hints.artifact_id` and `hints.disk_path` (explicit path). Never treat `mi2:` as a path.

Python API
- `register_artifact(state_dict, ...) -> (state_dict, commit_info)` with `commit_info.artifact_id` starting with `mi2:`.
- Helpers: `generate_artifact_id_from_state_dict(...)`, `generate_artifact_id_from_path(...)` for audit/migration.

## ArtifactDescriptor (returned by Commit)

Fields
- `artifact_id` (`mi2:...`), `index_multihash`, `data_multihash`.
- `schema_version`, `encoding` (recommend `cbor`).
- `total_size`, `hash_params` (e.g., `chunk_size`, `fanout`).

On-disk
- `artifact_descriptor.json` saved beside `tensor_index.(cbor|json)` for standard format and `safetensors` directories.

# Schema Changes

No immediate schema.sql changes are codified in-repo yet. When introducing/aligning the Global Store tables, link and update the canonical schema with:
- `artifacts` keyed by `artifact_id` with split `index_multihash` and `data_multihash`.
- `artifact_index` keyed by `index_multihash` for deduplicated index storage.
- `replicas` referencing `artifacts` with source typing and device/location metadata.

# Trade‑offs & Risks

Trade-offs
- `tensor_index_key` (structure-only) cannot distinguish identical layout with different bytes; the mi2 two-part design fixes this by including data digest.
- Runtime keeps fast KEY_POINTS/SEGMENT checks; full data hash is deferred to save/register/commit to contain costs.

Risks and mitigations
- Historical artifacts with unstable ordering → treat as distinct IDs; provide normalization/backfill tools.
- Commit overhead → GPU-first hashing, CPU/PCIe fallback, and async backfill; block only when strict consistency is required.
- Environment-sensitive grouping → defined to be environment-independent; strict 8B alignment and ordering rules.

# Compatibility & Acceptance Criteria

Compatibility
- Coexists with explicit disk loading: `disk_path` hints remain; `mi2:` never resolves as a path.
- Standard format directories must include `artifact_descriptor.json` and `tensor_index.(cbor|json)`; legacy layouts require migration.
- `safetensors` directories may be backfilled with `artifact_descriptor.json` when permitted.

Acceptance criteria
- Commit returns an `ArtifactDescriptor` with consistent mi2 fields; identical content across sources yields identical `artifact_id`.
- Global Store registers and queries artifacts exclusively by `artifact_id`.
- Disk loader enforces descriptor presence (or backfills during transition) and verifies consistency when strict mode is enabled.
- Daemon `materialize_replica` accepts `artifact_id` and routes optimally (memory > local disk > remote).

# References

- Architecture: [Architecture Overview](../architecture/architecture-overview.md), [P2P Transfer Strategies](../architecture/p2p-transfer-strategies.md), [Artifact Loading Workflow](../internals/model-loading.md).
- Related implementations: `core/store/loader/*`, `core/store/replica/*`, `daemon/*`, `tensorcast/global_store/*`, `tensorcast/api/*`.
 

