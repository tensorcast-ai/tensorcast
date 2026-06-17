---
slug: unified-reference-only-disk-import
title: Unified Reference-Only Disk Import And Source Authority Unification
areas: ["daemon", "core", "sdk"]
status: draft
created: 2026-02-14
last_updated: 2026-04-14
related_code:
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_disk_resolve_utils.{h,cc}
  - daemon/service/controllers/disk_artifact_service.{h,cc}
  - daemon/state/local_disk_import_catalog.{h,cc}
  - daemon/app/daemon_app.cc
  - daemon/state/daemon_options.h
  - daemon/state/daemon_kernel.{h,cc}
  - daemon/util/path_utils.{h,cc}
  - core/store/runtime/ingestion/materialization_service.cc
  - core/store/materialization/runtime/pipeline/verification_stage.cc
  - core/store/materialization/dataplane/verification/verification_utils.cc
  - core/store/materialization/dataplane/metadata/index_reader.cc
  - core/store/materialization/dataplane/metadata/safetensors_util.cc
  - core/store/materialization/contracts/loading_spec.h
  - tensorcast/api/store/__init__.py
  - tensorcast/daemon_ctl.py
  - tensorcast/api/store/README.md
  - daemon/README.md
  - docs/designs/0004-unified-runtime-config.md
  - docs/designs/0007-content-addressed-artifact-id.md
  - docs/designs/0071-managed-shared-disk-persistence.md
links:
  plan: ../plans/0077-unified-reference-only-disk-import.md
---

# Summary

Adopt one disk import model only: **Reference-Only Import** with **single source authority**.

`from_disk` becomes a daemon-owned registration flow that never performs payload
copy/link/reflink and never treats a filesystem path as retrieval identity.
Import writes daemon-owned registry metadata and source fingerprints into one
authority root. Materialization resolves disk bytes only through this registry
plus managed shared-disk locations.

The correction in this revision is narrow but important:

- source payload bytes remain read-only for import/materialization semantics,
- but daemon-owned metadata backfill for descriptor/index sidecars may still be
  used as a bounded optimization,
- and that optimization must stay subordinate to the same daemon-owned source
  authority rather than reintroducing path-return or sidecar-owned truth.

This design intentionally removes resolve-style path-return contracts and
compatibility wrappers. It also removes uncontrolled sidecar/backfill branching
by narrowing metadata backfill to one daemon-owned, non-identity-bearing
optimization.

Scope boundary note:

- this document owns **import** and daemon-owned registration semantics only,
- it does **not** own the later mounted-source artifact attestation path used by
  binding-native mounted serving,
- and a public `ResolvePublicDiskSource`-style API, if present, must remain a
  separate mounted-source artifact attestation contract rather than a
  compatibility alias for import (`0115`).

Long-term correction from `0115`:

- mounted-source resolve is `msa1:`-first and lightweight by default,
- it may attach a trusted `mi2:` hint when already known,
- but it must not mint primary `mi2:` identity from the default resolve fast
  path.

# Problem Statement

Current behavior still carries multiple disk semantics and owners:

- Resolve RPC + stream contracts still expose path-return semantics.
- Unbounded sidecar/backfill/source-write branches coexist with reference
  semantics.
- Stream error model is message-based but not machine-readable.
- Core/daemon responsibilities around canonical index and source mutation are not explicitly unified.
- Runtime root selection for import metadata is described separately from daemon runtime-state topology.

This causes cross-module drift, duplicate code paths, and operational ambiguity that will grow with future features.

# Goals / Non-Goals

## Goals

- One import mode only across daemon/core/sdk.
- One source authority model only across local import and managed shared-disk.
- Core is the only canonical-index/hash authority.
- Source payload bytes remain read-only for import/materialization flows.
- Any source-directory metadata write must be limited to daemon-owned
  descriptor/index backfill and must not create a second source authority.
- Machine-readable, fixed import error taxonomy with deterministic gRPC mapping.
- No client-provided disk path in retrieval/materialization RPCs.
- Single stream progress contract for progress bars.
- Breaking change accepted pre-launch; no compatibility retention.

## Non-Goals

- Snapshot durability independent of source lifecycle for local imports.
- Cross-node discovery for daemon-local imports.
- Compatibility wrappers for resolve/path-return legacy contracts.
- Runtime mode flags that re-enable removed behaviors.
- Metadata-first public mounted-source resolve for binding-native serving; that
  follow-up contract is owned separately from import.

# Architecture & Interfaces

## Normative Invariants

1. `artifact_id` is identity only (`mi2:` for this flow), never a filesystem path.
2. Import is registration only for payload bytes: never payload
   copy/link/reflink and never payload-data mutation.
3. Source authority is daemon-internal only; retrieval/materialization accepts `artifact_id`, not `disk_path`.
4. Core owns canonicalization and hashing logic; daemon orchestrates and persists decisions.
5. There is exactly one import metadata authority inside daemon runtime state.

## Scope boundary with metadata-first public disk resolve

`0077` governs `from_disk(...)` / `ImportArtifactFromPath*` only.

Normative boundary:

- import returns durable import results and stays `mi2:`-based,
- public metadata-first mounted-source resolve may attest a mounted source into
  a daemon-local artifact family without immediately producing final `mi2:`,
- public mounted resolve keeps primary `artifact_id = msa1:...` even when a
  trusted `mi2:` hint is already known,
- the two flows may share format helpers and canonicalization utilities,
  but they must not share contract semantics, caching semantics, or authority
  assumptions.

## Bounded metadata backfill

The repository permits one bounded source-write behavior in this flow:

- daemon-owned metadata backfill for trusted descriptor/index sidecars.

Normative rules:

- metadata backfill must be optional and subordinate to daemon-owned import
  authority,
- it must not change payload bytes or artifact identity,
- it must not become a second retrieval or publication contract,
- and callers must still reason about `from_disk(...)` as reference-only import
  for payload bytes.

## Unified Source Authority: `ArtifactSourceRegistry`

Introduce one daemon-owned registry abstraction:

- `ArtifactSourceRegistry` (single owner in daemon runtime)
  - `ManagedSharedDiskBinding` (Global Store-discovered)
  - `LocalImportBinding` (daemon-local reference import)
- Unified lookup key: `artifact_id`
- Unified value shape:
  - source kind (`MANAGED_SHARED_DISK` | `LOCAL_IMPORT`)
  - canonical source root/path
  - source file fingerprint map (`inode,size,mtime_ns`)
  - canonical index bytes
  - generation
  - lifecycle timestamps

Implementation detail: existing `LocalDiskImportCatalog` can be refactored into this registry in-place, but must not remain as a second semantic owner after cutover.

## RPC And API Contract

For import surfaces, replace resolve-style path APIs with import APIs:

- `ImportArtifactFromPath` (unary)
- `ImportArtifactFromPathStream` (server stream)

Response payload:

- `artifact_id`
- `generation`
- `canonical_index_bytes`
- `import_state` (`READY` on success)

Forbidden in import response:

- `disk_path`
- sidecar/debug path fallback fields

SDK:

- `Store.from_disk(path)` remains entrypoint.
- SDK must consume only import contracts and cache by `artifact_id`.
- SDK/daemon client methods for `ResolveArtifactFromDisk*` are removed, not wrapped.
- A later public `ResolvePublicDiskSource(...)` API may exist as a separate
  mounted-source artifact attestation contract, but it is not an import alias
  and must not be implemented as a compatibility wrapper around
  `ImportArtifactFromPath*`.
- That mounted resolve contract must remain lightweight by default and must not
  require import-style payload hashing to form its primary identity.

## Stream Progress Contract

`ImportArtifactFromPathStream` is canonical for progress bars.

Event fields:

- `seq`
- `phase`
- `processed_bytes`
- `total_bytes`
- `percent`
- `done`
- `error`
- `error_code` (machine-readable enum, required when `error=true`)
- optional `message`
- optional `result` (success terminal only)

Terminal rules:

- exactly one terminal event per stream (`done=true`)
- success terminal includes `result`
- error terminal includes stable `error_code`

Fixed global phases:

1. `PREPARE`
2. `SCAN_SOURCE`
3. `READ_HEADERS`
4. `BUILD_CANONICAL_INDEX`
5. `HASH_DATA`
6. `WRITE_REGISTRY`
7. `DONE`
8. `ERROR`

## Core Authority And Source Mutation Policy

Core remains canonical authority for:

- safetensors source-index parsing
- canonical index rebuilding/coalescing
- index/data multihash
- descriptor/index validation

Daemon must reuse core APIs directly (no duplicate builder implementation).

Introduce explicit mutation policy for disk-origin loads:

- `source_mutation_policy=PAYLOAD_READ_ONLY` for import/materialization reference
  paths.

`PAYLOAD_READ_ONLY` requires:

- no payload-data mutation,
- no payload copy/link/reflink,
- no sidecar mirroring/symlinking/write fallback that becomes a second source
  authority,
- optional metadata backfill, when enabled, is limited to daemon-owned
  descriptor/index helpers and does not change retrieval identity or artifact
  identity.

Source-write failure modes tied to payload mutation or fallback data copying are
removed. Metadata-backfill failures remain bounded optimization failures and
must not become a second correctness contract.

## Source Input Rules

- Input path must resolve to local directory.
- Directory must contain at least one `*.safetensors` or `tensor.data*` layout.
- Import RPC is loopback/UDS only.
- Non-loopback import gets `PERMISSION_DENIED`.

## Import Metadata Root (Unified With Runtime Topology)

Import metadata storage must live under daemon runtime root, not a parallel temp-root universe.

Effective root:

- `$TENSORCAST_HOME/hosts/<host_id>/runtime/daemons/<daemon_id>/import`

Startup checks:

- create dir (`0700`)
- create/write/fsync/rename/remove probe
- metadata DB bootstrap check

If unavailable, startup fails fast: `IMPORT_ROOT_UNAVAILABLE`.

No ad-hoc env/flag mode switches are introduced.

## Materialization Read Path

- Materialization disk resolution goes through `ArtifactSourceRegistry` only.
- For each referenced file, validate fingerprint before open/read.
- On fingerprint mismatch or missing file: fail `FAILED_PRECONDITION` with `SOURCE_MUTATED`.
- No self-heal copy/link/backfill path exists.
- Disk source selection is typed and internal; no string path hints crossing layers.

## Error Model (Fixed And Machine-Readable)

- `SOURCE_NOT_FOUND` -> `NOT_FOUND`
- `SOURCE_PERMISSION_DENIED` -> `PERMISSION_DENIED`
- `SOURCE_FORMAT_INVALID` -> `INVALID_ARGUMENT`
- `SOURCE_MUTATED` -> `FAILED_PRECONDITION`
- `IMPORT_ROOT_UNAVAILABLE` -> daemon startup failure
- `REGISTRY_IO_FAILURE` -> `UNAVAILABLE`
- `POLICY_DENIED_NON_LOCAL_PEER` -> `PERMISSION_DENIED`

No fallback mode can change these outcomes.

## Security Model

- Import RPC and import stream RPC are local-only (loopback/UDS).
- Materialization does not accept client disk paths.
- Source path canonicalization + root policy checks are centralized.
- Registry bindings are daemon-local authority and not externally writable.

## Hard Decommission (No Compatibility Retention)

The following are deleted in the same coordinated change set:

- RPC interfaces:
  - `ResolveArtifactFromDisk`
  - `ResolveArtifactFromDiskStream`
- Legacy stream enums/events:
  - `ResolveDiskPhase`
  - `ResolveArtifactFromDiskStreamEvent`
- SDK and daemon client methods for resolve-style APIs.
- Sidecar mirror and fallback helpers.
- Uncontrolled source backfill branches for descriptor/index writes.
- Compatibility env knobs used only by resolve legacy behavior.
- Any remaining code path that treats `disk_path` as retrieval identity.

No adapter layer and no dual runtime semantics are kept.

## Naming Compliance

```mermaid
flowchart LR
  A[Client from_disk] --> B[ImportArtifactFromPath]
  B --> C[Core builds canonical index and hashes]
  C --> D[Daemon writes ArtifactSourceRegistry]
  D --> E[Materialize by artifact_id]
  E --> F[Resolve source binding from registry]
  F --> G[Validate source fingerprint]
  G --> H[Read payload bytes]
```

Proposed interfaces follow repository naming rules:

- Classes/structs (`PascalCase`)
  - `ArtifactSourceRegistry`
  - `ArtifactSourceBinding`
  - `ImportedArtifactEntry`
- Functions/methods (`snake_case`)
  - `import_artifact_from_path`
  - `resolve_artifact_source_binding`
  - `validate_source_fingerprint`
- Constants (`ALL_CAPS`)
  - `IMPORT_ROOT_PROBE_FILENAME`
  - `IMPORT_REGISTRY_SCHEMA_VERSION`

# Schema Changes

- No change to repository root `schema.sql` (Global Store schema unchanged).
- Introduce daemon-local SQLite schema under daemon runtime root for `ArtifactSourceRegistry`.
- This local schema is daemon runtime state, not Global Store canonical schema.

# Trade-offs & Risks

- Pros:
  - single source authority and deterministic behavior
  - no split ownership across local-import vs managed-disk semantics
  - payload immutability remains the correctness rule
- Cons:
  - local import durability still depends on source continuity
  - stricter behavior removes permissive fallback ergonomics
  - bounded metadata backfill still requires a small source-write policy surface
- Risk:
  - source-file churn increases `FAILED_PRECONDITION`; mitigated via explicit `SOURCE_MUTATED` code and pre-read fingerprint checks.
  - hard deletion scope is large; mitigated by one coordinated merge and full test/doc gates.

# Compatibility & Acceptance Criteria

Compatibility policy:

- Breaking change accepted (pre-launch).
- Resolve/path-return semantics are removed.
- No compatibility wrappers remain in daemon or SDK.

Acceptance criteria:

- Import of safetensors-only directory succeeds without source metadata files.
- Import/materialization on imported sources never mutates payload bytes or
  treats source metadata writes as retrieval truth.
- If metadata backfill is enabled, only daemon-owned descriptor/index sidecars
  may be written and they remain an optimization rather than a required import
  contract.
- Materialization never consumes client-provided disk path hints.
- Unary and stream import share one internal phase model and one error mapper.
- Stream import emits exactly one terminal event with machine-readable `error_code` on failure.
- Source mutation after import returns `FAILED_PRECONDITION` with `SOURCE_MUTATED`.
- Daemon startup resolves import root under daemon runtime topology and fails fast with `IMPORT_ROOT_UNAVAILABLE` when unavailable.
- `ResolveArtifactFromDisk*` RPCs and SDK client methods are removed.
- Resolve and uncontrolled sidecar/backfill legacy codepaths are removed.
- Redundant compatibility env knobs for resolve legacy behavior are removed.
- No parallel import subsystem remains after refactor.
- Daemon and Python module docs are updated in the same change set:
  - `daemon/README.md`
  - `tensorcast/api/store/README.md`
- Cross-cutting architecture docs are updated in the same change set:
  - `docs/architecture/architecture-overview.md`
  - `docs/architecture/p2p-transfer-strategies.md`
  - `docs/architecture/high-availability-design.md`
  - `docs/internals/model-loading.md`
  - `docs/architecture/api/materialization-flow.md`

# Alternatives Considered

1. Keep dual authority (`LocalDiskImportCatalog` + managed disk separate)
- Rejected because it preserves duplicated ownership and drift.

2. Keep resolve-style APIs as wrappers
- Rejected because wrappers preserve duplicate interfaces and long-term dead code.

3. Re-implement canonical builder in daemon
- Rejected because canonical authority already belongs to core and duplication invites drift.

4. Separate temp-root import metadata layout
- Rejected because it conflicts with established daemon runtime root topology.

# References

- `docs/designs/0001-docs-system-design.md`
- `docs/designs/0004-unified-runtime-config.md`
- `docs/designs/0007-content-addressed-artifact-id.md`
- `docs/designs/0071-managed-shared-disk-persistence.md`
