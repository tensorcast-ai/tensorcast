---
slug: 0038-lazy-artifact-handle-hardening
title: Plan – Lazy Artifact Handle Hardening (prefetch, source policy, streaming)
links:
  design: ../designs/0036-03-lazy-artifact-handle.md
areas: ["sdk", "daemon"]
related_code:
  - tensorcast/api/_materialize.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/batch_context.py
  - tensorcast/api/store/README.md
  - daemon/grpc_service_impl.cc
  - daemon/service/controllers/materialization_controller.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
---

# Objective

Close the remaining gaps between the 0036 design series and the shipped code: make prefetch tickets usable end-to-end, honor source preferences for key-based materialization, stream `get_into` without rebuilding a full `state_dict`, expose the public `tc.artifact` surface at the top level, and align disk resolution bytes with the v2 canonical index contract. This plan hardens the lazy artifact stack without changing the public API surface.

# Current State & Gaps

- Prefetch: SDK issues `QueryReplicaStatus`/`ReleaseReplica` but the v2 daemon service does not implement these RPCs, so `PrefetchTicket.wait/cancel` fail with `UNIMPLEMENTED`.
- Source preference: `materialize_artifact_v2` ignores `SourcePreference` on the `materialize_by_key_v2` path; disk/local policies from `FallbackOptions` are dropped when fetching by key.
- Streaming copy: `MaterializationPipeline.get_into/get_into_async` rebuild full `state_dict` before copying, negating the descriptor iterator benefits.
- Public surface: `tc.artifact`/`artifact_async`/`from_disk` are not exported from `tensorcast/__init__.py`, diverging from the lazy handle design.
- Disk resolve bytes: `ResolveArtifactFromDisk` returns JSON in `canonical_index_bytes`; design calls for canonical bytes aligned with v2 materialization responses.

# Phases & Milestones

- [x] Prefetch RPC enablement
- [x] Source preference parity on key materialization
- [x] Streaming `get_into` copy path
- [x] Public API export alignment
- [x] Disk resolve canonical bytes alignment
- [ ] Tests, docs, and rollout

# Tasks

## Prefetch RPC enablement
- [x] Implement `QueryReplicaStatus` / `ReleaseReplica` in `StoreDaemonServiceV2Impl` and delegate to `MaterializationController`.
- [x] Add controller methods to read ticket status / release staged replicas (reuse existing session/replica bookkeeping).
- [ ] Update `tensorcast/daemon_ctl.py` error handling to surface retryable/non-retryable statuses cleanly for prefetch.
- [ ] Add Bazel tests (e.g., `//daemon:prefetch_ticket_test`) that issue `wait_for_completion=false` materialize, poll status, and release.

## Source preference parity
- [x] Pass `preference`, `verify_checksums`, `tensor_names`, and `view_subset_hash` through the key path in `materialize_artifact_v2` (tensorcast/api/_materialize.py).
- [ ] Add Python test ensuring `FallbackOptions.for_disk(...)` with key-backed fetch forces disk, and `prefer="local"` blocks P2P sources.

## Streaming `get_into`
- [x] Refactor `MaterializationPipeline.get_into` / `get_into_async` to copy directly from `payload.payload_iter()` without materializing a full dict; keep target validation via canonical/view index bytes.
- [ ] Add regression tests that gate iterator consumption to ensure no unused tensors are touched and that cancellation still releases replicas.

## Public API export alignment
- [x] Export `artifact`, `artifact_async`, `from_disk`, `BatchContext`, and `PrefetchTicket` from `tensorcast/__init__.py` and include in `__all__`.
- [x] Add a smoke test to import `tensorcast` and call `artifact(key=...)` using a stub runtime (no AttributeError).

## Disk resolve canonical bytes
- [x] Update `ResolveArtifactFromDisk` to return canonical index bytes encoded identically to v2 materialization responses (no `*_json` legacy), preserving `generation`.
- [ ] Add a parity test comparing `canonical_index_bytes` from `ResolveArtifactFromDisk` to the bytes returned by a `MaterializeReplica` call over the same artifact.

## Tests, docs, rollout
- [x] Python: `uv run pytest` new/updated suites (materialization pipeline, artifact handle, prefetch).
- [ ] C++: `bazel test //daemon:prefetch_ticket_test` and disk resolve parity test.
- [x] Update `tensorcast/api/store/README.md` with prefetch RPC availability, source preference behavior for keys, and streaming `get_into` note.
- [ ] Release notes/runbook: call out required daemon upgrade for prefetch RPCs and canonical index bytes change in `ResolveArtifactFromDisk`.

# Acceptance & Exit Criteria

- Prefetch tickets: `PrefetchTicket.wait()` succeeds against v2 daemon; `cancel()` releases replicas without UNIMPLEMENTED errors.
- Source policies: disk/local preferences honored for both artifact_id and key paths (validated by tests and logs).
- Streaming copy: `get_into` no longer constructs full `state_dict`; iterator-based copy verified by tests.
- Public API: `import tensorcast; tensorcast.artifact(...)` works and is discoverable.
- Disk resolve: `ResolveArtifactFromDisk.canonical_index_bytes` matches v2 materialization encoding and includes `generation`.

# Rollout / Backout

- Rollout: deploy daemon with new v2 RPC handlers first; then ship SDK changes (source preference fix, streaming copy, exports). Monitor prefetch metrics and materialization source labels.
- Backout: revert SDK to prior version (restores prefetch no-op behavior); if daemon issues arise, disable prefetch client-side (feature flag) and roll back daemon binaries.

# Risks & Mitigations

- RPC shape drift: ensure proto/buf regeneration and Bazel deps updated; add integration tests to lock expected fields.
- Prefetch resource leaks: require controller release path to be idempotent; add tests for double release and expired tickets.
- Behavior change for `get_into`: risk of target mismatch without full dict; mitigate with strict target validation and tests covering partial target maps.
- Disk index encoding change: ensure backward compatibility for callers expecting JSON by documenting the format and keeping parser tolerant to existing content.
