---
title: Piece Assembly v2 (Plan)
areas: ["core", "daemon", "global_store", "sdk", "proto"]
status: draft
created: 2026-01-26
last_updated: 2026-01-31
links:
  design: ../designs/0059-piece-assembly-v2.md
---

# Objective

Ship "Piece Assembly v2" on top of `0056-dense-piece-assembly-sealing` by adding:

- transform-aware assembly (starting with transpose)
- overlap tolerance with equality proofs (replicated overlaps)
- LIP pieces
- replica-level view routing/discovery (view replicas are first-class via replica records scoped by `view_id`);
  canonical `chunk_directory` remains canonical-only in v2 (view chunk directory is deferred)
- sealing as an `Operation[T]` (leases + status + progress + snapshot) and policy-driven post-seal behavior
- copy/transport optimizations (strided copy + direct write planning)
- `LayoutSpec` + bindings/attachments for expected view sets and overlap semantics (single source of truth)

This plan is intentionally phased so each phase is shippable while keeping v1 behavior as the default. All range-copy and
zero-fill execution MUST go through the unified byte-range engine from `0058-unified-byte-stream-plan`
(`ByteRangeMap` + `ByteRangeProgram`) so v2 does not introduce a parallel copy engine.

# Current State & Grounding (2026-01-26)

Baseline: `docs/plans/0056-dense-piece-assembly-sealing.md` is complete.

Key constraints in current code:

- Pieces accept selection + transpose view specs at registration time (placement-dependent).
- Assembly allows overlaps only when an active `LayoutSpec` opts in (`REPLICATE_EQUAL`) and GS has validated proof digests.
- Piece registration is supported for VRAM_LEASED/LIP (in-place only).
- The chunk directory is canonical-only and cannot index view replicas:
  - `schema.sql::chunk_directory`
  - `proto/tensorcast/global_store/v1/global_store.proto::ChunkStateUpdate`

# Phases & Milestones

- [x] Phase 1: Final-state contracts (schema + proto + GS skeleton)
  - [x] Milestone: Hard-cut schema rename: `variants` -> `views`, `variant_coverage_ranges` -> `view_coverage_ranges`
  - [x] Milestone: Add `LayoutSpec` storage + RPCs (`layout_specs`, put/get by `layout_id`)
  - [x] Milestone: Add layout bindings/attachments + RPCs (`assembly_layout_bindings`, `artifact_layout_attachments`)
  - [x] Milestone: Add per-assembly runtime policy storage + RPCs (quorum/timeout; non-hashed)
  - [x] Milestone: Unify `Operation[T]` model end-to-end (single proto shape + shared persistence conventions)
    - includes lease fencing token + throttled status/progress writes
    - sealing uses the unified Operation surfaces (no sealing-specific OperationStatus shape)
  - [x] Milestone: Add overlap proof tables (`assembly_proof_commitments`, `tensor_proof_commitments`,
    `piece_proof_digests`)
  - [x] Milestone: DigestGrid conventions + batching limits (shared digest encoding + batch-first writes + GS bounds)
  - [x] Milestone: Regenerate proto code (`bash tools/build_proto_python.sh`) and update callers

- [x] Phase 2: Overlap proofs for replicated layouts (selection-only)
  - [x] Milestone: Daemon computes proof digests from ingested bytes (daemon is the witness); SDK does not supply digests
  - [x] Milestone: v2 scope restriction: `REPLICATE_EQUAL` is tensor-level only (full tensor coverage required)
  - [x] Milestone: GS validates full-tensor coverage for replicated tensors and enforces commitment equality
  - [x] Milestone: Allow `REPLICATE_EQUAL` overlaps only for full replicated tensors when digests match; reject conflicts
    deterministically
  - [x] Milestone: Planner selects among multiple proven-equal sources deterministically (stable tie-breakers)

- [ ] Phase 3: Transform-aware assembly (transpose)
  - [x] Milestone: Allow transpose pieces at registration time; compute proof digests from canonical-order bytes via
    canonicalization (inverse transpose as needed)
  - [x] Milestone: Assembly supports inverse transpose for sources and forward transpose for targets (reuse existing executors)
  - [ ] Milestone: Optional caching policy for expensive transforms (daemon-owned; anchored to `mi2_id` post-seal)

- [x] Phase 4: LIP pieces (VRAM_LEASED)
  - [x] Milestone: Lift `RegPlan::LEASE` restriction for `registration_kind=PIECE`
  - [x] Milestone: Ensure all LIP export/lock/routing paths are ByteSpace-aware for view replicas
  - [x] Milestone: Tests (fake CUDA) that register/export/assemble LIP pieces safely

- [ ] Phase 5: Sealing as an Operation + post-seal alias policies
  - [x] Milestone: `seal_assembly(...) -> Operation[SealedArtifact]` end-to-end (start/status/wait; leases)
  - [x] Milestone: Snapshot semantics implemented; retries reuse snapshot; sources are pinned or stabilized (daemon-side
    pinning/leases; GS persists snapshot/status)
  - [x] Milestone: Post-seal policies implemented (redirect/migrate_views/reuse_views_if_safe/retire_pieces)
  - [x] Milestone: On seal success, compute MI2-scoped proof commitments (`tensor_proof_commitments`) from canonical bytes
    and attach one or more `layout_id` to the sealed artifact as desired
  - [x] Milestone: Retention/GC policies implemented for assembly-scoped proof commitments and operation snapshots

- [ ] Phase 6: Performance work (safe optimizations)
  - [x] Milestone: Use `ByteRangeProgram` strided runs to reduce range explosion (inner-dim narrow cases)
  - [x] Milestone: Direct-write planning for assembly where safe (single-source, chunk-aligned)
  - [ ] Milestone: Stress tests (multi-daemon, TP-like) to validate throughput and stability
  - [ ] Milestone (optional): View chunk directory (only if justified by measurements; must include TTL/GC + scale targets)

# Status (2026-01-31)

- Verified via:
  - Python: `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py::test_post_seal_reuse_views_if_safe tests/python/test_dense_piece_assembly_sealing_acceptance.py::test_post_seal_migrate_views`
  - Python: `uv run pytest tests/python/global_store/test_grpc_service.py::TestGRPCService::test_check_proof_commitments_match`
  - C++: `bazel build //daemon:tensorcast_daemon`
- Deferred / still open:
  - Transform caching policy (Phase 3)
  - Performance stress testing + view chunk directory (Phase 6)

# Tasks (Implementation Notes)

1. **Schema discipline**
   - Update `schema.sql` to the v2 final-state schema (hard cutover; no dual-write compatibility tables).
   - Prefer explicit renames to converge terminology (`VIEW`) and remove legacy ambiguity.
2. **Proto discipline**
   - After any `.proto` changes, run `bash tools/build_proto_python.sh`.
3. **Keep v1 default behavior**
   - Overlaps remain rejected unless an active `LayoutSpec` binding/attachment explicitly opts in.
   - Transpose behavior remains opt-in via view specs + placement rules; caching is still deferred.
4. **Avoid deadlocks**
   - All compute nodes and any blocking waits run on `blocking_executor()` only.
5. **Doc sync**
   - When code changes land, update owning module docs (`core/store/README.md`, `daemon/README.md`,
     `tensorcast/global_store/README.md`) and cross-links as required by repo `AGENTS.md`.

# Test / Rollout / Backout

## Tests (minimum per phase)

- Python:
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/...`
- C++:
  - `bazel test //core/store:... --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
  - `bazel test //daemon:... --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- Proto lint:
  - `bazel test //proto/... --test_output=errors`

## Rollout

- Phase gating via explicit `LayoutSpec` opt-in and config defaults:
  - no existing user code should observe new overlap behaviors unless opted in via layout binding/attachment.
- Deploy order (when implementing): schema/proto/GS first, then daemon/core, then SDK.

## Backout

- Because this is a hard cutover (schema/proto rename), backout is a revert. Behavioral backout is still possible by:
  - leaving `LayoutSpec` unbound (default `DISJOINT`)
  - disabling v2 sealing policies via config (post-seal reuse/migrate/retire)

# Risks & Tracking

- Digest/range metadata write amplification (mitigate with DigestGrid batching limits + commit semantics).
- Transpose canonicalization cost (mitigate with caching policies and placement rules).
- Multi-component rollout complexity (schema/proto/GS/daemon/core/SDK must be updated together).
