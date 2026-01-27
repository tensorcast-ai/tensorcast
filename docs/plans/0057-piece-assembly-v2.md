---
title: Piece Assembly v2 (Plan)
areas: ["core", "daemon", "global_store", "sdk", "proto"]
status: draft
created: 2026-01-26
last_updated: 2026-01-26
links:
  design: ../designs/0057-piece-assembly-v2.md
---

# Objective

Ship "Piece Assembly v2" on top of `0056-dense-piece-assembly-sealing` by adding:

- transform-aware assembly (starting with transpose)
- overlap tolerance with equality proofs (replicated overlaps)
- LIP pieces
- ByteSpace-aware per-chunk directory (`chunk_directory_v2`)
- robust sealing locks/status/cleanup policies
- copy/transport optimizations (strided copy + direct write planning)
- optional shard-set manifests for expected piece sets and semantics

This plan is intentionally phased so each phase is shippable while keeping v1 behavior as the default.

# Current State & Grounding (2026-01-26)

Baseline: `docs/plans/0056-dense-piece-assembly-sealing.md` is complete.

Key constraints in current code:

- Pieces reject transforms (`transpose`) at registration time:
  - `core/store/runtime/metadata/registration_backend.cc`
- Assembly rejects transforms and rejects overlaps:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `tensorcast/global_store/services/view_state_service.py`
- Piece registration is disallowed for VRAM_LEASED/LIP:
  - `daemon/service/controllers/registration_controller.cc`
- The chunk directory is canonical-only and cannot index view replicas:
  - `schema.sql::chunk_directory`
  - `proto/tensorcast/global_store/v1/global_store.proto::ChunkStateUpdate`

# Phases & Milestones

- [ ] Phase 1: Foundations (schema + proto)
  - [ ] Milestone: Add `chunk_directory_v2` (ByteSpace-aware) alongside legacy `chunk_directory`
  - [ ] Milestone: Add sealing lock + seal status tracking tables (and minimal RPCs)
  - [ ] Milestone: Add shard-set manifest storage + RPCs
  - [ ] Milestone: Regenerate proto code (`bash tools/build_proto_python.sh`) and update callers

- [ ] Phase 2: Overlap proofs for replicated layouts (selection-only)
  - [ ] Milestone: Persist per-piece canonical-chunk digests (reuse `leaves` with `space_kind=CANONICAL, space_id=view_id`)
  - [ ] Milestone: Allow `REPLICATE_EQUAL` overlaps when digests match; reject conflicts deterministically
  - [ ] Milestone: Assembly planner can select among multiple proven-equal sources (stable tie-breakers)

- [ ] Phase 3: Transform-aware assembly (transpose)
  - [ ] Milestone: Allow transpose pieces at registration time; compute canonical-chunk digests via canonicalization
  - [ ] Milestone: Assembly supports inverse transpose for sources and forward transpose for targets (reuse existing executors)
  - [ ] Milestone: Derived-view caching policy (optional) for expensive transforms

- [ ] Phase 4: LIP pieces (VRAM_LEASED)
  - [ ] Milestone: Lift `RegPlan::LEASE` restriction for `registration_kind=PIECE`
  - [ ] Milestone: Ensure all LIP export/lock/routing paths are ByteSpace-aware for view replicas
  - [ ] Milestone: Tests (fake CUDA) that register/export/assemble LIP pieces safely

- [ ] Phase 5: chunk_directory_v2 rollout and per-chunk integration
  - [ ] Milestone: Update chunk state update/query RPCs to carry ByteSpaceRef
  - [ ] Milestone: Dual-write/read rollout (v1 canonical flows remain stable)
  - [ ] Milestone: Optional: use chunk_directory_v2 for view-replica routing heuristics (hot/cold)

- [ ] Phase 6: Sealing robustness and lifecycle policies
  - [ ] Milestone: Coordinator lock (TTL + keepalive) gates sealers; sealing becomes resumable
  - [ ] Milestone: Seal status + progress observable via GS RPCs
  - [ ] Milestone: Post-seal policy knobs implemented (redirect/migrate/retire)

- [ ] Phase 7: Performance work (safe optimizations)
  - [ ] Milestone: Strided copy nodes to reduce range explosion (inner-dim narrow cases)
  - [ ] Milestone: Direct-write planning for assembly where safe (single-source, chunk-aligned)
  - [ ] Milestone: Stress tests (multi-daemon, TP-like) to validate throughput and stability

# Tasks (Implementation Notes)

1. **Schema discipline**
   - Update `schema.sql` for v2 tables with comments and indices.
   - Add migrations/backfills as needed (DuckDB).
2. **Proto discipline**
   - After any `.proto` changes, run `bash tools/build_proto_python.sh`.
3. **Keep v1 default behavior**
   - Overlaps remain rejected unless manifest explicitly opts in.
   - Transpose pieces remain rejected until Phase 3.
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

- Phase gating via explicit manifest opt-in and/or config defaults:
  - no existing user code should observe new overlap behaviors unless opted in.
- Deploy order (when implementing): schema/proto/GS first, then daemon/core, then SDK.

## Backout

- Keep legacy `chunk_directory` until `chunk_directory_v2` is validated in production.
- Keep overlap/transpose pieces behind manifest semantics so disabling the manifest restores v1 behavior.

# Risks & Tracking

- Range/chunk metadata blowup (mitigate with chunk sizing + batching).
- Transpose canonicalization cost (mitigate with caching policies and placement rules).
- Multi-component rollout complexity (schema/proto/GS/daemon/core/SDK must be updated together).
