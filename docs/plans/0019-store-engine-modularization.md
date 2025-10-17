---
slug: 0019-store-engine-modularization
title: StoreEngine Modularization — Implementation Plan
links:
  design: ../designs/0019-store-engine-modularization.md
areas: ["core"]
related_code:
  - core/store/store_engine.cc
  - core/store/loader/**
  - core/store/components/**
---

# Objective

Modularize `StoreEngine` by introducing `IndexService` (loader/), `VerificationService` (loader/), and `EvictionService` (components/), replacing duplicated inline logic and preserving public behavior. Reduce `store_engine.cc` size significantly while keeping tests green.

# Phases & Milestones

- [ ] Phase 1: Mechanical Extraction (no behavior changes)
  - [ ] Create `core/store/loader/index_reader.{h,cc}` with APIs from the design; move canonical index reading and total size derivation.
  - [ ] Create `core/store/loader/verification_utils.{h,cc}`; move data/view hash, verification.json reuse/generation, descriptor writeback.
  - [ ] Create `core/store/components/eviction_service.{h,cc}`; move GPU/CPU eviction logic.
  - [ ] Update Bazel BUILD rules for new libraries using `sc_cc_library`.
  - [ ] Refactor `store_engine.cc` to call services without changing external behavior or logs.

- [ ] Phase 2: Dedup and Simplify Orchestration
  - [ ] Replace repeated index canonicalization blocks in disk/P2P paths with `IndexService`.
  - [ ] Replace repeated verification/descriptor blocks with `VerificationService`.
  - [ ] Consolidate GPU retry-after-eviction to a single callsite using `EvictionService`.
  - [ ] Move `build_view_spec_json` and small helpers (align/filename sanitize) into appropriate utils.

- [ ] Phase 3: Copy & Variant Path Refinements
  - [ ] Optional: extract COPY_ONLY path into a `CopyService` (GPU→GPU) wrapping `Replica::copy_from`.
  - [ ] Ensure variant `view_data_hash` generation routes through `VerificationService` helpers.

- [ ] Phase 4: Tests & Docs
  - [ ] Unit tests for `IndexService` (safetensors/partition, malformed descriptors, size derivation).
  - [ ] Unit tests for `VerificationService` (CPU/GPU digest, verification.json reuse/mismatch, descriptor writeback, leaf digests).
  - [ ] Unit tests for `EvictionService` (GPU required bytes reclaimed; CPU pinned pool logic; metrics increments).
  - [ ] Update `core/store/README.md` and any linked docs per Doc Sync Rule.
  - [ ] Ensure existing tests (loader/components/daemon) pass in fake and real CUDA modes.

# Tasks

- Author service headers with precise includes (avoid forward declarations per repo guidelines).
- Implement functions by moving existing code; keep error messages and LOG/VLOG intact.
- Replace callsites in `store_engine.cc` incrementally; compile after each step.
- Update BUILD files: create `sc_cc_library` targets for each service; adjust deps to avoid transitive relies.
- Add Catch2 tests co-located with services; reuse existing test data where possible.
- Run linters and type checks:
  - `uv run ruff check .` (Python unaffected but kept in CI)
  - `uv run clang-tidy` on impacted C++ files
  - `bazel test //...` for C++ tree (with fake CUDA)

# Test / Rollout / Backout

Test
- Run targeted new unit tests and all existing loader/components tests.
- Run integration tests that exercise disk load, P2P load, and registration flows.

Rollout
- Land Phase 1 as a single PR (mechanical extraction only).
- Land Phase 2 in a second PR (dedup + orchestration simplification).
- Phase 3 optional PR for COPY_ONLY; Phase 4 adds tests and docs updates.

Backout
- Each phase is revertible via single‑commit reverts; no schema or proto changes.

# Risks & Tracking

Risks
- Subtle file system and descriptor edge cases during extraction.
- BUILD dependency mistakes causing hidden transitive includes.
- GPU hashing or device ordinal handling regressions if pointers are mishandled.

Mitigations
- Preserve messages/status codes; keep behavior identical in Phase 1.
- Add unit tests for malformed descriptors and empty indexes.
- Validate `gpu_device_id` handling paths; add fake CUDA coverage.

Owners
- Core Store maintainers; code reviewers from loader/components.

Success Criteria
- `store_engine.cc` reduced substantially (by >30%).
- All tests pass in fake CUDA; no regressions in real CUDA paths.
- Code ownership clear: services are independently testable and documented.


