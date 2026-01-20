---
slug: 0054-stride-aware-view-plan-source
title: Plan - Stride-Aware ViewPlanSource Coalescing for Efficient Narrow Loading
links:
  design: ../designs/0054-stride-aware-view-plan-source.md
areas:
  - core
  - daemon
  - sdk
related_code:
  - docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md
  - docs/designs/0004-unified-runtime-config.md
  - docs/designs/0016-artifact-view-v1.md
  - tensorcast/api/store/deferred_loader.py
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/materialization/dataplane/contracts/{source,sink}.h
  - core/store/materialization/dataplane/sources/segment_plan_source.{h,cc}
  - core/store/materialization/dataplane/view/view_plan_source.{h,cc}
  - core/store/materialization/dataplane/view/view_planner.{h,cc}
  - core/store/materialization/dataplane/runtime/pump.{h,cc}
  - core/store/materialization/dataplane/view/tests/view_plan_source_test.cc
---

# Objective

Make `MaterializeIntoTarget` + v1 `narrow(axis=1)` views fast enough for TP>1 vLLM deferred loading by turning fragmented
selection execution (IOPS-bound) into stride-aware coalesced reads + packing (bandwidth-bound), without introducing new
RPCs.

# Current State & Grounding

- `DeferredLoader.commit()` uses a single `MaterializeIntoTarget` call (`tensorcast/api/store/deferred_loader.py`).
- View planning emits per-row segments for `axis=1` narrow (`core/store/materialization/dataplane/view/view_planner.cc`).
- `ViewPlanSource::read_at` executes the selection with many base `read_at` calls and an O(N) scan of `ranges`
  (`core/store/materialization/dataplane/view/view_plan_source.cc`).
- `MaterializeIntoTarget` wraps the base source in a canonical segment-plan source (`PlanBackedSeekableSource` in
  `core/store/runtime/ingestion/materialization_facade.cc`), which also does a linear scan per `read_at` today.
- Benchmark grounding (`docs/benchmarks/20260118-qwen2.5-32b-safetensors-loading-strategies.md`): TP=4 Strategy B is ~8×
  slower than C due to fragmented reads.

# Phases & Milestones

- [x] Phase 1: Compile + index range execution (CPU scalability)
  - [x] Milestone 1: Replace per-call linear scans with indexed lookup so `read_at` is O(log N + K) where K is overlapped
        runs/pieces:
        - `ViewPlanSource` (view selection execution).
        - Canonical segment-plan source (`PlanBackedSeekableSource` or a refactored reusable source).
  - [x] Milestone 2: Add unit tests covering large plans (including PAD) and multi-range `read_at` spanning many segments.
  - [x] Milestone 3: Preserve correctness under concurrent `read_at` calls (multi-range pumping); avoid shared mutable
        state without synchronization.

- [x] Phase 2: Add stride-aware coalescing (I/O + pack)
  - [x] Milestone 1: Detect `axis=1`-like strided runs from `SelectionPlan::ranges` (equal width, constant src stride).
  - [x] Milestone 2: Implement row-block superset reads + pack-to-dst in `ViewPlanSource`, including partial reads that
        start/end mid-row and span row boundaries.
  - [x] Milestone 3: Add a small per-instance block cache to avoid redundant reads under `pump_ranges` chunking; keep it
        bounded and thread-safe.
  - [x] Milestone 4: Add gating heuristics (min run length, min row bytes, max amplification) + decision logging, and
        robust fallback to baseline execution.

- [ ] Phase 3: Validate, observe, and document
  - [ ] Milestone 1: Add a reproducible benchmark run (or extend existing benchmark harness) that exercises
        `MaterializeIntoTarget` with a view spec matching the vLLM TP=4 shape.
  - [x] Milestone 2: Add observability (VLOG summary + counters) so base read calls/bytes, pack bytes, amplification, and
        cache hit rates are visible in perf runs.
  - [x] Milestone 3: Update module docs if behavior/constraints change (e.g., `core/store/materialization/README.md` or
        `core/store/materialization/dataplane/README.md`).

- [ ] Phase 4: Follow-ons (post-0054; long-term consistency)
  - [ ] Milestone 1: Preserve direct-write capability through range wrappers where it is semantically safe (contiguous
        runs/pieces), and add integration coverage using `pump_ranges` direct-write tests.
  - [ ] Milestone 2: Add internal tuning / kill-switch via unified runtime config (`docs/designs/0004-unified-runtime-config.md`)
        once the optimization is stable (no ad-hoc ENV/flags).
  - [ ] Milestone 3: Track Strategy D (“materialize once, load many”) and view-aware routing as a separate follow-on
        design/plan (aligned with `docs/designs/0052-deferred-slice-materialization.md`).

# Tasks

- [x] Implementation: `core/store/materialization/dataplane/view/view_plan_source.{h,cc}`
- [x] Implementation: `core/store/runtime/ingestion/materialization_facade.cc` (canonical segment-plan wrapper indexing)
- [x] Tests: add/extend view dataplane tests under `core/store/materialization/dataplane/view/tests/`
- [ ] Benchmarks: validate against the Qwen2.5-32B TP=4 `loading-meta.json` plan shape
- [x] Telemetry: add minimal metrics for base read calls/bytes + cache hit/miss + pack bytes + amplification

# Test / Rollout / Backout

- Tests:
  - Determinism: byte-identical output vs. baseline for random `SelectionPlan` inputs.
  - Strided correctness: synthetic plan with `stride_bytes > row_len_bytes` and partial reads crossing row boundaries.
  - Large plan coverage: ensure no O(N) behavior remains for 600k+ ranges.
  - Concurrency: concurrent `read_at` over disjoint offsets does not race/crash and remains correct.
- Rollout:
  - Default-on (heuristic-detected) inside core; no user-facing config.
- Backout:
  - Preserve baseline execution paths so the optimization can be removed via a simple revert if needed.
  - Follow-on: once stable, prefer a unified-config kill-switch over a code revert.

# Risks & Tracking

- Heuristic false-positives: stride detection must preserve correctness; fallback must be robust.
- Shifted bottleneck: once `ViewPlanSource` is indexed, `PlanBackedSeekableSource` (or equivalent) must not remain O(N).
- Memory footprint: block cache size must be bounded and avoid fragmentation.
- Concurrency: ensure thread safety for `read_at` under potential multi-range pumping.
- Read amplification: bounded by policy; multi-rank cold-load contention should be monitored and motivates Strategy D.
