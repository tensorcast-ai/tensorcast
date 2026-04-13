---
slug: collective-first-binding-realization-for-tp-serving-startup
title: Collective-First Binding Realization for TP Serving Startup Plan
status: implemented
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "benchmarks", "serving"]
related_code:
  - docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
  - docs/plans/0113-step3p5-closure-and-sot-convergence.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/types.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/owned_binding_slot.py
  - daemon/service/controllers/representation_transform_builder.h
  - daemon/service/controllers/representation_transform_builder.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/materialization_target_plan_utils_test.cc
  - daemon/service/controllers/materialization_policy_utils.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/status_controller.h
  - core/store/materialization/contracts/representation_contract.h
  - core/store/materialization/contracts/representation_contract.cc
  - core/store/materialization/contracts/representation_contract_test.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_facade_test.cc
  - core/store/replica/collective_disk_loader.h
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/collective_disk_loader_test.cc
  - daemon/service/owned_binding_service_test.cc
  - tests/python/test_binding.py
links:
  design: ../designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
---

# Objective

Implement `0114` against the current repository as it actually exists today:

- keep `Binding.realize_from(...)` on the shared TensorCast runtime trunk,
- make the same-binding TP startup path `collective-first` by default,
- stop treating compatibility byte ranges as the primary execution graph,
- split planner truth from actual execution truth in the source-bound
  diagnostics surface,
- and use this document as the single active total execution plan for the
  remaining source-bound TP startup convergence work.

Plan ownership rule:

- `0113` remains the accepted closure design for capability and version handoff,
  single-mint identity constraints, and delete-gate invariants;
- active execution tracking is centralized here in `0114` plan;
- `docs/plans/0113-step3p5-closure-and-sot-convergence.md` remains a historical
  closure-handoff record rather than the active total checklist.

# Implementation Snapshot

As of 2026-04-01, the first landing described by this plan is implemented in
the repository:

- explicit execution-core `pad fill` now exists and is treated as local typed
  work rather than as hidden residual-pad coverage;
- source-bound mapped plan assembly no longer overwrites
  `RepresentationWorkPlan.residual_fallback_map` with the compatibility map;
- collective lane lowering now consumes a lane-local map on the mapped-target
  collective executor path instead of reading residual fallback as the primary
  request graph;
- strict `require_collective` now fails before generic execution starts on the
  binding-native path;
- additive `SourceBoundPlanDiagnostics` and actual-backend byte counters are
  now exposed through the daemon proto and Python SDK;
- `GetServerConfig.source_bound_contract_version` now advertises `3`.

Still open after this landing:

- mounted TP-startup evidence capture for at least two representative serving
  families;
- downstream integration summary updates that depend on the new planner
  diagnostics;
- deletion of transitional bridge logic only after the mounted evidence and
  migration gates pass.

## Verification Record

Local verification completed on 2026-04-01 for the landed first cut:

- `bash tools/build_proto_python.sh`
- `bazel test //core/store/materialization/contracts:representation_contract_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:materialization_target_plan_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source .venv/bin/activate && pytest tests/python/test_daemon_ctl_resolve_rpc_config.py`
- `source .venv/bin/activate && ruff check tensorcast/types.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/store/binding.py tensorcast/api/store/__init__.py tests/python/test_binding.py tests/python/test_daemon_ctl_resolve_rpc_config.py`

## Remaining Closure Gates

The remaining open work is not local unit or SDK correctness work anymore.
It is rollout and evidence work:

- capture mounted TP-serving evidence for at least two representative shared-source startup cases;
- update downstream summaries once the planner diagnostics are consumed end-to-end;
- delete transitional bridge logic only after those evidence and migration gates pass.

# Historical Baseline & Grounding

The bullets below capture the pre-landing baseline that motivated this plan.

- Source-bound collective ingress is already first-class on the daemon-owned
  binding path:
  - `proto/tensorcast/daemon/v2/store_daemon.proto`
  - `tensorcast/api/store/owned_binding_slot.py`
  - `tests/python/test_binding.py`
- The SDK already rejects `ctx.collective` for source-bound materialization and
  forwards `options.execution_topology.collective_group` plus
  `collective_policy` as explicit request fields. This plan must not reopen
  that ingress.
- `BindingRealizationPlan` lowering still emits a compatibility
  `generic_fallback_map`, and mapped-plan assembly still overwrites
  `RepresentationWorkPlan.residual_fallback_map` with that compatibility map:
  - `daemon/service/controllers/representation_transform_builder.h`
  - `daemon/service/controllers/representation_transform_builder.cc`
  - `daemon/service/controllers/materialization_target_plan_utils.cc`
- Current tests lock in the old behavior:
  - `daemon/service/controllers/materialization_target_plan_utils_test.cc`
    asserts that mixed copy plus const fill lowers to a `kData` segment plus a
    `kPad` segment and then reports `residual_fallback_map.total_bytes == 12`.
  - `core/store/replica/collective_disk_loader_test.cc`
    asserts that mapped collective skips pad-segment residual maps.
- The execution core already has partial typed-local support:
  - `RepresentationWorkPlan` already models `kConstFill` and
    `kScalarBroadcastFill` in
    `core/store/materialization/contracts/representation_contract.h`
  - `MaterializationFacade` already executes those work items locally in
    `core/store/runtime/ingestion/materialization_facade.cc`
- The missing execution-core piece is explicit `pad fill`:
  - there is no `RepresentationWorkItemKind` for pad work today,
  - the builder currently smuggles pad semantics through
    `ByteRangeSegment::Kind::kPad`.
- The current design stack still lacks one explicit layer between typed work and
  executor-private planning:
  - `0110` still describes a work IR using partly executor-shaped examples,
  - `0108` does not yet make lane allocation explicit enough,
  - `0109` still reads as though mixed execution might be finalized inside the
    collective executor rather than above it.
- This plan therefore depends on coordinated doc updates to
  `0110`, `0108`, `0109`, `0112`, and `0113` so the long-term owner boundaries
  are explicit before implementation work starts, with `0114` plan as the only
  active total checklist.
- Mapped collective still derives its execution input from
  `representation_work_plan.residual_fallback_map`:
  - `core/store/replica/collective_disk_loader.cc`
  - this is the direct reason pad bytes still block collective handling.
- Strict collective still fails too late on the binding path:
  - `daemon/service/controllers/owned_binding_service.cc` currently checks
    `collective_used` only after materialization returns.
  - there is no pre-execution "strict pure collective" eligibility gate yet.
- Diagnostics are still coarse:
  - `proto/tensorcast/daemon/v2/store_daemon.proto`
  - `daemon/service/controllers/materialization_policy_utils.cc`
  - `tensorcast/types.py`
  - current fields expose coarse collective outcome and hash facts only; they do
    not expose lane bytes, planner reject buckets, planner version, or plan
    hash.
- Current diagnostics also conflate planner intent and actual execution result:
  - the repo has no stable split today between planned lane accounting and
    actual backend accounting
  - this makes it too easy to declare success while generic backend bytes remain
    dominant on the hot path.
- at proposal time `GetServerConfig` advertised an older source-bound contract
  version and the existing typed-diagnostics capability bit:
  - `daemon/service/controllers/status_controller.h`
  - because the bit already means "typed diagnostics exist", the safer rollout
    lever for `0114` is a contract-version bump rather than redefining that bit.
- at proposal time the git state contained no new code beyond the design itself:
  - `git status --short` showed only
    `docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md`
    in staged/unstaged form.
  - This plan therefore uses the current trunk code, plus the in-progress `0114`
    design wording, as its implementation baseline.

## Recommended scope cut for the first landing

The first implementation cut should land the semantically necessary pieces
without blocking on optional collective shape expansion:

- in scope for the first cut:
  - replicated copy
  - dim0 copy
  - dim1 copy where the current collective executor already admits it
  - const fill
  - scalar fill
  - pad fill
  - additive planner/execution diagnostics and strict fail-fast policy
- explicitly not required for the first cut:
  - admitting `kConcatAssemble` into the collective lane in the same change

For `kConcatAssemble`, the first cut should keep explicit typed reject reasons
and true generic residual accounting. It must not silently erase typed identity
and relabel concat bytes as ordinary residual.

Additional first-cut rules:

- keep the current wire policy names and correct their semantics before any
  public rename cleanup;
- expose three executor-visible lanes in the first cut
  - collective
  - local typed
  - generic residual
- keep not-yet-admitted typed work as planner-owned accounting rather than
  forcing it into a fourth executor-visible contract before the runtime is ready.

## Owner-aligned implementation map

This plan executes one cross-module change set, but the long-term abstraction
owners stay split intentionally:

- `0110` owner slice: typed work inventory
  - own the shared execution-core vocabulary and invariants
  - code center:
    `core/store/materialization/contracts/representation_contract.h`
    `core/store/materialization/contracts/representation_contract.cc`
    `daemon/service/controllers/representation_transform_builder.cc`
- `0108` owner slice: lane planning
  - own the strategy-owned split between collective, local typed, deferred
    typed, and true residual lanes
  - code center:
    `core/store/runtime/ingestion/materialization_strategy_types.h`
    `core/store/runtime/ingestion/materialization_facade.cc`
- `0109` owner slice: collective-lane executor
  - own only the executor-private lowering and execution of the
    collective-admitted lane
  - code center:
    `core/store/replica/collective_disk_loader.h`
    `core/store/replica/collective_disk_loader.cc`
- `0113` owner slice: closure constraints and delete gates
  - own capability and version handoff, single-mint identity constraints, and
    final deletion gates that bound this rollout
  - code center:
    `daemon/service/controllers/status_controller.h`
    `daemon/state/lip_manager.cc`
    `docs/designs/0113-step3p5-closure-and-sot-convergence.md`
- `0114` owner slice: binding-native TP startup convergence
  - own the path-specific acceptance target for
    `Binding.realize_from(...)`, strict fail-fast behavior, and source-bound
    diagnostics surfacing, while this plan coordinates the total execution
    checklist across the owner slices above
  - code center:
    `daemon/service/controllers/owned_binding_service.cc`
    `daemon/service/controllers/materialization_policy_utils.cc`
    `proto/tensorcast/daemon/v2/store_daemon.proto`
    `tensorcast/types.py`
    `tensorcast/api/store/owned_binding_slot.py`

Normative rule:

- implementation may land in one or several commits,
- but any code change should still be explainable in terms of one of the owner
  slices above rather than by treating `0114` as the global owner of every
  touched abstraction.

# Phases & Milestones

- [x] Phase 1: Freeze the repo-local baseline and rollout contract
  - [ ] Milestone 1.0: Update the owner designs so `0110` owns typed work,
    `0108` owns lane planning, `0109` owns collective-lane execution, `0112`
    is narrowed to shipped correctness, `0113` owns closure constraints, and
    `0114` plan is the only active total checklist.
  - [x] Milestone 1.1: Keep first-class source-bound ingress unchanged and
    write `0114` against the current request contract rather than reopening
    `operation_id` side-channel work.
  - [x] Milestone 1.2: Freeze the current wire policy family and semantics:
    `allow_not_eligible_fallback` means collective-first mixed execution,
    `require_collective` means strict pure collective, and any public rename
    cleanup is deferred.
  - [x] Milestone 1.3: Decide the rollout signal:
    advance `source_bound_contract_version` to `3` for true residual semantics,
    split planner/execution diagnostics, and strict preflight; add a new
    capability bit only if downstream code truly needs an independent boolean
    gate.

- [x] Phase 2: Make residual fallback mean true residual fallback again
  - [x] Milestone 2.1: Extend the execution-core vocabulary with explicit
    `pad fill` in
    `core/store/materialization/contracts/representation_contract.h` and
    `core/store/materialization/contracts/representation_contract.cc`.
  - [x] Milestone 2.2: Update
    `daemon/service/controllers/representation_transform_builder.cc` so typed
    copy/fill coverage stays typed and does not use `generic_fallback_map` as
    the primary execution artifact on the binding-realization path.
  - [x] Milestone 2.3: Update
    `daemon/service/controllers/materialization_target_plan_utils.cc` so
    `RepresentationWorkPlan.residual_fallback_map` is populated only with true
    residual bytes.
  - [x] Milestone 2.4: Add invariant checks that forbid duplicate coverage
    between collective/local typed work and residual byte-range fallback.

- [x] Phase 3: Derive explicit lanes from typed work
  - [x] Milestone 3.1: Extend
    `core/store/runtime/ingestion/materialization_strategy_types.h` with lane
    inputs or summaries sufficient to represent three executor-visible lanes
    plus planner-owned accounting for not-yet-admitted typed work.
  - [x] Milestone 3.2: Update
    `core/store/runtime/ingestion/materialization_facade.cc` so collective
    eligibility is computed from typed work items and topology facts rather
    than from the historical shape of `residual_fallback_map`.
  - [x] Milestone 3.3: Reuse the existing local fill execution path for
    `kConstFill` and `kScalarBroadcastFill`, and extend it to explicit
    `pad fill` without creating a second write substrate.
  - [x] Milestone 3.4: Keep concat explicit even when not admitted:
    reject it with typed planner reasons and keep it in planner-owned typed
    accounting rather than silently widening it into true residual.

- [x] Phase 4: Make strict pure-collective fail fast
  - [x] Milestone 4.1: Introduce a pre-execution strict-eligibility check on
    the source-bound binding path in
    `daemon/service/controllers/owned_binding_service.cc`.
  - [x] Milestone 4.2: Ensure `require_collective` rejects before the
    expensive generic mapped path begins.
  - [x] Milestone 4.3: Preserve current mixed-mode success semantics for the
    default `allow_not_eligible_fallback` policy.

- [x] Phase 5: Converge the mapped collective executor on lane-local inputs
  - [x] Milestone 5.1: Refactor
    `core/store/replica/collective_disk_loader.h` and
    `core/store/replica/collective_disk_loader.cc` so the collective executor
    consumes collective-lane lowering rather than reading
    `residual_fallback_map` as the primary request graph.
  - [x] Milestone 5.2: Remove pad-segment-based whole-request blocking once pad
    work has become a local typed lane.
  - [x] Milestone 5.3: Keep any executor-private byte-range lowering internal
    to the collective lane; do not promote it back to shared semantic truth.

- [x] Phase 6: Split planner and execution diagnostics without breaking callers
  - [x] Milestone 6.1: Add a source-bound-scoped planner diagnostics contract on
    `proto/tensorcast/daemon/v2/store_daemon.proto` for:
    planned lane byte counters, reject-reason buckets, planner version,
    `plan_hash`, and material cost estimates that affect executor choice.
  - [x] Milestone 6.2: Update
    `daemon/service/controllers/materialization_policy_utils.cc` and
    `daemon/service/controllers/owned_binding_service.cc` to populate the new
    planner and execution fields from the actual strategy result instead of
    log-only counters.
  - [x] Milestone 6.2a: Keep shared `ExecutionDiagnostics` focused on actual
    execution facts:
    actual backend bytes, dominant executor, direct-write support, and existing
    hash or identity outcomes.
  - [x] Milestone 6.3: Update `tensorcast/types.py`,
    `tensorcast/api/store/owned_binding_slot.py`, and
    `tests/python/test_binding.py` so SDK-visible diagnostics stay additive and
    typed.
  - [x] Milestone 6.4: Bump `source_bound_contract_version` in
    `daemon/service/controllers/status_controller.h` and keep the current
    capability bits truthful.

- [ ] Phase 7: Closure, capability handoff, evidence, and deletion
  - [x] Milestone 7.1: Replace unit tests that currently assert pad-segment
    residual behavior with tests that assert explicit lane assignment and true
    residual semantics.
  - [x] Milestone 7.2: Preserve the `0113` single-mint and capability-handoff
    guarantees while advancing the source-bound readiness surface to version `3`.
  - [ ] Milestone 7.3: Re-run representative mounted TP startup evidence
    against Step3p5 and another TP-serving family such as Mixtral, and capture
    planned and actual byte breakdown plus dominant executor.
  - [ ] Milestone 7.4: Update downstream integration summaries only after the
    new typed diagnostics are actually available through the public SDK.
  - [ ] Milestone 7.5: Delete transitional bridge fields, residual-map-based
    collective gating, stale active references to `0113` plan, and old policy
    aliases only after the mounted evidence and downstream migration gates pass.

# Tasks

- [x] `0110`-owned execution-core contract work
  - [x] Add explicit `pad fill` to
    `core/store/materialization/contracts/representation_contract.h`.
  - [x] Teach
    `core/store/materialization/contracts/representation_contract.cc` to build,
    validate, and serialize the new work item without changing the shared
    semantic trunk or widening `representation_contract_hash` into executor-
    shaped padding artifacts.
  - [x] Add coverage validation that proves typed work and residual fallback do
    not overlap.
  - [x] Document and implement the coverage-closure rule that derives `pad fill`
    during work-plan construction and keeps it inside `plan_hash` rather than
    semantic contract hash.

- [x] Controller lowering work
  - [x] Refactor
    `daemon/service/controllers/representation_transform_builder.h` and
    `daemon/service/controllers/representation_transform_builder.cc` so
    `BuildRepresentationTransformResult` exposes the semantic distinction
    between typed work and true residual.
  - [x] Keep `generic_fallback_map` only as a transitional bridge if it is
    still needed for a short landing sequence.
  - [x] Update
    `daemon/service/controllers/materialization_target_plan_utils_test.cc` so
    it stops asserting that fill-covered bytes appear as residual pad segments.

- [x] `0108`-owned strategy and lane-planning work
  - [x] Extend
    `core/store/runtime/ingestion/materialization_strategy_types.h` with lane
    summaries or structured planner output.
  - [x] Update
    `core/store/runtime/ingestion/materialization_facade.cc` to:
    - compute collective candidate bytes from typed work,
    - execute local typed pad/fill work explicitly,
    - preserve planner-owned typed accounting for typed-but-not-yet-admitted
      work,
    - keep generic residual narrow and truthful,
    - split planner facts from actual execution facts,
    - bind the source-bound path to the unified `engine.materialization_strategy`
      policy family,
    - and reject strict pure-collective requests before generic execution.
  - [x] Keep existing local fill code paths and avoid introducing a serving-only
    write substrate.

- [x] `0109`-owned collective-lane executor work
  - [x] Refactor
    `core/store/replica/collective_disk_loader.cc` so collective planning is
    fed from collective-lane lowering rather than residual-map shape.
  - [x] Replace the current `pad_segments_not_supported` test expectation in
    `core/store/replica/collective_disk_loader_test.cc` with lane-aware
    expectations.
  - [ ] Emit executor evidence that captures actual collective bytes, unique
    source bytes, peer-transfer bytes, peak temporary bytes, and batch count for
    mounted validation.

- [x] `0114`-owned source-bound diagnostics and SDK work
  - [x] Expand
    `proto/tensorcast/daemon/v2/store_daemon.proto` and regenerate Python stubs
    with `bash tools/build_proto_python.sh`.
  - [x] Add a source-bound planner diagnostics child message or equivalent scoped
    contract instead of turning shared `ExecutionDiagnostics` into a permanent
    catch-all.
  - [x] Update `tensorcast/types.py` and
    `tensorcast/api/store/owned_binding_slot.py` so both planner and actual
    execution diagnostics are preserved and exposed.
  - [x] Keep old diagnostics fields during rollout; remove them only if a later
    design explicitly says they are obsolete.

- [x] `0114`-owned daemon binding-path convergence work
  - [x] Add strict preflight rejection in
    `daemon/service/controllers/owned_binding_service.cc`.
  - [x] Keep `Binding.realize_from(...)` as the only public ingress; do not add
    a new serving-specific API.
  - [x] Keep the current wire policy names in the first landing and correct
    their semantics before any public rename cleanup.

- [x] `0113`-owned closure and handoff work, tracked here as part of the total
  plan
  - [x] Advance `source_bound_contract_version` to `3` and document its meaning
    as true residual semantics plus split planner and execution diagnostics plus
    strict preflight.
  - [x] Keep single-mint identity guarantees and delete gates aligned with
    `docs/designs/0113-step3p5-closure-and-sot-convergence.md`.
  - [ ] Retire `docs/plans/0113-step3p5-closure-and-sot-convergence.md` as an
    active reference once the updated docs land.

- [ ] Cross-cutting docs and evidence work
  - [x] Update `docs/README.md` to index the `0114` design and plan.
  - [ ] Update any downstream or internal evidence notes once new planner
    fields and mounted traces exist.

# Test / Rollout / Backout

## Acceptance checks

- [x] `bazel test //core/store/materialization/contracts:representation_contract_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //daemon:materialization_target_plan_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [ ] mounted TP-serving evidence for at least two representative shared-source
  startup cases

## Rollout order

1. Land additive execution-core and diagnostics fields first.
2. Switch the binding-realization path to true residual semantics and
   lane-derived planning.
3. Turn on strict preflight rejection for `require_collective`.
4. Update downstream integration summaries once contract version `3` is
   available.
5. Delete transitional bridge logic and old policy aliases only after the new
   mounted evidence is captured.

## Backout

- If a regression is found before the delete gates pass, back out by reverting
  the `0114` implementation commits rather than leaving a new permanent runtime
  toggle behind.
- A short-lived bring-up guard is acceptable during development, but it must be
  deleted before this plan is closed.

# Risks & Tracking

- [ ] Risk: ordinary disk materialization and source-bound mapped materialization
  diverge further.
  Mitigation: keep `RepresentationWorkPlan` as the shared execution-core truth
  and limit `0114`-specific changes to lane derivation plus diagnostics.
- [ ] Risk: planner bytes and actual execution bytes remain conflated, leading to
  false confidence that generic execution is no longer dominant.
  Mitigation: split planner and execution diagnostics and require mounted
  evidence to report both planned and actual byte breakdowns.
- [ ] Risk: proto and SDK diagnostics churn breaks downstream callers.
  Mitigation: keep all new fields additive, prefer version bump over redefining
  existing capability bits, and keep old fields during the migration window.
- [ ] Risk: source-bound binding startup grows path-private strategy semantics.
  Mitigation: bind the path explicitly to the unified
  `engine.materialization_strategy` config family and forbid new environment-
  only toggles.
- [ ] Risk: concat admission expands scope and delays the fix for the real TP
  blocker.
  Mitigation: do not block the first cut on concat collective support; keep it
  explicit in reject buckets and residual accounting.
- [ ] Risk: "collective-first" still ends up meaning "generic dominant with
  better logs".
  Mitigation: require mounted evidence to show lane bytes and a dominant
  executor shift away from generic-only behavior before closing the plan.
