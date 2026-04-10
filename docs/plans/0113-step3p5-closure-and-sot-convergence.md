---
slug: step3p5-closure-and-sot-convergence
title: Step3p5 Closure and Single-SOT Convergence Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "benchmarks", "serving"]
related_code:
  - docs/designs/0113-step3p5-closure-and-sot-convergence.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/_config.py
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.cc
  - daemon/service/controllers/materialization_policy_utils.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/state/lip_manager.cc
  - /data/workspace/internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py
  - /data/workspace/internal-vllm/docs/design/tensorcast_step3p5_from_disk_cold_start_performance_followup.md
links:
  design: ../designs/0113-step3p5-closure-and-sot-convergence.md
---

# Objective

Finish the remaining Step3p5-facing closure work after the architectural cuts
from `0107` through `0112`, while preserving a clean handoff into the new total
plan arrangement.

Historical note:

- the completed `0114` and `0115` execution records have now been folded back
  into their implemented designs and their companion plans have been deleted;
- active rollout and cleanup tracking for the remaining work now lives in
  `docs/plans/0117-post-0114-mounted-rollout-and-delete-gate-cleanup.md`;
- this file remains as a closure-handoff record for already-landed work and for
  the rationale behind the still-active `0113` closure design constraints.

This plan owns:

- source-bound collective contract cutover,
- typed collective policy and typed failure-class contract,
- typed execution, hash, and identity diagnostics needed by downstream
  integrations,
- stable capability and version handoff so downstream code can switch without
  repo-local guesswork,
- single-mint closeout and second-stage hash removal,
- source-bound executor convergence,
- helper demotion, evidence recapture, and legacy deletion,
- and the documentation cleanup needed to keep those items in one place.

Execution policy for this plan:

- because the project has not launched yet, final-state cleanup is part of the
  deliverable rather than optional post-rollout polish;
- compatibility, temporary, and redundant code may be used only as narrowly
  scoped bring-up aids and must be deleted before `0113` is closed;
- when forced to choose, prefer the final contract and simpler end-state code
  shape over preserving broad repository-local compatibility.

# Current State & Grounding

- `0107` separation between retrieval policy and execution topology is already
  landed in daemon-side request normalization:
  - `daemon/service/controllers/materialization_policy_utils.h`
  - `daemon/service/controllers/materialization_policy_utils.cc`
- `0108` common-runtime strategy seams are already landed:
  - `core/store/runtime/ingestion/materialization_strategy_types.h`
  - `core/store/runtime/ingestion/materialization_facade.cc`
- `0109` owner-file collective phase-1 implementation is landed, but rollout
  evidence, mixed-residual policy, and prototype deletion remain open:
  - `core/store/replica/collective_disk_loader.cc`
- `0111` repo-owned builder/publication bridge is landed at base scope:
  - `tensorcast/api/store/serving_builder.py`
  - `daemon/service/controllers/assembly_operation_service.cc`
- `ServingAdmissionFacts.fast_path_validated` is already a correctness and
  admission gate for same-binding publication:
  - `tensorcast/types.py`
  - `docs/designs/0111-source-to-serving-builder-and-representation-publication.md`
- `0112` correctness path is landed:
  - public disk ingress
  - binding-native publication subject
  - fail-closed `canonical_full`
- the remaining blockers are not correctness blockers:
  - source-bound ingress still relies on `operation_id` side-channel metadata
    for collective lowering
  - closeout still performs a second-stage full-data hash through
    `LipManager::commit_lease_in_place(...)`
  - the mounted Step3p5 source-bound path can still show
    `dominant_executor=GenericByteRangeExecutor(source_ordered)`
- the current source-bound collective ingress is code-real but not
  contract-real:
  - `tensorcast/api/store/binding.py` still encodes collective context into
    `operation_id`
  - `tensorcast/api/store/owned_binding_slot.py` lowers `source_policy` from
    `GetArtifactOptions`, but does not yet forward
    `execution_topology.collective_group` as first-class source-bound request
    fields
  - `proto/tensorcast/daemon/v2/store_daemon.proto` already has
    `collective_load_group` on `MaterializeReplicaRequest`, but
    `RefillOwnedBindingRequest` still lacks an equivalent first-class contract
- current tests still prove the compatibility bridge rather than the future
  source-bound contract:
  - `tests/python/test_binding.py`
  - `tests/python/api/test_mapped_binding.py`
- `internal-vllm` has already moved further than the old TensorCast follow-up
  note claimed:
  - it now builds collective `CallContext` and execution-topology options in
    `vllm/model_executor/model_loader/tensorcast_loader.py`
  - but TensorCast still lacks a first-class source-bound collective contract,
    so the helper ultimately feeds a compatibility bridge.
- `internal-vllm` currently exposes bootstrap admission facts, but not stable
  upstream execution-quality facts:
  - `vllm/model_executor/model_loader/tensorcast_loader.py`
  - `/data/workspace/internal-vllm/docs/design/tensorcast_step3p5_from_disk_cold_start_performance_followup.md`
- Documentation SOT before this change was fragmented across the deleted
  companion plans:
  - `0108-tensor-aware-materialization-strategy-plane.md`
  - `0108-01-pre-109-strategy-plane-convergence.md`
  - `0109-batched-owner-file-collective-executor.md`
  - `0110-artifact-representation-contract-and-transform-unification.md`
  - `0111-source-to-serving-builder-and-representation-publication.md`
  - `0112-binding-native-serving-realization-and-publication-plan.md`
  - `0112-step3p5-binding-native-serving-bootstrap-performance-followup.md`

Current SOT rule for this work:

- architecture and invariants stay in `0107` through `0112` plus
  `0113` design;
- active execution tracking now stays in `0114` plan;
- this `0113` plan is retained as historical context until the updated design
  set is fully folded back and no active references remain.

# Phases & Milestones

- [x] Phase 1: Collapse Documentation SOT
  - [x] Milestone 1.1: Add `0113` design and plan as the only active closure
    record for residual `0107`-`0112` work.
  - [x] Milestone 1.2: Fold landed execution history back into the owning
    designs and delete the superseded companion plans.
  - [x] Milestone 1.3: Correct stale `internal-vllm` current-state wording so
    it matches the code that already passes collective context.

- [x] Phase 2: Source-Bound Contract Cutover
  - [x] Milestone 2.1: Freeze first-class source-bound topology and collective
    ingress on `RefillOwnedBindingRequest`.
  - [x] Milestone 2.2: SDK and daemon plumb
    `GetArtifactOptions.execution_topology` end-to-end instead of treating
    `operation_id` as the primary contract.
  - [x] Milestone 2.3: Add typed collective policy with explicit strict versus
    fallback semantics and fail-closed conflict handling.
  - [x] Milestone 2.4: Expose a stable capability or version surface that marks
    the first-class source-bound contract as ready for downstream use.

- [x] Phase 3: Diagnostics And Admission Semantics
  - [x] Milestone 3.1: Freeze `fast_path_validated` as a correctness and
    admission fact only, not a performance-quality claim.
  - [x] Milestone 3.2: Expose stable typed execution, hash, and identity facts
    for downstream consumers.
  - [x] Milestone 3.3: Remove any need for downstream code to parse daemon logs
    or `operation_id` payloads to understand source-bound execution quality.

- [ ] Phase 4: Identity And Hash Cutover
  - [x] Milestone 4.1: Binding-subject closeout reaches the chosen seal-mint
    reuse model and therefore has single-mint effect.
  - [x] Milestone 4.2: Second-stage full-data hash is removed from the steady
    same-binding path.
  - [ ] Milestone 4.3: Any remaining hash work is explicitly observable with
    location, round, bytes, wall time, and non-identity-forming semantics.

- [ ] Phase 5: Source-Bound Executor Convergence
  - [x] Milestone 5.1: Source-bound mapped realization can first-class attempt
    collective execution through the converged runtime seam.
  - [x] Milestone 5.2: If collective does not apply, the path falls back to a
    typed non-generic local executor shape instead of remaining indefinitely on
    generic dominant execution.
  - [ ] Milestone 5.3: `0109` residual policy and prototype deletion gates are
    decided against explicit mounted and benchmark evidence.

- [ ] Phase 6: Downstream Handoff, Evidence, And Deletion
  - [x] Milestone 6.1: `internal-vllm` can detect TensorCast readiness through a
    stable capability or version surface and switch to the first-class contract.
  - [ ] Milestone 6.2: mounted TP=8 serving and benchmark evidence is recaptured
    against the new contract, diagnostics surface, and executor path.
  - [ ] Milestone 6.3: compatibility-only collective side-channeling,
    second-stage hashing, and legacy prototype scaffolding are deleted once
    evidence gates pass.

# Tasks

- [x] Documentation consolidation
  - [x] Create `0113` design and plan.
  - [x] Update `0107`-`0112` designs so they point to `0113` for residual
    execution tracking.
  - [x] Delete the superseded `0108`-`0112` companion plans.
  - [x] Update `docs/README.md` to list the new closure pair.
  - [x] Update `internal-vllm` follow-up docs to remove stale references to the
    deleted Step3p5 plan and to current-state-correct collective context notes.

- [x] TensorCast contract work
  - [x] Extend `proto/tensorcast/daemon/v2/store_daemon.proto` source-bound
    request messages with first-class collective/topology fields and typed
    collective policy.
  - [x] Plumb `GetArtifactOptions.execution_topology` through daemon-owned
    binding helpers instead of dropping it at `OwnedBindingSlot.realize_from`.
  - [x] Reuse the normalized request-context lowering from
    `materialization_policy_utils.cc` instead of maintaining two parallel
    collective interpreters.
  - [x] Keep `operation_id` scoped to transport and tracing only rather than
    source-bound collective/topology semantics.
  - [x] Add fail-closed behavior for requests that supply contradictory
    first-class and compatibility-lowered collective facts.
  - [x] Delete compatibility-lowering code in the same closure sequence once the
    first-class source-bound contract and readiness surface are proven.

- [ ] Diagnostics and capability work
  - [x] Extend the source-bound execution report surface with stable typed facts
    for collective, executor, hash, and identity outcomes.
  - [x] Freeze `fast_path_validated` as correctness-only across `0111`, `0113`,
    and downstream-facing docs.
  - [x] Expose a stable capability or version surface so downstream
    integrations can detect when the first-class contract and diagnostics are
    ready.
  - [ ] Demote daemon log strings and `operation_id` encodings to debug-only
    evidence once the typed surfaces exist.
  - [ ] Remove duplicate or temporary diagnostics paths that exist only to keep
    old compatibility surfaces alive.

- [ ] TensorCast identity work
  - [x] Add typed identity reuse from seal into binding-subject closeout and
    promotion.
  - [x] Remove second-stage full-data hash from
    `daemon/state/lip_manager.cc` on the steady same-binding path.
  - [ ] Make hash observability explicit where hashing still exists, including
    round count, location, bytes, and wall time.
  - [x] Ensure any residual hash work does not remint steady-path identity.
  - [ ] Delete transitional duplicate identity or closeout helpers instead of
    preserving them as a second maintained path.

- [ ] Runtime execution work
  - [x] Make source-bound mapped realization prefer first-class collective when
    eligible and requested.
  - [x] Implement the typed non-generic local fallback shape if collective does
    not apply.
  - [x] Keep `ExecutionCommitReport` or its replacement diagnostics surface
    visible for both collective and non-collective outcomes.
  - [ ] Delete generic-dominant or prototype-only execution branches that are no
    longer part of the intended steady path once the new executor shape is
    proven.

- [ ] Integration and validation work
  - [x] Switch `internal-vllm` from bridge-style source-bound collective
    lowering to the first-class request contract once the capability surface is
    ready.
  - [x] Execute compatibility retirement in repo order:
    `internal-vllm` caller migration first, then TensorCast compatibility
    lowering deletion, then residual `internal-vllm` compat fallback cleanup.
  - [x] Expand bootstrap summary and profile fields only after TensorCast
    exposes stable typed facts and separates them from
    `fast_path_validated`.
  - [ ] Re-run mounted 8xH800 cold-start, TP=8 serving correctness, and the
    benchmark matrix before deleting compatibility scaffolding.
  - [x] Retire compatibility-only tests that assert `operation_id`-encoded
    collective semantics once the first-class request-field tests cover the
    supported path.
  - [x] Do not leave repo-local "old path" toggles, fallback branches, or
    adapter workarounds behind for compatibility once the final path is proven.

# Test / Rollout / Backout

## Acceptance checks

- [x] One active closure plan now exists for residual `0107`-`0112` work.
- [x] The superseded companion plans have been removed from the repo.
- [x] Stale `internal-vllm` wording about missing collective context has been
  corrected.
- [x] `RefillOwnedBinding` no longer relies on `operation_id` side-channeling
  as its primary collective contract.
- [x] a stable capability or version surface exists for source-bound first-class
  collective ingress and diagnostics.
- [x] `fast_path_validated` is documented and enforced as correctness-only.
- [x] downstream-consumable typed execution, hash, and identity diagnostics are
  available without log parsing.
- [x] same-binding closeout no longer pays a second-stage full-data hash.
- [x] same-binding closeout no longer remints content identity from the same
  immutable bytes.
- [x] mounted Step3p5 no longer remains indefinitely on
  `GenericByteRangeExecutor(source_ordered)` as dominant executor.
- [ ] final deletion of compatibility/prototype scaffolding is backed by one
  committed evidence package.
- [x] `0113` does not leave behind broad repository-local compatibility code,
  duplicate helper paths, or temporary toggles that are not part of the final
  intended architecture.

## Test plan

- [x] Link hygiene and stale-reference sweep:
  - `rg -n "0108-01-pre-109-strategy-plane-convergence.md|0108-tensor-aware-materialization-strategy-plane.md|0109-batched-owner-file-collective-executor.md|0110-artifact-representation-contract-and-transform-unification.md|0111-source-to-serving-builder-and-representation-publication.md|0112-binding-native-serving-realization-and-publication-plan.md|0112-step3p5-binding-native-serving-bootstrap-performance-followup.md" docs /data/workspace/internal-vllm/docs -S`
- [ ] Once contract code work starts:
  - [x] `bash tools/build_proto_python.sh`
  - [x] `source .venv/bin/activate && pytest tests/python/test_binding.py tests/python/test_daemon_ctl_resolve_rpc_config.py tests/python/api/test_config_models.py tests/python/api/test_retrieval_options.py`
  - [x] `bazel test //daemon:owned_binding_service_test //daemon:grpc_service_impl_startup_gate_test //daemon:grpc_service_impl_start_seal_assembly_test --test_output=errors`
  - [x] `bazel test //daemon:owned_binding_service_test //daemon:grpc_service_impl_startup_gate_test //daemon:materialize_into_mapped_target_test --test_output=errors`
  - [x] `rg -n "#tcg:|clid=|clws=|clrk=" tests/python /data/workspace/internal-vllm/tests tensorcast/api/store -S`
    only leaves non-source-bound transport/mapped-path coverage plus negative
    source-bound assertions that `operation_id` no longer carries collective
    topology.
  - mounted TP=8 `internal-vllm` cold-start validation after integration cutover

## Latest implementation status

- 2026-03-31:
  - landed the additive first-class source-bound contract on
    `RefillOwnedBindingRequest` / `RefillOwnedBindingResponse`, including
    `execution_topology`, `collective_policy`, typed execution diagnostics, and
    `GetServerConfig` readiness flags/version;
  - SDK `OwnedBindingSlot.swap(...)` / `realize_from(...)` now lower
    `GetArtifactOptions.execution_topology` end-to-end and reject
    `ctx.collective` on daemon-owned source-bound paths instead of treating it
    as a compatibility input;
  - daemon-side request normalization now consumes only first-class
    `execution_topology` for source-bound collective/topology semantics, so
    `operation_id` no longer acts as a hidden source-bound topology carrier;
  - typed execution/hash/identity diagnostics now surface on source-bound refill
    responses, binding promotion responses, and published model-version decode
    paths;
  - same-binding seal / local-only ready now precompute the serving-artifact
    identity once and later promote / binding-subject closeout reuse that
    identity rather than hashing identical bytes a second time;
  - identity diagnostics now distinguish `seal_mint`, `seal_reuse`,
    `closeout_mint`, and `not_applicable`, so existing-artifact publication no
    longer masquerades as seal reuse and local-only ready no longer hides its
    seal hash work;
  - strict collective failures now carry a stable
    `tc.collective_failure_class=...` marker through RPC errors so the SDK can
    surface typed `not_eligible` versus `execution_failed` failures without log
    parsing;
  - `GetServerConfig.source_bound_contract_version` now advances to `3`, which
    marks the additive `0114` landing for true residual semantics, strict
    preflight, and split planner/execution diagnostics;
  - `GetServerConfig` now advertises
    `SOURCE_BOUND_CAPABILITY_FLAG_SINGLE_MINT_BINDING_CLOSEOUT`, and the
    source-bound non-collective fallback path now reports typed executor names
    (`SourceOrderedMappedTargetExecutor`,
    `MappedTargetStreamingExecutor`,
    `SourceOrderedDirectTargetExecutor`,
    `DirectTargetStreamingExecutor`,
    `MappedLoaderTargetExecutor`) instead of surfacing as
    `GenericByteRangeExecutor...`;
  - repo-local source-bound compatibility lowering is now deleted on the
    TensorCast side; the remaining `operation_id` transport metadata surface is
    scoped to non-source-bound transport/tracing paths outside `0113`;
  - the active downstream runtime surface has advanced under `0114` to
    `source_bound_contract_version >= 4`, preserving the historical additive
    collective-first planner/execution split semantics while hard-cutting same-binding
    `realize_from(...)` to execution-only `BindingUpdateEpoch` behavior;
  - source-bound daemon tests now verify that `operation_id` collective
    metadata is ignored rather than treated as a compatibility collective
    ingress;
  - targeted Python revalidation passed:
    `pytest tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/test_daemon_ctl_resolve_rpc_config.py tests/python/api/test_config_models.py tests/python/api/test_retrieval_options.py`;
  - targeted daemon Bazel revalidation passed for
    `//daemon:owned_binding_service_test`,
    `//daemon:grpc_service_impl_startup_gate_test`, and
    `//daemon:materialize_into_mapped_target_test`;
  - mounted evidence-package capture remains the primary open closure item.

## Evidence package

- [ ] One committed evidence package must include:
  - mounted run identifiers and linked `status.json`
  - TensorCast typed execution diagnostics for collective, executor, hash, and
    identity outcomes
  - `internal-vllm` additive summary or profile output derived from those typed
    facts
  - baseline comparison against the default loader on the same host class
  - explicit delete-gate conclusion for compatibility bridge retirement
  - explicit confirmation that temporary, redundant, and compatibility-only code
    introduced during cutover was removed

## Rollout

- [x] Documentation rollout is complete in this change: `0113` is now the sole
  active closure plan.
- [x] Contract rollout should first land as an additive source-bound API with a
  minimal, short-lived cutover aid only if technically required, plus an
  explicit readiness surface.
- [x] Compatibility retirement order is normative:
  1. `internal-vllm` must first stop depending on compatibility inputs as its
     primary caller contract and switch to
     `GetArtifactOptions.execution_topology` plus stable
     capability/version gating.
  2. TensorCast may then delete repo-local compatibility lowering
     (`ctx.collective` / `operation_id` merge paths) once mounted evidence shows
     the caller no longer depends on them.
  3. Only after that should `internal-vllm` delete any residual compat fallback
     branches, summary shims, or tests that exist solely for the old path.
  Deleting TensorCast-side lowering before caller migration is not an intended
  rollout path unless both repos are updated atomically and no other caller
  depends on the compatibility surface.
- [ ] Identity and hash rollout should remove cost only after the first-class
  contract and diagnostics surface are proven on the mounted case.
- [ ] Prototype, compatibility, temporary, and redundant code deletion should
  happen as part of closure, not as a later cleanup backlog item, once the new
  path is evidenced and linked back into the owning designs.

## Backout

- [x] Documentation backout is a normal revert of the `0113` consolidation
  change if the single-SOT structure proves insufficient.
- [ ] Contract backout must not restore side-channel ingress as the long-term
  preferred API; if implementation regresses, narrow the first-class path
  instead.
- [ ] Diagnostics backout must not leave downstream integrations dependent on
  daemon log strings or encoded `operation_id` payloads.
- [ ] Identity backout must not restore duplicate hashing as a hidden permanent
  cost.
- [ ] Backout must not be used as justification to keep broad compatibility
  layers permanently maintained in the final design.

# Risks & Tracking

- [ ] Risk: the new closure plan becomes a stealth redesign of `0107`-`0112`.
  - Mitigation: keep architecture in the designs and use `0113` only for
    residual execution and deletion gates.
- [ ] Risk: deleting old plans hides useful implementation history.
  - Mitigation: keep landed outcomes in the owning designs and list deleted
    plans explicitly in this plan's grounding.
- [ ] Risk: `internal-vllm` keeps growing around the side-channel bridge while
  TensorCast delays first-class ingress.
  - Mitigation: treat first-class ingress as Phase 2 critical path and do not
    add new Python data-plane workaround logic.
- [ ] Risk: downstream integrations continue to treat `fast_path_validated` as a
  performance guarantee.
  - Mitigation: freeze correctness-only semantics in `0111`, `0113`, and
    downstream docs before code rollout.
- [ ] Risk: downstream code couples to daemon log strings because typed
  diagnostics arrive too late or too vaguely.
  - Mitigation: treat stable diagnostics and capability surfaces as first-class
    deliverables rather than optional polish.
- [ ] Risk: compatibility or temporary code survives because it "works" and no
  launch pressure forces deletion.
  - Mitigation: make final-state cleanup an explicit acceptance criterion and
    evidence-package requirement.
- [ ] Risk: `0109` prototype collective paths survive indefinitely because
  evidence and delete gates are not explicit.
  - Mitigation: keep their final deletion as an explicit Phase 6 milestone.
