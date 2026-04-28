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
  - /data/workspace/internal-vllm/docs/tensorcast/tensorcast_step3p5_from_disk_cold_start_performance_followup.md
links:
  design: ../designs/0113-step3p5-closure-and-sot-convergence.md
---

# Objective

Finish the remaining Step3p5-facing closure work after the architectural cuts
from `0107` through `0112`, while preserving a clean handoff into the new total
plan arrangement.

Historical note:

- the later standalone `0114` execution checklist has now been retired and its
  surviving closeout record lives in
  `docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md`
  plus
  `docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md`;
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
- `ServingAdmissionFacts.same_binding_fast_path_validated` is already a correctness and
  admission gate for same-binding publication:
  - `tensorcast/types.py`
  - `docs/designs/0111-source-to-serving-builder-and-representation-publication.md`
- `0112` correctness path is landed:
  - public disk ingress
  - binding-native publication subject
  - fail-closed `canonical_full`
- the first-class source-bound collective/topology contract is now landed on
  `RefillOwnedBindingRequest`, and daemon-owned source-bound callers are
  required to use `GetArtifactOptions.execution_topology` rather than
  `ctx.collective` or `operation_id` side-channel lowering.
- the live downstream readiness surface is now
  `source_bound_contract_version=4` with
  `source_bound_contract_path=collective_first_v4`.
- representative mounted closure evidence already exists for the surviving
  `0109` / `0112` / `0114` owner boundaries in:
  - `docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md`
  - `docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md`
- `internal-vllm` now exposes stable bootstrap summary fields for source-bound
  contract/version gating plus typed execution/hash/identity diagnostics:
  - `vllm/model_executor/model_loader/tensorcast_loader.py`
  - `/data/workspace/internal-vllm/docs/tensorcast/tensorcast_step3p5_from_disk_cold_start_performance_followup.md`
- the remaining closure blockers before the final 2026-04-28 TP8 run were:
  - broader Step3p5 mounted `TP=8` / `8xH800` evidence and performance signoff
    versus default and `fastsafetensors` loaders,
  - `0109` residual mixed-residual policy and prototype-delete gate decisions
    backed by explicit evidence,
  - and final retirement of legacy diagnostics fallbacks that remain only for
    backward-compatible error surfacing.
  The 2026-04-28 tensor-aware mapped executor evidence below closes the
  same-host TP8 performance blocker for this packet.
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
- the surviving mounted closeout record now stays in the `0114` design plus the
  `2026-04-15` mounted benchmark note;
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
  - [x] Milestone 3.1: Freeze `same_binding_fast_path_validated` as a correctness and
    admission fact only, not a performance-quality claim.
  - [x] Milestone 3.2: Expose stable typed execution, hash, and identity facts
    for downstream consumers.
  - [x] Milestone 3.3: Remove any need for downstream code to parse daemon logs
    or `operation_id` payloads to understand source-bound execution quality.

- [x] Phase 4: Identity And Hash Cutover
  - [x] Milestone 4.1: Binding-subject closeout reaches the chosen seal-mint
    reuse model and therefore has single-mint effect.
  - [x] Milestone 4.2: Second-stage full-data hash is removed from the steady
    same-binding path.
  - [x] Milestone 4.3: Any remaining hash work is explicitly observable with
    location, round, bytes, wall time, and non-identity-forming semantics.

- [x] Phase 5: Source-Bound Executor Convergence
  - [x] Milestone 5.1: Source-bound mapped realization can first-class attempt
    collective execution through the converged runtime seam.
  - [x] Milestone 5.2: If collective does not apply, the path falls back to a
    typed non-generic local executor shape instead of remaining indefinitely on
    generic dominant execution.
  - [x] Milestone 5.3: `0109` residual policy and prototype deletion gates are
    decided against explicit mounted and benchmark evidence.

- [ ] Phase 6: Downstream Handoff, Evidence, And Deletion
  - [x] Milestone 6.1: `internal-vllm` can detect TensorCast readiness through a
    stable capability or version surface and switch to the first-class contract.
  - [x] Milestone 6.2: mounted TP=8 serving and benchmark evidence is recaptured
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

- [x] Diagnostics and capability work
  - [x] Extend the source-bound execution report surface with stable typed facts
    for collective, executor, hash, and identity outcomes.
  - [x] Freeze `same_binding_fast_path_validated` as correctness-only across `0111`, `0113`,
    and downstream-facing docs.
  - [x] Expose a stable capability or version surface so downstream
    integrations can detect when the first-class contract and diagnostics are
    ready.
  - [x] Demote daemon log strings and `operation_id` encodings to debug-only
    evidence once the typed surfaces exist.
  - [x] Remove duplicate or temporary diagnostics paths that exist only to keep
    old compatibility surfaces alive.

- [ ] TensorCast identity work
  - [x] Add typed identity reuse from seal into binding-subject closeout and
    promotion.
  - [x] Remove second-stage full-data hash from
    `daemon/state/lip_manager.cc` on the steady same-binding path.
  - [x] Make hash observability explicit where hashing still exists, including
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
    `same_binding_fast_path_validated`.
  - [x] Re-run mounted 8xH800 cold-start, TP=8 serving correctness, and the
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
- [x] `same_binding_fast_path_validated` is documented and enforced as correctness-only.
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

- 2026-04-27:
  - the first-class source-bound contract is live on
    `RefillOwnedBindingRequest` / `RefillOwnedBindingResponse`, including
    `execution_topology`, `collective_policy`, typed `ExecutionDiagnostics`,
    and typed `SourceBoundPlanDiagnostics`;
  - daemon-owned source-bound callers now require
    `GetArtifactOptions.execution_topology`, reject `ctx.collective`, and no
    longer use `operation_id` as a source-bound topology carrier;
  - `GetServerConfig.source_bound_contract_version` is now `4`, with
    `source_bound_contract_path=collective_first_v4` and capability bits for
    first-class ingress, typed execution diagnostics, and single-mint binding
    closeout;
  - same-binding closeout now reuses seal-minted identity on the steady path,
    and hash diagnostics explicitly surface round count, location, backend,
    bytes, wall time, identity-forming semantics, and mint strategy;
  - source-bound non-collective fallback now reports typed executor names
    (`SourceOrderedMappedTargetExecutor`,
    `MappedTargetStreamingExecutor`,
    `SourceOrderedDirectTargetExecutor`,
    `DirectTargetStreamingExecutor`,
    `MappedLoaderTargetExecutor`) instead of surfacing as
    `GenericByteRangeExecutor...`;
  - representative mounted closure evidence for the `v4` path is committed in
    `docs/benchmarks/20260415-qwen2.5-32b-mounted-collective-first-v4-serving-evidence.md`,
    which closes the `0114` owner boundary but not the broader Step3p5 `TP=8`
    signoff required by this plan;
  - strict collective failures now surface
    `collective_failure_class` through structured gRPC trailing metadata only;
  - TP8 mounted evidence is now committed in
    `docs/benchmarks/20260427-step3p5-fp8-mounted-tp8-cold-start-evidence.md`,
    including run identifiers, typed `/weight_version` facts, baseline
    comparison, and the explicit delete-gate conclusion; and
  - broader Step3p5 `TP=8` performance signoff remains open because the
    committed 2026-04-27 packet measured TensorCast ready time `370.346s`
    against same-host `safetensors` `220.242s` and `fastsafetensors`
    `222.243s`.
- 2026-04-28:
  - the mounted TP8 follow-up on
    `dev-yuchu-lxnsr-358366-worker-0` moved the dominant
    `w13_weight` family into the concat fast path with
    `mapped_expert_dim0_concat_job_summary requested=168 accepted=168`;
  - `collective_mapped_target prepared` now reports `concat_jobs=258`,
    `concat_job_source_bytes=127,656,296,448`, and
    `concat_job_exec_sec=73.948`;
  - TensorCast ready time improved to `326.319s`, with rank-local
    `Tensorcast load_model timings` around `167.526s` to `171.516s`; but
  - broader Step3p5 `TP=8` performance signoff still remains open because
    same-host `safetensors` measured `158.172s` and same-host
    `fastsafetensors` measured `162.179s`;
  - daemon-side profiling identified the remaining gap as disk-loader strategy:
    the same-host path selected `OwnerFileCollectiveExecutor`, executed `258`
    concat jobs over `127,656,296,448` source bytes, then processed `428`
    mapped residual windows and `906` chunks over another `197,099,083,520`
    read bytes, for `149.071s` in the mapped collective executor; and
  - the remaining gate should be treated as a same-host loader-shape problem:
    TensorCast must avoid turning this case into a root-rank staged read plus
    peer/NCCL distribution path if it needs to match the same-host
    `safetensors` / `fastsafetensors` baselines.
- 2026-04-28:
  - a follow-up 8xH800 no-collective strategy probe ran at
    `/data/tc/0113-tp8-8gpu-ab-20260428-064325/0113-tp8-nocollective-rerun-20260428-064445`
    on `dev-yuchu-kk2x2-362559-worker-0`;
  - the probe used
    `{"tensorcast_collective_policy":"disable_collective"}` and selected
    `SourceOrderedMappedTargetExecutor` with
    `bootstrap_realize_collective_requested=false`,
    `bootstrap_realize_collective_used=false`, and
    `bootstrap_realize_actual_generic_backend_bytes=25,550,556,928`;
  - ready time improved to `262.284s`, and rank-local
    `Tensorcast load_model timings` improved to `117.313s` to `128.150s`; but
  - the gate still remains open because this best measured existing strategy
    remains much slower than same-host `safetensors` `158.172s` and
    `fastsafetensors` `162.179s`, with daemon profile showing
    `375,214` to `375,310` mapped byte-range segments per rank.
- 2026-04-28:
  - the tensor-aware mapped executor closure ran on
    `ws-7681b3683947089e-worker-k68dd`
    (`dev-yuchu-4thvk-367119-worker-0`) at
    `/data/tc/0113-tp8-rect2d-final-20260428-125500`;
  - the fix routes same-host disjoint TP shard work to the local typed mapped
    executor while the owner-file collective lane only carries the small
    replicated overlap selected by the overlap/dedup gate;
  - daemon profile shows `accepted_rect2d=137`,
    `accepted_bytes=873,562,112`, `residual_segments=0`,
    `residual_bytes=0`, and
    `actual_generic_backend_bytes=0`;
  - rank-local `Tensorcast load_model timings` improved to `17.434s` to
    `19.549s`; and
  - the same-host TP8 ready gate now passes with TensorCast `136.271s` against
    same-host `safetensors` `144.155s` and `fastsafetensors` `152.174s`.
- 2026-04-28:
  - the same-prompt TP8 model-output comparison is recorded in
    `docs/benchmarks/20260427-step3p5-fp8-mounted-tp8-cold-start-evidence.md`;
  - the compared request is
    `{"prompt":"Say hi in five words.","max_tokens":8,"temperature":0}`;
  - TensorCast artifact
    `/data/tc/0113-tp8-20260428-004448/completion.json`,
    same-host `safetensors`
    `/data/tc/0113-safetensors-tp8-final-20260428-125212/completion.json`,
    and same-host `fastsafetensors`
    `/data/tc/0113-fastsafetensors-tp8-final-20260428-125809/completion.json`
    all return `"\tCHITCHAT\nI'm going"` with `finish_reason=length`,
    `prompt_tokens=7`, `completion_tokens=8`, and `total_tokens=15`;
  - normalized response equality is `true` after excluding only expected
    volatile envelope fields `id`, `created`, and `model`; and
  - the final `136.271s` TensorCast performance packet only issued a shorter
    smoke completion. Rerunning that optimized packet with the same prompt was
    attempted on 2026-04-28 but blocked by 8-GPU scheduling capacity:
    `codesign` quota check reported `gpu: 133/128`, and `tensorcast_dev`
    returned `no machine available`.

## Evidence package

- [x] One committed evidence package must include:
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
- [x] Identity and hash rollout should remove cost only after the first-class
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
- [x] Diagnostics backout must not leave downstream integrations dependent on
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
- [ ] Risk: downstream integrations continue to treat `same_binding_fast_path_validated` as a
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
