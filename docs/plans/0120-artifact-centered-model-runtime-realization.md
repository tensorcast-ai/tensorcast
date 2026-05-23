---
slug: artifact-centered-model-runtime-realization
title: Artifact-Centered Model Runtime Realization Plan
status: draft
areas: ["sdk", "serving", "daemon", "core", "integrations", "docs", "tests"]
created: 2026-05-23
last_updated: 2026-05-23
related_code:
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/designs/0121-unified-artifact-realization-kernel.md
  - docs/plans/0121-unified-artifact-realization-kernel.md
  - docs/designs/0116-prefetch-serving-binding-target.md
  - docs/plans/0116-prefetch-serving-binding-target.md
  - tensorcast/api/store/artifact.py
  - tensorcast/types.py
  - tensorcast/serving/runtime.py
  - tensorcast/serving/config.py
  - tensorcast/serving/policy.py
  - tensorcast/serving/hosts.py
  - tensorcast/serving/binding_plan.py
  - tensorcast/serving/retained_binding.py
  - tensorcast/serving/runtime_attachment.py
  - tensorcast/serving/replica_publication.py
  - tensorcast/serving/_runtime_impl/lifecycle.py
links:
  design: ../designs/0120-artifact-centered-model-runtime-realization.md
  dependencies:
    - ../designs/0121-unified-artifact-realization-kernel.md
---

# Objective

Plan the successor work from the current serving-runtime baseline: move
TensorCast TensorDict retrieval, binding, prefetch, and model-runtime loading
from parallel surfaces toward one artifact-centered realization model while
preserving all vLLM behavior and performance-sensitive semantics.

This plan intentionally starts with scenario inventory and API ownership before
renaming or deleting code. API incompatibility is expected where it makes the
final model simpler; semantic regression is not allowed.

# Current State & Grounding

The current serving-runtime baseline is implemented and folded into `0120` as
context. It is not a separate long-term public model.
- `tensorcast.serving.runtime` is the narrow framework-facing runtime API.
- `tensorcast.serving.config` selects exactly one startup plan.
- `tensorcast.serving.policy` owns serving locator and policy normalization.
- `tensorcast.serving.binding_plan` centralizes trace/recipe/spec/layout/schema
  identity.
- `tensorcast.serving.retained_binding` owns retained acquire validation,
  reservation bytes, lease restore, and runtime ownership transfer.
- `tensorcast.serving.runtime_attachment` owns process-local attachment state.
- `tensorcast.serving.replica_publication` owns artifact-backed runtime replica
  publication and retirement.
- `tensorcast.serving._runtime_impl.lifecycle` still performs most orchestration.

The Store SDK baseline also matters for the target model:

- `Artifact.tensor_dict(...)` and `Artifact.tensor_dict_into(...)` are ordinary
  retrieval paths.
- `Artifact.bind(...)` and `Artifact.bind_into(...)` are daemon-owned local
  realization paths.
- `Artifact.prefetch(device=...)` prepares ordinary replicas, while
  `Artifact.prefetch(target=ServingBindingTarget(...))` prepares retained
  binding state. `0116` is superseded as a standalone public serving-target
  design; its retained residency and acquire semantics are absorbed into `0121`.

The plan must converge these paths into one realization spec/handle model before
renaming serving-centered APIs broadly.

The concrete kernel work is tracked in
[`0121-unified-artifact-realization-kernel`](../designs/0121-unified-artifact-realization-kernel.md).
This `0120` plan remains the umbrella migration plan; `0121` owns the
anti-split-brain implementation sequence for selection, target, strategy,
representation, lifecycle, execution lowering, reports, and TP target sets.

Execution order:

1. Execute `0121` first until TensorDict, binding, prefetch, runtime attach, and
   TP share the same realization kernel.
2. Then use this `0120` plan to finish public vocabulary, serving-name
   retirement, runtime adapter cleanup, and external API consolidation.
3. Do not rename broad public APIs before the shared kernel prevents split-brain.

The internal-vLLM baseline depends on these surfaces:

- `vllm/tensorcast/loader.py`: session start, attachment storage, in-place
  reload, replica publication, local-ready durable promotion.
- `vllm/tensorcast/placement.py`: TP/PP/DP member identity, EP/EPLB digests,
  materialization execution facts.
- `vllm/tensorcast/source.py`: local source catalog and cache policy.
- `vllm/tensorcast/collective.py`: same-node source coordination and local-ready
  TP barrier.
- `vllm/tensorcast/adapter.py`: meta/runtime model construction, trace capture,
  tensor attach/finalize, runtime-only tensors, semantic probes.
- `vllm/model_executor/model_loader/memory_accounting.py`: trusted retained
  reservation bytes before vLLM memory admission.
- `vllm/v1/worker/gpu_model_runner.py`: reload endpoint, runtime view,
  shutdown retirement, EP/EPLB reload safety.

Feasibility result for the current scenario:

- internal-vLLM can adapt cleanly to the artifact-centered model because its
  TensorCast loader is already concentrated behind model loader, placement,
  source, adapter, and worker reload/publication surfaces;
- vLLM does not need a direct TensorCast TensorDict model-loading API, but it
  does need TensorDict to become a first-class projection of the same realization
  kernel so selection, strategy, diagnostics, lifetime, and P2P behavior cannot
  diverge from binding/runtime attach;
- retained reservation credit before vLLM memory admission is the main timing
  constraint and must be represented before any runtime attachment exists;
- local HF/safetensors bootstrap is feasible when represented as a
  daemon-attested mounted-source artifact subject such as `msa1:...`, not as a
  vLLM-owned source authority;
- reload/publication semantics are feasible if active attachment or binding-value
  generation checks remain explicit;
- EP/EPLB reload safety must combine static semantic digests with live framework
  checks from vLLM before reload;
- main and draft TensorCast model reload must be made explicit as either a
  target-set transaction or the current sequential main-then-draft behavior with
  unhealthy marking on partial failure.

# Phases & Milestones

- [ ] Phase 1: Freeze The Current Serving Runtime Baseline
  - [x] Remove the standalone serving-centered design and fold baseline context
        into `0120`.
  - [ ] Record the current serving-runtime code/module state and mark which fields are
        behavior contracts versus temporary names.
  - [ ] Capture a vLLM scenario matrix with owner files and expected behavior.
  - [ ] Record that current vLLM does not use TensorCast TensorDict as its steady
        model-loading path; TensorDict is still required as the equivalence proof
        for the shared realization kernel.
  - [ ] Verify no Python SDK path added direct Global Store access.

- [ ] Phase 2: Classify vLLM Configuration Semantics
  - [ ] Classify `serving.artifact_locator` as artifact locator/selection input.
  - [ ] Classify `serving.policy` as manifest/build/contract preflight.
  - [ ] Classify `bootstrap.*` as source artifact and recipe-cache policy.
  - [ ] Classify `materialization.collective` as realization strategy policy.
  - [ ] Classify `retained_binding_acquire.*` as retained realization claim.
  - [ ] Classify `replica_publication.*` as artifact replica publication policy.
  - [ ] Classify `diagnostics.*` as realization diagnostics and operator
        profiling.
  - [ ] Classify reload request `artifact_locator`, `policy`, and any model
        overrides as artifact selection/admission inputs rather than serving-only
        control data.

- [ ] Phase 3: Define The Successor API Boundary
  - [ ] Draft `ArtifactRealizationSpec` as the public/professional intent concept
        for TensorDict, binding, retained prefetch, and model runtime.
  - [ ] Draft `ArtifactRealizationHandle` as the result concept with projection
        and lifecycle actions.
  - [ ] Decide which `RealizationTarget` and `ModelRuntimeProfile` fields live
        inside the spec rather than as separate public concepts.
  - [ ] Draft `Artifact.realize(spec=...)` as the long-term artifact-centered
        realization model.
  - [ ] Define `tensor_dict()`, `binding()`, `attach(adapter=...)`,
        `prefetch_handoff()`, `publish_replica(...)`, and `promote(...)`
        projections/actions on the handle.
  - [ ] Define `RetainedRealizationClaim` pre-admission APIs for trusted
        reservation bytes and expected-member validation.
  - [ ] Define framework admission hooks for live facts that cannot be durable
        artifact identity, including vLLM EP/EPLB reload checks.
  - [ ] Define the common `ArtifactRealizationReport` fields required by
        TensorDict diagnostics, binding/runtime attach, retained acquire,
        publication, and TP target sets.
  - [ ] Keep `RuntimeAttachment` as the process-local framework attachment and do
        not move model object state into `Artifact` or the realization handle.

- [ ] Phase 4: Map Existing SDK And Serving Paths To Artifact Realization
  - [ ] Map `Artifact.tensor_dict(...)` to TensorDict projection over a
        realization handle.
  - [ ] Map `Artifact.tensor_dict_into(...)` to caller-owned TensorDict target
        realization.
  - [ ] Add TensorDict projection lifetime ownership so wrapper-returned tensors
        cannot outlive their payload lease/export owner.
  - [ ] Map `Artifact.bind(...)` and `Artifact.bind_into(...)` to binding
        projection over a realization handle.
  - [ ] Map `Artifact.prefetch(device=...)` to ordinary replica realization or
        replica preparation.
  - [ ] Map `Artifact.prefetch(target=...)` to retained realization handoff.
  - [ ] Move remaining `0116` retained residency/acquire TODOs into retained
        realization lifecycle, report, and runtime-admission tasks.
  - [ ] Map `artifact_bind` to durable artifact realization.
  - [ ] Map `source_bootstrap_to_binding` to mounted-source artifact realization
        with `msa1:` admission and no source-handle-only bypass.
  - [ ] Map `retained_binding_acquire` to retained realization acquire.
  - [ ] Map `ServingBindingPlan` fields to realization-plan identity.
  - [ ] Map runtime replica publication to artifact replica publication with
        active attachment or binding-value generation checks.
  - [ ] Keep durable local-ready promotion as explicit representation
        publication.

- [ ] Phase 5: Preserve TP And P2P Correctness
  - [ ] Represent TP as `RealizationTargetSet` plus member-local target layouts.
  - [ ] Keep same-node collective-first as a strategy-plane choice.
  - [ ] Permit direct P2P only for compatible representation/topology/member
        layout/schema.
  - [ ] Route incompatible representation changes through transform or
        realization plans, not P2P fallback.
  - [ ] Preserve source coordination and local-ready barrier semantics for vLLM.

- [ ] Phase 6: Migrate internal-vLLM
  - [ ] Port `TensorcastModelLoader` startup from serving session naming to the
        successor artifact-runtime API.
  - [ ] Preserve retained reservation byte credit before vLLM admission through
        `RetainedRealizationClaim` validation, not late runtime attachment state.
  - [ ] Port vLLM source bootstrap to the mounted-source artifact contract and
        remove any source authority path that bypasses artifact selection.
  - [ ] Keep `VLLMTensorcastAdapter` as the owner for model construction,
        trace capture, runtime-only tensor rehydration, finalize hooks, and
        semantic probes.
  - [ ] Preserve in-place reload response projection and stale/duplicate reload
        handling.
  - [ ] Preserve after-ready publication, required-publication failure state,
        active-generation checks, stale publication retirement, and shutdown
        retirement.
  - [ ] Preserve EP/EPLB reload safety through both semantic placement digests and
        live EP/EPLB framework checks.
  - [ ] Preserve drafter model reload behavior, either by target-set transaction
        semantics or by explicitly keeping the current sequential
        main-then-draft failure/unhealthy contract.

- [ ] Phase 7: Narrow Serving-Centered Names
  - [ ] Reshape or remove `tensorcast.serving` public imports where the
        artifact-centered API has a clearer owner.
  - [ ] Rename or absorb serving-rooted DTO names such as `ServingBindingTarget`,
        `PrefetchedServingBinding`, and `ServingArtifactManifest` unless the
        payload is specifically tied to serving ABI.
  - [ ] Move public docs toward artifact/runtime terminology.
  - [ ] Keep internal names only where the implementation still specifically
        means model serving ABI or serving readiness.
  - [ ] Remove broad facades that expose private lifecycle helpers as public API.

- [ ] Phase 8: Extend To A Second Runtime
  - [ ] Use SGLang or a minimal mock runtime adapter to prove the API is not
        vLLM-shaped.
  - [ ] Confirm source catalog, target layout, runtime-only tensors, and
        publication can be represented without vLLM-specific public names.

# Tasks

- Build a vLLM contract table from current internal-vLLM imports and runtime
  call sites.
- Build a TensorDict/Binding contract table from current SDK call sites and
  tests, including selection identity, diagnostics, publication eligibility, and
  P2P behavior.
- Add an explicit vLLM feasibility table that maps early memory credit,
  `msa1:` source bootstrap, runtime adapter responsibilities, reload, publication,
  EP/EPLB safety, and drafter reload to TensorCast-owned contracts.
- Create a semantic field map for `model_loader_extra_config`.
- Define an `ArtifactRealizationSpec` and `ArtifactRealizationHandle` DTO/API
  draft that can represent TensorDict retrieval, binding, current vLLM placement,
  source, collective, and retained acquire behavior.
- Define the pre-admission retained claim DTO/API and the exact trusted
  reservation validation inputs vLLM needs before allocation.
- Define the publication generation/CAS contract shared by after-ready
  publication, reload retirement, and shutdown retirement.
- Refactor current `ServingRuntimeSession` semantics toward the
  artifact-centered vocabulary without preserving old import or parameter
  compatibility.
- Decide which current names stay because they are accurate implementation
  terms, which are renamed, and which are deleted.
- Add failure-mode checks for artifact locator resolution, retained acquire
  member mismatch, active-publication reload, P2P representation mismatch, and
  local-ready publication misuse.

# Test / Rollout / Recovery

Validation should focus on behavior, not old API compatibility:

- Python serving runtime tests under `tests/python/...` for the three startup
  paths, reload, retained acquire, runtime view, and replica publication.
- Python SDK tests under `tests/python/...` showing TensorDict, bind, prefetch,
  and model-runtime realization use the same selection and strategy semantics.
- TensorDict projection lifetime tests proving wrapper-returned tensors retain
  or transfer payload ownership correctly.
- vLLM integration tests or smoke scripts for:
  - TensorDict projection equivalence with binding/runtime realization;
  - local source cold start;
  - durable artifact startup;
  - retained prefetch attach with memory credit;
  - TP same-node collective-first startup;
  - in-place reload;
  - after-ready publication generation checks and shutdown retirement;
  - local-ready durable promotion;
  - EP/EPLB reload rejection on topology mismatch;
  - draft model reload success and partial-failure/unhealthy behavior.
- C++ daemon/core tests only when proto, materialization, binding, or P2P
  behavior changes.
- Execute TensorCast and internal-vLLM changes together, because both codebases
  are under our control.
- Recovery is behavior-based: if the new API shape is wrong, revert or revise
  the incomplete refactor branch before landing rather than keeping a parallel
  compatibility layer.

# Risks & Tracking

- Artifact API overgrowth: track whether proposed methods are durable artifact
  lifecycle operations or realization-handle projection/actions.
- TensorDict split-brain: track whether TensorDict tests still exercise a
  separate materialization path that bypasses realization specs, strategy
  selection, diagnostics, or P2P compatibility checks.
- vLLM timing regression: specifically track retained reservation memory credit
  before admission.
- Source authority split-brain: track whether local source bootstrap can run
  without an admitted `msa1:` or durable artifact subject.
- Reload/publication race: track whether publication, reload, and shutdown
  retirement compare the active attachment or binding-value generation before
  mutating state.
- Drafter partial reload: track whether main/draft behavior is target-set atomic
  or explicitly sequential with worker-unhealthy marking.
- TP/P2P semantic confusion: require explicit compatibility validation before
  direct P2P member reuse.
- Hidden framework leakage: keep vLLM model attributes and finalize hooks inside
  runtime adapters and attachments.
- Naming churn without semantic gain: do not rename a class until ownership is
  clear and the vLLM migration path is documented.
