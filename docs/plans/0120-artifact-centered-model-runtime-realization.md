---
slug: artifact-centered-model-runtime-realization
title: Artifact-Centered Model Runtime Realization Plan
status: draft
areas: ["sdk", "serving", "daemon", "core", "integrations", "docs", "tests"]
created: 2026-05-23
last_updated: 2026-05-25
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

The Store SDK baseline has moved from target model to implemented kernel
baseline:

- `Artifact.tensor_dict(...)`, `tensor_dict_with_diagnostics(...)`,
  `tensor_dict_into(...)`, `tensor_into(...)`, `bind(...)`, and `bind_into(...)`
  lower through `Artifact.realize(...)`.
- `Artifact.prefetch(device=...)` and `Artifact.prefetch(target=...)` lower
  through `Artifact.realize_async(...)` for retained replica, retained binding,
  and target-set operation semantics.
- `ArtifactRealizationSpec`, `ArtifactRealizationHandle`,
  `ArtifactRealizationReport`, `RealizationResourceEnvelope`,
  `RealizationReleaseContract`, target-set reports, mounted-source reports,
  runtime-attachment reports, model-runtime report wrappers, and publication
  reports are implemented and exported from the Store SDK.
- Direct `Artifact.realize(ArtifactRealizationSpec.model_runtime(...))` still
  fails closed. Serving lifecycle code creates runtime-attachment and
  model-runtime handles internally while internal-vLLM still enters through
  `ServingRuntimeSession`.

The plan no longer needs to converge the main SDK paths before broad serving
cleanup; `0121` did that. This plan now tracks the remaining naming/API boundary
work after the kernel convergence.

The concrete kernel work is tracked in
[`0121-unified-artifact-realization-kernel`](../designs/0121-unified-artifact-realization-kernel.md).
This `0120` plan remains the umbrella migration plan; `0121` owns the
anti-split-brain implementation sequence for selection, target, strategy,
representation, lifecycle, execution lowering, reports, and TP target sets.

Execution order:

1. Treat the implemented `0121` kernel as the baseline.
2. Finish direct public/professional model-runtime realization instead of routing
   framework integrations through serving-named session APIs.
3. Port internal-vLLM in the same execution window as TensorCast API changes.
4. Then retire or narrow serving-centered public names where the artifact/runtime
   owner is clear.
5. Prove the boundary with SGLang or a minimal second runtime adapter.

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
- vLLM does not need a direct TensorCast TensorDict model-loading API, and
  TensorDict is now a first-class projection of the shared realization kernel;
- retained reservation credit before vLLM memory admission is the main timing
  constraint and must be represented before any runtime attachment exists;
- local HF/safetensors bootstrap is feasible through daemon-attested mounted
  source subjects such as `msa1:...`, not as a vLLM-owned source authority;
- reload/publication semantics are implemented in serving lifecycle and remain a
  migration constraint for the public artifact-runtime API;
- EP/EPLB reload safety must combine static semantic digests with live framework
  checks from vLLM before reload;
- main and draft TensorCast model reload must be made explicit as either a
  target-set transaction or the current sequential main-then-draft behavior with
  unhealthy marking on partial failure.

# Phases & Milestones

- [x] Phase 1: Freeze The Current Serving Runtime Baseline
  - [x] Remove the standalone serving-centered design and fold baseline context
        into `0120`.
  - [x] Record the current serving-runtime code/module state and mark behavior
        contracts versus temporary names.
  - [x] Capture the vLLM scenario matrix with owner files and expected behavior
        in the design.
  - [x] Record that current vLLM does not use TensorCast TensorDict as its steady
        model-loading path; TensorDict is the equivalence proof for shared
        realization semantics.
  - [x] Verify no Python SDK artifact metadata or realization path added direct
        Global Store access; `0121` guardrails now cover this.

- [x] Phase 2: Land The Shared Realization Kernel Through `0121`
  - [x] Define and export `ArtifactRealizationSpec`,
        `ArtifactRealizationHandle`, `ArtifactRealizationReport`, selection,
        target, strategy, representation, lifecycle, resource-envelope,
        release-contract, and report DTOs.
  - [x] Lower `Artifact.tensor_dict(...)`,
        `tensor_dict_with_diagnostics(...)`, `tensor_dict_into(...)`,
        `tensor_into(...)`, `bind(...)`, and `bind_into(...)` through
        `Artifact.realize(...)`.
  - [x] Lower retained replica, retained binding, and target-set prefetch through
        `Artifact.realize_async(...)` while preserving `Operation[T]`.
  - [x] Add TensorDict projection ownership and release-contract lifecycle
        coverage.
  - [x] Add retained binding/target-set reports, mounted-source realization,
        runtime-attachment reports, model-runtime report wrappers, and
        publication reports.
  - [x] Add direct Global Store guardrails for SDK artifact metadata and
        realization paths.

- [x] Phase 3: Preserve TP, P2P, Publication, And Runtime-Attachment Correctness
  - [x] Represent TP as target-set realization with member-local layouts and
        source-selection modes.
  - [x] Keep same-node collective-first as strategy-plane state.
  - [x] Keep P2P direct reuse gated by compatible
        representation/topology/member/layout/schema.
  - [x] Route runtime attachment, retained acquire close, and publication
        projection close through realization release contracts.
  - [x] Preserve source coordination, local-ready barrier, active-generation
        publication, stale-publication retirement, and shutdown retirement
        semantics.

- [ ] Phase 4: Expose Direct Model-Runtime Realization
  - [ ] Decide whether direct
        `Artifact.realize(ArtifactRealizationSpec.model_runtime(...))` becomes
        the professional framework API or remains internal behind a different
        artifact-runtime session facade.
  - [ ] If direct API is chosen, lower model-runtime realization through the
        serving/runtime attachment implementation instead of returning
        `UNIMPLEMENTED`.
  - [ ] Define the framework adapter surface for construction, trace capture,
        runtime-only tensors, finalize hooks, semantic probes, reload admission,
        and live EP/EPLB checks.
  - [ ] Preserve `RuntimeAttachment` as the process-local framework boundary and
        keep model object state out of `Artifact`.
  - [ ] Add tests that direct model-runtime handle reports match
        serving-lifecycle model-runtime reports.

- [ ] Phase 5: Migrate internal-vLLM To The Successor Boundary
  - [ ] Port `TensorcastModelLoader` startup from serving session naming to the
        successor artifact-runtime API.
  - [ ] Preserve retained reservation byte credit before vLLM admission through
        retained realization claim validation.
  - [ ] Port vLLM source bootstrap to the mounted-source artifact contract and
        keep `msa1:`/durable artifact admission explicit.
  - [ ] Keep `VLLMTensorcastAdapter` as the owner for model construction,
        trace capture, runtime-only tensor rehydration, finalize hooks, and
        semantic probes.
  - [ ] Preserve in-place reload response projection, stale/duplicate reload
        handling, after-ready publication, required-publication failure state,
        stale publication retirement, shutdown retirement, EP/EPLB reload
        safety, and drafter sequential failure/unhealthy behavior.

- [ ] Phase 6: Narrow Serving-Centered Names
  - [ ] Classify `serving.artifact_locator`, `serving.policy`, `bootstrap.*`,
        `materialization.collective`, `retained_binding_acquire.*`,
        `replica_publication.*`, and `diagnostics.*` into artifact selection,
        representation preflight, source artifact bootstrap, realization
        strategy, retained claim, publication policy, and diagnostics fields.
  - [ ] Decide which names remain because they describe serving ABI semantics
        and which move to artifact/runtime vocabulary.
  - [ ] Rename or absorb serving-rooted DTOs such as `ServingBindingTarget`,
        `PrefetchedServingBinding`, and `ServingArtifactManifest` only after
        replacements exist.
  - [ ] Move public docs toward artifact/runtime terminology.
  - [ ] Remove broad facades that expose private lifecycle helpers as public API.

- [ ] Phase 7: Extend To A Second Runtime
  - [ ] Use SGLang or a minimal mock runtime adapter to prove the API is not
        vLLM-shaped.
  - [ ] Confirm source catalog, target layout, runtime-only tensors, and
        publication can be represented without vLLM-specific public names.

# Tasks

- Keep this plan as the post-`0121` migration ledger; do not duplicate the
  completed `0121` kernel checklist.
- Draft the direct model-runtime API decision and implementation path:
  `Artifact.realize(ArtifactRealizationSpec.model_runtime(...))` versus an
  explicit artifact-runtime session facade.
- Build a current internal-vLLM import/call-site table showing every remaining
  `ServingRuntimeSession`, `ServingConfig`, retained-binding, publication, and
  runtime-view dependency.
- Create the semantic field map from current `model_loader_extra_config` to
  artifact/runtime terminology.
- Define the retained realization claim public/professional naming path while
  preserving current trusted reservation validation inputs.
- Define the publication generation/CAS contract shared by after-ready
  publication, reload retirement, and shutdown retirement as the vLLM migration
  acceptance rule.
- Add direct model-runtime realization tests once the public lowering exists.
- Add or update internal-vLLM smoke/integration tests for startup, retained
  memory credit, local source cold start, durable artifact startup, in-place
  reload, after-ready publication, shutdown retirement, EP/EPLB rejection, and
  draft partial-failure/unhealthy behavior.
- Add second-runtime adapter tests or fixtures before retiring serving-centered
  public vocabulary broadly.

# Test / Rollout / Recovery

Validation now splits completed `0121` kernel guardrails from remaining `0120`
migration checks.

Completed kernel guardrails:

- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_replica_publication.py`

Remaining migration checks:

- direct model-runtime realization tests once `Artifact.realize(model_runtime)`
  no longer fails closed;
- internal-vLLM smoke/integration tests for startup, reload, publication, and
  retained credit;
- second-runtime adapter proof before broad serving vocabulary retirement;
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
