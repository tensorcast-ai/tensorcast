---
slug: artifact-centered-model-runtime-realization
title: Artifact-Centered Model Runtime Realization Plan
status: draft
areas: ["sdk", "serving", "daemon", "core", "integrations", "docs", "tests"]
created: 2026-05-23
last_updated: 2026-05-23
related_code:
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/designs/0119-serving-runtime-concept-deepening-and-vllm-migration.md
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
    - ../designs/0119-serving-runtime-concept-deepening-and-vllm-migration.md
---

# Objective

Plan the successor work after `0119`: move TensorCast TensorDict retrieval,
binding, prefetch, and model-runtime loading from parallel surfaces toward one
artifact-centered realization model while preserving all vLLM behavior and
performance-sensitive semantics.

This plan intentionally starts with scenario inventory and API ownership before
renaming or deleting code. API incompatibility is expected where it makes the
final model simpler; semantic regression is not allowed.

# Current State & Grounding

`0119` has produced the current baseline:

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
  binding state.

The plan must converge these paths into one realization spec/handle model before
renaming serving-centered APIs broadly.

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

# Phases & Milestones

- [ ] Phase 1: Freeze The `0119` Baseline
  - [ ] Record the executed `0119` code/module state and mark which fields are
        behavior contracts versus temporary names.
  - [ ] Confirm the missing or stale `0119` companion-plan link is resolved by
        either restoring the completed plan or documenting that the final design
        now embeds execution outcome.
  - [ ] Capture a vLLM scenario matrix with owner files and expected behavior.
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
  - [ ] Keep `RuntimeAttachment` as the process-local framework attachment and do
        not move model object state into `Artifact` or the realization handle.

- [ ] Phase 4: Map Existing SDK And Serving Paths To Artifact Realization
  - [ ] Map `Artifact.tensor_dict(...)` to TensorDict projection over a
        realization handle.
  - [ ] Map `Artifact.tensor_dict_into(...)` to caller-owned TensorDict target
        realization.
  - [ ] Map `Artifact.bind(...)` and `Artifact.bind_into(...)` to binding
        projection over a realization handle.
  - [ ] Map `Artifact.prefetch(device=...)` to ordinary replica realization or
        replica preparation.
  - [ ] Map `Artifact.prefetch(target=...)` to retained realization handoff.
  - [ ] Map `artifact_bind` to durable artifact realization.
  - [ ] Map `source_bootstrap_to_binding` to source artifact realization.
  - [ ] Map `retained_binding_acquire` to retained realization acquire.
  - [ ] Map `ServingBindingPlan` fields to realization-plan identity.
  - [ ] Map runtime replica publication to artifact replica publication.
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
  - [ ] Preserve retained reservation byte credit before vLLM admission.
  - [ ] Preserve in-place reload response projection and stale/duplicate reload
        handling.
  - [ ] Preserve after-ready publication, required-publication failure state,
        stale publication retirement, and shutdown retirement.
  - [ ] Preserve EP/EPLB reload safety and semantic placement digests.
  - [ ] Preserve drafter model reload behavior.

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
- Create a semantic field map for `model_loader_extra_config`.
- Define an `ArtifactRealizationSpec` and `ArtifactRealizationHandle` DTO/API
  draft that can represent TensorDict retrieval, binding, current vLLM placement,
  source, collective, and retained acquire behavior.
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
- vLLM integration tests or smoke scripts for:
  - TensorDict-dependent internal paths;
  - local source cold start;
  - durable artifact startup;
  - retained prefetch attach with memory credit;
  - TP same-node collective-first startup;
  - in-place reload;
  - after-ready publication and shutdown retirement;
  - local-ready durable promotion;
  - EP/EPLB reload rejection on topology mismatch.
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
- TP/P2P semantic confusion: require explicit compatibility validation before
  direct P2P member reuse.
- Hidden framework leakage: keep vLLM model attributes and finalize hooks inside
  runtime adapters and attachments.
- Naming churn without semantic gain: do not rename a class until ownership is
  clear and the vLLM migration path is documented.
