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

No source compatibility guarantee is required. This plan is delete-forward:
replacement artifact-runtime paths should absorb behavior, prove equivalence
with focused tests, and then remove or internalize old serving-rooted public
entrypoints, compatibility adapters, duplicate diagnostics, and redundant tests.
Semantic regression is not allowed; maintaining two long-term stacks is also not
allowed.

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
- Daemon materialization already has the key performance primitives the target
  API must preserve: `MaterializeReplica` tries the artifact LIP/local-replica
  fast path before engine-backed materialization, binds CUDA IPC or CPU memfd
  leases into the response, and reports the selected materialization source.
- Binding materialization already attempts direct byte-space planning for
  compatible mapped source artifacts and exposes source-bound plan diagnostics;
  direct model-runtime realization should reuse this behavior rather than
  routing through Python TensorDict materialization.
- Python materialization reconstructs tensor views from daemon CUDA IPC or memfd
  handles and records IPC/restore timings. Those views are acceptable
  projections but must not become a required intermediate for vLLM weight
  loading.

Current gaps against the `0120` target state:

- Top-level `tensorcast` does not yet expose `ArtifactRealizationSpec`,
  `ArtifactRealizationHandle`, or `ArtifactRealizationReport`, even though the
  target design treats them as public SDK peers of `Artifact`.
- Direct model-runtime realization still returns `UNIMPLEMENTED` from
  `Artifact.realize(...)`; runtime attachment lowering is only reachable through
  serving lifecycle code.
- `ArtifactRealizationHandle.attach(...)` currently exists as a delegation hook,
  but the target design needs completed-handle runtime attachment projection
  semantics. The implementation must either add `attachment()` or make
  `attach(...)` explicitly projection-only.
- Runtime host capabilities exist under serving host/integration names
  (`IntegrationHost`, framework/tensor-surface/placement/source/collective
  protocols). They need to become the artifact-runtime professional boundary
  instead of a serving-session dependency.
- `ArtifactRealizationSpec` still carries serving-rooted fields such as
  `serving_runtime_policy`; those fields need a neutral target/profile policy or
  a binding-specific options object.
- Retained pre-admission memory credit is still exposed through
  retained-serving-binding helpers. A neutral retained realization claim wrapper
  must absorb that behavior before vLLM migration.
- Publication generation, active-generation checks, replay, reload rejection,
  and shutdown retirement already exist in serving publication code but are not
  yet formalized as the shared artifact-runtime publication/CAS contract.
- A fake second framework already proves the serving runtime is not purely
  vLLM-shaped, but the proof still enters through `ServingRuntimeSession`; it
  must be repeated on the direct artifact-runtime API.
- Serving-rooted public DTOs and helpers remain broadly exported. They must be
  removed or internalized after replacement, not kept as compatibility aliases.

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
2. Expose the public realization symbols at the intended package level and make
   the docs/examples use the real target API shape.
3. Finish direct public/professional model-runtime realization instead of routing
   framework integrations through serving-named session APIs.
4. Port internal-vLLM in the same execution window as TensorCast API changes.
5. Delete or internalize serving-centered public names, compatibility wrappers,
   redundant diagnostics, and duplicate tests once replacements are wired.
6. Prove the boundary with SGLang or a minimal second runtime adapter through the
   direct artifact-runtime API.

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
- weight loading must keep the current best data path: retained acquire,
  local-replica/LIP, compatible P2P, local mounted-source/disk streaming, or
  explicit transform. The direct artifact-runtime API must not introduce a
  TensorDict or Python state-dict intermediate;
- local HF/safetensors bootstrap is feasible through daemon-attested mounted
  source subjects such as `msa1:...`, not as a vLLM-owned source authority;
- reload/publication semantics are implemented in serving lifecycle and remain a
  migration constraint for the public artifact-runtime API;
- EP/EPLB reload safety must combine static semantic digests with live framework
  checks from vLLM before reload;
- main and draft TensorCast model reload must be made explicit as either a
  target-set transaction or the current sequential main-then-draft behavior with
  unhealthy marking on partial failure.

# Migration Decision Logic

Every migration change should classify the touched concept before renaming or
deleting it. Use this order:

1. If it is durable identity, discovery, routing, replica visibility, or
   lifecycle, move it to artifact selection or artifact replica metadata.
2. If it is target/device/member/layout/strategy/admission intent, move it to
   `ArtifactRealizationSpec`, target plans, strategy plans, representation
   admission, or target-set realization.
3. If it is framework construction, trace capture, tensor surface,
   runtime-only tensor handling, finalize hooks, placement facts, source
   catalog, collective behavior, or live EP/EPLB checks, move it to the runtime
   host capability surface or `RuntimeAttachment`.
4. If it exists to credit memory or acquire a prepared value later, move it to
   retained realization claim or prefetch handoff naming.
5. If it creates a reusable source for later loads/P2P, move it to
   artifact-runtime publication/promote actions.
6. If the name is serving-rooted only for source compatibility, delete or
   internalize it once the replacement behavior and tests exist.
7. If a serving name truly describes a model-serving ABI payload, it may remain,
   but only as a profile/ABI field or private implementation detail.

The default decision is not "rename everything first." The default is:
classify behavior, wire the artifact-runtime replacement, prove behavior with
tests, then delete or internalize the old public serving surface in the same
cleanup window. A phase is incomplete if old and new public paths both remain as
supported peer entrypoints.

# internal-vLLM Migration Slice

TensorCast and internal-vLLM changes should land in one coordinated window
because both sides are under our control and no source compatibility guarantee is
required.

| internal-vLLM owner | Current TensorCast dependency | Target interaction | Completion signal |
| --- | --- | --- | --- |
| `vllm/tensorcast/loader.py` | `ServingConfig`, `IntegrationHost`, `ServingRuntimeSession`, `RuntimeAttachment` | build artifact/runtime request, call direct `Artifact.realize(... model_runtime ..., runtime_host=...)`, store `handle.attachment()` | startup, reload, required-publication, and local-ready promotion smoke tests no longer instantiate `ServingRuntimeSession` |
| `vllm/tensorcast/host.py` | `tensorcast.serving.hosts.IntegrationHost` | construct `RuntimeHostCapabilities` or transitional alias with deletion trigger | host construction has no public serving-session dependency |
| `vllm/tensorcast/adapter.py` | serving host/tensor-surface DTOs | framework capability implementation for construction, trace, runtime-only tensors, finalize hooks, semantic probes | adapter tests pass through direct artifact-runtime handle |
| `vllm/tensorcast/placement.py` | serving placement/local-ready DTOs | target/member/admission facts plus publication context for artifact-runtime actions | placement no longer creates public serving targets for normal runtime startup |
| `vllm/tensorcast/source.py` | `ServingConfig` and serving source catalog | mounted-source or durable artifact selection input | local HF/safetensors cold start admits an `msa1:` subject before planning |
| `vllm/tensorcast/collective.py` | serving-local collective coordination | realization strategy and target-set coordination facts | TP same-node startup uses shared target-set strategy reports |
| `vllm/tensorcast/retained_binding.py` | retained-serving-binding helpers | retained realization claim helpers | retained startup validates claim through neutral naming |
| `vllm/model_executor/model_loader/memory_accounting.py` | `tensorcast.serving.retained_binding` trusted bytes | retained realization claim trusted reservation bytes | memory credit still occurs before vLLM admission without public serving imports |
| `vllm/tensorcast/runtime_view.py` and `gpu_model_runner.py` | serving runtime view, session shutdown retirement, serving policy helpers | runtime attachment/view projection and artifact-runtime retirement actions | runtime view, reload, and shutdown tests use artifact-runtime actions |
| `vllm/tensorcast/builder/*` | serving builder/publication helpers | keep only if the payload is serving-ABI-specific; otherwise move to artifact publication actions | remaining builder imports are documented as internal/offline ABI-specific paths |

# Performance Migration Gates

The migration is not complete until the direct artifact-runtime path proves that
the user-facing API change did not move model loading onto a slower or larger
path.

| Gate | What to prove | Concrete check |
| --- | --- | --- |
| No TensorDict intermediate | `Artifact.realize(... model_runtime ...)` attaches a binding/retained value directly instead of first calling TensorDict materialization. | Direct API tests and internal-vLLM startup tests fail if normal model-runtime startup calls TensorDict projection helpers, Python builder materializers, or full state-dict loaders. |
| Fast source selection preserved | Retained, local replica/LIP, P2P, disk, mounted-source/direct-write, and explicit-transform cases report the expected selected source and fallback status. | Artifact-realization reports assert source kind, fallback reason bucket, copy bytes, temporary bytes, retained bytes, and direct-write bytes for each representative path. |
| No extra GPU weight residency | Steady-state runtime attach owns one TensorCast weight residency plus framework runtime-only tensors; direct API migration does not keep both serving and artifact-runtime owners. | vLLM smoke/profile captures CUDA allocated/reserved deltas around startup and reload, checks `_vllm_external_weight_bytes`/retained credit, and verifies old attachment/binding handles are retired. |
| No full host-memory staging | Normal durable, retained, and mounted-source startup do not build a full Python `dict[str, torch.Tensor]`, full safetensors state dict, or full CPU copy of weights. | RSS/profile events and call-site audit keep full host materialization limited to explicit offline builder workflows. |
| Admission timing preserved | Retained reservation bytes are credited before vLLM calculates requested KV/cache memory. | `memory_accounting.py` tests use the neutral claim helper and assert credit before `gpu_worker` startup admission. |
| Latency remains stage-local | Direct artifact-runtime start adds no extra data-plane RPC, session start, retained acquire, or IPC restore beyond the chosen source strategy. | Profile events compare current serving baseline and direct API for startup, IPC open, attach/finalize, source-bound plan, reload, and publication stages; any added stage needs an explicit reason in the report. |
| Reload overlap bounded | In-place reload may temporarily overlap old and new weights only under declared swap semantics. | Reload tests assert active-generation CAS, stale publication retirement, and resource-envelope overlap accounting. |

These gates are intentionally behavior-based. A rename can pass only when the
resolved source, memory ownership, and timing shape match the current optimal
path for the same compatibility class.

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
  - [ ] Export `ArtifactRealizationSpec`, `ArtifactRealizationHandle`, and
        `ArtifactRealizationReport` from the intended public SDK package level
        and add import smoke tests.
  - [ ] Adopt direct
        `Artifact.realize(ArtifactRealizationSpec.model_runtime(...), runtime_host=...)`
        or an equivalent artifact-rooted signature as the professional framework
        API. Do not introduce a new public artifact-runtime session facade.
  - [ ] Lower model-runtime realization through the existing runtime attachment
        implementation instead of returning `UNIMPLEMENTED`.
  - [ ] Ensure that lowering calls the daemon binding/retained/source-bound
        paths directly. Direct model-runtime realization must not first call
        TensorDict projection helpers, materialize a Python state dict, or start
        a second public serving session.
  - [ ] Define the runtime host capability surface for construction, trace
        capture, tensor surface, runtime-only tensors, finalize hooks, placement
        facts, source catalog, collective behavior, semantic probes, reload
        admission, and live EP/EPLB checks.
  - [ ] Define completed-handle projection semantics: add
        `ArtifactRealizationHandle.attachment()` or make `attach(...)`
        explicitly projection-only, not a second execution step.
  - [ ] Preserve `RuntimeAttachment` as the process-local framework boundary and
        keep model object state out of `Artifact`.
  - [ ] Add tests that direct model-runtime handle reports match
        serving-lifecycle model-runtime reports.
  - [ ] Add report assertions for selected source kind, fallback reason,
        copy bytes, temporary bytes, retained bytes, direct-write bytes,
        IPC-open timing, and attach/finalize timing.
  - [ ] Update public examples to use the actual target API shape and current
        binding arguments (`mapping` / target-plan DTOs), not stale `layout=...`
        placeholders.

- [ ] Phase 5: Migrate internal-vLLM To The Successor Boundary
  - [ ] Port `TensorcastModelLoader` startup from serving session naming to the
        successor artifact-runtime API.
  - [ ] Port `vllm/tensorcast/host.py` from public `IntegrationHost`
        construction to `RuntimeHostCapabilities` construction or a transitional
        alias with an explicit deletion trigger.
  - [ ] Preserve retained reservation byte credit before vLLM admission through
        retained realization claim validation.
  - [ ] Introduce neutral retained realization claim helpers and migrate vLLM
        memory accounting off retained-serving-binding public helpers.
  - [ ] Add vLLM memory-admission tests proving retained credit is applied
        before startup admission and is not double-counted after acquire.
  - [ ] Port vLLM source bootstrap to the mounted-source artifact contract and
        keep `msa1:`/durable artifact admission explicit.
  - [ ] Keep durable, retained, and local-source startup off TensorDict and full
        Python state-dict paths; direct API startup should attach daemon-owned
        tensors through the selected binding/retained/source-bound path.
  - [ ] Keep `VLLMTensorcastAdapter` as the owner for model construction,
        trace capture, runtime-only tensor rehydration, finalize hooks, and
        semantic probes.
  - [ ] Add internal-vLLM profile/smoke coverage for CUDA allocated/reserved
        deltas, host RSS deltas, selected source kind, attach/finalize timing,
        reload overlap, and old-handle retirement.
  - [ ] Preserve in-place reload response projection, stale/duplicate reload
        handling, after-ready publication, required-publication failure state,
        stale publication retirement, shutdown retirement, EP/EPLB reload
        safety, and drafter sequential failure/unhealthy behavior.
  - [ ] Remove normal internal-vLLM startup, reload, memory-accounting,
        runtime-view, and shutdown imports of public `tensorcast.serving.*`
        APIs after replacement paths pass.

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
        replacements exist and tests cover the replacement behavior.
  - [ ] Move public docs toward artifact/runtime terminology.
  - [ ] Remove broad facades that expose private lifecycle helpers as public API.
  - [ ] Delete compatibility aliases and duplicate helper functions in the same
        cleanup window; do not leave old and new public surfaces as peers.
  - [ ] Rewrite or delete tests that primarily assert the old serving surface
        instead of the artifact-runtime contract.
  - [ ] Close every applicable entry in the deletion ledger below; Phase 6 is
        not complete while any old public serving path remains as a supported
        peer of the artifact-runtime path.

- [ ] Phase 7: Extend To A Second Runtime
  - [ ] Use SGLang or a minimal mock runtime adapter to prove the direct
        artifact-runtime API is not vLLM-shaped.
  - [ ] Confirm source catalog, target layout, runtime-only tensors, and
        publication can be represented without vLLM-specific public names.

# Deletion Ledger

The migration is delete-forward. Each old surface below must either be removed
from the public API or narrowed to an explicitly internal implementation detail
after its replacement is wired.

| Current surface | Replacement owner | Delete/internalize after | Guardrail |
| --- | --- | --- | --- |
| `ServingRuntimeSession` public runtime root | `Artifact.realize(... model_runtime ..., runtime_host=...)` plus completed `ArtifactRealizationHandle.attachment()` | internal-vLLM startup/reload/shutdown and second-runtime fixture use the direct API | import/call-site search shows no normal public startup path instantiates `ServingRuntimeSession`; smoke tests show no extra binding owner or TensorDict intermediate |
| `ServingConfig` as public runtime request | artifact/runtime request DTOs and profile policy fields | loader/source/reload paths parse the new request and preserve behavior | semantic field-map tests cover durable, source, retained, diagnostics, publication, and reload inputs |
| `serving_runtime_policy` on generic realization specs | neutral runtime profile/preflight policy or binding-specific options object | binding/runtime preflight no longer needs serving-rooted field names | spec construction tests use neutral field names; old field rejected or private |
| `ServingArtifactLocator` | artifact locator or artifact selection locator | durable startup and reload resolve through artifact selection | reload/startup tests assert artifact selection digest and no serving locator authority |
| `ServingBindingTarget` and `ServingBindingSetTarget` | `RealizationTarget` and `RealizationTargetSet` | retained prefetch, TP target-set, and direct runtime startup accept replacement targets | target-set tests cover member layout, source reuse, and collective strategy |
| `PrefetchedServingBinding` and `PrefetchedServingBindingSet` | `PrefetchHandoff` or `RetainedRealizationClaim` | vLLM retained acquire and memory accounting use neutral claim helpers | trusted reservation byte tests pass before admission; old public names absent from normal API |
| `RetainedServingBindingAuthority` and retained-serving helpers | retained realization claim parser/validator | acquire validation, lease restore, and reservation credit are represented neutrally | retained startup tests validate member/device/layout at credit and acquire time |
| serving publication helpers used by normal runtime startup | handle/attachment artifact-runtime publication actions | after-ready publication, reload retirement, shutdown retirement, and local-ready promotion use shared actions | generation/CAS tests cover stale result, duplicate reload, required-publication failure, and shutdown |
| `ServingArtifactManifest` for non-ABI metadata | runtime representation manifest or artifact representation metadata | manifest fields are reclassified into representation/runtime profile terms | preflight tests assert schema/build/topology/contract admission without public serving manifest authority |
| `serving_build_digest` when not serving-ABI-specific | `runtime_build_digest` or `representation_build_digest` | build identity is owned by representation/runtime profile | manifest/build tests explain any remaining serving ABI field |
| `ServingRealizationReport` and serving diagnostics aliases | `ArtifactRealizationReport` and target-specific report payloads | direct model-runtime reports match current serving lifecycle reports | diagnostics tests assert one report model and no duplicate path-specific assertions |
| old tests that assert public serving-session behavior | direct artifact-runtime tests plus private lowering tests where needed | replacement tests pass and internal lowerings are covered directly | test inventory has no compatibility-only public serving tests |

# Tasks

- Keep this plan as the post-`0121` migration ledger; do not duplicate the
  completed `0121` kernel checklist.
- Add public SDK exports for the realization spec/handle/report symbols and
  verify `import tensorcast as tc; tc.ArtifactRealizationSpec` works.
- Implement the direct model-runtime API path from `Artifact.realize(...)` to the
  existing runtime attachment lowerings, preserving the current report and
  release-contract behavior.
- Preserve the current optimal weight-loading data paths while changing the API:
  retained acquire, local-replica/LIP, compatible P2P, local mounted-source/disk
  streaming, and explicit transforms must remain distinguishable in reports.
- Define the completed-handle runtime attachment projection API and update tests
  so `attach(...)`, if retained, is not treated as a separate execution step.
- Rename or wrap `IntegrationHost`-style serving host capabilities into the
  artifact-runtime professional API without changing framework behavior.
- Move `serving_runtime_policy` out of the generic realization spec or fence it
  behind a transitional binding/runtime-profile options object with a deletion
  trigger.
- Build a current internal-vLLM import/call-site table showing every remaining
  `ServingRuntimeSession`, `ServingConfig`, retained-binding, publication, and
  runtime-view dependency.
- Create the semantic field map from current `model_loader_extra_config` to
  artifact/runtime terminology.
- Define the retained realization claim public/professional naming path while
  preserving current trusted reservation validation inputs.
- Formalize the existing publication generation/CAS contract shared by
  after-ready publication, reload retirement, and shutdown retirement as the
  vLLM migration acceptance rule.
- Add direct model-runtime realization tests once the public lowering exists.
- Add or update internal-vLLM smoke/integration tests for startup, retained
  memory credit, local source cold start, durable artifact startup, in-place
  reload, after-ready publication, shutdown retirement, EP/EPLB rejection, and
  draft partial-failure/unhealthy behavior.
- Add internal-vLLM profile checks for CUDA allocated/reserved deltas, host RSS,
  selected source kind, copy/temporary/direct-write bytes, attach/finalize
  timing, reload overlap, and old-handle retirement.
- Add direct artifact-runtime second-framework tests or fixtures before retiring
  serving-centered public vocabulary broadly.
- Delete old serving-session public tests, compatibility wrappers, and redundant
  diagnostics assertions after the replacement tests pass; keep only tests that
  exercise internal lowerings still intentionally owned by serving modules.
- Maintain the deletion ledger above as implementation work proceeds; every
  temporary serving compatibility object must have a replacement, owner,
  guardrail, and removal trigger.
- Add an internal-vLLM call-site search check before cleanup completion so
  normal startup, reload, memory accounting, runtime view, and shutdown do not
  import public `tensorcast.serving.*` APIs.

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

- public SDK import/export smoke tests for `tc.ArtifactRealizationSpec`,
  `tc.ArtifactRealizationHandle`, and `tc.ArtifactRealizationReport`;
- direct model-runtime realization tests once `Artifact.realize(model_runtime)`
  no longer fails closed;
- completed-handle runtime attachment projection tests proving no second attach
  execution path exists;
- internal-vLLM smoke/integration tests for startup, reload, publication, and
  retained credit;
- performance migration gates for no TensorDict intermediate, source selection,
  no extra GPU/host full-weight residency, pre-admission retained credit, stage
  timing, and bounded reload overlap;
- direct artifact-runtime second-framework adapter proof before broad serving
  vocabulary retirement;
- cleanup guardrails proving old public serving-session entrypoints,
  compatibility aliases, duplicate diagnostics paths, and redundant tests are
  deleted or internalized after replacement;
- deletion-ledger audit showing each old public serving surface is removed,
  private, or serving-ABI-specific with an owner;
- internal-vLLM import/call-site audit proving normal startup, reload,
  memory-accounting, runtime-view, and shutdown paths use artifact-runtime
  APIs;
- C++ daemon/core tests only when proto, materialization, binding, or P2P
  behavior changes.

- Execute TensorCast and internal-vLLM changes together, because both codebases
  are under our control.
- Recovery is behavior-based and delete-forward: if the new API shape is wrong,
  revise the refactor before landing rather than preserving a parallel
  compatibility layer.

# Risks & Tracking

- Artifact API overgrowth: track whether proposed methods are durable artifact
  lifecycle operations or realization-handle projection/actions.
- TensorDict split-brain: track whether TensorDict tests still exercise a
  separate materialization path that bypasses realization specs, strategy
  selection, diagnostics, or P2P compatibility checks.
- vLLM timing regression: specifically track retained reservation memory credit
  before admission.
- Weight-loading fast-path regression: track whether direct model-runtime
  startup keeps retained acquire, local-replica/LIP, P2P, disk, mounted-source,
  and explicit-transform cases on their intended paths.
- Hidden memory duplication: track GPU allocated/reserved deltas, host RSS,
  live binding owners, reload overlap, and compatibility wrappers that keep old
  handles alive after artifact-runtime attachment.
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
- Dual-stack drift: every compatibility adapter or old public entrypoint must
  have an owner, replacement, test coverage, and deletion trigger. A migration
  phase is not complete while old and new public paths both remain supported.
- Redundant test drift: tests that assert old serving-session behavior must be
  rewritten to assert artifact-runtime behavior or deleted after internal
  lowering coverage exists.
