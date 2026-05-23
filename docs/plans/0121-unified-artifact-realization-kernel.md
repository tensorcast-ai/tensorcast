---
slug: unified-artifact-realization-kernel
title: Unified Artifact Realization Kernel Plan
status: draft
areas: ["sdk", "daemon", "core", "serving", "integrations", "tests", "docs"]
created: 2026-05-23
last_updated: 2026-05-23
related_code:
  - docs/designs/0121-unified-artifact-realization-kernel.md
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0039-artifact-first-sdk.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0115-trusted-disk-source-format-aware-source-handle-and-metadata-first-resolve.md
  - docs/designs/0116-prefetch-serving-binding-target.md
  - docs/designs/0117-group-realization-transaction.md
  - docs/plans/0116-prefetch-serving-binding-target.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/_materialize.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/realization_plan.py
  - tensorcast/api/plan/plan.py
  - tensorcast/serving/local_ready.py
  - tensorcast/serving/runtime_attachment.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/representation_transform_builder.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
links:
  design: ../designs/0121-unified-artifact-realization-kernel.md
  dependencies:
    - ../designs/0120-artifact-centered-model-runtime-realization.md
    - ../designs/0055-programmable-framework.md
    - ../designs/0039-artifact-first-sdk.md
    - ../designs/0108-tensor-aware-materialization-strategy-plane.md
    - ../designs/0115-trusted-disk-source-format-aware-source-handle-and-metadata-first-resolve.md
    - ../designs/0116-prefetch-serving-binding-target.md
    - ../designs/0117-group-realization-transaction.md
---

# Objective

Implement the shared realization kernel required by `0120`: TensorDict
retrieval, caller-tensor writes, binding, retained prefetch, runtime attachment,
publication, and TP target-set startup should all lower through one selection,
target, strategy, representation, lifecycle, resource-envelope, execution, and
report pipeline.

The goal is not source compatibility. The goal is one deeper and simpler kernel
that reaches the target state directly, deletes old parallel paths, and enables
stronger external features.

# Current State & Grounding

The current SDK has parallel realization paths:

- `docs/designs/0039-artifact-first-sdk.md` is the unified public artifact
  entrance. TensorDict methods, binding helpers, and prefetch helpers are
  convenience wrappers below `Artifact`, not independent materialization owners.
- `docs/designs/0116-prefetch-serving-binding-target.md` is superseded as a
  standalone public serving target. Its retained GPU residency, reservation,
  acquire, TTL, and validation semantics are preserved here as retained
  realization lifecycle behavior.
- current serving-runtime code is the behavior baseline for runtime attachment,
  retained acquire, reload, publication, runtime view, and shutdown retirement.
- `tensorcast/api/store/artifact.py`: `Artifact.tensor_dict(...)`,
  `tensor_dict_with_diagnostics(...)`, `tensor_dict_into(...)`, `bind(...)`,
  `bind_into(...)`, `_build_owner_source_selection(...)`,
  `_build_artifact_selection(...)`, and `_build_region_layout_selection(...)`.
- `tensorcast/api/store/materialization.py`: `MaterializationPipeline` owns
  `materialize_subset(...)`, `get_into(...)`, region-backed writes, retry,
  payload release, and payload-to-state-dict conversion.
- `tensorcast/api/_materialize.py`: `materialize_artifact_v2(...)` builds an
  `ArtifactSelection`, calls `MaterializeReplica`, decodes CUDA IPC or CPU memfd
  handles, and returns `MaterializationPayload`.
- `tensorcast/api/store/binding.py` and
  `tensorcast/api/store/owned_binding_slot.py`: binding current/staged values,
  swap/reload, publication, group acquire, and lifecycle checks.
- `tensorcast/api/store/inplace_slot.py` and `tensorcast/api/plan/plan.py`:
  refill, swap, `prefetch_many`, and plan lowering still build or consume
  selection identity outside a shared resolver.
- `tensorcast/api/store/__init__.py`: representation publication layout helpers
  still resolve a Global Store address and open direct Global Store gRPC channels;
  these paths must move behind daemon APIs before the kernel is considered
  authoritative.
- `tensorcast/serving/local_ready.py`: source-to-binding local-ready realization
  and freeze/promotion behavior.
- `PublicDiskSourceHandle` and local-ready mounted-source flows need the `0115`
  interpretation: successful trusted mounted-source resolution becomes an
  `msa1:` artifact subject, not a source-handle-only bypass.
- `Artifact.prefetch(...)` is already an `Operation[T]` surface from `0055` and
  `0116`; the new realization handle must not replace operation status/wait/cancel
  semantics with a second continuation model.
- `proto/tensorcast/daemon/v2/store_daemon.proto`: separate RPC lowerings for
  `MaterializeReplica`, `MaterializeIntoTarget`, `CreateBinding`,
  `CreateOwnedBinding`, `PrefetchServingBinding`, and `PublishTargetReplica`.
- `core/checkpoint/checkpoint.cc`: restored CUDA IPC and CPU memfd tensors
  already carry C++ deleter ownership that closes mappings and releases daemon
  handle-lease tokens when the last tensor owner is dropped.
- `daemon/state/handle_lease_registry.cc`,
  `daemon/state/session_lifecycle.cc`, and
  `daemon/state/lifecycle_kernel.h`: export, placement, retention, and
  publication capability concepts already exist, but are not yet surfaced as
  one realization resource contract.
- `daemon/service/body_backing_types.h` and
  `daemon/service/body_backing_manager.cc`: byte-body backing, retention,
  locality, capability resolution, and observation types provide a daemon-side
  basis for a unified backing model.
- `core/store/runtime/ingestion/materialization_strategy_types.h`: execution
  strategy and commit-report types already track lanes, residuals, fallback,
  and cost estimates that should feed the unified realization report.

The active risk is split-brain:

- multiple selection builders can diverge;
- source policy and fallback can diverge across TensorDict and binding;
- TensorDict diagnostics and binding diagnostics are not one operator model;
- publishability and retained lifecycle are attached to binding paths but not
  represented as general realization capabilities;
- resource ownership is path-specific: TensorDict payloads, `get_into`
  temporary payloads, region-backed direct writes, owned/adopted bindings,
  retained claims, and runtime attachments each encode backing/export/release
  rules in different places;
- TP/group realization can drift into serving-specific orchestration rather
  than target-set realization.

Risk closure is part of the implementation, not a separate review checklist.
Every migrated path must close its applicable risks through one of these shared
mechanisms: admission gate, `RealizationResourceEnvelope` adapter, unified
report field, or deletion guardrail. A path that calls a common helper but still
keeps private cleanup, fallback, mutability, or authority policy is not migrated.

# Phases & Milestones

These phases are workstreams, not a strict execution order. Implementers may
advance, split, merge, or reorder TODOs when the repository state makes that the
better path. The hard rule is semantic closure: a migrated path must have its
selection identity, resource envelope, lifecycle release model, strategy
fallback, report fields, and deletion guardrails in place.

- [ ] Phase 1: Contract Inventory And Semantic Freeze
  - [ ] Build a contract table for TensorDict, TensorDict-into, bind,
        bind-into, prefetch-device, prefetch-target, runtime attach, reload,
        publication, local-ready promotion, mounted-source bootstrap, and TP
        startup.
  - [ ] Record current behavior for CPU vs CUDA, ephemeral vs retained,
        publishable vs non-publishable, source policy, fallback, diagnostics,
        cleanup, operation ids, and direct control-plane dependencies.
  - [ ] Build a resource-envelope matrix for every path with normalized
        `backing_kind`, `export_kind`, `projection_kind`, `owner_kind`,
        `release_policy`, `mutability_contract`, planned cost fields, and
        deterministic token-backed export behavior.
  - [ ] Build a risk-closure matrix mapping each design risk to an admission
        field, envelope field, report field, guardrail test, and blocking
        condition.
        Risks without a closure mechanism block migration of the affected path.
  - [ ] Record current lifecycle release triggers separately for export release,
        backing release, and workflow release.
  - [ ] Record current copy/reference behavior: Python dict projection,
        CUDA IPC mapping, CPU memfd mmap/private mapping, region-backed direct
        write, temporary-payload fallback, and binding copy/fill plans.
  - [ ] Import the non-obsolete `0116` TODOs as retained realization behavior:
        reservation bytes, acquire validation, TTL/idle retirement, metrics,
        status/debug visibility, and transform-required fail-closed behavior.
  - [ ] Pin `0055` operation semantics for prefetch and async realization:
        `Operation[T]`, deterministic idempotency, scoped cancel, `NO_LEASE`,
        and degraded/timeout behavior.
  - [ ] Pin `0115` mounted-source semantics: successful trusted source resolve
        produces `msa1:` artifact identity and no source-handle-only execution
        bypass.
  - [ ] Pin `0117` target-set semantics: `same_selection` and
        `per_part_selection`, staged values, publish barriers, and group-aware
        acquire.
  - [ ] Add or update tests that pin current behavior before refactoring.

- [ ] Phase 2: Control-Plane Authority Cleanup
  - [ ] Inventory every SDK direct Global Store import/channel/RPC used by
        realization, publication, layout provisioning, key resolution, operation
        observation, and group admission.
  - [ ] Add daemon API coverage or daemon-client wrappers for layout
        provisioning and representation publication helpers currently using
        Global Store directly.
  - [ ] Migrate SDK publication/layout helpers to daemon-mediated APIs.
  - [ ] Add static or unit guardrails that fail on direct Global Store access in
        SDK realization paths.
  - [ ] Keep runtime orchestration Global Store startup/connect logic out of this
        ban; the ban is for SDK artifact metadata and realization authority.

- [ ] Phase 3: Canonical Selection Resolver
  - [ ] Introduce `ResolvedArtifactSelection` and a single resolver in the SDK.
  - [ ] Move artifact id/key resolution, canonical index fetch, artifact profile
        and authority scope, view/subset preparation, view index bytes, view id,
        view subset hash, logical layout hash, and generation hint handling into
        the resolver.
  - [ ] Admit `msa1:` mounted-source artifacts through the same resolver while
        preserving same-daemon, non-routable authority.
  - [ ] Migrate `tensor_dict`, `tensor_dict_into`, `bind`, `bind_into`,
        `prefetch`, retained/runtime paths, `Plan` prefetch-set lowering,
        inplace refill, owned-binding refill, and local-ready source views to use
        the resolver.
  - [ ] Add a guardrail test that fails when SDK realization paths build
        `ArtifactSelection` outside the resolver.
  - [ ] Add tests for canonical, subset, view, mapped-source, `msa1:`, and
        group-member source selection identity.

- [ ] Phase 4: Target, Strategy, Representation, Lifecycle, And Resource Plans
  - [ ] Define internal DTOs for `RealizationTargetPlan`,
        `RealizationStrategyPlan`, `RepresentationAdmissionPlan`, and
        `RealizationLifecyclePlan`.
  - [ ] Define internal `RealizationResourceEnvelope` fields shared by every
        target: backing, export, projection, owner, release policy,
        mutability contract, release strictness, and cost model.
  - [ ] Define the common risk-relevant admission fields:
        `authority_evidence`, `source_selection_digest`, `target_layout_digest`,
        `copy_plan_digest`, `fallback_policy`, `release_strictness`,
        `export_lifetime_kind`, and `mutability_contract`. For process-visible
        CUDA IPC and CPU memfd exports, `export_lifetime_kind` must be
        token-backed; mint failure is a hard error.
  - [ ] Represent existing targets as target kinds: `tensor_dict`,
        `caller_tensors`, `binding_owned`, `binding_adopted`,
        `retained_replica`, `runtime_attachment`, and `target_set`.
  - [ ] Separate source selection digest from target layout digest, mapped view
        id, binding layout id, and copy-plan digest.
  - [ ] Move source policy, P2P/disk preference, verification, retry, deadline,
        region-backed fallback, wait-for-shared-disk, lease/export policy, and
        collective policy into `RealizationStrategyPlan`.
  - [ ] Carry `0108` lane allocation and residual fallback accounting, or the
        lowered daemon/core equivalent, when tensor-aware strategy is involved.
  - [ ] Move representation/layout/schema/member compatibility checks into
        `RepresentationAdmissionPlan`.
  - [ ] Model publishability, retained claims, current/staged binding values,
        borrowed caller tensors, TensorDict payload ownership, and ephemeral
        TensorDict as lifecycle capabilities.
  - [ ] Map daemon `BodyBacking*`, `HandleLeaseRegistry`,
        `SessionLifecycleManager`, `LifecycleKernel`, and core
        `ExecutionCommitReport` concepts into the envelope instead of adding
        path-specific resource managers.
  - [ ] Reject executable plans that are missing risk-relevant fields required
        by their target kind or strategy. Missing fields fail admission rather
        than selecting a broad fallback.

- [ ] Phase 5: ArtifactRealizationSpec, Handle Facade, And Projection Lifetime
  - [ ] Introduce `ArtifactRealizationSpec` constructors for TensorDict,
        caller tensors, binding, adopted binding, retained replica, retained
        binding, mounted-source realization, model runtime, and target set.
  - [ ] Introduce `Artifact.realize(spec=..., ctx=...)` for completed
        realization and `Artifact.realize_async(...)` only where operation
        semantics are required.
  - [ ] Introduce `ArtifactRealizationHandle` projections/actions:
        `tensor_dict()`, `binding()`, `attach(...)`, `prefetch_handoff()`,
        `publish_replica(...)`, `promote(...)`, and `report`.
  - [ ] Implement TensorDict projection ownership so projected tensors cannot
        outlive the payload lease/export owner unsafely.
  - [ ] Implement shared projection-owner adapters so raw tensor escape, handle
        close, binding close, runtime detach, and retained acquire close all use
        the envelope release contract.
  - [ ] Enforce token-backed export release policies. If CUDA IPC or CPU memfd
        handle-lease minting fails, fail before returning tensors. Do not add a
        PID-bound export fallback.
  - [ ] Classify CPU TensorDict as read-only/read-mostly private mapping and
        reject TensorDict write semantics.
  - [ ] Delete or rewrite existing `Artifact.tensor_dict(...)`,
        `tensor_dict_into(...)`, `bind(...)`, and `bind_into(...)` entry points
        to target-state realization APIs. Do not keep independent compatibility
        wrappers.
  - [ ] Keep `Artifact.prefetch(...)` as `Operation[T]` and lower it through
        async realization without changing public wait/cancel/status semantics.
  - [ ] Keep the `0039` SDK target surface aligned so convenience methods are
        documented and tested as realization-handle wrappers below `Artifact`.
  - [ ] Ensure unsupported handle actions fail with clear
        `FAILED_PRECONDITION` messages.

- [ ] Phase 6: Unified Report And Diagnostics
  - [ ] Define `ArtifactRealizationReport`.
  - [ ] Map current `MaterializationDiagnostics` fields into the report.
  - [ ] Map binding materialization diagnostics, execution diagnostics,
        binding value ids, publication eligibility, and retained acquire facts
        into the report.
  - [ ] Add source selection digest, target layout digest, copy-plan digest,
        operation id/backend, artifact profile/authority scope, and
        publishability facts.
  - [ ] Add resource-envelope fields: backing kind, export kind, projection
        kind, owner kind, release policy, release strictness, export lifetime
        kind, mutability contract, retained bytes, and temporary-replica bytes.
  - [ ] Add cost fields: direct-write bytes, copy bytes, copy count, mmap bytes,
        CUDA IPC open count, CPU memfd fd count, and fallback reason buckets.
  - [ ] Include `ExecutionCommitReport` facts where available: lane allocation,
        committed ranges, residual fallback ranges, actual executor path, and
        reject buckets.
  - [ ] Add risk-closure labels to report diagnostics for authority, identity,
        lifecycle, lease strength, mutability, hidden movement cost, async
        continuation, and target-set/group behavior. These labels should be
        derived from the plan, not hand-written per target.
  - [ ] Emit consistent profile events for all realization targets.
  - [ ] Update tests to assert comparable report fields across TensorDict,
        binding, retained handoff, runtime attach, and target-set paths.

- [ ] Phase 7: TP As Target-Set Realization
  - [ ] Define target-set plan shape with group selection plan, shared
        group/version-set context, shared strategy, and member-local targets.
  - [ ] Support `same_selection` and `per_part_selection` as first-class target-set
        selection modes.
  - [ ] Map TP rank/member, target layout, device UUID, runtime profile, semantic
        placement digests, and per-member source selection into member target
        plans.
  - [ ] Map same-node source coordination, collective-first loading, group
        barriers, staged values, publish barrier, and acquire claims into
        strategy/lifecycle plans.
  - [ ] Produce one group realization report with per-member diagnostics.
  - [ ] Prevent TP code from adding a TP-only materialization path.

- [ ] Phase 8: Daemon And Core Controller Convergence
  - [ ] Add shared daemon-controller planning structures that mirror the SDK
        realization plan where appropriate.
  - [ ] Lower daemon controller resource decisions through existing
        `BodyBackingManager`, `HandleLeaseRegistry`, `SessionLifecycleManager`,
        `LifecycleKernel`, and core strategy/report structures rather than
        adding target-specific resource managers.
  - [ ] Route existing RPC handlers through common target-plan,
        representation-admission, strategy, and lifecycle helpers.
  - [ ] Keep protocol messages stable until the shared controller path is
        proven.
  - [ ] Consider proto cleanup only after SDK and daemon controllers share one
        plan model.

- [ ] Phase 9: Delete Split-Brain Paths And Enforce Guardrails
  - [ ] Remove or narrow old path-specific selection helpers.
  - [ ] Remove path-specific fallback behavior that is not represented in
        `RealizationStrategyPlan`.
  - [ ] Remove path-specific cleanup and cost-report behavior that is not
        represented in `RealizationResourceEnvelope`.
  - [ ] Remove duplicate diagnostics construction once reports cover all paths.
  - [ ] Keep guardrails for direct selection construction, direct Global Store
        access, unmanaged source-handle bypasses, second operation-continuation
        models, and TP-only materialization bypasses.
  - [ ] Update README/design references after code convergence lands.

# Tasks

- [ ] Add SDK realization package/module ownership for the kernel, distinct from
      the current binding-only `tensorcast/api/store/realization_plan.py` copy/fill
      helper.
- [ ] Add `RealizationResourceEnvelope` DTOs and adapters so TensorDict,
      caller-tensor direct write, caller-tensor copy fallback, owned binding,
      adopted binding, retained prefetch, runtime attachment, and target-set
      member realization all expose the same backing/export/projection/owner/
      release/cost fields.
- [ ] Add an adapter matrix from existing implementation objects
      (`MaterializationPayload`, region-backed `MaterializeIntoTarget`,
      `OwnedBindingSlot`, retained acquire state, runtime attachment state,
      group realization state) into `RealizationResourceEnvelope`.
- [ ] Add a risk-closure matrix fixture or generated table that ties every
      design risk to the concrete admission field, envelope adapter, report
      assertion, and deletion guardrail that closes it.
- [ ] Add plan-admission tests that reject execution when risk-relevant fields
      are absent: source authority, target layout digest for mapped targets,
      fallback policy for optional fallback, release strictness for exports, and
      mutability contract for CPU TensorDict.
- [ ] Add export-lease failure tests that force CUDA IPC and CPU memfd
      handle-lease mint failures and assert hard realization errors before any
      tensor projection is returned.
- [ ] Add daemon-mediated layout provisioning and representation publication
      helpers, then remove direct `GlobalStoreCompositeStub` usage from SDK
      realization/publication helpers.
- [ ] Add a guardrail test for forbidden direct Global Store access in SDK
      artifact metadata, realization, publication, operation, and group
      authority paths.
- [ ] Add focused unit tests for `ResolvedArtifactSelection`, including
      canonical id/key, subset, view, mapped source view, generation hint,
      artifact profile, authority scope, and digest stability.
- [ ] Migrate `artifact.py`, `materialization.py`, `plan.py`,
      `inplace_slot.py`, and `owned_binding_slot.py` selection construction to
      the shared resolver, then add a guardrail for direct
      `build_artifact_selection(...)` calls outside resolver/adapters.
- [ ] Add integration tests covering TensorDict and Binding source-selection
      equivalence while asserting separate target layout and copy-plan digests
      for mapped/adopted targets.
- [ ] Add `msa1:` mounted-source tests for same-daemon realization, rejected
      durable GS routing/key activation, explicit promotion to `mi2:`, and
      local-ready pending-verification admission.
- [ ] Add TensorDict projection lifetime tests proving wrapper-returned tensors
      cannot outlive the payload lease/export owner unsafely and daemon
      materialized payloads are released after projection/handle close.
- [ ] Add lifecycle release tests that separately assert export release, backing
      release, and workflow release for TensorDict, temporary copy fallback,
      owned binding, retained acquire, and runtime attachment.
- [ ] Add handle-lease failure tests proving CUDA IPC and CPU memfd exports fail
      closed when token-backed lease minting fails.
- [ ] Add CPU TensorDict mutability tests that document read-only/read-mostly
      private mapping behavior and reject TensorDict write semantics.
- [ ] Add `get_into` cost tests proving region-backed direct write reports
      direct-write bytes and temporary-payload fallback reports source export,
      copy bytes, temp bytes, unload, and fallback reason.
- [ ] Add prefetch and async realization tests proving `Operation[T]` status,
      wait, cancel, deterministic idempotency, `NO_LEASE`, degraded, and timeout
      behavior are preserved.
- [ ] Add retained prefetch/report tests with source selection digest, target
      layout digest, copy-plan digest, operation id/backend, and lifecycle
      capability fields.
- [ ] Add retained acquire capability, TTL, idle-retire, and status/debug tests
      from the retired `0116` plan where still relevant.
- [ ] Add strategy/report tests for `0108` lane allocation, committed ranges,
      residual fallback ranges, executor path, and reject buckets when
      tensor-aware strategy is involved.
- [ ] Add TP target-set planning tests with real CUDA coverage in this
      environment, using fake CUDA only for narrow unit cases that do not need
      IPC/device behavior.
- [ ] Add TP target-set tests for `same_selection` and `per_part_selection`,
      staged values, publish barriers, group acquire claims, and group reports
      with per-member diagnostics.
- [ ] Add daemon-controller tests for common target-plan lowering.
- [ ] Update vLLM integration tests or scenario fixtures once runtime attach
      lowers through the kernel.
- [ ] Update docs that mention serving-specific target names after replacement
      APIs exist.

# Test / Rollout / Backout

Required Python test command for implementation work:

```bash
source .venv/bin/activate
pytest tests/python/...
```

Required C++ test command pattern for daemon/core implementation work:

```bash
bazel test //core/component:xxx_test
```

Targeted acceptance tests to add or update:

- risk-closure tests proving every migrated path has the required admission
  gate, envelope adapter, report assertion, or deletion guardrail for its
  applicable risk classes;
- control-plane authority tests proving SDK realization, publication, layout,
  key, operation, and group paths do not open direct Global Store channels;
- selection identity tests for TensorDict, binding, bind-into, prefetch,
  retained/runtime attach, `Plan.prefetch_many`, inplace refill, owned-binding
  refill, local-ready source views, and target-set inputs;
- resource-envelope tests asserting normalized backing, export, projection,
  owner, release, mutability, and cost fields for TensorDict, caller-tensor
  direct write, caller-tensor copy fallback, owned binding, adopted binding,
  retained prefetch, runtime attachment, and target-set member paths;
- lifecycle tests that separate export release, backing release, and workflow
  release and prove raw returned tensors never point at freed backing;
- lease-policy tests for CUDA IPC and CPU memfd export showing token-backed
  deterministic release and hard failure on mint errors, with no silent
  fallback;
- TensorDict behavior tests for CPU, CUDA, subset, view, mapped, region-backed
  optional/required modes, and projection lifetime cleanup;
- CPU TensorDict mutability tests for read-only/read-mostly private mapping and
  rejection of TensorDict write semantics;
- `get_into` movement-cost tests for region-backed direct write versus
  temporary-payload copy fallback;
- target layout tests proving mapped/adopted layout identity and copy-plan
  digest can vary without changing source selection identity;
- mounted-source tests proving `msa1:` same-daemon realization, GS-routing
  rejection, explicit durable promotion, and local-ready verification policy;
- binding lifecycle tests for current value, staged value, publication token,
  dirty rejection, publishability, and daemon-issued authority tokens;
- retained handoff tests for reservation bytes, acquire validation, `NO_LEASE`,
  TTL/idle retirement, operation semantics, and report fields;
- strategy/report tests for planned fallback lanes, committed ranges, residual
  fallback, executor path, and reject buckets;
- runtime attachment tests for adapter attach, reload admission, shutdown
  retirement, and publication;
- TP target-set tests for `same_selection`, `per_part_selection`, member layout,
  collective strategy, group failure, staged/publish barriers, acquire claims,
  and per-member diagnostics.

Execution should prefer semantic closure over fixed phase order. The following
sequence is a useful default, but implementers may reorder or combine steps when
that removes split-brain faster without weakening risk closure:

1. land contract inventory tests and semantic freeze fixtures;
2. land daemon-mediated control-plane helpers and direct-GS guardrails;
3. land selection resolver with no behavior change;
4. land target, strategy, representation, lifecycle, and resource-envelope DTOs
   behind existing methods;
5. land `Artifact.realize(...)`, `Artifact.realize_async(...)`, and the completed
   handle facade;
6. switch TensorDict/binding/caller-tensor wrappers to the facade;
7. switch retained, local-ready, runtime attach, and `msa1:` mounted-source
   paths;
8. switch TP/group realization to target-set planning;
9. converge daemon/core controllers, then delete duplicate logic and keep
   guardrails.

Backout should happen at coherent semantic boundaries, not by preserving old and
new implementations indefinitely. Once a target-state path is proven, delete or
rewrite the old independent path instead of maintaining compatibility code.

# Risks & Tracking

- [ ] Selection resolver becomes too broad.
      Track by keeping artifact identity, view/subset, and index facts in the
      resolver while target layout remains in target planning.
- [ ] SDK direct Global Store access survives behind helper APIs.
      Track by blocking Phase 3 until publication, layout provisioning, key,
      operation, and group authority calls have daemon-mediated replacements and
      guardrail coverage.
- [ ] `PublicDiskSourceHandle` becomes a permanent source authority.
      Track by requiring every successful trusted mounted-source path to carry
      an `msa1:` identity, source format facts, generation, and daemon policy
      evidence into `ResolvedArtifactSelection`.
- [ ] Mapped target layout is confused with source selection.
      Track by asserting separate source selection digest, target layout digest,
      and copy-plan digest for mapped/adopted cases.
- [ ] TensorDict accidentally inherits binding lifecycle.
      Track by capability tests proving TensorDict handles cannot publish,
      swap, or retain beyond their lease.
- [ ] TensorDict projections release daemon payloads too early or leak them.
      Track by tests covering wrapper-returned tensors, explicit handle close,
      projection close, and daemon release calls.
- [ ] Resource lifecycle remains path-specific under a unified API.
      Track by requiring every existing path to emit a
      `RealizationResourceEnvelope` before execution and by deleting cleanup or
      cost-report code that cannot be explained through envelope fields.
- [ ] Handle-lease mint failure silently weakens export lifetime.
      Track by deterministic-release tests that force CUDA IPC and CPU memfd
      mint failures and assert hard errors before tensor projection.
- [ ] CPU TensorDict mutability stays ambiguous.
      Track by classifying CPU TensorDict as read-only/read-mostly private
      mapping and by rejecting TensorDict write semantics.
- [ ] `get_into` hides expensive fallback copies.
      Track by strategy/report tests that compare region-backed direct write
      with temporary-payload fallback and assert direct-write bytes, copy bytes,
      temporary bytes, unload, and fallback reason fields.
- [ ] Prefetch grows a second continuation model.
      Track by preserving `Operation[T]` for public async realization and
      prefetch status/wait/cancel/idempotency semantics.
- [ ] Binding paths bypass strategy planning.
      Track by requiring every daemon materialization call to receive a
      `RealizationStrategyPlan` or its lowered equivalent.
- [ ] Tensor-aware strategy loses lane/residual visibility.
      Track by requiring `ExecutionCommitReport` fields in reports whenever the
      lower strategy plane uses mixed execution or fallback.
- [ ] TP grows special-case orchestration.
      Track by requiring TP additions to be target-set fields, strategy policy,
      or lifecycle state.
- [ ] RPC cleanup is attempted too early.
      Track by forbidding protocol unification until SDK and daemon-controller
      planning are already shared.
- [ ] Target-state behavior regresses while compatibility code is deleted.
      Track by scenario tests for vLLM startup, reload, publication, retained
      acquire, local-ready promotion, and shutdown retirement, with real CUDA
      coverage for IPC/runtime-sensitive cases.
