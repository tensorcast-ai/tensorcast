---
slug: artifact-representation-contract-and-transform-unification
title: Artifact Representation Semantic Core and Transform Unification Plan
status: draft
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests"]
created: 2026-03-24
last_updated: 2026-03-25
related_code:
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/designs/0058-communicator-topology-model.md
  - docs/architecture/api/materialization-flow.md
  - docs/internals/disk-load-strategy.md
  - docs/internals/model-loading.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/mapped_binding.py
  - tensorcast/types.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/representation_layout_types.h
  - daemon/service/controllers/representation_transform_builder.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
links:
  design: ../designs/0110-artifact-representation-contract-and-transform-unification.md
---

# Objective

Hard-cut TensorCast onto one semantic-core representation-contract architecture
 so the repository stops carrying parallel semantic systems for:

- mapped-target semantic planning,
- executor-local tensor-job recovery,
- builder-side private trace semantics,
- and ordinary tensor-aware materialization lowering.

The target outcome of this plan is narrower than the earlier draft:

- one resolved `RepresentationTransformContract`,
- one common strategy-plane lowering path,
- one executor-neutral work IR,
- explicit cleanup of `MappedCopyContract` and executor-local semantic recovery,
- an explicit source-to-serving builder boundary,
- explicit deferral of topology-scoped reshard executor work and broad durable
  artifact-catalog expansion to follow-on designs.

# Current State & Grounding

Current code and docs already contain the pieces of the target architecture, but
they are split across incompatible seams.

## Existing grounding that stays valid

- `0055` already defines `TransformSpec` and names reshard as a compute
  transform rather than a pure view:
  [0055-programmable-framework.md](/data/workspace/tensorcast-280/docs/designs/0055-programmable-framework.md#L1134)
- `0078` already fixes `ArtifactSelection` as the only public selection
  contract.
- `0084` already keeps binding, artifact, and assembly planes separate.
- `0105` already provides the right externally visible lineage carrier:
  `PublishedModelVersion`.
- `0108` already establishes the semantic truth -> source binding -> strategy
  plane split:
  [0108-tensor-aware-materialization-strategy-plane.md](/data/workspace/tensorcast-280/docs/designs/0108-tensor-aware-materialization-strategy-plane.md#L294)
- `0109` already states that owner-file collective is execution strategy, not
  semantic truth:
  [0109-batched-owner-file-collective-executor.md](/data/workspace/tensorcast-280/docs/designs/0109-batched-owner-file-collective-executor.md#L367)
- `0058` already separates topology reachability and routing from semantic
  intent:
  [0058-communicator-topology-model.md](/data/workspace/tensorcast-280/docs/designs/0058-communicator-topology-model.md#L22)

## Current duplication that must still be removed

- architecture docs still describe mapped-target semantic truth in mapped-only
  terms:
  [materialization-flow.md](/data/workspace/tensorcast-280/docs/architecture/api/materialization-flow.md#L282)
- `internal-vllm` still carries its own representation planning concepts and
  hash semantics instead of lowering into a TensorCast-defined semantic core.

## Explicit deferrals in this plan

This plan does **not** implement:

- topology-scoped participant planning for TP4<->TP8 group execution,
- communicator or routing-aware reshard executor behavior,
- durable artifact-catalog schema expansion for every representation family.

Those remain follow-on design work. This plan exists to make that future work
consume one semantic core instead of inheriting today's duplicated semantic
stacks.

# Latest Status

## 2026-03-25

Implemented in this change:

- added common semantic-core types in
  `core/store/materialization/contracts/representation_contract.{h,cc}`
- added normalization, validation, `tensor_schema_hash`, and
  `representation_contract_hash` helpers
- extended `ResolvedMaterializationPlan` to carry:
  - `fallback_map`
  - `representation_transform_contract`
- added a controller-side `build_representation_transform_contract(...)` path
  and switched mapped-target planning to emit the new contract
- switched mapped binding and mapped-target execution plan publication to the
  shared contract/work-plan path
- removed `MappedCopyContract` and common `TensorJobDistribution` from common
  runtime strategy types
- refactored mapped collective execution to consume
  `RepresentationWorkPlan` items instead of `tensor_job_candidates` /
  `concat_job_candidates`
- refactored ordinary local-batched and collective replica disk loading so
  `build_tensor_jobs(...)` now lowers executor-local `TensorJob` objects from a
  participant `RepresentationWorkPlan` instead of acting as the semantic source
  of truth
- made `MaterializationFacade::materialize_into_target(...)` invoke the shared
  index-backed representation/work-plan builder for ordinary non-transform
  loads, while still executing the generic byte-range data path
- made ordinary `materialize_into_target(...)` route eligible disk-backed
  collective requests through the same owner-file collective executor path used
  by mapped-target loads
- made `MaterializationFacade` derive ordinary and mapped-target work plans
  through the same `build_representation_work_plan(...)` seam
- deleted `materialization_mapped_copy_plan_utils.*` and rehomed the surviving
  neutral layout helpers in
  `daemon/service/controllers/representation_layout_types.h`
- moved ordinary replica collective/local-batched work-plan derivation up into
  `core/store/replica/replica.cc` so `collective_disk_loader.cc` now consumes
  prebuilt `RepresentationWorkPlan` inputs only
- refactored source-bound binding create/refill lowering in
  `daemon/service/controllers/owned_binding_service.cc` so startup/refill paths
  reuse one shared semantic-core preparation flow before execution
- updated `0108`, `0109`, `materialization-flow`, `disk-load-strategy`, and
  `model-loading` docs so they describe `RepresentationTransformContract` /
  `RepresentationWorkPlan` as the converged semantic/runtime seam and make
  explicit that runtime execution must not invent implicit fallback work
- added unit coverage for representation-contract normalization and work-plan
  derivation

Not completed in this change:

- full repository-wide end-to-end Bazel verification has not been run; targeted
  isolated-output-base verification completed for:
  - `//daemon:materialization_target_plan_utils_lib`
  - `//daemon:target_materialization_service_lib`
  - `//daemon:owned_binding_service_test`
  - `//core/store/replica:replica`
  - `//core/store/materialization/contracts:representation_contract_test`
  - `//core/store/replica:collective_disk_loader_test`
  - `//core/store/runtime/ingestion:materialization_facade_test`

# Phases & Milestones

- [ ] Phase 0: Freeze Terminology And Design Ownership
  - [ ] Milestone 0.1: Land the rewritten `0110` design and companion plan.
  - [ ] Milestone 0.2: Update `0055`, `0108`, `0105`, and `0109` so accepted
    docs describe the same semantic-core boundary.
  - [ ] Milestone 0.3: Mark `MappedCopyContract` and executor-local semantic
    recovery as target-state removals in the updated docs.

- [ ] Phase 1: Define The Semantic Core Contract Family
  - [x] Milestone 1.1: Add core semantic contract types:
    - `RepresentationDescriptor`
    - `BindingOpKind`
    - `RepresentationTensorBinding`
    - `RepresentationTransformContract`
  - [x] Milestone 1.2: Extend `ResolvedMaterializationPlan` to carry the new
    contract as the only transform-semantic truth in the phase-1 scope.
  - [ ] Milestone 1.3: Define canonical normalization and hashing for:
    - exact copy
    - slice copy
    - concat
    - scalar fill-from-source
    - const fill
    - residual fallback accounting
  - [ ] Milestone 1.4: Define logical-topology participation in
    `representation_contract_hash` while explicitly excluding physical topology
    planning from the semantic contract.

- [ ] Phase 2: Replace Mapped-Only Semantic Builders
  - [x] Milestone 2.1: Replace
    `daemon/service/controllers/materialization_mapped_copy_plan_utils.*` with a
    common `representation_transform_builder` family.
  - [x] Milestone 2.2: Make mapped binding, mapped-target materialization, and
    source-bind startup lowering all use the same builder output.
  - [x] Milestone 2.3: Remove semantic `tensor_job_candidates` and
    `concat_job_candidates` from common runtime contracts; keep any equivalent
    structures executor-private only.

- [ ] Phase 3: Unify `0108` Strategy Lowering
  - [x] Milestone 3.1: Introduce one executor-neutral work IR derived from the
    new representation contract plus source binding.
  - [x] Milestone 3.2: Refactor `MaterializationFacade` to lower both ordinary
    tensor-aware loads and mapped-target loads through the same work-plan
    generator.
  - [x] Milestone 3.3: Make local tensor-aware execution consume the shared work
    IR instead of re-parsing source and view indexes.
  - [x] Milestone 3.4: Make owner-file collective consume the shared work IR and
    remove remaining bespoke semantic recovery from `0109`-owned paths.

- [ ] Phase 4: Hard-Cut Executor Local Semantic Recovery
  - [x] Milestone 4.1: Delete or collapse
    `core/store/replica/collective_disk_loader.cc::build_tensor_jobs(...)` as
    semantic truth.
  - [x] Milestone 4.2: Remove `TensorJobDistribution` from common semantic
    contracts; if an executor still needs a local enum, keep it executor-local.
  - [x] Milestone 4.3: Ensure runtime lowering flows share one authoritative
    semantic-to-work seam and executors only consume prebuilt work plans.

- [ ] Phase 5: Builder Boundary And Integration Convergence
  - [ ] Milestone 5.1: Document source artifact versus serving artifact as
    sibling representations linked by the common semantic core.
  - [ ] Milestone 5.2: Define the TensorCast-side producer boundary expected by
    `internal-vllm` so `TracePlan` lowers into the common semantic family rather
    than a private runtime truth.
  - [ ] Milestone 5.3: Keep externally visible publish lineage anchored in
    `0105` closeout and `PublishedModelVersion` rather than re-inventing a
    second publication truth inside materialization docs.

- [ ] Phase 6: Cleanup And Concept Deletion
  - [x] Milestone 6.1: Remove `MappedCopyContract` from
    `materialization_strategy_types.h`.
  - [x] Milestone 6.2: Delete compatibility-only names, builder shims, and
    duplicated planner utilities that were only needed by the old mapped path.
  - [ ] Milestone 6.3: Simplify `0108` and `0109` docs to describe the converged
    semantic-core model only.
  - [ ] Milestone 6.4: Update SDK and internals docs so old semantic terms are
    no longer presented as current truth.

# Tasks

## Code tasks by module

- [x] Add `core/store/materialization/contracts/representation_contract.h` and
  related normalization or validation utilities.
- [x] Update
  `core/store/runtime/ingestion/materialization_strategy_types.h` to replace:
  - `MappedCopyContract`
  - common semantic `TensorJobDistribution`
  with the new semantic-core contract types.
- [ ] Add a common representation-transform builder in daemon or core runtime
  and delete mapped-only builder ownership from the old mapped helper family.
- [x] Refactor
  [materialization_target_plan_utils.cc](/data/workspace/tensorcast-280/daemon/service/controllers/materialization_target_plan_utils.cc)
  to emit the new contract for mapped-target and target-layout paths.
- [x] Refactor
  [materialization_facade.cc](/data/workspace/tensorcast-280/core/store/runtime/ingestion/materialization_facade.cc)
  to derive one shared work IR.
- [x] Refactor
  [collective_disk_loader.cc](/data/workspace/tensorcast-280/core/store/replica/collective_disk_loader.cc)
  so executors consume lowered work items rather than recovering semantic truth.
- [ ] Audit binding and mapped-binding helpers under `tensorcast/api/store/` so
  public mapping surfaces lower to the new contract.

## Documentation tasks

- [ ] Update `docs/designs/0055-programmable-framework.md`
  - `TransformSpec` remains the public request
  - `RepresentationTransformContract` becomes resolved semantic truth
  - topology-scoped reshard execution remains follow-on work
- [x] Update `docs/designs/0108-tensor-aware-materialization-strategy-plane.md`
  - remove `MappedCopyContract` as the long-term common semantic family
  - rewrite executor sections around shared work IR
  - keep group-scoped topology execution explicitly out of the first strategy
    convergence wave
- [ ] Update `docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
  - keep `PublishedModelVersion` as the publication-lineage carrier
  - clarify that future serving publication consumes the semantic core but is
    not owned by materialization runtime contracts
- [x] Update `docs/designs/0109-batched-owner-file-collective-executor.md`
  - owner-file collective consumes shared work-plan output
  - owner-file collective is not the future topology-scoped reshard executor
- [x] Update `docs/architecture/api/materialization-flow.md`
  - ordinary load and mapped-target both use the same semantic contract family
- [x] Update `docs/internals/disk-load-strategy.md`
  - remove semantic reliance on executor-local tensor-job recovery
- [x] Update `docs/internals/model-loading.md`
  - describe source-to-serving build and runtime serving representation boundary
- [x] Update `docs/README.md`
  - add discoverability links for the new design and plan

## External integration coordination tasks

- [ ] Coordinate follow-up work in `/data/workspace/internal-vllm`:
  - lower `TracePlan.copy_plan` into TensorCast representation contracts
  - align `representation_contract_hash` semantics to the TensorCast-defined
    normalized contract inputs
  - keep source-bind startup as a producer or builder path
  - keep steady-state runtime on serving-artifact bind or swap

## Deferred follow-on design tasks

- [ ] Write a dedicated design for topology-scoped representation transform
  execution:
  - group participant model
  - communicator/routing integration
  - reshard executor commit and failure semantics
- [ ] Write a dedicated design for durable representation publication or catalog
  metadata when the repository is ready to persist those fields beyond
  `PublishedModelVersion` lineage.

# Test / Rollout / Backout

## Test plan

- [ ] Add C++ unit coverage for representation-contract normalization and
  validation:
  - multi-range source and destination
  - concat and split
  - scalar fill and const fill
  - residual coverage accounting
- [ ] Add C++ strategy-plane tests proving:
  - ordinary tensor-aware and mapped-target requests lower through the same
    semantic family
  - local tensor-aware executor and owner-file collective both consume the same
    work IR
  - generic fallback remains byte-exact for residual work
- [ ] Extend Python binding and mapped-binding tests to verify:
  - mapped copy plans lower to the new contract
  - no public semantic regression in `bind` and `bind_into`
- [ ] Re-run existing `0108` and `0109` regression suites after the hard cut:
  - `bazel test //core/store/runtime/ingestion:materialization_facade_test`
  - `bazel test //core/store/runtime/ingestion:materialization_service_test`
  - `bazel test //daemon:materialize_into_mapped_target_test`
  - relevant `collective_disk_loader` tests
- [ ] Run benchmark or regression evidence for:
  - host-local disk subset loads
  - shared-FS owner-file collective loads
  - mapped-target workloads

## Rollout

- [ ] Use a hard cut behind the branch, not a dual semantic stack.
- [ ] Do not introduce a second public checkpoint shard object or selector
  model.
- [ ] Land the semantic core first, then delete old semantic builders before
  enabling broader builder or topology follow-on work.
- [ ] Do not preserve `MappedCopyContract` as a long-lived compatibility shell.
- [ ] Do not preserve executor-local semantic reconstruction after the common
  lowering path is available.

## Backout

- [ ] Backout is repository-level revert before the hard cut merges.
- [ ] Do not add runtime compatibility toggles or dual planner modes as a
  backout mechanism.
- [ ] If a phase proves under-specified, revise `0110` design or this plan and
  continue from the converged direction rather than reviving deprecated
  concepts.

# Risks & Tracking

- [ ] Risk: the contract becomes too abstract and stalls code motion.
  - Mitigation: land it by replacing concrete duplicated builders immediately,
    not as a speculative type layer.
- [ ] Risk: source-to-serving builder and publication docs drift away from the
  semantic core.
  - Mitigation: keep `0105` and external integration guidance explicitly linked
    to this semantic family.
- [ ] Risk: topology-scoped reshard support is over-scoped too early.
  - Mitigation: explicitly defer participant and communicator planning to a
    follow-on design after semantic-core convergence.
- [ ] Risk: accepted docs drift during the multi-phase cleanup.
  - Mitigation: Phase 0 is mandatory and must land before semantic deletion
    starts.
