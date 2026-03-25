---
slug: semantic-core-builder-publication-program
title: Semantic Core, Builder, and Publication Program Plan
status: draft
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests"]
created: 2026-03-25
last_updated: 2026-03-25
related_code:
  - docs/designs/0110-artifact-representation-contract-and-transform-unification.md
  - docs/designs/0111-source-to-serving-builder-and-representation-publication.md
  - docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0058-communicator-topology-model.md
  - docs/internals/model-loading.md
  - docs/architecture/api/materialization-flow.md
  - tensorcast/types.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/replica.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/representation_layout_types.h
  - daemon/service/controllers/representation_transform_builder.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
links:
  design: ../designs/0110-artifact-representation-contract-and-transform-unification.md
---

# Objective

Provide one high-level execution program for the representation-contract work so
the repository lands it in dependency order instead of mixing semantic-core,
builder-publication, and future topology-executor scopes.

This program treats the work as three stacked tracks:

- semantic core
  - `0110`
  - normalized transform semantics
  - strategy-plane convergence
- builder and publication bridge
  - `0111`
  - serving manifest
  - `serving_build_digest`
  - typed `RepresentationPublishContract`
  - `representation_publish`
  - `PublishedModelVersion` serving lineage
- future topology execution
  - separate follow-on design
  - group-scoped reshard executor
  - communicator or routing-aware participant planning

The execution rule is:

- freeze semantic-core truth first,
- then standardize builder or publication carriers,
- then land `PURE_TRANSFORM`,
- then bootstrap builder,
- then `BINDING_FINALIZE`,
- and only after that start topology-scoped execution design.

# Latest Status

## 2026-03-25

Semantic-core implementation advanced in-repo:

- landed `RepresentationDescriptor`, `RepresentationTensorBinding`,
  `RepresentationTransformContract`, and `RepresentationWorkPlan`
- landed normalization plus `tensor_schema_hash` /
  `representation_contract_hash` helpers
- migrated mapped-target plan building and mapped binding runtime preparation to
  the shared semantic family
- removed `MappedCopyContract` / common `TensorJobDistribution` from common
  runtime strategy types
- switched mapped collective execution to consume shared work-plan items
- switched ordinary local-batched and collective replica disk loading to derive
  executor-local tensor jobs from participant `RepresentationWorkPlan` objects
  instead of reparsing source/view semantics directly
- made `MaterializationFacade::materialize_into_target(...)` call the shared
  index-backed work-plan builder for ordinary non-transform loads, while
  leaving the ordinary execution path on the generic byte-range backend
- made ordinary `materialize_into_target(...)` route eligible collective disk
  loads through the shared owner-file collective executor path
- made `MaterializationFacade` lower ordinary and mapped-target requests through
  the same work-plan generator seam
- deleted `materialization_mapped_copy_plan_utils.*`, moved neutral layout
  helpers into `representation_layout_types.h`, and made replica-side
  collective/local executors consume prebuilt work plans only
- refactored source-bound binding startup/refill in
  `owned_binding_service.cc` so it uses one shared semantic-core preparation
  flow before materialization
- updated `0108`, `0109`, `materialization-flow`, `disk-load-strategy`, and
  `model-loading` docs so they no longer describe mapped-only semantic truth as
  the current architecture and instead require explicit residual accounting with
  no implicit runtime fallback widening

Still pending before Phase 5:

- builder / publication phases remain blocked; this change does not start
  `0111`, `PURE_TRANSFORM`, or `representation_publish`
- only targeted isolated-output-base verification has run so far; this change
  has not yet been validated with a broad repository-wide Bazel sweep

# Current State & Grounding

## Repository facts that now anchor the program

- `0110` now owns the semantic transform core and explicitly defers publication
  lineage and topology-scoped execution:
  [0110-artifact-representation-contract-and-transform-unification.md](/data/workspace/tensorcast-280/docs/designs/0110-artifact-representation-contract-and-transform-unification.md#L264)
- `0111` now owns the builder and publication bridge:
  [0111-source-to-serving-builder-and-representation-publication.md](/data/workspace/tensorcast-280/docs/designs/0111-source-to-serving-builder-and-representation-publication.md#L217)
- `0105` remains the only publication-lineage trunk:
  [0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md](/data/workspace/tensorcast-280/docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md#L317)
- `0108` still carries prototype semantic contracts and needs convergence onto
  the `0110` semantic family:
  [0108-tensor-aware-materialization-strategy-plane.md](/data/workspace/tensorcast-280/docs/designs/0108-tensor-aware-materialization-strategy-plane.md#L337)
- `0109` should consume shared work IR but is not the future topology-scoped
  reshard executor:
  [0109-batched-owner-file-collective-executor.md](/data/workspace/tensorcast-280/docs/designs/0109-batched-owner-file-collective-executor.md#L367)
- `0055` already provides the public transform request family:
  [0055-programmable-framework.md](/data/workspace/tensorcast-280/docs/designs/0055-programmable-framework.md#L1134)

## Current code and integration gaps

- serving publication proto fields exist but are not yet dependency-ready in the
  current wave:
  [store_daemon.proto](/data/workspace/tensorcast-280/proto/tensorcast/daemon/v2/store_daemon.proto#L1943)
- current serving-facing closeout code still rejects serving publication inputs:
  [assembly_coordination_utils.cc](/data/workspace/tensorcast-280/daemon/service/controllers/assembly_coordination_utils.cc#L189)
- `/data/workspace/internal-vllm` still carries private serving-manifest and
  builder/publication identity logic

## Program constraints

- do not start builder-publication implementation before semantic normalization
  and hash inputs are stable enough to consume
- do not start `BINDING_FINALIZE` before manifest, closeout, and runtime
  preflight contracts are typed
- do not start topology-scoped TP4<->TP8 executor work inside this program
- do not require broad durable artifact-catalog schema changes inside this
  program

# Execution Order

1. Freeze documentation, terminology, and ownership boundaries.
2. Land `0110` semantic-core types, normalization, and hashing.
3. Migrate current producers onto the `0110` semantic family.
4. Converge `0108` and `0109` on shared work IR and delete executor-local
   semantic recovery.
5. Freeze `0111` manifest, `serving_build_digest`, and
   `RepresentationPublishContract` semantics.
6. Land `PURE_TRANSFORM` serving publication first.
7. Support node-local bootstrap builder as a temporary bridge.
8. Land `BINDING_FINALIZE` only behind explicit admission and validation gates.
9. Converge admitted integrations on serving-artifact bind or swap.
10. Start a separate design for topology-scoped reshard execution.

Repository rule:

- later phases may refine earlier ones,
- but they must not bypass earlier phase exit gates,
- and they must not revive deprecated semantic islands to move faster locally.

# Phase Gates

Before Phase 2 can start:

- `RepresentationTransformContract` shape is frozen enough for consumers,
- normalization rules exist for copy, slice, concat, scalar-fill, const-fill,
  and residual fallback,
- `representation_contract_hash` inputs are explicitly defined and scoped to the
  semantic core only.

Before Phase 5 can start:

- semantic-core producer migration is far enough along that builder-publication
  can rely on one canonical semantic family,
- `MappedCopyContract` is no longer treated as the long-term semantic truth.

Before Phase 8 can start:

- serving manifest carrier and runtime preflight are typed,
- `representation_publish` closeout semantics are implementation-ready,
- `PURE_TRANSFORM` publication path has passed end-to-end tests.

# Phases & Milestones

- [ ] Phase 0: Freeze Terminology, Ownership, And Deferrals
  - [ ] Milestone 0.1: Land and align `0110`, `0111`, `0105`, `0108`, `0109`,
    and `0055` docs.
  - [ ] Milestone 0.2: Freeze the meaning of:
    - `RepresentationTransformContract`
    - `representation_contract_hash`
    - `serving_build_digest`
    - `serving_manifest_ref`
    - `RepresentationPublishContract`
    - `PublishedModelVersion` serving fields
  - [ ] Milestone 0.3: Mark topology-scoped TP4<->TP8 executor work and broad
    durable artifact-catalog expansion as explicit follow-on scopes.

- [ ] Phase 1: Land The `0110` Semantic Core
  - [x] Milestone 1.1: Define:
    - `RepresentationDescriptor`
    - `RepresentationTensorBinding`
    - `RepresentationTransformContract`
    - normalization and validation helpers
  - [x] Milestone 1.2: Freeze canonical `representation_contract_hash` inputs.
  - [ ] Milestone 1.3: Add semantic-core tests for normalization, coverage, and
    stable hashing.
  - [ ] Milestone 1.4: Keep all builder-publication work blocked on semantic-core
    freeze.

- [ ] Phase 2: Migrate Producers To The Semantic Core
  - [x] Milestone 2.1: Make mapped binding lower to the `0110` semantic family.
  - [x] Milestone 2.2: Make mapped-target materialization lower to the same
    semantic family.
  - [x] Milestone 2.3: Make source-bind startup lowering consume the same
    semantic family.
  - [x] Milestone 2.4: Remove semantic executor shapes from shared contracts.

- [ ] Phase 3: Converge `0108` Strategy Lowering
  - [x] Milestone 3.1: Introduce one executor-neutral work IR.
  - [x] Milestone 3.2: Refactor `MaterializationFacade` to own semantic-to-work
    lowering for both ordinary tensor-aware and mapped-target requests.
  - [x] Milestone 3.3: Make local tensor-aware execution consume shared work IR.
  - [x] Milestone 3.4: Make owner-file collective consume shared work IR.

- [ ] Phase 4: Delete Executor-Local Semantic Recovery
  - [x] Milestone 4.1: Delete or collapse
    `collective_disk_loader.cc::build_tensor_jobs(...)` as semantic truth.
  - [x] Milestone 4.2: Remove `MappedCopyContract` as long-term semantic truth.
  - [x] Milestone 4.3: Ensure common runtime has one authoritative semantic path.

- [ ] Phase 5: Freeze `0111` Publication Inputs
  - [ ] Milestone 5.1: Define `ServingBuildIntent`.
  - [ ] Milestone 5.2: Define `ServingArtifactManifest` and the reserved manifest
    carrier tensor.
  - [ ] Milestone 5.3: Define `serving_build_digest` as a builder/publication
    digest distinct from `representation_contract_hash`.
  - [ ] Milestone 5.4: Make typed `RepresentationPublishContract` closeout
    semantics precise.
  - [ ] Milestone 5.5: Make `PublishedModelVersion` serving lineage semantics
    precise.
  - [ ] Milestone 5.6: Add runtime preflight contract for serving artifacts.

- [ ] Phase 6: Land `PURE_TRANSFORM` Publication First
  - [ ] Milestone 6.1: Support source-to-serving builds whose canonical bytes are
    fully described by `0110`.
  - [ ] Milestone 6.2: Register or promote serving artifacts plus validated
    manifests.
  - [ ] Milestone 6.3: Publish source and serving lineage through `0105`.
  - [ ] Milestone 6.4: Bind or swap the resulting serving artifact in runtime for
    admitted families.

- [ ] Phase 7: Support Node-Local Bootstrap Builder
  - [ ] Milestone 7.1: Define node-local bootstrap builder workflow as an
    explicit migration or bootstrap mode.
  - [ ] Milestone 7.2: Require bootstrap completion to produce a serving artifact
    before steady-state switch.
  - [ ] Milestone 7.3: Document source-bind bootstrap and steady-state serving
    bind or swap as different modes with different guarantees.

- [ ] Phase 8: Add `BINDING_FINALIZE` Under Admission Gates
  - [ ] Milestone 8.1: Define the minimal framework facts consumed from the
    shared Torch layer.
  - [ ] Milestone 8.2: Define `FinalizeClass` and `ServingSupportLevel`.
  - [ ] Milestone 8.3: Enforce runtime-only finalize invariants.
  - [ ] Milestone 8.4: Add semantic validation gates before publication or
    steady-state admission.
  - [ ] Milestone 8.5: Admit only explicitly validated families.

- [ ] Phase 9: Integration Convergence
  - [ ] Milestone 9.1: Move `internal-vllm` off private serving-manifest JSON.
  - [ ] Milestone 9.2: Move `internal-vllm` off private builder/publication
    identity computation and onto `representation_contract_hash` plus
    `serving_build_digest`.
  - [ ] Milestone 9.3: Converge admitted families on serving-artifact startup and
    reload through bind or swap.
  - [ ] Milestone 9.4: Keep source-bind bootstrap only as an explicit migration
    or bootstrap mode.

- [ ] Phase 10: Kick Off Follow-On Design Work
  - [ ] Milestone 10.1: Start a dedicated design for topology-scoped
    representation transform execution.
  - [ ] Milestone 10.2: Start a separate design for durable artifact-catalog
    metadata if the repository still needs it after serving publication is
    stable.

# Tasks

## Documentation workstream

- [ ] Keep `0110`, `0111`, `0105`, `0108`, `0109`, and `0055` aligned whenever
  execution intent changes
- [ ] Update `docs/internals/model-loading.md`
- [ ] Update `docs/architecture/api/materialization-flow.md`
- [ ] Update integration-facing docs in `/data/workspace/internal-vllm`

## Semantic-core code workstream

- [ ] Implement `0110` semantic-core contract types
- [ ] Implement normalization and hash helpers
- [ ] Migrate current producers to the semantic core
- [ ] Remove executor-local semantic truth

## Builder-publication code workstream

- [ ] Implement manifest carriers and validation
- [ ] Implement `representation_publish` closeout path
- [ ] Implement runtime preflight
- [ ] Implement `PURE_TRANSFORM`
- [ ] Implement bootstrap builder
- [ ] Later implement `BINDING_FINALIZE`

## Integration workstream

- [ ] Coordinate adapter changes with `/data/workspace/internal-vllm`
- [ ] Move manifest ownership into TensorCast
- [ ] Move hash ownership into TensorCast
- [ ] Roll families through explicit support levels

# Test / Rollout / Backout

## Test plan

- [ ] Run semantic-core unit tests before builder-publication work
- [ ] Run strategy-lowering and executor-convergence tests before deleting old
  semantic recovery
- [ ] Run manifest, closeout, and runtime-preflight tests before admitting any
  serving publication
- [ ] Run end-to-end `PURE_TRANSFORM` publication tests before bootstrap-builder
  rollout
- [ ] Run `BINDING_FINALIZE` admission tests before allowing those families into
  steady-state serving bind or swap

## Rollout

- [ ] Do not overlap semantic-core deletion with unresolved manifest or hash
  semantics
- [ ] Do not overlap `BINDING_FINALIZE` rollout with unresolved runtime-only
  finalize classification
- [ ] Do not begin topology-scoped executor work until the semantic core and
  builder-publication bridge are stable enough to consume
- [ ] Prefer freezing interfaces before broadening family coverage

## Backout

- [ ] Back out by reverting incomplete phase work at the branch level
- [ ] Do not add dual semantic stacks, dual manifest formats, or dual publication
  truth as a compatibility shell
- [ ] If a later phase proves under-specified, stop at the previous phase's exit
  gate rather than weakening invariants

# Risks & Tracking

- [ ] Risk: phases get executed out of order and the builder bridge hardcodes
  unstable semantic-core behavior.
  - Mitigation: enforce phase gates and keep `0111` blocked on semantic-core
    freeze.
- [ ] Risk: `PURE_TRANSFORM` and `BINDING_FINALIZE` get mixed too early.
  - Mitigation: always land `PURE_TRANSFORM` first and keep
    `BINDING_FINALIZE` behind admission.
- [ ] Risk: bootstrap builder becomes a permanent second runtime truth.
  - Mitigation: treat it as an explicit migration mode only and track runtime
    convergence on serving bind or swap.
- [ ] Risk: topology-scoped reshard work restarts private semantic ownership.
  - Mitigation: require future executor design to consume `0110` contracts and
    `0111` publication semantics rather than redefining them.
