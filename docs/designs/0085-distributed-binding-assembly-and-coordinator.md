---
slug: distributed-binding-assembly-and-coordinator
title: Distributed Binding Assembly on the Existing Assembly and Layout Trunk
status: proposed
created: 2026-03-14
last_updated: 2026-03-18
areas: ["sdk", "daemon", "core", "proto", "global_store"]
related_code:
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/registration.py
  - tensorcast/api/_register.py
  - tensorcast/types.py
  - proto/tensorcast/layout/v1/layout.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - schema.sql
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/assembly_coordination_utils.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/registration_controller.cc
  - core/store/runtime/metadata/registration_backend.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/materialization/dataplane/view/view_identity.h
  - tensorcast/global_store/services/view_state_service.py
  - tensorcast/global_store/repositories/assembly_layout_binding_repository.py
  - tensorcast/global_store/repositories/artifact_binding_repository.py
  - tensorcast/global_store/repositories/assembly_contribution_repository.py
links:
  plan: ../plans/0085-distributed-binding-assembly-and-coordinator.md
  schema: ../../schema.sql
  successors:
    - ./0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  predecessors:
    - ./0084-binding-unified-model-and-contract.md
    - ./0055-programmable-framework.md
    - ./0078-selection-first-artifact-retrieval.md
    - ./0011-unified-session-lifecycle-leases.md
    - ./0090-existence-semantics-and-single-authority-truth.md
    - ./0094-unified-lifecycle-kernel-and-capability-families.md
    - ./0096-workflow-companion-admission-and-fencing.md
    - ./0100-distributed-authority-handoff-security-and-public-surfaces.md
---

# Summary

Define distributed binding-backed publish by reusing the repository's existing
assembly and layout trunk.

`0085` is the parent thesis only.
It is not the executable attempt-carrier specification.

Its job is to lock the repository-wide invariants that must remain true:

- one structural assembly trunk,
- one frontend-agnostic attempt domain above that trunk,
- one workflow and closeout line above immutable attempt truth,
- and one binding frontend that lowers onto the same trunk instead of inventing
  a second publish implementation.

`0105` owns the executable attempt model.
`0085` owns the parent-level rules that child designs must not violate.

# Goals / Non-Goals

## Goals

- Publish one full distributed model version from many local sealed values.
- Keep `LayoutSpec` as the canonical structural layout and overlap or proof
  contract.
- Reuse the existing structural assembly substrate:
  - `LayoutSpec`
  - one assembly workspace trunk
  - deterministic `view_id`
  - coverage and proof persistence
  - `artifact_bindings`
  - the existing structural seal substrate
- Keep binding-backed contribution as one frontend onto the existing structural
  registration and seal path.
- Keep semantic kernels separated so the same question never has two competing
  authority roots.
- Preserve a clean extension path for planners, future frontend families, and
  richer closeout scopes without creating a second assembly implementation.
- Keep SDK control-path operations daemon-mediated only.

## Non-Goals

- Redefine the local binding contract from `0084`.
- Redefine repository-wide distributed continuation rules from `0100`.
- Redefine lifecycle semantics from `0011` or `0094`.
- Widen `LayoutSpec` with frontend topology, workflow, or closeout metadata.
- Introduce a second persistent assembly implementation.
- Promise `TP > 1` assembly semantics in this phase.

# Problem Statement

The repository already contains the structural substrate we want:

- `LayoutSpec` is content-addressed and layout-scoped,
- `ViewSpec` canonically maps to deterministic `view_id`,
- structural registration persists facts through `view_state_service`,
- the structural seal path already exists,
- and `artifact_bindings` already make sealed lineage authoritative.

The historical inconsistency was not the trunk.
It was the join points around the trunk.

Previous drafts and partial implementations blurred:

1. requirement truth and binding-local lowering shape,
2. requirement identity and readiness policy,
3. attempt identity and structural workspace identity,
4. durable attempt truth and workflow snapshots,
5. workflow observation and workflow transition,
6. source seal completion and final published success,
7. public continuation metadata and bare internal ids.

That blurring is the real design bug.

# Repository Alignment

`0085` must be read under four repository-wide rules that are already owned
elsewhere.

## One authority root per question

`0092` and `0090` established the repository rule that different questions must
have different authority roots.

For this domain:

- structural layout truth belongs to `LayoutSpec`,
- structural target truth belongs to `ViewSpec` plus deterministic `view_id`,
- requirement truth belongs to the canonical requirement kernel,
- readiness-policy truth belongs to a separate readiness-policy kernel,
- closeout-contract truth belongs to a separate closeout-contract kernel,
- immutable per-attempt truth belongs to an immutable attempt record,
- workflow currentness belongs to the coordinator operation runtime,
- live occupancy belongs to durable slot occupancy plus lifecycle-backed
  liveness,
- closeout result truth belongs to the final published lineage.

No row, proto, or helper may silently answer more than one of those questions by
accident.

## Lifecycle is not workflow truth

`0094` and `0011` established that lifecycle protects bounded runtime promises.
It does not own workflow truth.

Therefore:

- leases and keepalives may protect live contributions,
- finalizers may accelerate cleanup,
- but lifecycle state does not replace requirement truth,
- does not replace readiness policy,
- and does not replace coordinator workflow truth.

## Workflow semantics stay above lifecycle

`0096` established that replay, wait, currentness, fencing, and completion are
workflow semantics.

Therefore:

- mutation fences may be derived from live slot occupancy,
- explicit transition APIs advance attempt workflow,
- observation APIs report workflow state,
- and `wait` must not become an implicit transition helper.

## Public continuation follows `0100`

`0100` established that public continuation must converge on `Operation[T]` or
another explicit public family surface.

Therefore:

- assembly attempts may not define a private continuation dialect,
- public reentry must carry durable attempt scope,
- and child designs must not weaken the continuation contract merely because the
  attempt path began life as a local daemon workflow.

# Architectural Thesis

## Hard Parent-Child Split

The repository needs both a parent thesis and an executable hard cut.
They answer different questions.

| Design | Owns | Must not own |
| --- | --- | --- |
| `0085` | one-trunk thesis, semantic-kernel separation, frontend-agnostic lowering rules, parent invariants, high-level success model | concrete proto carriers, public continuation payloads, exact runtime or durable-row schema |
| `0105` | executable attempt carriers, public attempt surface, durable attempt-row direction, readiness-cut shape, continuation metadata | second structural trunk, alternative layout root, alternate lifecycle system |

## One Structural Trunk, Multiple Semantic Kernels

The parent design should be read through the following table.

| Semantic kernel | Carrier family | Authoritative question answered |
| --- | --- | --- |
| Structural layout kernel | `LayoutSpec` | what canonical structure must final bytes satisfy |
| Structural target kernel | `ViewSpec` plus deterministic `view_id` | which structural projection is being addressed |
| Requirement kernel | canonical requirement-set contract | which requirements must be satisfied for this attempt |
| Readiness-policy kernel | explicit readiness-policy contract | what liveness facts must still hold at transition time |
| Closeout-contract kernel | typed closeout contract | what success boundary must final closeout satisfy |
| Attempt instance kernel | immutable attempt record | which immutable attempt instance exists |
| Workflow fence kernel | coordinator operation runtime | who is currently allowed to accept contributions and transition to seal |
| Occupancy kernel | durable slot occupancy plus lifecycle-backed liveness | which contributor currently occupies one required slot |
| Mutation-fence kernel | live occupancies pointing at contributor-value identity | may this local sealed value become mutable again |
| Assembly workspace kernel | structural workspace id | where unsealed structural bytes are accumulated |
| Closeout result kernel | `PublishedModelVersion` lineage | what externally visible result did the attempt produce |

No kernel replaces another.

## Structural Kernel

`LayoutSpec` remains narrow but deep.
It continues to own:

- canonical index binding,
- overlap policy,
- proof policy,
- and reusable structural target semantics.

It must not absorb:

- binding identity,
- slot occupancy,
- attempt workflow state,
- or attempt closeout policy.

## Requirement Kernel

The authoritative attempt root is not `LayoutSpec.expected_view_ids`.
It is the frontend-agnostic requirement kernel that `0105` defines.

Parent-level rules:

- attempt completeness is checked against explicit canonical requirements,
- frontend-specific planner or binding state must canonicalize once into that
  requirement kernel before attempt creation completes,
- `LayoutSpec.expected_view_ids` may seed one frontend bridge but must not remain
  canonical attempt truth after the attempt exists,
- `canonical_full`-style semantics prove that not every requirement is naturally
  a structural `view_id`,
- no controller may rebuild a weaker substitute requirement set from layout hints
  after the attempt exists.

## Binding Frontend Reuses The Structural Registration Trunk

`SealedBindingValue` from `0084` becomes one frontend onto the existing
structural registration trunk.

It must not bypass:

- deterministic structural identity computation,
- coverage validation,
- overlap and proof checks,
- `view_state_service`,
- or the existing structural seal substrate.

The binding bridge may still use binding-local carriers such as:

- contribution kind,
- coverage-plan hash,
- or binding-local slot metadata.

But parent rule:

- those bridge-local carriers are not the canonical attempt kernel.
- and frontend code must not own the final structural lowering for one
  attempt-scoped contribution.

This is how "one trunk" should be interpreted.
The shared trunk is the structural commit and seal substrate, not one overloaded
submit carrier.

## Workflow, Readiness, And Closeout Boundary

Parent-level ordering rule:

1. one immutable attempt record is created before contributors are accepted,
2. contributors may submit only while the attempt is open,
3. explicit transition APIs advance the attempt from open to sealing,
4. the system captures one readiness cut before structural seal begins,
5. structural seal consumes that captured cut,
6. closeout consumes the seal result and produces final published lineage.

That ordering is required even though the concrete runtime and snapshot carriers
are owned by `0105`.

## Pre-Attempt Closeout Profile Boundary

Mutable closeout-profile configuration, if it exists at all, is a pre-attempt
input, not attempt truth.

Parent-level rules:

- mutable profile sources must be keyed by a pre-attempt scope visible before the
  attempt exists,
- post-start workspace identity must not remain the sole lookup key for
  pre-attempt closeout semantics,
- once the attempt exists, the snapped closeout contract is authoritative.

## Parent-Level Public Surface Direction

`0085` does not define concrete public carriers.
That is owned by `0105`.

The parent-level API direction remains:

- callers use an explicit assembly-attempt surface,
- binding-backed contribution stays on `SealedBindingValue`,
- attempt transition is explicit,
- wait and status are observation only,
- final success returns published lineage rather than a bare source artifact,
- and public continuation aligns with `0100`.

# Consistency Model

## Authority Separation

Durable rows and runtime projections may each carry facts from several kernels,
but no single row replaces all of them.

Assembly attempts must keep the following split explicit:

- structural authority
  - `LayoutSpec`
  - structural targets
  - coverage and proof persistence
- requirement authority
  - canonical requirement kernel
- readiness authority
  - readiness-policy contract
- attempt authority
  - immutable attempt record
- workflow authority
  - coordinator operation runtime
- occupancy authority
  - durable slot occupancy plus lifecycle-backed liveness
- closeout authority
  - typed closeout contract plus final published lineage result

## Seal Correctness Is A Conjunction

Seal correctness is not one scalar predicate.
The attempt may progress only when all relevant kernels agree:

- `structural_ready`
- `workflow_current`
- `occupancy_live` when readiness policy requires live contributors
- `closeout_contract_satisfied` before final success is reported

This conjunction is the deepest parent invariant in the design.

## Mutation Fencing

If a `SealedBindingValue` currently occupies one live required slot in an open
attempt, that same binding value must not become mutable again.

Therefore:

- mutation fencing is keyed by contributor-value identity,
- it is derived from live slot occupancy,
- and it must remain coherent with the same liveness model used by readiness
  transition and seal.

## Recovery Boundary

The attempt domain must make its recovery boundary explicit.

Parent-level rule:

- before the readiness cut, live slot occupancy and coordinator currentness are
  still part of correctness,
- after the readiness cut, seal correctness must depend on the captured cut
  rather than on post-cut mutations,
- terminal closeout may still fail,
- but later contributor loss or policy edits must not retroactively change the
  semantic contents of the already-captured cut.

# Invariants And Error Model

## Invariants

- one structural assembly trunk remains in the system,
- `LayoutSpec` remains the single structural layout root,
- binding publish remains one frontend onto that trunk,
- the attempt boundary remains contract-first,
- requirement identity, readiness policy, closeout contract, attempt identity,
  workspace identity, and contributor identity remain distinct,
- one immutable attempt root exists per attempt,
- one publish attempt uses one structural workspace,
- mutable pre-attempt closeout configuration is resolved before contributors are
  accepted,
- only `SealedBindingValue` may contribute binding-backed bytes,
- a live contributor may not reopen for mutation,
- final success means published lineage completion, not source seal alone,
- SDK code never talks to Global Store directly for this workflow.

## Error Model

Parent-level error expectations:

- `INVALID_ARGUMENT`
  - malformed layout or contribution mapping
  - malformed continuation payload
  - invalid readiness or closeout input
- `FAILED_PRECONDITION`
  - requirement mismatch
  - slot already occupied by another live contributor
  - binding no longer has the current sealed value
  - required readiness facts are missing
- `ABORTED`
  - coordinator generation advanced
  - contributor liveness was lost before the readiness cut
  - explicit attempt abort occurred
- `DATA_LOSS`
  - structural bytes or seal evidence are inconsistent after safe recovery is no
    longer possible
- `DEADLINE_EXCEEDED`
  - coordinator wait, seal, or closeout exceeded the budget

The concrete public and proto-facing mapping is owned by `0105`.

# Schema Direction

Parent-level schema rules:

- structural tables remain the structural substrate,
- workflow rows remain workflow substrate,
- slot occupancy must converge on a first-class slot model rather than on a
  `view_id` alias,
- immutable attempt truth must converge on first-class durable attempt rows
  rather than on workflow snapshots,
- and the next evolution step is a first-class attempt domain, not a second
  assembly trunk.

# Alternatives And Rationale

## Collapse Attempt Truth Into Binding-Shaped Contract

Rejected.

Reasons:

- binding publish is only one frontend,
- future frontends need the same canonical attempt kernel,
- and a strong-consistency domain must not make one bridge enum the
  authoritative attempt truth.

## Collapse Attempt Truth Into `LayoutSpec.expected_view_ids`

Rejected.

Reasons:

- layout hints are structural bridge input, not canonical requirement truth,
- richer requirement families need explicit attempt-level semantics,
- completeness must be checked against immutable attempt truth, not a weaker
  reconstruction.

## Treat Structural Identity, Requirement Identity, And Contributor Identity As One Plane

Rejected.

Reasons:

- they answer different questions,
- not every requirement is naturally a structural view,
- and mutation fencing is already keyed by contributor-value identity in `0084`.

## Let Workflow Snapshots Replace Durable Attempt Truth

Rejected.

Reasons:

- workflow is a projection, not semantic truth storage,
- public continuation must project durable truth honestly,
- and a strong-consistency domain cannot keep its only immutable root inside a
  workflow snapshot blob.

## Let `wait` Perform Transition

Rejected.

Reasons:

- `0096` already separates observation from transition,
- hidden transition inside `wait` creates a private workflow dialect,
- and it makes public continuation semantics dishonest.

# Trade-offs And Risks

- **More named domain objects**
  - The domain becomes more explicit.
  - That is less risky than continuing to overload one carrier.
- **Hard-cut discipline**
  - The parent-child split only works if `0085` stops restating executable
    carriers and `0105` becomes the sole source of truth for those carriers.
- **Schema cost**
  - Strong-consistency attempt semantics require more durable structure.
  - That is an intentional cost of making the domain honest.

# Compatibility & Acceptance Criteria

Parent-level acceptance requires:

- distributed publish is implemented on the existing structural assembly and
  layout trunk,
- `0105` is the sole executable carrier specification for the attempt domain,
- canonical requirement truth is frontend-agnostic,
- readiness policy is separate from requirement identity,
- closeout contract is separate from workflow state,
- binding-backed contribution, direct `register_view`, and future frontends all
  lower onto the same structural commit and seal substrate,
- mutable pre-attempt closeout configuration is resolved before contributors are
  accepted,
- attempt transition is explicit,
- observation is side-effect-free,
- public continuation aligns with `0100`,
- published success means the required lineage and closeout facts exist,
- future frontend families can all reuse the same trunk without creating
  separate assembly implementations.

# References

- `docs/designs/0084-binding-unified-model-and-contract.md`
- `docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
- `docs/plans/0085-distributed-binding-assembly-and-coordinator.md`
- `docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
- `docs/designs/0011-unified-session-lifecycle-leases.md`
- `docs/designs/0055-programmable-framework.md`
- `docs/designs/0090-existence-semantics-and-single-authority-truth.md`
- `docs/designs/0092-artifact-profiles-shared-dataplane-and-truth-layering.md`
- `docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md`
- `docs/designs/0096-workflow-companion-admission-and-fencing.md`
- `docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md`
- `docs/architecture/view-replicas-and-assembly.md`
- `docs/architecture/api/api-design.md`
