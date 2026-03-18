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

Define distributed publish for binding-backed training by reusing the
repository's existing assembly and layout trunk.

`0085` now has one job only:

- define the parent thesis for the assembly-attempt domain,
- lock the repository-wide invariants that must remain true,
- and define how binding-backed publish fits onto the existing structural trunk.

`0085` no longer tries to be the executable carrier specification for the
attempt domain.
That executable responsibility now belongs to `0105`.

The long-term rule set is:

- TensorCast has one structural assembly trunk.
- TensorCast does not have one universal identity plane for all attempt
  semantics.
- Frontend-specific planner or topology languages must canonicalize into one
  explicit attempt contract before attempt creation completes.
- Distributed binding publish is one frontend onto the existing structural
  trunk, not a second assembly implementation.

# Status Update

`0085` is the parent design for:

- one structural assembly trunk,
- one layout contract trunk,
- one frontend-agnostic structural commit path,
- and the semantic separation between structural truth, attempt truth,
  liveness, workflow fencing, and closeout truth.

`0105` is the active child design for:

- `AssemblyAttemptSpec`,
- `AssemblyAttemptRuntime`,
- `SealReadinessSnapshot`,
- `slot_key`,
- snapshotted closeout-policy authority,
- and the public continuation contract for assembly attempts.

Normative ownership split:

- `0085` owns the parent invariants and the one-trunk integration model.
- `0105` owns the executable carrier model and the hard cut at the join points.

# Goals / Non-Goals

## Goals

- Publish one full distributed model version from many local sealed values.
- Keep `LayoutSpec` as the canonical structural layout and overlap or proof
  contract.
- Reuse the existing structural assembly substrate:
  - `assembly_id`
  - `LayoutSpec`
  - `assembly_layout_bindings`
  - `ViewSpec` and deterministic `view_id`
  - coverage and proof persistence
  - `artifact_bindings`
  - `StartSealAssembly`
- Keep binding-backed contribution as one frontend onto the existing structural
  registration and seal path.
- Keep the semantic kernels separated so the same question never has two
  competing authority roots.
- Preserve a clean extension path for `PP`, `EP`, `canonical_full`, future
  byte-in frontends, and later richer contract families.
- Keep SDK control-path operations daemon-mediated only.
- Keep final success defined by published lineage, not by source seal alone.

## Non-Goals

- Redefine the local binding contract from `0084`.
- Redefine repository-wide distributed continuation rules from `0100`.
- Redefine lifecycle semantics from `0011` or `0094`.
- Widen `LayoutSpec` with frontend topology or ownership metadata.
- Introduce a second persistent assembly implementation.
- Introduce a training-only layout plane.
- Promise `TP > 1` assembly semantics in phase 1.

# Problem Statement

The repository already contains the structural substrate we want:

- `LayoutSpec` is content-addressed and layout-scoped.
- `ViewSpec` canonically maps to deterministic `view_id`.
- view registration already persists structural facts through
  `view_state_service`.
- `assembly_id` already names an unsealed structural workspace.
- `artifact_bindings` already make `assembly_id -> mi2` the post-seal authority.

The historical inconsistency was not the trunk.
It was the join points around the trunk.

Previous drafts and partial implementations blurred:

1. immutable attempt truth and mutable seal-time truth,
2. structural projection identity and required-slot identity,
3. durable occupancy projection and lifecycle authority,
4. source seal completion and final published success,
5. public continuation metadata and bare internal workflow ids,
6. layout-scoped structural hints and attempt-contract authority,
7. pre-attempt policy scope and post-start workspace identity.

That blurring is the real design bug.

# Repository Alignment

`0085` must now be read under four repository-wide rules that are already owned
elsewhere.

## One authority root per question

`0090` established the repository rule that different questions must have
different authority roots.
Assembly attempts must follow the same rule.

For this domain:

- structural layout truth belongs to `LayoutSpec`,
- structural projection truth belongs to `ViewSpec` plus `view_id`,
- attempt contract truth belongs to the immutable attempt spec,
- workflow currentness belongs to the coordinator operation runtime,
- live occupancy belongs to the durable slot projection plus lifecycle-backed
  liveness,
- closeout truth belongs to the final attempt result.

No row, proto, or helper may silently answer more than one of those questions
by accident.

## Lifecycle is not workflow truth

`0094` and `0011` established that lifecycle protects a bounded runtime promise.
It does not own workflow truth.

Therefore:

- leases and keepalives may protect live contributions,
- finalizers may accelerate cleanup,
- but lifecycle state does not replace attempt contract truth,
- and lifecycle state does not replace coordinator workflow truth.

## Workflow semantics stay above lifecycle

`0096` established that replay, currentness, fencing, wait, and completion are
workflow semantics.
Assembly attempts must follow the same split.

Therefore:

- mutation fences may be derived from live slot occupancy,
- but whether an attempt is still open, sealing, failed, or complete is a
  workflow question,
- and whether final success may be reported is a closeout-workflow question.

## Public continuation follows `0100`

`0100` established that public continuation must converge on `Operation[T]` or
another explicit public family surface.
Bare internal attachment carriers and bare string ids are not enough.

Therefore:

- assembly attempts may not define a private continuation dialect,
- public reentry must carry enough metadata for honest recovery,
- and child designs must not weaken the repository's continuation contract just
  because the attempt path started life as a local daemon workflow.

# Architectural Thesis

## Hard Parent-Child Split

The repository needs both a parent thesis and an executable hard cut.
They answer different questions.

| Design | Owns | Must not own |
| --- | --- | --- |
| `0085` | one-trunk thesis, semantic kernel separation, frontend-agnostic structural lowering, parent invariants, high-level success model | concrete public carriers, operation snapshot shape, public continuation payloads, exact slot-key or proto fields |
| `0105` | immutable attempt spec, runtime projection, readiness cut, public slot identity, continuation metadata, closeout-policy hard cut | second structural trunk, alternative layout root, alternate lifecycle system |

## One Structural Trunk, Multiple Semantic Kernels

The design should be read through the following table.

| Semantic kernel | Carrier family | Authoritative question answered |
| --- | --- | --- |
| Structural layout kernel | `LayoutSpec` | what canonical structure must final bytes satisfy |
| Structural projection kernel | `ViewSpec` plus deterministic `view_id` | which structural projection is being registered |
| Attempt contract kernel | immutable attempt spec | which required slots must be satisfied for this attempt |
| Workflow fence kernel | coordinator operation runtime | who is currently allowed to accept contributions and transition to seal |
| Occupancy kernel | `assembly_contributions` plus lifecycle-backed liveness | which contributor currently occupies one required slot |
| Mutation-fence kernel | live occupancies that point at `(binding_id, binding_value_id)` | may this local binding value become mutable again |
| Assembly workspace kernel | `assembly_id` | where unsealed structural bytes are accumulated |
| Closeout kernel | `PublishedModelVersion` lineage | what externally visible result did the attempt produce |

No kernel replaces another.

## Structural Kernel

`LayoutSpec` remains narrow but deep.
It continues to own:

- canonical index binding,
- overlap policy,
- proof policy,
- and optional layout-scoped expected piece projections.

It must not absorb:

- local binding identity,
- contributor liveness state,
- attempt-specific workflow state,
- or serving closeout policy for one specific attempt.

`ViewSpec` plus deterministic `view_id` remain the reusable structural
projection identity.
They answer the structural question only.

## Attempt Contract Kernel

The authoritative attempt root is not `LayoutSpec.expected_view_ids`.
It is the immutable attempt specification defined by `0105`.

Within that immutable attempt spec, the contract kernel remains the explicit
`ContributionContractSnapshot`.

Parent-level rules:

- attempt completeness is checked against explicit required entries,
- frontend-specific planner or topology state must canonicalize once into those
  explicit required entries before attempt creation completes,
- `LayoutSpec.expected_view_ids` is only the layout-scoped seed for the
  disjoint `piece_partial` family,
- `canonical_full` is a legal contract family member and must not be disguised
  as a full-coverage piece,
- and no controller may rebuild a weaker substitute contract from layout hints
  after the attempt exists.

## Attempt Entry Boundary

The attempt domain is contract-first at its entry boundary.

Parent-level rules:

- a frontend may present either explicit required-slot data or a
  frontend-specific template that canonicalizes to it,
- `layout_id` alone is acceptable only as a phase-1 shorthand for the
  layout-seeded `piece_partial` family,
- once the attempt exists, the canonical contract is the only authoritative
  required-slot answer.

## Slot, Occupancy, And Contributor Identity

The design must distinguish at least three related but non-equal identities:

1. structural projection identity
   - usually `view_id`
   - answers which structural bytes exist
2. required-slot identity
   - answers which attempt-contract entry must be satisfied
3. contributor-value identity
   - `(binding_id, binding_value_id)`
   - answers which sealed local value currently occupies one required slot

Phase 1 may still reuse existing carriers, but only as an explicit alias.
No design or implementation may describe those identities as one universal
plane.

## Binding Contribution Reuses The Structural Registration Trunk

`SealedBindingValue` from `0084` becomes one frontend onto the existing
structural registration trunk.

It must not bypass:

- deterministic structural identity computation,
- coverage validation,
- overlap and proof checks,
- `view_state_service`,
- or the existing assembly seal flow.

The lowering rules are:

- `piece_partial` lowers onto the structural piece or view commit path,
- `canonical_full` lowers onto the canonical structural registration path,
- both use the same trunk-level validation and sealing semantics,
- neither creates a second publish path.

This is how "one trunk" should be interpreted.
The shared trunk is the structural commit and seal substrate, not one
overloaded submit carrier.

## Workflow, Readiness, And Closeout Boundary

Parent-level ordering rule:

1. one fresh `assembly_id` names one attempt workspace,
2. contributors may submit only while the attempt is open,
3. the system must capture one readiness cut before structural seal begins,
4. structural seal consumes that readiness cut,
5. closeout consumes the seal result and produces final published lineage.

That ordering is required even though the concrete runtime and snapshot carriers
are owned by `0105`.

## Pre-Attempt Policy Source Boundary

Mutable closeout-policy configuration is a pre-attempt input, not attempt
truth.

Parent-level rules:

- mutable policy sources must be keyed by a pre-attempt scope visible before the
  attempt workspace exists,
- post-start `assembly_id` may identify the frozen attempt snapshot but must not
  remain the sole mutable lookup key for pre-attempt policy resolution,
- once the attempt exists, the snapshotted closeout policy is authoritative.

## Parent-Level Public Surface Direction

`0085` no longer defines concrete public carriers.
That is now owned by `0105`.

The parent-level API direction remains:

- callers use an explicit assembly-attempt surface,
- binding-backed contribution stays on `SealedBindingValue`,
- wait returns a published lineage result rather than a bare source artifact,
- and public continuation must align with `0100`.

## Naming Compliance

`0085` no longer introduces executable API carriers of its own.
Concrete naming compliance for public and proto-facing attempt carriers now
lives in `0105`.

# Consistency Model

## Authority Separation

Durable rows and runtime projections may each carry facts from several kernels,
but no single row replaces all of them.

Assembly attempts must keep the following split explicit:

- structural authority
  - `LayoutSpec`
  - `views`
  - coverage and proof persistence
- attempt authority
  - immutable attempt spec
- workflow authority
  - coordinator operation runtime
- occupancy authority
  - durable slot projection plus lifecycle-backed liveness
- closeout authority
  - final published lineage

## Seal Correctness Is A Conjunction

Seal correctness is not one scalar predicate.
The attempt may progress only when all relevant kernels agree:

- `structural_ready`
- `workflow_current`
- `occupancy_live` when the contract requires live contributors
- `closeout_policy_satisfied` before final success is reported

This conjunction is the deepest parent invariant in the design.

## Mutation Fencing

If a `SealedBindingValue` currently occupies one live required slot in an open
attempt, that same binding value must not become mutable again.

Therefore:

- mutation fencing is keyed by contributor-value identity,
- it is derived from live slot occupancy,
- and it must remain coherent with the same liveness model used by seal.

## Recovery Boundary

The attempt domain must make its recovery boundary explicit.

Parent-level rule:

- before the readiness cut, live slot occupancy and coordinator currentness are
  still part of correctness,
- after the readiness cut, seal correctness must depend on the captured cut
  rather than on post-cut mutations,
- terminal closeout may still fail,
- but later contributor loss or policy edits must not retroactively change the
  semantic contents of the already-captured readiness cut.

# Invariants and Error Model

## Invariants

- one structural assembly trunk remains in the system,
- one layout contract trunk remains in the system,
- `LayoutSpec` remains the single structural layout root,
- `LayoutSpec.expected_view_ids` remains only the layout-scoped seed for the
  disjoint `piece_partial` family,
- the attempt boundary remains contract-first even if a phase-1 layout-seeded
  shorthand remains,
- one immutable attempt root exists per attempt,
- slot identity, structural identity, and contributor identity remain distinct,
- one publish attempt uses one fresh `assembly_id`,
- mutable closeout-policy lookup uses pre-attempt scope and is snapshotted
  before contributors are accepted,
- only `SealedBindingValue` may contribute binding-backed bytes,
- a live contributor may not reopen for mutation,
- final success means published lineage completion, not source seal alone,
- SDK code never talks to Global Store directly for this workflow.

## Error Model

Parent-level error expectations:

- `INVALID_ARGUMENT`
  - malformed layout or contribution mapping
  - malformed continuation payload
  - invalid closeout policy input
- `FAILED_PRECONDITION`
  - attempt contract mismatch
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

# Schema Changes

`0085` still reuses the existing structural and occupancy tables.

Parent-level schema rules:

- `layout_specs`, `assembly_layout_bindings`, `views`,
  `view_coverage_ranges`, `artifact_bindings`, and `operations` remain the
  structural and workflow substrate,
- `assembly_contributions` remains the durable occupancy projection,
- phase-1 carrier reuse is acceptable only if code and docs remain explicit
  about any storage alias,
- and the next evolution step is a first-class slot representation, not a
  second assembly trunk.

# Alternatives and Rationale

## Collapse The Attempt Contract Into `LayoutSpec.expected_view_ids`

Rejected.

Reasons:

- `expected_view_ids` is only the layout-scoped seed for one contract family,
- `canonical_full` and later richer families require explicit attempt-level
  entry semantics,
- and completeness must be checked against the immutable attempt contract, not
  against a weaker reconstruction.

## Treat Structural Identity, Slot Identity, And Contributor Identity As One Plane

Rejected.

Reasons:

- they answer different questions,
- `canonical_full` proves not every slot is naturally a structural view,
- and mutation fencing is already keyed by contributor-value identity in
  `0084`.

## Create A Second Persistent Assembly Implementation

Rejected.

Reasons:

- the repository already has a viable structural assembly trunk,
- the real problem is semantic overloading at the join points,
- and a second assembly implementation would hide that problem rather than fix
  it.

## Let Source Seal Alone Define Final Success

Rejected.

Reasons:

- serving-facing workflows may require additional closeout stages,
- immutable version keys and manifest facts are part of the externally visible
  result,
- and callers should not infer success from side effects outside the attempt
  result.

# Trade-offs and Risks

- **More named objects**
  - The domain becomes more explicit, but the explicit split is less risky than
    continuing to overload one carrier.
- **Phase-1 storage alias risk**
  - Reusing old columns as temporary slot carriers keeps migration cost down,
    but the alias must stay explicit or drift returns quickly.
- **Hard-cut discipline**
  - The parent-child split only works if `0085` stops restating executable
    carriers and `0105` becomes the sole source of truth for those carriers.
- **Readiness-cut rigor**
  - If implementations continue to reread mutable policy or post-cut occupancy
    during seal, the design becomes inconsistent again even if the prose is
    correct.

# Compatibility & Acceptance Criteria

Parent-level acceptance requires:

- distributed publish is implemented on the existing structural assembly and
  layout trunk,
- `0105` is the sole executable carrier specification for the attempt domain,
- `ContributionContractSnapshot` is explicit and authoritative for attempt
  completeness,
- attempt creation lowers one canonical contract before the attempt domain
  begins,
- `canonical_full` is modeled as a legal contract family member rather than as
  a fake full-coverage piece,
- binding-backed contribution, direct `register_view`, and future frontends all
  lower onto the same structural commit and seal substrate,
- mutable closeout-policy configuration is resolved on pre-attempt scope and
  snapshotted before contributors are accepted,
- slot identity, structural identity, and contributor identity are no longer
  described or implemented as one plane,
- seal consumes a captured readiness cut rather than post-cut ambient state,
- public continuation aligns with `0100`,
- published success means the required lineage and closeout facts exist,
- future `PP`, `EP`, single-rank `canonical_full`, and later frontends can all
  reuse the same trunk without creating separate assembly implementations.

# References

- `docs/designs/0084-binding-unified-model-and-contract.md`
- `docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
- `docs/plans/0085-distributed-binding-assembly-and-coordinator.md`
- `docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md`
- `docs/designs/0011-unified-session-lifecycle-leases.md`
- `docs/designs/0055-programmable-framework.md`
- `docs/designs/0090-existence-semantics-and-single-authority-truth.md`
- `docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md`
- `docs/designs/0096-workflow-companion-admission-and-fencing.md`
- `docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md`
- `docs/architecture/view-replicas-and-assembly.md`
- `docs/architecture/api/api-design.md`
