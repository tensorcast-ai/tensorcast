---
slug: volatile-publication-subjects-and-multi-replica-semantics
title: Volatile Target Publication Subjects and Multi-Replica Semantics (Plan)
links:
  design: ../designs/0103-volatile-publication-subjects-and-multi-replica-semantics.md
---

# Objective

Ground the already-landed target-backed publish path on an explicit target-publication projection over the existing
selection, workflow, lifecycle, and continuation kernels, keep today's publish closeout valid as `local_ephemeral`,
move same-subject non-reuse onto existing `0094` carriers, and avoid turning `0103` into either:

- a second repository-wide owner of bare `publish`,
- or a second publication-private identity or fence lattice.

# Current State & Grounding

- `TargetPublicationRegistry` already tracks per-target currentness via `latest_by_target_`, but its key is still an
  implementation-shaped precursor rather than an explicit publication-subject contract.
  - [target_publication_registry.h](/data/workspace/tensorcast-1/daemon/state/target_publication_registry.h)
  - [target_publication_registry.cc](/data/workspace/tensorcast-1/daemon/state/target_publication_registry.cc)
- Current `publication_key` still mixes stable target-domain identity and owner-local fencing facts such as `owner_pid`,
  which is acceptable as a bridge but not as the long-term semantic model.
  - [target_publish_service.cc](/data/workspace/tensorcast-1/daemon/service/controllers/target_publish_service.cc)
- Current workflow and lifecycle projections already exist, but are not yet aligned:
  - `WorkflowCompanionRef.currentness_key` mirrors `publication_key`
  - `publication_target` lifecycle minting still fixes `subject_generation = 1`
  - [target_publish_service.cc](/data/workspace/tensorcast-1/daemon/service/controllers/target_publish_service.cc)
- `0094` already owns the runtime carriers that should express same-subject rebinding:
  - `subject_id`
  - `subject_generation`
  - optional `fencing_context`
  - [0094-unified-lifecycle-kernel-and-capability-families.md](/data/workspace/tensorcast-1/docs/designs/0094-unified-lifecycle-kernel-and-capability-families.md)
- `TargetPublishService` already implements:
  - target-memory publish execution,
  - workflow replay/currentness checks,
  - `attach_existing -> OperationRef` projection,
  - and fail-closed public operation admission.
  - [target_publish_service.cc](/data/workspace/tensorcast-1/daemon/service/controllers/target_publish_service.cc)
- `OperationRef` already carries publish continuation metadata:
  - `authority_scope_kind`
  - `authority_scope_id`
  - `attachment_kind`
  - `recovery_class`
  - optional `fencing_digest`
  - [operation.py](/data/workspace/tensorcast-1/tensorcast/api/operation.py)
- `0096` and `0100` already ground the first landed publish `attach_existing -> Operation[T]` path; this plan should
  not reopen that closeout.
  - [0096-workflow-companion-admission-and-fencing.md](/data/workspace/tensorcast-1/docs/designs/0096-workflow-companion-admission-and-fencing.md)
  - [0100-distributed-authority-handoff-security-and-public-surfaces.md](/data/workspace/tensorcast-1/docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md)
- `0102` already keeps multi-target orchestration and canonical action vocabulary outside workflow and continuation
  ownership.
  - [0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md](/data/workspace/tensorcast-1/docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md)
- `0104` now owns worker-local `realize_set` and cluster rollout layering above optional child target publication.
  - [0104-artifact-realization-and-cluster-rollout.md](/data/workspace/tensorcast-1/docs/designs/0104-artifact-realization-and-cluster-rollout.md)
- `MetadataGateway` already has a useful publish-context idempotency pattern tied to `ReplicaKey`.
  - [metadata_gateway.h](/data/workspace/tensorcast-1/core/store/runtime/metadata/metadata_gateway.h#L101)
  - [metadata_gateway.cc](/data/workspace/tensorcast-1/core/store/runtime/metadata/metadata_gateway.cc#L186)

# Status Update

- [x] Plan sequencing is narrowed so `0103` follows `0056` substrate, `0102` bridge closeout, and `0104` rollout
      layering.
- [x] Phase 1-3 are the dependency-ready closeout path for target publication only.
- [ ] Publication currentness, lifecycle-generation projection, idempotency, and fail-closed hardening are still pending
      implementation work.

# Execution Position

- [ ] Stage 1: start only after `0056` set substrate and `0102` bridge semantics are stable enough that target-
      publication terminology cleanup is no longer blocking them.
- [ ] Stage 2: execute only Phase 1-3 as the main closeout path.
- [ ] Stage 3: treat any future coordinated-slot work as a separate low-cardinality design follow-up rather than part
      of this plan.

# Phases & Milestones

- [ ] Phase 1: Current-reality object-model closeout
  - [ ] Milestone 1.1: land `0103` as target-publication currentness binding over existing kernels rather than as a new
        semantic kernel.
  - [ ] Milestone 1.2: document the landed publish currentness, generation, instance, and continuation crosswalk without
        reopening the first `attach_existing -> Operation[T]` closeout.
  - [ ] Milestone 1.3: document that `0103` does not retake repository-wide ownership of bare `publish`.
  - [ ] Milestone 1.4: reconcile current `publication_key`, `currentness_key`, `publication_id`, `subject_id`,
        `subject_generation`, and `OperationRef` terminology in docs and internal type naming.
  - [ ] Milestone 1.5: add `0103` cross-linking to `0094`, `0102`, `0104`, and `0087`.

- [ ] Phase 2: Publication currentness and lifecycle projection extraction
  - [ ] Milestone 2.1: define explicit internal `PublicationSubjectKey` and `PublicationInstanceId` types plus one
        canonical cross-layer projection table.
  - [ ] Milestone 2.2: keep `PublicationOwnerFence` as a logical same-subject relation carried by
        `subject_generation` and optional `fencing_context`, not as a standalone parallel carrier.
  - [ ] Milestone 2.3: keep current landed path explicitly in `local_ephemeral` mode with current-instance semantics.
  - [ ] Milestone 2.4: factor implementation-shaped names toward currentness, generation, and instance semantics
        without regressing current owner-local fail-closed behavior.
  - [ ] Milestone 2.5: keep richer history and coordinated-slot semantics explicitly out of dependency-ready scope.

- [ ] Phase 3: Correctness alignment for `local_ephemeral` target publication
  - [ ] Milestone 3.1: make current-instance attach behavior depend on currentness key plus `subject_generation` plus
        instance relation rather than artifact id.
  - [ ] Milestone 3.2: make stale-current and wrong-subject rejection depend on publication currentness and publication
        instance semantics.
  - [ ] Milestone 3.3: harden owner-loss fail-closed behavior through generation-aware rebinding without weakening
        owner-local observations.
  - [ ] Milestone 3.4: preserve the current `attach_existing -> Operation[T]` public surface and keep `OperationRef`
        instance-scoped.
  - [ ] Milestone 3.5: align idempotency with stable target-domain identity plus binding generation and existing
        `ReplicaKey`-grounded publish-context patterns.
  - [ ] Milestone 3.6: keep `owner_pid` and similar facts as local observations only and avoid introducing a second
        publication-private fence dialect.
  - [ ] Milestone 3.7: keep non-current historical replay and richer reusable-terminal semantics explicitly out of
        dependency-ready scope unless bounded-history substrate lands.

# Tasks

- Audit current publish call sites and classify each as:
  - target-publication family,
  - source-side publish alias or barrier,
  - or higher-level orchestration intent.
- Define the minimum stable fields for `PublicationSubjectKey` and the stable target-domain scope it represents.
- Define one canonical projection table across:
  - `WorkflowCompanionRef.currentness_key`
  - `LifecycleSubjectRecord.subject_id`
  - `LifecycleEpochs.subject_generation`
  - `publication_id`
  - `OperationRef`
- Reconcile current `publication_key`, `currentness_key`, `publication_id`, `subject_id`, `subject_generation`, and
  operation metadata with the new currentness, generation, and instance split.
- Keep `PublicationOwnerFence` as a logical relation over existing lifecycle carriers rather than as a new persisted
  publication-private type.
- Keep current `owner_pid`-based owner-local checks honest as local observations until generation-driven rebinding is
  explicit.
- Keep `OperationRef` instance-scoped unless a separate public-surface proposal proves otherwise.
- Do not expand this plan into repo-wide publish renaming, richer history, or coordinated-slot truth unless a separate
  design explicitly requires it.
- Add tests for:
  - same artifact, different daemon, different publish subject
  - same subject replay attach with same generation
  - same current subject replay terminal with same generation
  - same subject stale-current
  - same stable subject with newer generation and owner replacement or loss
  - local-ephemeral owner loss
- Update design references and wording without regressing already-landed publish continuation behavior.

# Test / Rollout / Backout

- Documentation-only phase:
  - verify links resolve and the design or plan pair are discoverable.
- Local-ephemeral alignment phase:
  - `bazel test //daemon:grpc_service_impl_publish_target_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //daemon:grpc_service_impl_operation_rpc_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `pytest tests/python/api/test_operation_semantics.py tests/python/test_binding.py tests/python/test_inplace_slot.py`
- Rollout:
  - land current-reality documentation changes first
  - then land internal currentness, generation, and instance model refactors without changing public API names
  - then harden behavior and tests
- Backout:
  - retain current `StartPublishTargetReplica` public API
  - revert internal subject or fence extraction separately from documentation if needed

# Risks & Tracking

- Risk: document cleanup accidentally suggests the current landed publish closeout is less real than it already is.
  - Tracking: keep tests and wording aligned with the existing `attach_existing -> Operation[T]` path.
- Risk: publication-subject extraction accidentally serializes valid multi-daemon same-artifact publish flows.
  - Tracking: add explicit multi-daemon, same-artifact, different-target tests.
- Risk: `0103` still reads like a new semantic kernel instead of a specialization over existing workflow and lifecycle
  carriers.
  - Tracking: keep the design centered on cross-layer projection and `0094` reuse rather than on new standalone types.
- Risk: stable subject identity and same-subject rebinding diverge between `currentness_key` and
  `subject_generation`.
  - Tracking: define one explicit projection table and test same-currentness plus generation transitions directly.
- Risk: owner-local fencing is abstracted away too early and weakens fail-closed behavior for `local_ephemeral`.
  - Tracking: preserve `owner_pid`-backed checks as local observations until generation-aware rebinding is explicit.
- Risk: `0103` is still read as repository-wide ownership of bare `publish` and keeps cross-document ambiguity alive.
  - Tracking: make scope explicit in `0103` and keep adjacent-doc amendments narrow and concrete.
- Risk: idempotency uses the wrong domain and either over-joins across rebinding or forks retries that should join.
  - Tracking: align idempotency with stable target-domain identity plus binding generation before expanding callers.
- Risk: this plan expands into workflow history or continuation redesign that `0096` and `0100` already own.
  - Tracking: keep execution scoped to Phase 1-3 unless a new blocker is documented explicitly.
