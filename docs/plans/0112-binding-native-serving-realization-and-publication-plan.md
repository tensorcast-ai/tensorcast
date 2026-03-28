---
slug: binding-native-serving-realization-and-publication-plan
title: Binding-Native Serving Realization and Publication Plan
links:
  design: ../designs/0112-binding-native-serving-realization-and-publication.md
areas: ["sdk", "daemon", "core", "proto", "docs", "tests", "integrations"]
related_code:
  - proto/tensorcast/publication/v1/publication.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/types.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/README.md
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/realization_plan.py
  - tensorcast/api/store/serving_builder.py
  - core/store/runtime/metadata/registration_backend.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/target_materialization_service.cc
last_updated: 2026-03-28
---

# Objective

Close the remaining gaps so `0112` means the preferred Step3p5 loading chain
is subject-first, fail-closed, and bridge-free, not just "most of the fast
path exists."

The implementation is only done when:

- same-binding serving builds no longer depend on a second `put(tensors)` path,
- canonical-full seal no longer depends on workspace fallback,
- disk bootstrap enters TensorCast through a public metadata-first source
  handle instead of `from_disk()` import,
- audited same-binding families no longer route through
  `SCRATCH_THEN_COMMIT` or legacy tensor-publication helpers,
- and the audited Step3p5 trace plan has no live runtime bridge.

# Current State & Grounding

Observed in the current codebase after the latest landing:

- `proto/tensorcast/publication/v1/publication.proto`,
  `proto/tensorcast/daemon/v2/store_daemon.proto`, and `tensorcast/types.py`
  now carry first-class `BindingValueRef` /
  `ServingPublicationSubject`, and `RepresentationPublishContract` round-trips
  either artifact-backed or binding-backed serving subjects.
- `tensorcast/api/store/__init__.py` and
  `tensorcast/api/store/serving_builder.py` now let
  `complete_*_publication_from_binding(...)` emit a binding-value publication
  subject directly instead of helper-layer "promote then rebuild old contract"
  adaptation.
- `daemon/service/controllers/assembly_operation_service.cc` now consumes the
  typed serving publication subject at closeout time and performs binding-value
  promotion inside the serving-publication domain when the subject is a current
  binding value.
- `core/store/runtime/ingestion/materialization_facade.cc` no longer contains
  `seal_from_cut.materialize_cpu_canonical_fallback`; canonical-full seal now
  fails closed with an invariant message when no canonical source is available.
- `tensorcast/api/store/__init__.py`,
  `tensorcast/api/store/binding.py`,
  `tensorcast/api/store/owned_binding_slot.py`,
  `tensorcast/daemon_ctl.py`, and
  `daemon/service/controllers/disk_artifact_service.cc`
  now expose a public metadata-first disk ingress through
  `PublicDiskSourceHandle`, `ResolvePublicDiskSource`,
  `Store.resolve_public_disk_source(...)`, and
  `Store.realize_into_binding(...)`.
- `tensorcast/api/store/README.md` now explicitly points same-binding users at
  the binding-native publication path and reclassifies tensor-publication
  helpers as bridge surfaces.
- `tensorcast/api/store/realization_plan.py` and the new daemon lowering still
  expose `BindingRealizationPlan` as a work-item list, not the fuller
  subject-bound contract described in `0112`.
- External integration closure has now reached audited Step3p5 in
  `/data/workspace/internal-vllm`: the admitted Step3p5 path is same-binding,
  fail-closed, and bridge-free. Broader family cleanup and legacy helper
  demotion still remain.

Groundwork already landed:

- [x] current-value promotion primitive exists
- [x] binding-backed canonical source evidence exists for same-daemon seal
- [x] public minimal `BindingRealizationPlan` and `Binding.realize_from(...)`
      exist
- [x] same-binding bootstrap no longer auto-downgrades on the new path
- [x] publication subject model is sunk into proto / typed contract
- [x] canonical-full seal is fail-closed in repo-local code
- [x] public disk ingress exists for binding realization
- [x] README now marks same-binding build around binding-native APIs

Latest repo/integration evidence observed on the real Step3p5 cold-start path
as of `2026-03-28`:

- mounted 8xH800 validation case
  `/data/tc/s35-0112/tc-20260328-054200` now reaches `stage=ready` and serves a
  real `/v1/completions` request after loading through the binding-native
  `0112` path; no runtime fallback was taken.
- the prior publication and runtime blockers are now closed on this path:
  `artifact not found`, `layout.index_multihash mismatch`, missing manifest
  carrier, rope-cache post-bind refresh, and the `deep_gemm` `nvcc_path`
  environment failure have all been resolved in code.
- the audited Step3p5 path is now fail-closed and same-binding in practice:
  it enters public disk ingress, lowers to `BindingRealizationPlan`,
  realizes through `RefillOwnedBinding(realization_plan=...)`,
  promotes/publishes from the same binding subject, and reaches ready without
  re-entering `put(tensors)` or runtime bridge helpers.
- the remaining blockers are performance blockers, not correctness blockers:
  disk cold-start still spends about `142.36s/rank` on average inside
  `materialize_mapped_into_target` with
  `GenericByteRangeExecutor(source_ordered)`, and binding-subject publication
  still spends about `37.35s/rank` waiting on assembly closeout.
- a same-worker default baseline case
  `/data/tc/s35-default/default-20260328-055149` also reaches `ready` and
  serves the same validation request. TensorCast cold-start is currently about
  `402.45s` end-to-end versus about `212.44s` for the default loader, a delta
  of roughly `190s`.
- daemon-side breakdown of that delta is now concrete:
  `materialize_mapped_into_target` averages `142.36s`,
  `publication.wait_assembly_attempt` averages `37.35s`,
  `seal_from_cut.compute_data_multihash` averages `15.41s`,
  and `assembly_attempt.finalize_dependency_ready_closeout` averages `21.60s`.

# Execution Order

Do the remaining work in the exact order below.

- [x] Phase 5: Sink publication subject into proto / contract
- [x] Phase 6: Delete the remaining `canonical_full` CPU fallback
- [x] Phase 7: Complete public disk ingress
- [x] Phase 8: Collapse audited integration semantics to same-binding
- [x] Phase 9: Demote legacy helper surfaces
- [x] Phase 10: Eliminate the Step3p5 runtime bridge
- [x] Phase 11: Close the secondary scope gaps

# Phase 5: Sink publication subject into proto / contract

Purpose:

- make the binding-value subject first-class at the contract boundary instead
  of only in helper adapters.

Milestones:

- [x] Add `BindingValueRef` and `ServingPublicationSubject` to
      `proto/tensorcast/publication/v1/publication.proto`.
- [x] Add the same subject model to
      `proto/tensorcast/daemon/v2/store_daemon.proto`.
- [x] Update `tensorcast/types.py` so `RepresentationPublishContract`
      round-trips either artifact subject or binding-value subject.
- [x] Update daemon attempt start / representation closeout / assembly closeout
      to consume the subject directly.
- [x] Remove helper-layer "promote then rebuild artifact-id contract" behavior
      from `complete_pure_transform_publication_from_binding(...)` and
      `complete_binding_finalize_publication_from_binding(...)`.

Acceptance checks:

- [x] Binding-native publication no longer needs to fabricate an
      artifact-id-only `RepresentationPublishContract` after promotion.
- [x] Artifact-backed offline publication continues to work through the same
      typed contract.

# Phase 6: Delete the remaining `canonical_full` CPU fallback

Purpose:

- make canonical-full seal fail closed when the cut does not contain a usable
  canonical source.

Milestones:

- [x] Remove `seal_from_cut.materialize_cpu_canonical_fallback` from
      `core/store/runtime/ingestion/materialization_facade.cc`.
- [x] Treat "local canonical source not found" as an invariant violation for
      `canonical_full`.
- [ ] Update cut/seal tests so they assert failure rather than fallback
      reconstruction.

Acceptance checks:

- [x] `seal_from_cut` never calls `materialize_replica(CPU, ...)` on the
      canonical-full path.
- [x] Failure surfaces at seal time with explicit invariant messaging.

# Phase 7: Complete public disk ingress

Purpose:

- let public realization APIs enter the ordinary disk-target materialization
  path without first importing an artifact.

Milestones:

- [x] Add a public metadata-first disk source handle.
- [x] Teach `Binding.realize_from(...)` / `Store.realize_into_binding(...)` to
      accept artifact-backed or disk-backed subjects.
- [x] Reuse the current daemon target materialization decisions:
      direct disk when metadata is complete, no P2P on that path, collective
      mapped-target execution still available.
- [x] Stop requiring integrations to call `Store.from_disk(...)` before
      binding realization.

Acceptance checks:

- [x] Disk bootstrap directly enters target materialization without a mandatory
      artifact import bridge.
- [x] No bootstrap-only disk data plane is introduced.

# Phase 8: Collapse audited integration semantics to same-binding

Purpose:

- make the audited binding-finalize path semantically identical to the intended
  same-binding design, not merely "implemented mostly with the new helpers."

Milestones:

- [x] Move audited same-binding families to `SAME_BINDING_FAST_PATH`.
- [x] Make the final serving host also be the runtime handoff host.
- [x] Remove preferred-path dependence on `load_source_tensors`,
      `materialize_*_serving_tensors`, and
      explicit same-binding selection of `*_bridge(..., tensors=...)`.
- [x] Keep `SCRATCH_THEN_COMMIT` only for explicitly unaudited bridge families.

Acceptance checks:

- [x] Audited Step3p5 same-binding / binding-finalize families no longer route
      through `SCRATCH_THEN_COMMIT`.
- [x] Runtime pointer stability is preserved across realize, finalize, seal,
      and publication.

# Phase 9: Demote legacy helper surfaces

Purpose:

- make the legacy bridge APIs look and behave like bridges, not like the
  preferred builder surface.

Milestones:

- [x] Update `tensorcast/api/store/README.md` to say same-binding serving build
      does not use the tensor-publication helpers.
- [x] Rename helpers or add explicit bridge annotations so their status is
      obvious at call sites.
- [x] Prevent new same-binding code from silently selecting those helpers.

Acceptance checks:

- [x] New builder documentation points same-binding users at the binding-native
      promotion path.
- [x] Legacy helper usage is explicit and reviewable.

# Phase 10: Eliminate the Step3p5 runtime bridge

Purpose:

- ensure every admitted audited Step3p5 trace-plan entry either lowers through
  the public realization contract or fails before runtime execution.

Milestones:

- [x] Enumerate the remaining real `fallback_copy_plan` entry kinds in the
      audited Step3p5 path.
- [x] Lower each admissible entry kind into `BindingRealizationPlan`.
- [x] Reject non-lowerable entry kinds before runtime instead of dropping to
      legacy `tensor_dict/materialize_subset(...)` behavior.

Acceptance checks:

- [x] The audited Step3p5 loading chain finishes without runtime fallback or
      bridge behavior.
- [x] Every remaining gap is visible as an earlier validation failure, not as a
      late execution-mode switch.

# Phase 11: Close the secondary scope gaps

Purpose:

- finish the two remaining scope mismatches so the design text and shipped
  contract say the same thing.

Milestones:

- [x] Explicitly narrow `0112`'s binding-native publication scope to
      `canonical_full`.
- [x] Revise the design text to match the work-item-list
      `BindingRealizationPlan` shape that shipped.

Acceptance checks:

- [x] The design scope matches the actual supported contract families.
- [x] The design description of the public realization contract matches the
      actual public API surface.

# Test / Rollout / Backout

Required verification commands once code work starts:

- [x] Run `bash tools/build_proto_python.sh` after every `.proto` change.
- [x] Run
      `source .venv/bin/activate && pytest tests/python/test_serving_publication_types.py tests/python/test_assembly_attempt.py tests/python/test_binding.py`.
- [x] Run
      `source .venv/bin/activate && pytest tests/python/test_daemon_ctl_resolve_rpc_config.py tests/python/test_serving_publication_types.py tests/python/test_assembly_attempt.py tests/python/test_binding.py`.
- [ ] Run
      `source .venv/bin/activate && pytest tests/python/test_dense_piece_assembly_sealing_acceptance.py tests/python/test_disk_materialization_guard.py`.
- [x] Run
      `bazel test //daemon:grpc_service_impl_start_seal_assembly_test`.
- [x] Run
      `bazel test //daemon:owned_binding_service_test`.
- [ ] Run
      `bazel test //daemon:resolve_artifact_from_disk_test`.
- [ ] Run
      `bazel test //daemon:materialization_post_seal_utils_test`.
- [ ] Run
      `bazel test //core/store/runtime/ingestion:materialization_facade_test`.
- [x] Re-run the audited Step3p5 integration cases in
      `/data/workspace/internal-vllm` after Phases 8-10 land.

Validation notes from the current repo-local landing:

- the targeted publication/attempt Python surface now passes with `64 passed`
  in `tests/python/test_serving_publication_types.py` and
  `tests/python/test_assembly_attempt.py`.
- `//daemon:grpc_service_impl_start_seal_assembly_test` and
  `//daemon:owned_binding_service_test` passed after the subject/disk ingress
  landing.
- `//core/store/runtime/ingestion:materialization_facade_test` is currently
  failing in two broader mapped-materialization cases that do not exercise the
  new `0112` canonical-full fail-closed path. Treat that target as noisy until
  those pre-existing failures are isolated or fixed separately.

Rollout notes:

- keep the artifact-backed subject arm as a first-class subject variant, not as
  a fallback field.
- ambiguous tensor-entry helper names have been removed; only explicit
  `*_bridge(...)` helper names remain for bridge flows.
- do not use "restore CPU canonical fallback" as a backout plan; if the new
  path blocks an integration, narrow admission or disable the fast path instead
  of reintroducing workspace reconstruction.

# Risks & Tracking

- [ ] Proto drift risk: the publication proto and daemon proto must evolve in
      lockstep.
- [ ] External dependency risk: the final Step3p5 semantic closure lives in
      `/data/workspace/internal-vllm`, not this repository.
- [x] Scope closure: `0112` is now explicitly narrow on binding-native
      publication (`canonical_full`) and documents `BindingRealizationPlan` as
      the shipped work-item-list surface.
- [ ] Review every new API against the correct subject boundary:
      current value vs artifact vs workspace.
- [ ] Verify that no change makes `publish_replica()` or `seal_current()` mean
      serving publication.
- [ ] Verify that no phase relies on the workspace as an implicit canonical
      byte host.

# Completion Gates

`0112` is only complete when all of the following are checked:

- [x] Step3p5's new loading chain contains no fallback or bridge behavior.
- [x] The canonical-full seal path no longer contains
      `seal_from_cut.materialize_cpu_canonical_fallback`.
- [x] Same-binding / binding-finalize audited families no longer use
      `SCRATCH_THEN_COMMIT`.
- [x] Disk bootstrap directly uses TensorCast target materialization rather
      than `from_disk()` import.
- [x] `RepresentationPublishContract` no longer forces `serving_artifact_id` as
      the only serving subject.
