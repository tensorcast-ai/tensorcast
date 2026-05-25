---
slug: unified-artifact-realization-kernel
title: Unified Artifact Realization Kernel Plan
status: draft
areas: ["sdk", "daemon", "core", "serving", "integrations", "tests", "docs"]
created: 2026-05-23
last_updated: 2026-05-25
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
  - tensorcast/serving/runtime_view.py
  - tensorcast/serving/_runtime_impl/lifecycle.py
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

Completion requires cleanup, not just compatibility: redundant legacy paths,
temporary compatibility adapters, and tests that primarily assert old split
paths must be deleted or rewritten to assert the unified realization kernel.
Ergonomic public syntax such as `tensor_dict(...)` and `bind(...)` remains, but
those entrypoints must stay thin facades over the unified path.

# Implementation Status

Last synced: 2026-05-25.

Implemented in the current worktree:

- Added the first SDK realization kernel module at
  `tensorcast/api/store/realization_kernel.py` with
  `ResolvedArtifactSelection`, `ArtifactRealizationSpec`,
  `ArtifactRealizationHandle`, `ArtifactRealizationReport`,
  target/strategy/representation/lifecycle DTOs, resource-envelope DTOs, and a
  TensorDict projection owner.
- Rewired the high-volume SDK entrypoints `Artifact.tensor_dict(...)`,
  `tensor_dict_with_diagnostics(...)`, `tensor_dict_into(...)`, `bind(...)`, and
  `bind_into(...)` to call `Artifact.realize(...)` for TensorDict,
  caller-tensor, owned-binding, and adopted-binding target kinds.
- Rewired `Artifact.tensor_into(...)` through a single-tensor subset
  `caller_tensors` realization, preserving explicit subset lowering to avoid
  rematerializing unrelated tensors.
- Added `envelope_for_caller_tensors(...)` so caller tensor writes report
  planned registered-region direct-write bytes for CUDA targets and temporary
  copy bytes/counts for CPU targets through the shared resource envelope.
- Added `envelope_for_retained_replica(...)` and attached an
  `ArtifactRealizationReport` to `PrefetchedReplica` operation results for
  retained device prefetch while preserving the existing `Operation` wait,
  status, and cancel contract.
- Added retained-binding report/envelope adapters:
  `RealizationRetainedBindingReport`, `retained_binding_reports_for(...)`, and
  `envelope_for_retained_binding(...)`. `PrefetchedServingBinding` and
  `PrefetchedServingBindingSet` operation results now carry
  `ArtifactRealizationReport` data for binding value refs, member identity,
  reservation capability, readiness, verification state, staged group acquire
  facts, target layout digest, copy-plan digest, and retained reservation
  bytes while preserving the public `Operation` contract.
- Hardened retained-binding acquire semantics from `0116`: expired
  `BindingReservationCapability` authorities now fail closed before daemon
  acquire, acquire/attached/runtime-owned retained handles expose `status()` and
  `debug_status()` for lifecycle visibility, retained reports/profile payloads
  preserve capability expiry, and TTL/idle-retire behavior is covered by unit
  tests plus the real-CUDA retained binding E2E.
- Added target-set report/envelope adapters for retained binding set prefetch:
  `RealizationTargetSetReport`, `RealizationTargetSetMemberReport`,
  `envelope_for_target_set(...)`,
  `target_set_report_for_retained_bindings(...)`, and
  `report_for_target_set(...)`. `PrefetchedServingBindingSet` operation results
  now report top-level `target_kind="target_set"` with one group report,
  per-member binding value ids, target layout digests, source-selection mode
  (`same_selection` / `per_part_selection`), device UUIDs, readiness,
  verification state, reservation capability facts, staged group acquire facts,
  total retained bytes, and daemon-session grouping. Individual members keep the
  same report attached so existing retained handoff callers can still inspect
  member-level authority facts without a parallel side channel.
- Extended target-set realization reports with the first explicit
  strategy/lifecycle plan surface:
  `ArtifactRealizationReport.target_plan`, `strategy_plan`,
  `representation_admission`, and `lifecycle_plan` now carry target-set group
  selection mode, same-daemon/cross-daemon source coordination,
  collective-first candidate policy for multi-member checkpoint sources, group
  barriers, staged values, publish barriers, acquire claim ids, release policy,
  member release policies, member-local device/layout/copy-plan/runtime-profile
  digests, and placement digests. SDK tests now assert both
  `same_selection`/`per_part_selection` report behavior and the attached
  target-set strategy/lifecycle plans.
- Extended target-set facade coverage for TP/per-part source sets: SDK
  `realize_async(target_set)` now has a guardrail for `serving_artifact_set`
  member sources, per-member artifact refs, per-member copy-plan digests,
  staged values, group acquire claims, publish barriers, and member release
  policies. The operation proto now preserves
  `GroupRealizationAcquireRef.wait_for_publish` and `wait_timeout_ms` across
  `PrefetchedServingBinding` round trips so publish-barrier planning survives
  daemon operation result transport.
- Propagated staged group-acquire publish-barrier facts through the real daemon
  retained serving prefetch path: `CreateOwnedBindingResponse` now carries
  `wait_for_publish` plus an acquire wait budget, `PrefetchServingBindingResult`
  preserves those facts for single and target-set members, and operation RPC
  tests assert the daemon result matches the SDK report/operation proto shape.
  The reference serving consumer also has a typed target-set prefetch helper,
  covered by a real-CUDA daemon E2E that prefetches two target-set members and
  attaches them from a worker process.
- Updated the real-CUDA serving binding set E2E to use vLLM-shaped target-set
  identity: `runtime="vllm"`, `vllm://parallelism?tp=2&pp=1&dp=1` topology,
  and `dp0:pp0:tp*` member ids. The daemon target-set prefetch, serialized
  operation result, cross-process worker acquire, CUDA IPC lease, and member
  diagnostics now execute through the same target-set realization shape used by
  TP startup planning without requiring an external vLLM dependency in the
  repository tests.
- Updated the serving runtime integration fixture to use a vLLM/TP-shaped
  placement for runtime attachment finalization. The model-runtime realization
  report now has test coverage for vLLM framework identity, adapter/ABI
  versions, topology/member digests, and `framework:vllm` risk labeling while
  sharing the runtime-attachment release contract.
- Added an SDK fail-closed target-set bypass guardrail: a
  `ServingBindingSetTarget` can no longer be passed through
  `ArtifactRealizationSpec.retained_binding(...)`; callers must lower TP/group
  prefetch through `ArtifactRealizationSpec.target_set(...)` so admission,
  strategy, lifecycle, report, and resource-envelope state all carry
  `target_kind="target_set"`.
- Updated the live Store SDK README so retained serving prefetch is documented
  as lowering through `ArtifactRealizationSpec.retained_binding(...)` for
  single retained bindings and `ArtifactRealizationSpec.target_set(...)` for
  TP/group sets. `ServingBindingTarget` and `ServingBindingSetTarget` are now
  described as ergonomic `artifact.prefetch(target=...)` wrappers, not the
  source authority for group realization. Historical 0116/0120 design/plan
  references intentionally keep their original DTO names until the broader
  protocol/API cleanup phase.
- Added a shared resource-envelope admission guardrail:
  `RealizationResourceEnvelope.validate_for_target(...)` now rejects executable
  plans that are missing backing/export/projection/owner, release policy,
  mutability, release strictness, or export-lifetime facts before target-specific
  fallback behavior can run.
- Added lowered strategy-plan reporting for binding and runtime attachment
  reports. `report_for_binding_realization(...)` and
  `report_for_runtime_attachment(...)` now derive a `RealizationStrategyPlan`
  from the shared envelope plus `RealizationExecutionCommitReport`, carrying
  fallback and collective policy facts instead of leaving them only in
  path-local diagnostics.
- Extended binding and runtime attachment reports with lowered
  `RepresentationAdmissionPlan` and `RealizationLifecyclePlan` facts, so
  binding layout admission, runtime attachment admission, publishability,
  retained/resource ownership, and release policies are carried alongside the
  target plan and resource envelope.
- Extended the same representation/lifecycle surface to TensorDict,
  caller-tensor, retained-replica, retained-binding, model-runtime,
  publication, and mounted-source reports. Retained binding lifecycle plans now
  carry reservation acquire claim ids, staged value counts, publish barriers,
  group transaction/version facts, and per-member release policies; TensorDict
  and caller-tensor reports carry payload ownership and borrowed caller-target
  lifecycle facts through the shared plan.
- Extended `RealizationStrategyPlan` derivation for SDK materialization reports:
  TensorDict, caller tensor, binding, retained-replica, retained-binding, and
  runtime attachment reports now carry lowered source/retrieval policy,
  P2P/disk preference, wait-for-shared-disk, verification, region-backed,
  export-policy, wait-for-completion, transport-hold, lease-mode, deadline,
  fallback, and collective-policy facts where available.
- Removed the legacy `Artifact._build_artifact_selection()`,
  `Artifact._build_owner_source_selection()`, and
  `Artifact._resolve_owner_source_selection()` private adapters. Artifact and
  owned-binding source selection lowering now goes through
  `_resolve_realization_selection(...)` and `resolve_artifact_selection(...)`.
  The standalone materialization path renamed its local selector helper from
  `_build_artifact_selection(...)` to `_resolve_materialization_selection(...)`.
  Guardrail coverage now asserts the old adapter methods are not exposed on
  `Artifact` and no SDK path calls the removed helpers.
- Added an AST guardrail proving ergonomic `Artifact` entrypoints
  `tensor_dict(...)`, `tensor_dict_with_diagnostics(...)`,
  `tensor_dict_into(...)`, `tensor_into(...)`, `bind(...)`, and
  `bind_into(...)` enter the realization facade instead of maintaining separate
  public implementation paths.
- Synced the live Store SDK README with the `0039` public target surface:
  convenience `Artifact` entrypoints are documented as ergonomic wrappers over
  `ArtifactRealizationSpec` and the shared `Artifact.realize(...)` handle
  facade, with direct handle use reserved for callers that need report,
  projection-owner, or lifecycle-handle access.
- Synced the `0121` design references to the landed implementation by updating
  `last_updated`, adding `realization_kernel.py`, daemon controller
  `materialization_policy_utils`, and the real-CUDA target-set E2E to
  `related_code`, and documenting controller resource-manager linkage as part
  of the resource-envelope contract.
- Added `risk_labels_for_target(...)` so migrated reports derive authority,
  identity, lifecycle, lease-strength, mutability, movement-cost,
  async-continuation, target-set, and publication risk labels from
  `RealizationTargetPlan` plus `RealizationResourceEnvelope`. TensorDict,
  caller-tensor, binding, retained prefetch, target-set, runtime attachment, and
  runtime-owned publication reports now use the shared helper; runtime and
  publication call sites only pass contextual extra labels such as
  `retained_acquire` or publication state.
- Added report-level operation backend and publishability facts:
  `ArtifactRealizationReport.operation_backend` and
  `RealizationPublishabilityReport`. Migrated TensorDict, caller-tensor,
  binding, retained prefetch, target-set, runtime attachment, and runtime-owned
  publication paths now expose the daemon/runtime backend used for the
  realization plus normalized publishability state. Binding reports derive
  eligibility/requested/published facts from the binding value, publication
  reports derive active publication state from the projection, and
  non-publishable targets report an explicit non-publishable reason instead of
  omitting the field.
- Added a risk-closure matrix test fixture that mirrors the plan's
  `Risks & Tracking` list and requires every risk entry to name an admission
  field, envelope field, report field, guardrail test, and blocking condition.
- Added a source/target identity boundary guardrail proving
  `resolve_artifact_selection(...)` owns artifact/key/view/subset/generation
  source identity while target layout and copy-plan digests remain
  `RealizationTargetPlan` facts.
- Added SDK mounted-source guardrails proving `mounted_source` realization
  rejects non-`msa1:` subjects and key-backed activation before promotion.
  The shared resolver now rejects `msa1:` key mapping, durable authority
  overrides, and Global Store-style index fetch routing; explicit same-daemon
  promotion records daemon-local mounted-source authority and the promoted
  durable `mi2:` artifact. A local-ready pending-verification report guardrail
  records runtime-local-ready profile, local-ready authority, verification job,
  and local serving ref instead of treating local-ready state as an ordinary
  durable source.
- Completed the shared cost field surface for migrated resource envelopes:
  direct-write bytes, copy bytes, copy count, mmap bytes, CUDA IPC open count,
  CPU memfd fd count, retained/temporary bytes, and fallback reason buckets are
  present on `RealizationResourceEnvelope`. Binding envelopes now derive
  `copy_count` from fallback/residual/generic backend execution components,
  matching the existing direct-write/copy byte accounting used by the report.
- Added an implementation-object adapter matrix guardrail covering
  `MaterializationPayload`, region-backed `MaterializeIntoTarget` target state,
  `OwnedBindingSlot` state, retained acquire state, runtime attachment state,
  group realization state, publication state, and mounted-source state. The
  matrix asserts each path lowers into the same
  `RealizationResourceEnvelope` backing/export/projection/owner/release/cost
  contract instead of keeping path-specific lifecycle facts out of band.
- Added a `GetIntoResult` execution DTO for caller-tensor materialization so
  region-backed direct writes and temporary-payload fallback copies can report
  actual source, replica, byte count, and fallback reason buckets while keeping
  existing callers compatible. `Artifact.realize(caller_tensors)` now consumes
  those facts in its report/envelope, and focused tests assert fallback source,
  temporary bytes, fallback buckets, and fallback replica unload behavior.
- Caller-tensor `RealizationResourceEnvelope` construction now distinguishes
  actual region-backed direct-write completion from temporary materialization
  fallback through `GetIntoResult.used_region_backed`. CUDA direct-write reports
  registered-region bytes and only target-region cleanup; CUDA fallback reports
  `temporary_copy`, fallback copy bytes/counts, temporary replica bytes, fallback
  buckets, and temporary replica unload policy through the shared envelope.
- Removed the report-level `legacy_diagnostics` compatibility field. TensorDict,
  binding, runtime attachment, and retained replica reports now expose
  materialization facts through `materialization_diagnostics`, alongside typed
  execution and source-bound plan diagnostics.
- Centralized SDK materialization/execution/source-bound diagnostic decoding in
  `realization_kernel.py`: TensorDict and binding source labels,
  binding-response materialization facts, `ExecutionDiagnostics`, and
  `SourceBoundPlanDiagnostics` now share kernel helpers instead of private
  `OwnedBindingSlot` or `Artifact` adapters. The old owned-binding diagnostics
  unit file was folded into `test_realization_kernel.py`.
- Removed the old TensorDict-specific profile side channel:
  `_execute_tensor_dict_with_diagnostics(...)` no longer emits
  `artifact.tensor_dict_with_diagnostics`; public TensorDict realization now
  records the unified report-shaped `artifact.realize` event only, with a
  guardrail test blocking reintroduction of the old event.
- Deleted the retired target-publication capability path:
  `TargetPublicationScope` and `CAPABILITY_AUDIENCE_TARGET_PUBLICATION` were
  removed from `capability_token.proto` (reserved for wire safety), and the
  daemon test that minted old target-publication tokens was deleted. Target
  publication now uses only binding-current-value publication authority. The
  daemon routed-authority wire parser also rejects the retired
  `target_publication_token` front-door label instead of aliasing it to the new
  binding-current-value token label.
- Removed the Runtime-side compatibility hydrate rewrite/cache. `Runtime`
  no longer caches `PublishManifest` results or rewrites
  `hydrate(engine_request_id=...)` into `hydrate(publish_manifest=...)`; the
  engine request id now remains the request source handed to the daemon/node
  agent, avoiding a second local publication authority.
- Removed the no-op `cache` compatibility parameter from
  `Artifact.tensor(...)`. Single-tensor access remains as a thin subset
  TensorDict facade, without carrying an unused compatibility switch.
- Removed `runtime.signals().list_workers()` and
  `runtime.signals().list_instances()` compatibility delegation shims. Directory
  routing now has one public SDK entrypoint through `runtime.directory()`, while
  `runtime.signals()` remains scoped to connected-worker status snapshots.
- Removed the pure-transform `tc_serving_*` string-arg publication fallback and
  the `build_pure_transform_serving_args*` helpers. Transform registration now
  carries serving publication intent only through typed
  `TransformSpec.publication_spec`, and the engine adapter no longer
  reconstructs serving lineage from transform args. A focused guardrail test
  asserts the helper and string-arg markers stay out of SDK/engine adapter code.
- Adopted binding realization now records a mapped copy-plan digest separately
  from both source selection and target layout identity. SDK facade tests assert
  TensorDict and adopted binding entrypoints share the same source-selection
  digest while mapped/adopted target layout and copy-plan digests remain
  distinct.
- Added `RealizationExecutionCommitReport` and
  `execution_commit_report_for(...)` so migrated binding and runtime attachment
  reports normalize lower strategy facts from `ExecutionDiagnostics` and
  `SourceBoundPlanDiagnostics`: actual executor path, direct/committed/fallback
  bytes, residual bytes, collective metrics, lane-allocation bytes,
  committed-range bytes, residual-fallback bytes, and planner reject buckets.
- Extended `artifact.realize` profile payloads to include the same 0108
  strategy visibility fields from `RealizationExecutionCommitReport`: execution
  plan kind/hash, lane allocation, committed ranges, residual fallback ranges,
  residual bytes, and planner reject buckets.
- Added `Artifact.realize_async(...)` for retained replica, retained binding,
  and target-set specs, and rewired `Artifact.prefetch(...)` to lower
  `ServingBindingSetTarget` through `ArtifactRealizationSpec.target_set(...)`
  while keeping the public `Operation[PrefetchedReplica]` /
  `Operation[ServingPrefetchResult]` behavior.
- Added mounted-source realization through `ArtifactRealizationSpec.mounted_source(...)`.
  `Store.promote_mounted_source(...)` now lowers through `Artifact.realize(...)`
  and returns the promoted `Artifact` from the shared handle facade. The
  mounted-source report/envelope records the `msa1:` source subject, promoted
  `mi2:` artifact id, attested canonical-index byte length, generation,
  checksum policy, promotion target digest, daemon promotion backend, and
  artifact-identity release policy.
- Added `ArtifactRealizationSpec.model_runtime(...)` and `ArtifactRealizationHandle`
  `attach(...)` delegation hooks, giving the artifact-centered model-runtime
  facade the target shape required by `0120`. Direct `Artifact.realize(...)`
  calls for model runtime still fail closed with `UNIMPLEMENTED` until serving
  runtime attachment lowering is migrated.
- Routed SDK selection construction in `tensorcast/api/_materialize.py`,
  `tensorcast/api/store/materialization.py`,
  `tensorcast/api/store/inplace_slot.py`, `tensorcast/api/plan/plan.py`, and
  `Artifact._resolve_realization_selection(...)` through
  `resolve_artifact_selection(...)`.
- Deleted the legacy Artifact selection adapter names and moved retained
  prefetch plus owned-binding refill/swap source-selection lowering onto
  `_resolve_realization_selection(...)`; guardrails now block production SDK
  paths outside `Artifact` from calling the removed adapters.
- Expanded resolver coverage for daemon key resolution, daemon index fetch,
  view/subset selected-index identity, mapped source view hints,
  daemon-attested `msa1:` admission, and same/per-part group-member source
  selection identity.
- Extended the completed `ArtifactRealizationHandle` binding facade so binding
  handles can delegate `publish_replica(...)`, `promote(...)`, and handle
  `close()` through the shared realization handle surface when the realized
  value supports those capabilities.
- Added `RealizationReleaseContract` and `release_contract_for(...)` as the
  first shared projection-owner release adapter. `ArtifactRealizationHandle`
  now exposes a release contract derived from the resource envelope, runs
  release actions through the contract, and makes TensorDict projection
  `close()` and binding handle `close()` share the same idempotent release path.
  Runtime detach, retained acquire close, and publication projection close now
  also route through the shared contract. Phase 5 still needs old independent
  entrypoint deletion/alignment work before it can close.
- Extended the release-contract lifecycle matrix to assert raw tensor owner
  escape, export release, backing release, caller-target cleanup, runtime
  workflow cleanup, retained acquire close, and publication projection close as
  distinct idempotent release paths.
- Bound ergonomic TensorDict entrypoints to deterministic payload cleanup:
  `Artifact.tensor_dict(...)` now returns a projection whose close path unloads
  the daemon materialized payload through the shared realization handle, raw
  tensors keep the projection owner reachable for tensor escape cases, and
  `tensor_dict_with_diagnostics(...)` exposes the same close path through
  `TensorDictMaterializationResult.release()`.
- Removed broad `contextlib.suppress(Exception)` cleanup from the API
  realization and registration paths. Best-effort close, abort, revoke,
  region-cache, runtime, and future-cancel cleanup now logs contextual failures
  instead of silently hiding them; TensorDict projection owner attachment fails
  closed when it cannot preserve the handle lifetime.
- Removed broad `contextlib.suppress(Exception)` cleanup from touched Global
  Store production control-plane paths. Cluster-runtime channel close now logs
  cleanup failures, packaged schema lookup narrows expected missing-resource
  errors, span-attribute callers rely on the centralized observability helper,
  instance heartbeat capability updates fail closed instead of silently
  skipping state, and DuckDB cursor finalization logs failures at debug level.
- Removed broad `contextlib.suppress(Exception)` cleanup from serving retained
  binding, local-ready, recipe-cache, and runtime attachment lifecycle paths.
  Lease rollback and close cleanup now log failures; descriptor fallback
  catches only expected metadata shape errors; placement payload construction
  fails closed instead of silently dropping framework facts; deferred recipe
  cache write failures are no longer hidden.
- Deleted the old non-offset CUDA IPC export fallback from target-region
  registration. `Store.register_vram_region(...)` and
  `MaterializationPipeline.get_into(...)` now use only the offset-aware
  `get_cuda_memory_handle_with_offset(...)` exporter, and tests no longer patch
  the retired `get_cuda_memory_handle(...)` path for region-backed realization.
- Deleted the retired core/pybind non-offset CUDA IPC exporter itself.
  `core/checkpoint` now exposes only
  `get_cuda_memory_handle_with_offset(...)`; address-range lookup failure is a
  hard IPC export error instead of a zero-offset fallback, and the Python C
  extension plus `_C.pyi` no longer expose `get_cuda_memory_handle(...)`.
- Deleted the mounted-source absolute-path compatibility fallback from daemon
  config and core policy resolution. `public_disk_source` now has only trusted
  root policies; unmatched absolute mounted-source paths fail closed, and the
  old `unmatched_path_mode` / allow-absolute-fallback proto and option path is
  reserved rather than executable.
- Removed the daemon canonical-index disk fallback helper. Canonical-index
  loading now takes an explicit authority: local imports, mounted sources, and
  GS-unavailable disk paths read from disk; GS-managed artifacts read only from
  engine metadata and fail closed instead of silently falling back to disk.
- Extended the same explicit canonical-index authority rule to
  `MaterializeReplica`. The replica path now uses
  `load_canonical_index_from_authority(...)`, dropped the inline
  engine-then-disk retry, and renamed pre-binding/local-import disk-source
  variables so post-seal view reuse no longer looks like a compatibility
  fallback. C++ no-lease tests now model GS-managed artifacts by exposing
  canonical index metadata through the fake Global Store.
- Canonicalized the daemon loaded-replica status surface by removing the
  misleading `GetLoadedReplicasV2` RPC/message/helper names. The only active
  daemon API name is now `GetLoadedReplicas`, while the service package version
  remains `tensorcast.daemon.v2`.
- Added token-backed export fail-closed coverage for CPU memfd and CUDA IPC
  materialization: missing CPU lease tokens, missing local-handle socket,
  disabled CPU shared memory, empty CPU memfd handles, and CUDA lease-token
  paths without local-handle authority all fail before tensor restore/open.
- Classified CPU TensorDict projection storage as `read_mostly_private_copy`
  and added CPU memfd restore coverage proving raw tensor writes are private to
  the process mapping and do not write back to the daemon/export backing.
- TensorDict projections now reject mapping-level writes (`__setitem__`,
  `update`, `pop`, `clear`, and related mutators) with
  `FAILED_PRECONDITION`, matching the resource envelope's read-only/read-mostly
  mutability contract. Tensor storage-level write protection still needs lower
  backing/export enforcement before the CPU TensorDict mutability item is fully
  closed.
- Added TensorDict handle capability guardrails proving TensorDict realization
  handles can return a projection but cannot expose binding, prefetch handoff,
  publication, promotion, attachment, or caller-completion capabilities.
- Added binding-aware report and resource-envelope adapters:
  `RealizationBindingReport`, `envelope_for_binding(...)`,
  `binding_report_for(...)`, and `report_for_binding_realization(...)`.
  `binding_owned` and `binding_adopted` realization now capture binding ids,
  binding layout ids, current/staged value ids, publication eligibility,
  materialization source/operation diagnostics, execution diagnostics, source
  bound planner diagnostics, retained/direct-write/copy bytes, temporary bytes,
  fallback reason buckets, and publish-lease release policy through the shared
  report/envelope surface.
- Added runtime-attachment report and resource-envelope adapters:
  `envelope_for_runtime_attachment(...)` and
  `report_for_runtime_attachment(...)`. Direct serving-artifact load, reload,
  prepared local-ready restore, retained binding acquire, and local-ready
  finalize now attach `ArtifactRealizationReport` data to
  `RuntimeBindingView.diagnostics`, including source selection digest, target
  layout digest, artifact profile/authority scope, binding layout ids, retained
  acquire/member reservation facts when available, runtime attachment release
  policy, CUDA IPC / CPU memfd open counts, retained bytes, and runtime
  attachment risk labels.
- Centralized serving runtime source-selection projection in
  `RuntimeWorkerView`: endpoint payloads now derive `source_selection` from
  explicit projections, materialization diagnostics, execution diagnostics, or
  `ServingRealizationReport` execution fields at projection time.
  `ServingIntegration._state_seed(...)` no longer stores a duplicated
  `source_selection` diagnostic copy.
- Added daemon byte-artifact put-if-absent authority preflight. `HomeBatch*`
  and the local `BatchPutIfAbsentFromRegion` front-door path now classify
  joined/conflicting claims through `ByteArtifactAuthorityService` before
  source or target lowering runs. Existing ready backings return `joined`
  without opening payload transports or staging a duplicate deterministic
  physical backing; invisible/lost backings still lower only when they need a
  new backing, preserving claim-truth conflict behavior.
- Runtime attachment state now carries a concrete
  `ArtifactRealizationHandle(target_kind="runtime_attachment")` where an
  `ArtifactRealizationReport` is available. Direct serving load, reload without
  a model reattach, retained acquire, and local-ready prepared runtime state all
  install the handle facade and use its release contract as the runtime close
  owner while preserving existing binding and ownership-handle fields.
- Added model-runtime report/handle adapters over completed runtime attachment
  realization. `RuntimeBindingMaterialization.attach_and_finalize(...)` now
  installs `ArtifactRealizationHandle(target_kind="model_runtime")` alongside
  the runtime-attachment handle, sharing the same release contract and exposing
  `attach(...)` as a completed runtime-state projection. Local-ready prepared
  state also carries the model-runtime handle when framework context is
  available.
- Retained acquire lease state now preserves the parsed retained-binding
  authority through `AttachedRetainedBinding`, runtime-owned retained attachment
  handles, and serving `RestoredRetainedBinding`, so framework runtime
  attachment can report acquisition authority without re-parsing side-band
  configuration.
- Added publication report/envelope adapters:
  `RealizationPublicationReport`, `envelope_for_publication(...)`,
  `publication_report_for(...)`, and `report_for_publication(...)`.
  Runtime-owned replica publication and non-authoritative publishing
  projections now attach `ArtifactRealizationReport` data to
  `RuntimeWorkerView.diagnostics["artifact_publication_report"]`, including
  published replica state, operation id, replica id, lease id, byte-space facts,
  binding value id, target layout digest, publication release policy, and
  publication risk labels. This covers the serving runtime publication
  projection without changing the lower-level representation publish attempt
  state machine.
- Runtime-owned and non-authoritative publication projections now also lower
  through `ArtifactRealizationSpec.publication(...)` and install
  `ArtifactRealizationHandle(target_kind="publication")` on
  `RuntimeBindingState`. The publication handle shares the projection release
  contract, exposes the publication binding via `binding()`, and makes
  `RuntimeBindingState.close()` / handle close idempotently retire active
  publication before releasing the underlying runtime owner.
- Added daemon-mediated canonical layout provisioning for representation
  publication: `EnsureCanonicalLayout` in the daemon proto/service, a
  `GlobalStoreClient::put_layout_spec(...)` lowering, daemon controller
  admission/attach handling, and `DaemonClient.ensure_canonical_layout(...)`.
- Added the first daemon/controller realization planning structures in
  `materialization_policy_utils`: `ControllerRealizationPlan` with target,
  strategy, lifecycle, and resource-envelope subplans. `MaterializeReplica`
  now builds a controller plan after artifact, view, and selection identity
  resolution but before LIP or engine execution, covering both TensorDict
  handle-export materialization and no-lease retained-replica prefetch with
  explicit target layout digest, source selection digest, collective policy,
  lease/export lifetime, mutability, and operation-ticket release facts.
  `MaterializeIntoTarget`
  and `MaterializeIntoMappedTarget` now build this shared controller plan during
  common request preparation, attach realization plan facts to spans, and route
  mapped-target collective policy through the plan rather than an ad hoc local
  decision. Focused daemon policy tests cover single caller-target plans and
  target-set group strategy/lifecycle plans with version-set, transaction,
  barrier, collective, and release facts.
- Extended daemon/controller realization planning to retained serving prefetch
  and retained acquire paths. `PrefetchServingBinding` now builds controller
  plans for both retained single binding and retained target-set prefetch,
  including source selection mode, target-set layout digest, group barriers,
  staged publish barrier, retained reservation release policy, and per-member
  acquire-claim counts. `AcquireBindingValue` now builds a retained-acquire
  controller plan with runtime-attachment projection, CUDA IPC lease export,
  attachment-ref release policy, group acquire/version-set facts, and publish
  barrier state before minting local handle leases.
- Extended daemon/controller realization planning to target publication.
  `PublishTargetReplica` and `StartPublishTargetReplica` now build controller
  plans after token/front-door inspection, using the decoded publication scope
  and normalized byte space to record publication target identity, source
  selection digest, publication lease lifetime, lifecycle use-guard release,
  and published-replica resource envelope facts before continuing the existing
  lifecycle-kernel redemption path.
- Extended daemon/controller realization planning to binding controller flows.
  `CreateBinding`, `CreateOwnedBinding`, and `RefillOwnedBinding` now build the
  same `ControllerRealizationPlan` model before allocating/exporting binding
  memory or executing source-bound refill work. The plans record owned versus
  adopted binding targets, daemon-binding versus caller-region backing,
  CUDA-IPC or publication-token export semantics, staged group publication
  barriers, refill replacement/mutable-epoch release policy, source-selection
  digests, collective policy, and binding resource-envelope facts.
- Extended daemon/controller realization planning to assembly controller flows.
  `StartAssemblyAttempt`, `SealAssemblyAttempt`, `SealAssembly`, and
  `StartSealAssembly` now build the same `ControllerRealizationPlan` model
  before operation-lease acquisition or engine sealing. The plans record
  assembly workspace/attempt/seal target identity, requirement/member counts,
  closeout coordination, readiness/seal/post-seal barriers, operation-lease
  lifetime, sealed artifact mutability, and operation/artifact projection
  resource-envelope facts.
- Deepened daemon/controller resource envelopes with normalized
  `ControllerBodyBackingIntentPlan` fields and explicit `resource_authorities`
  that map current controller plans onto caller allocation, `BindingRegistry`,
  `BodyBackingManager`, `HandleLeaseRegistry`, `SessionLifecycleManager`,
  `LifecycleKernel`, assembly registry, and operation-lease authority concepts.
  All controller plans now finalize through `validate_controller_realization_plan(...)`,
  which keeps lifecycle/resource release policy aligned and rejects
  process-visible CUDA IPC / CPU memfd exports unless the lifecycle is
  token-backed.
- Extended daemon/controller resource envelopes with
  `ControllerResourceManagerLinkagePlan`, derived from the admitted controller
  plan rather than path-local branches. The envelope now records the expected
  BodyBackingManager intent, HandleLeaseRegistry export path,
  SessionLifecycleManager lease shape, LifecycleKernel capability path, and
  core `ExecutionCommitReport` shape for source-bound execution. Validation
  rejects mismatched linkage so future controller paths cannot add a resource
  manager decision outside the shared envelope.
- Centralized daemon controller realization span attributes in
  `attach_controller_realization_plan_span_attrs(...)`. Materialization,
  replica materialization, target materialization, owned-binding, assembly, and
  target-publication controllers now emit the same realization diagnostics,
  including resource projection/owner and manager-linkage fields, from the
  shared controller plan instead of carrying duplicate path-local span builders.
- Started converting daemon execution branches to consume the admitted
  controller resource envelope rather than recomputing target-specific export
  behavior. `MaterializeReplica` now passes
  `ControllerRealizationResourceEnvelope.export_kind` into the replica handle
  binding path, so `cuda_ipc_lease`, `cpu_memfd_lease`, and `none` select the
  corresponding HandleLeaseRegistry / CPU memfd / retained-ticket behavior.
  NO_LEASE retained replica realization now keeps the session/ticket backing
  without requiring a process-visible CUDA IPC handle.
- Added shared controller execution-admission helpers for resource authority
  and export-kind checks. `CreateBinding`, `CreateOwnedBinding`,
  `AcquireBindingValue`, and `MaterializeReplica` now fail before CUDA
  IPC/export-lease work unless the admitted controller plan carries the
  expected process-visible export kind and `HandleLeaseRegistry` /
  `SessionLifecycleManager` / `LifecycleKernel` authorities. `CreateBinding`
  also now derives its daemon-owned CUDA export path versus adopted
  publication-token path from the admitted envelope rather than branching
  directly on request ownership at the resource-export site.
- Extended the same execution-admission pattern to operation/publication
  lifecycle work. Assembly start/seal operation-lease acquisition now requires
  the admitted controller plan to expose `export_kind="operation_lease"` and
  `OperationLeaseRegistry` authority, and publication start/sync redemption now
  requires `export_kind="publication_lease"` plus `LifecycleKernel` authority
  before continuing the existing lifecycle-kernel path.
- Extended execution-admission gating to caller-target direct-write
  materialization. `MaterializeIntoTarget` and `MaterializeIntoMappedTarget`
  now require the admitted controller plan to expose
  `export_kind="registered_region_direct_write"` and `caller_allocation`
  authority before they continue into external target storage leases and engine
  direct-write execution.
- Extended execution-admission gating to binding refill and retained serving
  prefetch paths. `RefillOwnedBinding` now consumes the admitted refill export
  kind (`publication_token_or_none` for ready-artifact replacement, `none` for
  execution-only mutable refill) plus `BodyBackingManager` and
  `BindingRegistry` authorities before storage layout and execution. Retained
  `PrefetchServingBinding` now consumes member/set reservation export kinds
  (`binding_reservation` / `binding_reservation_set`) plus retained binding
  registry/session authorities before reservation execution.
- Expanded focused `ResolvedArtifactSelection` coverage for canonical
  artifact-id/key exclusivity, daemon key/index resolution, subset/view identity,
  mapped source view hints, generation-sensitive digest stability, and explicit
  artifact profile / authority-scope digest inputs.
- Added handle-level realization admission guards. `ArtifactRealizationHandle`
  now fails closed when the report target kind disagrees with the handle, source
  selection authority is absent, release strictness or mutability contracts are
  missing, an attached target plan fails envelope admission, or observed
  fallback diagnostics are paired with an empty fallback policy.
- Added a shared synchronous-facade profile event for `Artifact.realize(...)`
  targets. TensorDict, caller-tensor direct write, owned/adopted binding, and
  mounted-source realization now emit one `artifact.realize` event shaped from
  `ArtifactRealizationReport`, including source/target identity, operation
  backend, risk labels, resource-envelope facts, cost counters, publishability,
  and target/strategy/lifecycle summaries when those plans are present.
- Extended the same report-shaped `artifact.realize` profile event to async and
  serving-backed targets. Retained replica operation results, retained binding
  prefetch, target-set prefetch, runtime attachment handles, model-runtime
  handles, and publication projections now emit through the shared realization
  profile payload helper instead of per-path field mappings.
- Extended comparable report-field coverage across TensorDict, binding,
  retained handoff, runtime attachment, and target-set paths. Retained replica
  and retained binding reports now carry `target_plan` like the synchronous
  facade, binding, runtime attachment, publication, mounted-source, and
  target-set report helpers, and the SDK test matrix asserts common identity,
  backend, envelope, publishability, risk-label, and admission fields across
  migrated report families.
- Added caller-tensor/get-into cost coverage through the
  `Artifact.realize(ArtifactRealizationSpec.caller_tensors(...))` facade. CPU
  caller tensors assert temporary-payload copy bytes, copy count, temp bytes,
  release policy, and target layout digest; CUDA caller tensors assert
  registered-region direct-write bytes when CUDA is available.
- Extended CPU TensorDict mutability coverage. The test matrix now asserts the
  read-mostly private-copy envelope and rejects `dict` mutation APIs including
  item set/delete, clear, pop, popitem, setdefault, update, and in-place union
  while preserving the realized tensor owner.
- Extended async realization operation-contract coverage. Retained replica
  prefetch now asserts `Operation` status, wait/result handoff, cancel release,
  deterministic idempotency, `NO_LEASE`, and degraded wait timeout behavior;
  completed retained-binding prefetch asserts `done()`, status, result, and
  non-cancellable completed-operation behavior.
- Extended release lifecycle coverage so the same contract matrix explicitly
  checks TensorDict export/backing release, caller-tensor temporary-copy fallback
  cleanup, owned-binding close/retire cleanup, retained-acquire runtime cleanup,
  and ordinary runtime attachment cleanup. Each case asserts the normalized
  release policy and idempotent action execution.
- Migrated SDK representation publication layout helpers in
  `tensorcast/api/store/__init__.py` away from direct Global Store stubs and
  channels. Canonical layout provisioning now goes through the Store Daemon
  client boundary.
- Audited SDK representation publication/layout convenience call sites after
  the publication handle migration: `tensorcast/api` contains no direct
  `GlobalStoreCompositeStub`, generated Global Store stub, or
  `grpc.insecure_channel` usage, and canonical layout provisioning remains
  daemon-mediated through `ensure_canonical_layout(...)`.
- Updated representation-publication tests to exercise the daemon client
  canonical-layout wrapper directly instead of monkeypatching the private SDK
  helper.
- Added focused guardrail/unit coverage in
  `tests/python/api/test_realization_kernel.py` for resolver digest stability,
  `msa1:` mounted-source admission fail-closed behavior, target-envelope
  admission, TensorDict projection ownership, direct selection-builder imports,
  direct Global Store channel usage in the migrated realization files, and
  forbidden direct Global Store access across `tensorcast/api`. The Global Store
  guardrail rejects direct generated GS stubs plus sync/aio insecure and secure
  gRPC channel construction in SDK API code; daemon-mediated operation helpers
  remain allowed.
- Added daemon controller coverage for `EnsureCanonicalLayout` provisioning and
  artifact attachment, backed by recording Global Store client test hooks.
- Exported the new SDK realization types from `tensorcast.api.store`.

Verified:

- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/test_store_pipelines_unit.py tests/python/test_store_session_api.py -q`
- `source .venv/bin/activate && pytest tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_store_pipelines_unit.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py tests/python/test_store_pipelines_unit.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/_materialize.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && ruff check tensorcast/daemon_ctl.py tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/_materialize.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && ruff check tensorcast/daemon_ctl.py tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/_materialize.py tensorcast/api/plan/plan.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/test_assembly_attempt.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/_materialize.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/_materialize.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tensorcast/daemon_ctl.py tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/_materialize.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tensorcast/api/store/owned_binding_slot.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/daemon_ctl.py tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/_materialize.py tensorcast/api/plan/plan.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py tests/python/test_store_pipelines_unit.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_binding.py -q`
- `source .venv/bin/activate && pytest tests/python/test_assembly_attempt.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_plan_spec.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py tests/python/api/test_plan_spec.py tests/python/test_assembly_attempt.py tests/python/test_store_pipelines_unit.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && python -m py_compile tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_tensor_into_materializes_subset_only tests/python/api/test_artifact_handle.py::test_bind_coerces_serving_manifest_into_runtime_policy tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff format tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py::test_tensor_into_materializes_subset_only -q`
- `source .venv/bin/activate && ruff format tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_prefetch_operation.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_artifact_handle.py tests/python/test_binding.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && python -m py_compile tensorcast/types.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && ruff check tensorcast/types.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_prefetch_operation.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_serving_binding_target.py`
- `source .venv/bin/activate && pyright tensorcast/types.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/serving/retained_binding.py tensorcast/serving/_runtime_impl/lifecycle.py tests/python/api/test_realization_kernel.py tests/python/test_serving_integration.py tests/python/test_serving_retained_binding_acquire.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/serving/retained_binding.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_load_prepared_local_ready_uses_restore tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core tests/python/test_serving_state.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_state.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_load_prepared_local_ready_uses_restore tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core tests/python/test_binding.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py -q`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py::test_prefetch_serving_binding_real_cuda_worker_read_and_release -q`
- `source .venv/bin/activate && pytest tests/python/test_dual_daemon_global_store_tp_view.py -q`
- `source .venv/bin/activate && pytest tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/serving/replica_publication.py tests/python/test_serving_replica_publication.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/serving/replica_publication.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_replica_publication.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_state.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/replica_publication.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py tests/python/test_serving_retained_binding_acquire.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/serving/replica_publication.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py tests/python/test_serving_replica_publication.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_load_prepared_local_ready_uses_restore tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/__init__.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/serving/replica_publication.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py tests/python/test_serving_replica_publication.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_load_prepared_local_ready_uses_restore tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/api/store/artifact.py tensorcast/serving/_runtime_impl/lifecycle.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_load_prepared_local_ready_uses_restore tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core -q`
- `git diff --check`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py tests/python/test_store_session_api.py::test_promote_mounted_source_function_delegates_to_session tests/python/test_store_session_api.py::test_promote_mounted_source_function_delegates_timeout_override -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && ruff check tensorcast/serving/retained_binding.py tensorcast/serving/runtime_attachment.py tensorcast/serving/replica_publication.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py`
- `source .venv/bin/activate && pyright tensorcast/serving/retained_binding.py tensorcast/serving/runtime_attachment.py tensorcast/serving/replica_publication.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core -q`
- `source .venv/bin/activate && ruff check tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/runtime_attachment.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/runtime_attachment.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership tests/python/test_serving_integration.py::test_runtime_binding_materialization_closes_runtime_handle_on_state_failure tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership tests/python/test_serving_integration.py::test_runtime_binding_materialization_closes_runtime_handle_on_state_failure tests/python/test_serving_integration.py::test_serving_integration_load_and_reload_use_materialization tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/runtime_attachment.py tests/python/api/test_realization_kernel.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/runtime_attachment.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_in_core tests/python/test_serving_integration.py::test_serving_integration_acquire_retained_binding_uses_materialization -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_integration.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/serving/runtime_attachment.py tensorcast/serving/replica_publication.py tests/python/api/test_realization_kernel.py tests/python/test_serving_replica_publication.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/serving/runtime_attachment.py tensorcast/serving/replica_publication.py tests/python/api/test_realization_kernel.py tests/python/test_serving_replica_publication.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_replica_publication.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py::test_realize_async_retained_replica_operation_status_wait_and_cancel tests/python/api/test_prefetch_operation.py::test_realize_async_retained_replica_degraded_wait_raises_timeout tests/python/api/test_prefetch_operation.py::test_realize_async_retained_binding_completed_operation_status_and_cancel -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_release_contract_lifecycle_matrix_runs_policy_actions_once -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_materialization_token_guards.py`
- `source .venv/bin/activate && pyright tests/python/api/test_materialization_token_guards.py`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_token_guards.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py tests/python/test_cpu_memfd_lease_raii.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py tests/python/test_cpu_memfd_lease_raii.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_cpu_memfd_lease_raii.py -q`
- `source .venv/bin/activate && bash tools/build_proto_python.sh`
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=real --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:owned_binding_service_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_no_lease_materialize_test //daemon:materialization_policy_utils_test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `git diff --check`
- `bazel test //daemon:materialization_policy_utils_test //daemon:grpc_service_impl_start_seal_assembly_test //daemon:assembly_closeout_identity_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `git diff --cached --check && git diff --check`
- `bazel test //daemon:materialization_policy_utils_test //daemon:grpc_service_impl_start_seal_assembly_test //daemon:assembly_closeout_identity_utils_test //daemon:owned_binding_service_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_no_lease_materialize_test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=real --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_start_seal_assembly_test //daemon:grpc_service_impl_publish_target_replica_test //daemon:materialization_policy_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test //daemon:materialize_into_target_validation_test //daemon:materialize_into_mapped_target_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=real --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_operation_rpc_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py::test_prefetch_serving_binding_real_cuda_worker_read_and_release -q`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=real --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff format tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff format tensorcast/api/store/artifact.py tensorcast/api/store/realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tensorcast/api/store/realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/replica_publication.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py::test_realize_async_prefetch_targets_emit_report_shaped_profile_events tests/python/test_serving_replica_publication.py::test_publish_current_replica_returns_published_projection tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership -q`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/replica_publication.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py`
- `source .venv/bin/activate && python -m py_compile tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/test_serving_replica_publication.py tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership -q`
- `git diff --check && git diff --cached --check`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_tensor_dict_projection_rejects_mapping_mutations tests/python/api/test_realization_kernel.py::test_cpu_tensor_dict_envelope_reports_private_copy_mutability -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_tensor_subset_materialization_and_release tests/python/api/test_artifact_handle.py::test_tensor_dict_with_diagnostics_release_is_idempotent -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/materialization.py tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/test_store_pipelines_unit.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/materialization.py tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/test_store_pipelines_unit.py`
- `source .venv/bin/activate && pytest tests/python/test_store_pipelines_unit.py::test_get_into_returns_fallback_result_and_unloads tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_temporary_copy_costs -q`
- `source .venv/bin/activate && pytest tests/python/test_store_pipelines_unit.py tests/python/api/test_materialization_pipeline_v2.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_tensor_dict_handle_rejects_binding_lifecycle_capabilities -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_risk_closure_matrix_covers_plan_risks_and_enforcement_fields -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_binding_envelope_and_report_capture_identity_diagnostics -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_tensor_dict_and_adopted_binding_share_source_selection_with_separate_target_digests -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && ruff check tensorcast/serving/retained_binding.py tensorcast/api/store/realization_kernel.py tests/python/test_serving_retained_binding_acquire.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/serving/retained_binding.py tensorcast/api/store/realization_kernel.py tests/python/test_serving_retained_binding_acquire.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_retained_binding_acquire.py::test_acquire_retained_binding_rejects_expired_capability_before_daemon_call tests/python/test_serving_retained_binding_acquire.py::test_retained_binding_debug_status_tracks_capability_ttl_and_lifecycle tests/python/test_serving_retained_binding_acquire.py::test_retained_prefetch_retention_policy_round_trips_ttl_and_idle_retire tests/python/api/test_realization_kernel.py::test_retained_binding_report_captures_capability_expiry tests/python/api/test_realization_kernel.py::test_reports_share_core_realization_fields_across_targets -q`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py::test_prefetch_serving_binding_real_cuda_worker_read_and_release -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_retained_binding_acquire.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_realization_kernel_paths_do_not_open_global_store_channels tests/python/api/test_realization_kernel.py::test_sdk_api_paths_do_not_open_global_store_channels tests/python/api/test_realization_kernel.py::test_risk_closure_matrix_covers_plan_risks_and_enforcement_fields -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_resolve_artifact_selection_keeps_target_plan_identity_separate tests/python/api/test_realization_kernel.py::test_risk_closure_matrix_covers_plan_risks_and_enforcement_fields -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/test_serving_retained_binding_acquire.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_mounted_source_realize_rejects_non_msa1_subject tests/python/api/test_realization_kernel.py::test_risk_closure_matrix_covers_plan_risks_and_enforcement_fields -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_resource_envelope_adapter_matrix_covers_implementation_objects -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_mounted_source_selection_rejects_key_activation tests/python/api/test_realization_kernel.py::test_mounted_source_selection_rejects_global_store_index_routing tests/python/api/test_realization_kernel.py::test_mounted_source_selection_rejects_durable_authority_override tests/python/api/test_realization_kernel.py::test_local_ready_pending_verification_report_records_admission_state tests/python/api/test_artifact_handle.py::test_mounted_source_realize_promotes_and_reports_identity tests/python/api/test_artifact_handle.py::test_mounted_source_realize_rejects_key_mapping_activation -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py -q`
- `source .venv/bin/activate && bash tools/build_proto_python.sh`
- `source .venv/bin/activate && ruff check tensorcast/types.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py`
- `source .venv/bin/activate && pyright tensorcast/types.py tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py::test_realize_async_target_set_per_part_selection_reports_group_lifecycle tests/python/api/test_prefetch_serving_binding_target.py::test_prefetched_serving_binding_staged_result_proto_roundtrip -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_prefetch_serving_binding_target.py -q`
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_operation_rpc_test //daemon:owned_binding_service_test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && ruff check tensorcast/api/store/serving_binding_reference_consumer.py tensorcast/api/store/__init__.py tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/serving_binding_reference_consumer.py tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py::test_prefetch_serving_binding_set_real_cuda_worker_reads_members -q`
- `source .venv/bin/activate && pytest tests/python/api/test_serving_binding_reference_consumer.py -q`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py -q`
- `source .venv/bin/activate && ruff check tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_runtime_binding_materialization_attaches_and_transfers_ownership -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py::test_retained_binding_realization_rejects_target_set_bypass tests/python/api/test_prefetch_operation.py::test_realize_async_retained_binding_set_attaches_target_set_report tests/python/api/test_prefetch_operation.py::test_prefetch_target_set_uses_target_set_realization_spec -q`
- `source .venv/bin/activate && pytest tests/python/api/test_prefetch_operation.py -q`
- `rg -n "ServingBindingTarget|ServingBindingSetTarget|PrefetchedServingBinding|PrefetchedServingBindingSet|serving binding target|serving target" README.md docs tensorcast/api/store/README.md -g'*.md'`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_resource_envelope_requires_shared_risk_fields -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_artifact_no_longer_exposes_removed_selection_adapters tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_artifact_selection_adapter tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_owner_source_selection_adapter tests/python/api/test_artifact_handle.py::test_selection_reuses_eager_view_metadata tests/python/api/test_prefetch_operation.py::test_prefetch_uses_deterministic_operation_id tests/python/api/test_plan_spec.py::test_prefetch_many_lowers_to_inline_artifact_set_ref -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/api/test_plan_spec.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/api/test_plan_spec.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py tests/python/api/test_prefetch_operation.py tests/python/api/test_plan_spec.py -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_public_artifact_entrypoints_call_realization_facade -q`
- `rg -n 'Convenience materialization APIs|ArtifactRealizationSpec|test_public_artifact_entrypoints_call_realization_facade' tensorcast/api/store/README.md docs/plans/0121-unified-artifact-realization-kernel.md tests/python/api/test_realization_kernel.py`
- `bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_no_lease_materialize_test //daemon:materialize_into_target_validation_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_operation_rpc_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialization_policy_utils_test //daemon:grpc_service_impl_no_lease_materialize_test //daemon:materialize_into_target_validation_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_operation_rpc_test //daemon:grpc_service_impl_start_seal_assembly_test //daemon:grpc_service_impl_publish_target_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py tests/python/test_dual_daemon_global_store_tp_view.py -q`
- `source .venv/bin/activate && ruff check tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py`
- `source .venv/bin/activate && pyright tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py::test_prefetch_serving_binding_set_real_cuda_worker_reads_members -q`
- `rg -n 'realization_kernel.py|materialization_policy_utils|test_prefetch_serving_binding_real_cuda_e2e|resource-manager linkage' docs/designs/0121-unified-artifact-realization-kernel.md tensorcast/api/store/README.md docs/plans/0121-unified-artifact-realization-kernel.md`
- `source .venv/bin/activate && ruff check tensorcast/serving/runtime_view.py tensorcast/serving/_runtime_impl/lifecycle.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/serving/runtime_view.py tensorcast/serving/_runtime_impl/lifecycle.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_runtime_worker_view_accepts_typed_source_selection_projection tests/python/test_serving_integration.py::test_source_selection_projection_from_materialization_diagnostics tests/python/test_serving_integration.py::test_execution_diagnostics_seed_runtime_source_selection_projection tests/python/test_serving_integration.py::test_materialization_diagnostics_seed_runtime_source_selection_projection tests/python/test_serving_integration.py::test_runtime_binding_result_captures_materialization_diagnostics -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_temporary_copy_costs tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_cuda_direct_write_costs tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_cuda_fallback_copy_costs tests/python/api/test_realization_kernel.py::test_resource_envelope_adapter_matrix_covers_implementation_objects tests/python/api/test_realization_kernel.py::test_release_contract_lifecycle_matrix_runs_policy_actions_once -q`
- `source .venv/bin/activate && ruff check tensorcast/api/_materialize.py tensorcast/api/store/artifact.py tensorcast/api/store/owned_binding_slot.py tests/python/api/test_realization_kernel.py tests/python/test_binding.py`
- `source .venv/bin/activate && pyright tensorcast/api/_materialize.py tensorcast/api/store/artifact.py tensorcast/api/store/owned_binding_slot.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_artifact_no_longer_exposes_removed_selection_adapters tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_artifact_selection_adapter tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_owner_source_selection_adapter tests/python/test_binding.py::test_binding_append_publish_uses_view_routing tests/python/test_binding.py::test_binding_view_reuse tests/python/test_binding.py::test_binding_swap_preserves_data_ptr tests/python/test_binding.py::test_bind_does_not_delegate_to_bind_into -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tests/python/api/test_realization_kernel.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_tensor_dict_with_diagnostics_reports_source_and_bytes tests/python/api/test_realization_kernel.py::test_binding_envelope_and_report_capture_identity_diagnostics tests/python/api/test_realization_kernel.py::test_runtime_attachment_envelope_and_report_capture_release_contract tests/python/api/test_realization_kernel.py::test_realization_reports_do_not_expose_legacy_diagnostics_field -q`
- `source .venv/bin/activate && ruff check tensorcast/api/runtime.py tests/python/api/test_runtime.py`
- `source .venv/bin/activate && pyright tensorcast/api/runtime.py tests/python/api/test_runtime.py`
- `source .venv/bin/activate && pytest tests/python/api/test_runtime.py::test_runtime_hydrate_engine_request_id_goes_to_daemon_unrewritten tests/python/api/test_runtime.py::test_runtime_publish_result_does_not_install_hydrate_rewrite_cache tests/python/api/test_runtime.py::test_runtime_execute_plan_propagates_call_deadline_to_daemon_timeout -q`
- `source .venv/bin/activate && pytest tests/python/api/test_runtime.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_tensor_subset_materialization_and_release tests/python/api/test_artifact_handle.py::test_tensor_dict_with_diagnostics_reports_source_and_bytes -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/serving_builder.py tensorcast/engine_adapter/adapter.py tensorcast/serving/builder/publication.py tensorcast/serving/builder/__init__.py tensorcast/api/plan/__init__.py tensorcast/api/store/__init__.py tensorcast/api/__init__.py tensorcast/__init__.py tests/python/test_serving_publication_types.py tests/python/test_serving_builder_publication.py tests/python/test_binding.py tensorcast/api/signals.py tests/python/api/test_runtime.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/serving_builder.py tensorcast/engine_adapter/adapter.py tensorcast/serving/builder/publication.py tensorcast/serving/builder/__init__.py tensorcast/api/plan/__init__.py tensorcast/api/store/__init__.py tensorcast/api/__init__.py tensorcast/__init__.py tests/python/test_serving_publication_types.py tests/python/test_serving_builder_publication.py tensorcast/api/signals.py tests/python/api/test_runtime.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_builder_publication.py tests/python/test_serving_publication_types.py::test_build_pure_transform_transform_spec_wraps_transform_args tests/python/test_serving_publication_types.py::test_build_pure_transform_transform_spec_can_omit_representation_hash tests/python/test_serving_publication_types.py::test_build_pure_transform_publication_spec_wraps_typed_inputs tests/python/api/test_plan_spec.py::test_plan_transform_register_pure_transform_builds_repo_owned_spec tests/python/node_agent/test_plan_execution.py::test_engine_adapter_identity_transform_register_can_return_pure_transform_bundle -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_publication_types.py::test_pure_transform_publication_no_longer_exposes_string_arg_fallback -q`
- `source .venv/bin/activate && pytest tests/python/api/test_runtime.py::test_runtime_directory_reads_daemon_served_routes tests/python/test_binding.py::test_binding_swap_does_not_encode_transport_group_tags_into_operation_id -q`
- `source .venv/bin/activate && pytest tests/python/api/test_runtime.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_publication_types.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_builder_publication.py -q`
- `source .venv/bin/activate && pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py tests/python/test_dual_daemon_global_store_tp_view.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_runtime.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py tests/python/test_serving_runtime_contract.py -q`
- `bash tools/build_proto_python.sh`
- `bazel test //proto/... //daemon:grpc_service_impl_disk_index_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py -q`
- `source .venv/bin/activate && ruff check tensorcast/daemon_ctl.py`
- `source .venv/bin/activate && pyright tensorcast/daemon_ctl.py`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py tests/python/api/test_materialization_token_guards.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/_materialize.py tests/python/api/test_materialization_pipeline_v2.py tests/python/api/test_materialization_token_guards.py`
- `source .venv/bin/activate && pyright tensorcast/api/_materialize.py tests/python/api/test_materialization_token_guards.py`
- `bazel test //daemon:materialization_policy_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_runtime_worker_view_projection_is_typed_not_diagnostics_only tests/python/test_serving_integration.py::test_runtime_worker_view_ignores_redundant_source_selection_diagnostics tests/python/test_serving_integration.py::test_source_selection_projection_from_materialization_diagnostics tests/python/test_serving_integration.py::test_execution_diagnostics_seed_runtime_source_selection_projection tests/python/test_serving_integration.py::test_materialization_diagnostics_seed_runtime_source_selection_projection -q`
- `source .venv/bin/activate && ruff check tensorcast/serving/runtime_view.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/serving/runtime_view.py`
- `bash tools/build_proto_python.sh`
- `bazel test //proto/... //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_binding.py::test_binding_append_publish_uses_view_routing tests/python/test_binding.py::test_binding_view_reuse tests/python/test_binding.py::test_create_binding_treats_empty_current_value_as_absent tests/python/test_binding.py::test_binding_missing_seal_current_value_fails_fast tests/python/api/test_mapped_binding.py tests/python/test_inplace_slot.py tests/python/api/test_placement_pin.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/__init__.py tensorcast/api/store/artifact.py tensorcast/api/store/inplace_slot.py tensorcast/api/store/owned_binding_slot.py tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/test_inplace_slot.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/__init__.py tensorcast/api/store/artifact.py tensorcast/api/store/inplace_slot.py tensorcast/api/store/owned_binding_slot.py`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_runtime_worker_view_ignores_redundant_source_selection_diagnostics tests/python/test_serving_publication_types.py::test_pure_transform_publication_no_longer_exposes_string_arg_fallback tests/python/api/test_realization_kernel.py::test_artifact_no_longer_exposes_removed_selection_adapters tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_artifact_selection_adapter tests/python/api/test_realization_kernel.py::test_sdk_lowering_does_not_call_removed_owner_source_selection_adapter tests/python/api/test_realization_kernel.py::test_realization_reports_do_not_expose_legacy_diagnostics_field -q`
- `source .venv/bin/activate && ruff check tests/python/test_serving_integration.py tests/python/test_serving_publication_types.py tests/python/api/test_realization_kernel.py`
- `bazel test //daemon:grpc_service_impl_batch_runtime_test --test_arg='[daemon][batch][frontdoor]' --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py tests/python/test_store_pipelines_unit.py::test_get_into_returns_fallback_result_and_unloads tests/python/api/test_artifact_region_cleanup.py tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_cuda_direct_write_costs tests/python/api/test_artifact_handle.py::test_caller_tensors_realization_reports_cuda_fallback_copy_costs tests/python/api/test_mapped_binding.py::test_bind_into_surfaces_region_registry_capacity_errors tests/python/test_binding.py::test_create_client_binding_uses_region_backed_layout_on_rpc -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/materialization.py tensorcast/api/store/target_region_lifecycle.py tensorcast/api/store/realization_kernel.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_artifact_region_cleanup.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/materialization.py tensorcast/api/store/target_region_lifecycle.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_artifact_region_cleanup.py`
- `bash tools/build_proto_python.sh`
- `bazel run @rules_buf_toolchains//:buf -- format ./proto -w`
- `bazel test //proto/... //daemon:grpc_service_impl_disk_index_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_store_region_registration.py tests/python/api/test_artifact_region_cleanup.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/realization_kernel.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/realization_kernel.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/store/artifact.py tensorcast/api/store/__init__.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_binding_materialization_diagnostics_use_realization_kernel_helper tests/python/api/test_realization_kernel.py::test_binding_envelope_and_report_capture_identity_diagnostics tests/python/api/test_artifact_handle.py::test_tensor_subset_materialization_and_release tests/python/test_assembly_attempt.py::test_register_pure_transform_publication_registers_manifest_bearing_artifact -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py -q`
- `source .venv/bin/activate && ruff check tensorcast/api/store/inplace_slot.py tests/python/test_inplace_slot.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/inplace_slot.py tensorcast/api/store/target_region_lifecycle.py`
- `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py -q`
- `bash tools/build_proto_python.sh`
- `bazel run @rules_buf_toolchains//:buf --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error -- format ./proto -w`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py tests/python/api/test_artifact_handle.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_target_publication_legacy_capability_proto_is_removed tests/python/api/test_artifact_handle.py::test_realize_emits_report_shaped_profile_event tests/python/api/test_artifact_handle.py::test_tensor_dict_with_diagnostics_reports_source_and_bytes -q`
- `bazel test //daemon:distributed_security_kernel_test //proto/... //daemon:grpc_service_impl_publish_target_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //core/store/replica:replica_placement_test //core/store/replica:replica_p2p_registration_test //core/store/runtime/replica:replica_runtime_test //core/store/materialization/dataplane:disk_mmap_test //core/store/materialization/dataplane:disk_cpu_load_test //core/store/materialization/dataplane:disk_cpu_gpu_transfer_test //core/store/materialization/dataplane:disk_error_test //core/store/materialization/dataplane:disk_multi_gpu_test //core/store/materialization/dataplane:artifact_gpu_auto_release_test //core/store/materialization/dataplane:artifact_verification_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //core/store/runtime/replica:replica_runtime_test //core/store/materialization/dataplane:disk_cpu_gpu_transfer_test //core/store/materialization/dataplane:disk_multi_gpu_test //core/store/materialization/dataplane:artifact_gpu_auto_release_test //core/store/replica:replica_p2p_registration_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_artifact_realization_report_seeds_runtime_source_selection_projection tests/python/test_serving_integration.py::test_artifact_realization_report_fallback_uses_strategy_and_envelope_facts tests/python/test_serving_integration.py::test_execution_diagnostics_seed_runtime_source_selection_projection tests/python/test_serving_integration.py::test_materialization_diagnostics_seed_runtime_source_selection_projection -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_artifact_realization_report_seeds_runtime_source_selection_projection tests/python/test_serving_integration.py::test_artifact_realization_report_fallback_uses_strategy_and_envelope_facts tests/python/test_serving_integration.py::test_runtime_worker_view_ignores_redundant_source_selection_diagnostics -q`
- `source .venv/bin/activate && ruff check tensorcast/serving/runtime_view.py tensorcast/serving/_runtime_impl/lifecycle.py tests/python/test_serving_integration.py`
- `source .venv/bin/activate && pyright tensorcast/serving/runtime_view.py tensorcast/serving/_runtime_impl/lifecycle.py`
- `bazel test //daemon:grpc_service_impl_batch_runtime_test --test_filter='[daemon][batch][home],[daemon][batch][host_shared][put]' --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_batch_runtime_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tests/python/test_inplace_slot.py tests/python/api/test_mapped_binding.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/api/store/artifact.py`
- `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py::test_bind_into_uses_create_binding_publication_token tests/python/test_inplace_slot.py::test_bind_into_releases_regions_when_create_binding_fails tests/python/test_inplace_slot.py::test_bind_into_closes_binding_and_releases_regions_for_malformed_create_binding -q`
- `source .venv/bin/activate && pytest tests/python/api/test_mapped_binding.py::test_mapped_binding_uses_materialize_into_mapped_target tests/python/api/test_mapped_binding.py::test_mapped_bind_into_closes_binding_and_releases_regions_for_malformed_create_binding -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_client_binding_rollbacks_log_cleanup_failures_instead_of_suppressing -q`
- `source .venv/bin/activate && ruff check tensorcast/daemon_ctl.py tensorcast/api/_materialize.py tensorcast/api/store/__init__.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/node_agent/executor.py tests/python/test_daemon_ctl_resolve_rpc_config.py tests/python/test_store_session_api.py tests/python/test_inplace_slot.py tests/python/test_binding.py tests/python/node_agent/test_plan_execution.py tests/python/api/test_artifact_handle.py tests/python/api/test_materialization_pipeline_v2.py tests/python/api/test_materialization_token_guards.py tests/python/api/test_mapped_binding.py`
- `source .venv/bin/activate && pyright tensorcast/daemon_ctl.py tensorcast/api/_materialize.py tensorcast/api/store/__init__.py tensorcast/api/store/artifact.py tensorcast/api/store/materialization.py tensorcast/api/store/inplace_slot.py tensorcast/node_agent/executor.py`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_token_guards.py -q`
- `source .venv/bin/activate && pytest tests/python/test_daemon_ctl_resolve_rpc_config.py::test_import_artifact_from_path_uses_configurable_timeout_and_retries tests/python/test_daemon_ctl_resolve_rpc_config.py::test_import_artifact_from_path_stream_uses_configurable_timeout -q`
- `source .venv/bin/activate && pytest tests/python/api/test_artifact_handle.py::test_from_disk_progress_mode_uses_stream_resolution tests/python/api/test_artifact_handle.py::test_import_from_disk_uses_import_stream_and_publishes_key -q`
- `source .venv/bin/activate && pytest tests/python/api/test_materialization_pipeline_v2.py::test_materialize_subset_preserves_generation tests/python/api/test_materialization_pipeline_v2.py::test_disk_fallback_verify_flag_passed -q`
- `source .venv/bin/activate && pytest tests/python/node_agent/test_plan_execution.py::test_node_agent_executes_artifact_actions -q`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_sdk_materialization_client_surface_has_no_retired_v2_aliases -q`
- `source .venv/bin/activate && ruff check tensorcast/api/_register.py tensorcast/api/store tests/python/api/test_realization_kernel.py tests/python/test_store_region_registration.py tests/python/test_inplace_slot.py tests/python/test_binding.py tests/python/api/test_mapped_binding.py`
- `source .venv/bin/activate && pyright tensorcast/api/_register.py tensorcast/api/store`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_realization_lifecycle_code_does_not_silently_suppress_broad_exceptions tests/python/api/test_realization_kernel.py::test_target_region_registration_uses_offset_aware_cuda_ipc_export_only tests/python/api/test_realization_kernel.py::test_tensor_dict_projection_fails_closed_when_owner_attachment_fails tests/python/test_store_region_registration.py tests/python/test_inplace_slot.py::test_bind_into_uses_create_binding_publication_token tests/python/test_inplace_slot.py::test_bind_into_releases_regions_when_create_binding_fails tests/python/test_inplace_slot.py::test_bind_into_closes_binding_and_releases_regions_for_malformed_create_binding tests/python/api/test_mapped_binding.py::test_mapped_binding_uses_materialize_into_mapped_target tests/python/api/test_mapped_binding.py::test_mapped_bind_into_closes_binding_and_releases_regions_for_malformed_create_binding -q`
- `bash tools/build_proto_python.sh`
- `source .venv/bin/activate && ruff check tensorcast/global_store/cluster_runtime_rpc.py tensorcast/global_store/db_utils.py tensorcast/global_store/repositories/base.py tensorcast/global_store/rpc/replica_registration_rpc_handler.py tensorcast/global_store/rpc/transport_rpc_handler.py tensorcast/global_store/services/instance_service.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/global_store/cluster_runtime_rpc.py tensorcast/global_store/db_utils.py tensorcast/global_store/repositories/base.py tensorcast/global_store/rpc/replica_registration_rpc_handler.py tensorcast/global_store/rpc/transport_rpc_handler.py tensorcast/global_store/services/instance_service.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_realization_lifecycle_code_does_not_silently_suppress_broad_exceptions tests/python/api/test_realization_kernel.py::test_daemon_loaded_replica_status_surface_has_no_retired_v2_name -q`
- `source .venv/bin/activate && pytest tests/python/global_store/test_instances.py tests/python/global_store/test_grpc_service.py::TestGRPCService::test_instance_capability_flags_clear_on_heartbeat tests/python/global_store/test_grpc_service.py::TestGRPCService::test_request_replica_transport_requires_request_id tests/python/global_store/test_grpc_service.py::TestGRPCService::test_request_replica_transport_request_id_replay -q`
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_status_mixed_residency_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && ruff check tensorcast/_c_ext.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/_c_ext.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_target_region_registration_uses_offset_aware_cuda_ipc_export_only tests/python/api/test_realization_kernel.py::test_core_cuda_ipc_export_no_longer_exposes_non_offset_handle_path -q`
- `bazel test //core/checkpoint:duplicate_buffer_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && BUILD_CORE=1 BUILD_EXTENSION=1 python setup.py build_ext`
- `source .venv/bin/activate && python -c "from tensorcast import _c_ext; assert 'get_cuda_memory_handle' not in _c_ext.__all__; assert not hasattr(_c_ext, 'get_cuda_memory_handle'); module = _c_ext.get_c_ext(); assert hasattr(module, 'get_cuda_memory_handle_with_offset'); assert not hasattr(module, 'get_cuda_memory_handle')"`
- `bash tools/build_proto_python.sh`
- `source .venv/bin/activate && ruff check tensorcast/daemon_runtime_config.py tests/python/api/test_realization_kernel.py tests/python/test_daemon_runtime_config.py tests/python/utils/daemon.py`
- `source .venv/bin/activate && pyright tensorcast/daemon_runtime_config.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_mounted_source_config_no_longer_exposes_absolute_fallback_mode tests/python/test_daemon_runtime_config.py::test_load_daemon_config_defaults_public_disk_source_from_storage_path -q`
- `bazel test //core/common:daemon_config_io_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:import_artifact_from_path_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && ruff check tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_daemon_canonical_index_loading_uses_explicit_authority_not_disk_fallback -q`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_core_cuda_ipc_export_no_longer_exposes_non_offset_handle_path tests/python/api/test_realization_kernel.py::test_daemon_canonical_index_loading_uses_explicit_authority_not_disk_fallback -q`
- `bazel test //daemon:grpc_service_impl_disk_index_test //daemon:grpc_service_impl_no_lease_materialize_test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && ruff check tensorcast/serving/retained_binding.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/local_ready.py tensorcast/serving/recipe_build.py tests/python/api/test_realization_kernel.py`
- `source .venv/bin/activate && pyright tensorcast/serving/retained_binding.py tensorcast/serving/_runtime_impl/lifecycle.py tensorcast/serving/local_ready.py tensorcast/serving/recipe_build.py`
- `source .venv/bin/activate && pytest tests/python/api/test_realization_kernel.py::test_realization_lifecycle_code_does_not_silently_suppress_broad_exceptions -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_framework_context_preserves_optional_host_placement_payloads tests/python/test_serving_integration.py::test_runtime_binding_materialization_closes_binding_on_finalize_failure tests/python/test_serving_integration.py::test_runtime_binding_materialization_closes_runtime_handle_on_state_failure tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_closes_on_error tests/python/test_serving_retained_binding_acquire.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py::test_serving_integration_prepare_local_ready_owns_contract_and_options tests/python/test_serving_integration.py::test_serving_integration_finalizes_local_ready_runtime_closes_on_error tests/python/test_serving_integration.py::test_serving_integration_prepare_local_ready_builds_recipe_from_source tests/python/test_serving_integration.py::test_serving_integration_prepare_local_ready_builds_framework_context tests/python/test_serving_recipe_build_session.py -q`
- `source .venv/bin/activate && pytest tests/python/test_serving_integration.py tests/python/test_serving_retained_binding_acquire.py tests/python/test_serving_replica_publication.py -q`
- `source .venv/bin/activate && LD_LIBRARY_PATH=/data/cuda/compat:${LD_LIBRARY_PATH:-} pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py -q`
- `source .venv/bin/activate && LD_LIBRARY_PATH=/data/cuda/compat:${LD_LIBRARY_PATH:-} pytest tests/python/daemon/test_inplace_slot_swap_publish_e2e.py::test_inplace_slot_swap_publish_e2e -q`
- `git diff --check`
- `git diff --cached --check`

Closeout state:

- Phase 9 split-brain deletion work is closed for the audited 0121 surfaces:
  old path-specific fallback and cleanup/cost paths were deleted, narrowed, or
  rewritten to the shared strategy/envelope/report surfaces.
- API realization, registration cleanup, and touched Global Store
  control-plane cleanup no longer silently suppress broad exceptions, and
  serving retained-binding/runtime cleanup now follows the same rule.
  Target-region/core CUDA IPC export no longer falls back to or exposes the
  retired non-offset exporter; guardrails block these regressions.
- Mounted-source resolve no longer has an absolute-path compatibility fallback;
  operators must express allowed local paths through `public_disk_source`
  trusted root policies.
- Canonical-index lookup no longer retries disk as an implicit fallback after
  engine metadata lookup fails; the disk-vs-engine authority is selected before
  loading in target, owned-binding, and replica materialization paths.
- Binding publication tokens no longer flow through materialization responses;
  binding publication paths now use the same lifecycle/cost report surfaces
  rather than a second token source.
- Misleading legacy/compatibility names have been removed from the touched
  daemon proto comments, batch runtime test, and Python guardrail test names.
- Final 0121 completion now includes a compatibility cleanup gate: delete or
  rewrite redundant legacy adapters and old-path tests instead of preserving two
  maintenance paths. Public ergonomic sugar remains only as a facade over the
  unified realization kernel.
- Protocol and client cleanup has started: `MaterializeReplicaResponse` now
  exposes realized view indexes only through `view_index_bytes`, the old
  disk-path `DaemonCtl.materialize_by_artifact_id(...)` wrapper has been
  replaced by the canonical artifact-selection method name, and
  daemon VRAM registration now uses only `RegisterRegion(memory_kind=VRAM)` /
  `UnregisterRegion` with `register_vram_region(...)` kept as SDK syntax
  sugar. Unused daemon proto shells `RegisterRequest`, `RegisterResponse`, and
  `DiskFallbackHint` have also been deleted. The old
  `TargetPublicationScope`/`CAPABILITY_AUDIENCE_TARGET_PUBLICATION` capability
  path is reserved but no longer generated or tested, and daemon routed
  authority no longer accepts `target_publication_token` as a compatibility
  label. Daemon loaded-replica status now also uses only the canonical
  `GetLoadedReplicas` RPC/message/helper names; the retired `V2` suffix is
  blocked by a guardrail.
- SDK materialization decode now fails closed when the daemon omits
  `canonical_index_bytes`, rather than reusing a pre-RPC index hint as a local
  fallback.
- Runtime source selection no longer accepts a duplicated
  `diagnostics["source_selection"]` side channel; endpoint source choice is
  projected from materialization/execution/report facts. Runtime endpoint
  projection now also consumes `ArtifactRealizationReport` directly, using
  report materialization diagnostics first, execution-commit facts second, and
  `RealizationStrategyPlan`/`RealizationResourceEnvelope` fallback and cost
  fields when execution diagnostics are absent.
- Runtime compatibility hydrate rewrite/cache has been deleted; hydrate by
  `engine_request_id` is now a daemon/node-agent responsibility instead of a
  local Runtime fallback.
- Adopted-binding, low-level client-owned binding, and
  `MaterializationPipeline.get_into` target-region registration and stale-region
  invalidation cleanup now flow through `target_region_lifecycle.py`,
  `envelope_for_target_region_registration(...)`, and a
  `RealizationReleaseContract`; successful `get_into` registrations still keep
  region-cache ownership for reuse, while retry/failure paths release daemon
  region ids instead of only dropping the local cache.
- Adopted and mapped `bind_into` post-materialization `CreateBinding` rollback
  now releases caller target regions through the same target-region release
  contract. Malformed `CreateBinding` values close the newly-created binding
  before releasing regions, and rollback close failures are logged instead of
  being silently suppressed.
- `InplaceSlot` region refresh and close now use the same
  `target_region_lifecycle.py` release helper instead of private
  unregister/cache cleanup loops, keeping slot refresh failures and normal close
  on the shared target-region envelope/release contract path.
- Runtime directory compatibility shims and pure-transform `tc_serving_*`
  string-arg publication fallback have been deleted; tests now exercise
  `runtime.directory()` and typed `TransformSpec.publication_spec` directly.
- Resource lifecycle risk is closed for the audited API/serving paths: old
  cleanup/cost-reporting paths that were not explainable through
  `RealizationResourceEnvelope` were deleted or made envelope-driven.
- Core replica placement compatibility cleanup is closed in this pass:
  `resolve_replica_config_device_key(...)` is now the shared core resolver,
  `device_type` is the placement authority, GPU replicas require an explicit
  `local_device_id`, CPU replicas require `local_device_id=-1`, and runtime /
  load-controller normalization no longer defaults ordinal-less GPU keys to
  GPU0. Core/dataplane tests now use explicit `device_type=GPU` for GPU
  residency and `CPU/-1` for CPU residency.
- Full external vLLM engine startup remains outside the in-repo dependency set.
  The in-repo target-state risk is closed by vLLM-shaped startup/reload/runtime
  attachment, retained acquire, local-ready promotion, publication, shutdown
  retirement, and real CUDA IPC coverage.
- Final verification has real-GPU retained binding daemon coverage, real-GPU
  reference target-set prefetch/acquire coverage, in-place swap publication and
  retirement coverage, and dual-daemon TP view materialization coverage.
  Runtime, retained-binding acquire, replica publication, and serving runtime
  contract Python suites are passing; broader
  real-GPU runtime attachment and vLLM TP startup behavior coverage is still
  pending.

# Coordination Plan

This section is the execution rhythm for the remaining work. The current
implementation has enough shared SDK surface and report adapters to stop
expanding report-only migrations as independent wins. The next priority is to
make the facade and release contract own the paths that currently only attach
realization reports.

Parallel tracks:

- Track A: SDK facade closure. Publication projections now lower through
  `ArtifactRealizationSpec` / `ArtifactRealizationHandle`; continue hardening
  public entrypoint alignment and profile events after the release-contract,
  token-backed export, publication/layout audit, and CPU TensorDict mutability
  items covered in this pass.
- Track B: target-set / TP realization. The SDK report plane now carries
  target-set strategy/lifecycle state for `same_selection`,
  `per_part_selection`, member layout, device UUIDs, group barriers, staged
  values, publish barrier, acquire claims, and per-member reports. Daemon
  operation results now preserve the staged publish barrier, and reference
  target-set prefetch has real-CUDA worker attach coverage. The next step is to
  verify the actual vLLM TP startup path with real CUDA.
- Track C: daemon/core convergence. Replica materialization, main
  target materialization, binding creation/refill, retained prefetch/acquire,
  publication, and assembly entrypoints now have shared controller planning
  structures with first-pass body-backing/resource-authority mapping. The next
  convergence work is to make those resource-authority facts drive deeper
  decisions through `BodyBackingManager`, `HandleLeaseRegistry`,
  `SessionLifecycleManager`, `LifecycleKernel`, and core execution commit
  concepts without changing stable RPC shapes prematurely.
- Track D: deletion and guardrails. Delete old selection, fallback, cleanup,
  and diagnostics construction only after the affected path actively lowers
  through the realization facade and has tests for the relevant risk labels.
  Keep direct Global Store, direct selection-builder, unmanaged mounted-source,
  second continuation model, and TP-only bypass guardrails in place.

Next implementation slices:

1. Extend the daemon/controller realization plan into execution-time
   resource-decision lowering for lifecycle-kernel capability paths without
   changing protocol shapes prematurely. MaterializeReplica, publication,
   binding creation/refill, and assembly controller flows now use the same
   controller plan model with resource-authority/admission facts; Phase 8 still
   needs those facts to drive deeper manager-owned decisions, report/envelope
   linkage, and cleanup of any controller-specific decisions.
2. Run targeted daemon/proto builds plus real-GPU runtime/TP verification while
   SDK tests continue in parallel, then use those results to decide which old
   TP-only or controller-specific paths can be narrowed.

Serial boundaries:

- Do not delete an old implementation path until that path actively lowers
  through `ArtifactRealizationSpec` / `ArtifactRealizationHandle` and has
  risk-closure tests.
- Do not unify or remove protocol messages until SDK and daemon controllers
  share one stable plan model.
- Do not close CPU TensorDict mutability until storage-level behavior is either
  enforced by backing/export code or verified as read-only/read-mostly with
  explicit tests.
- Do not close the Phase 5 projection-owner item until runtime detach, retained
  acquire close, binding close, TensorDict projection close, and publication
  projection close all use `RealizationReleaseContract`.
- Do not classify TP target-set realization as complete until same/per-part
  selection, staged publish barriers, group acquire claims, and real-CUDA
  member diagnostics are covered.

Phase closure criteria:

- Phase 5 closes when every public SDK/runtime/publication target either returns
  or delegates through a realization handle, unsupported handle actions fail
  closed, and release behavior is contract-owned.
- Phase 6 closes when migrated targets emit comparable report fields and
  profile events, including operation backend, publishability, resource
  envelope, risk labels, and execution commit facts where available.
- Phase 7 closes when TP is represented as target-set strategy/lifecycle state
  in both SDK and daemon/controller execution, not a TP-only materialization
  path, and has real-CUDA coverage for group behavior.
- Phase 8 closes when daemon RPC handlers share controller planning helpers and
  daemon/core resource decisions are explainable through the same envelope and
  report model.

# Current State & Grounding

The current SDK has parallel realization paths:

- `docs/designs/0039-artifact-first-sdk.md` is the unified public artifact
  entrance. TensorDict methods, binding helpers, and prefetch helpers are
  ergonomic public entrypoints below `Artifact`, not independent materialization
  owners or compatibility shims.
- `docs/designs/0116-prefetch-serving-binding-target.md` is superseded as a
  standalone public serving target. Its retained GPU residency, reservation,
  acquire, TTL, and validation semantics are preserved here as retained
  realization lifecycle behavior.
- current serving-runtime code is the behavior baseline for runtime attachment,
  retained acquire, reload, publication, runtime view, and shutdown retirement.
- `tensorcast/api/store/artifact.py`: `Artifact.tensor_dict(...)`,
  `tensor_dict_with_diagnostics(...)`, `tensor_dict_into(...)`, `bind(...)`,
  `bind_into(...)`, `_resolve_realization_selection(...)`, and
  `_build_region_layout_selection(...)`.
- `tensorcast/api/store/materialization.py`: `MaterializationPipeline` owns
  `materialize_subset(...)`, `get_into(...)`, region-backed writes, retry,
  payload release, and payload-to-state-dict conversion.
- `tensorcast/api/_materialize.py`: `materialize_artifact(...)` builds an
  `ArtifactSelection`, calls `MaterializeReplica`, decodes CUDA IPC or CPU memfd
  handles, and returns `MaterializationPayload`.
- `tensorcast/api/store/binding.py` and
  `tensorcast/api/store/owned_binding_slot.py`: binding current/staged values,
  swap/reload, publication, group acquire, and lifecycle checks.
- `tensorcast/api/store/inplace_slot.py` and `tensorcast/api/plan/plan.py`:
  refill, swap, `prefetch_many`, and plan lowering still build or consume
  selection identity outside a shared resolver.
- `tensorcast/api/store/__init__.py`: representation publication layout helpers
  now provision canonical layouts through the Store Daemon client. The
  surrounding representation-publication workflow remains assembly-attempt
  based until daemon/core controller convergence introduces the shared plan
  model.
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

# Contract Inventory Snapshot

This is the current semantic freeze for deletion and convergence work. The
implementation may keep ergonomic public methods, but the authority, target,
resource, lifecycle, and report facts below must stay represented through the
realization kernel rather than path-local side channels.

| Path | Realization target | Authority and selection | Resource and release contract | Strategy/report facts |
| --- | --- | --- | --- | --- |
| TensorDict retrieval | `tensor_dict` | `ResolvedArtifactSelection` owns artifact id/key, view/subset, generation, and canonical index identity. | Temporary daemon replica, CUDA IPC or CPU memfd token-backed export, TensorDict projection owner, release export token and unload temporary replica. CPU projections are read-mostly private. | `MaterializationDiagnostics` map into the report with source, bytes, replica id, backend, publishability, movement-cost fields, and risk labels. |
| TensorDict-into / tensor-into | `caller_tensors` | Same resolver as TensorDict, with subset lowering for single-tensor writes. | Caller-owned target region; CUDA uses registered-region direct write, CPU uses temporary-payload copy fallback; release unregisters target regions and unloads fallback replicas. | Direct-write bytes, copy bytes/counts, fallback buckets, fallback unload, source, and replica facts are reported through `GetIntoResult` and the resource envelope. |
| Owned and adopted bindings | `binding_owned` / `binding_adopted` | Source selection digest stays separate from target layout, mapped view, binding layout, and copy-plan digests. | Binding-owned daemon value or caller-adopted target tensors; binding close/swap is the lifecycle action exposed through the handle release contract. | Binding diagnostics, execution diagnostics, source-bound plan diagnostics, lane allocation, committed ranges, residual fallback ranges, reject buckets, publication eligibility, and operation backend are normalized into the report. |
| Device prefetch | `retained_replica` | Resolver supplies artifact/view identity; public async contract remains `Operation[PrefetchedReplica]`. | Daemon-retained replica ticket with daemon-retained export lifetime; public prefetch defaults to `NO_LEASE` and releases the replica ticket on cleanup. | Operation id/status/wait/cancel/idempotency semantics remain the continuation model; report carries retained bytes, backend, source digest, target placement, and fallback buckets. |
| Retained serving prefetch | `retained_binding` | Resolver supplies source identity; runtime target supplies resolved layout/index metadata and capability authority. | Binding reservation capability with TTL/idle-retire validation, staged acquire release when present, and fail-closed acquire after capability expiry. | Reports carry binding value refs, layout/copy-plan digests, reservation bytes, readiness, verification state, staged group acquire, publish barrier, wait budget, and status/debug visibility. |
| TP/group serving prefetch | `target_set` | `same_selection` and `per_part_selection` are first-class target-set selection modes; target sets do not lower through `retained_binding`. | Binding reservation capability set with per-member release and optional staged-acquire release; group state owns member-local device/layout/profile facts. | One group report carries per-member diagnostics, source-selection mode, same-daemon/cross-daemon coordination, collective-first policy, barriers, staged values, publish barrier, acquire claim ids, and member release policies. |
| Runtime attach, reload, and local-ready restore | `runtime_attachment` with `model_runtime` wrapper reports | Runtime paths consume resolved artifact or retained binding state; local-ready promotion records runtime-local authority and pending verification instead of durable source authority. | Runtime attachment owns restored tensors and closes the runtime attachment plus binding or placement lease. Retained acquire close uses the same release contract. | Runtime reports carry framework, adapter/ABI, topology/member digests, attachment backend, release policy, verification/local-ready facts, and model-runtime wrapper labels such as `framework:vllm`. Direct `Artifact.realize(model_runtime)` remains fail-closed until full lowering lands. |
| Runtime-owned publication | `publication` | Publication is attached to runtime-owned/binding-owned state, not a source-selection authority. | Published replica lease with retire-published-replica and optional publication-lease release actions. | Publishability report records eligibility, requested/published state, active publication facts, backend, and release contract. |
| Mounted-source bootstrap | `mounted_source` | Only explicit `msa1:` artifact subjects are admitted; key-backed or Global Store-routed mounted-source activation fails closed. | Daemon-mounted metadata attestation promotes to `mi2:` and drops the promotion handle after returning the durable artifact identity. | Report carries `msa1:` source subject, promoted `mi2:` artifact id, canonical-index bytes length, generation, checksum policy, daemon promotion backend, and mounted-source risk labels. |

# Phases & Milestones

These phases are workstreams, not a strict execution order. Implementers may
advance, split, merge, or reorder TODOs when the repository state makes that the
better path. The hard rule is semantic closure: a migrated path must have its
selection identity, resource envelope, lifecycle release model, strategy
fallback, report fields, and deletion guardrails in place.

- [x] Phase 1: Contract Inventory And Semantic Freeze
  - [x] Build a contract table for TensorDict, TensorDict-into, bind,
        bind-into, prefetch-device, prefetch-target, runtime attach, reload,
        publication, local-ready promotion, mounted-source bootstrap, and TP
        startup.
  - [x] Record current behavior for CPU vs CUDA, ephemeral vs retained,
        publishable vs non-publishable, source policy, fallback, diagnostics,
        cleanup, operation ids, and direct control-plane dependencies.
  - [x] Build a resource-envelope matrix for every path with normalized
        `backing_kind`, `export_kind`, `projection_kind`, `owner_kind`,
        `release_policy`, `mutability_contract`, planned cost fields, and
        deterministic token-backed export behavior.
  - [x] Build a risk-closure matrix mapping each design risk to an admission
        field, envelope field, report field, guardrail test, and blocking
        condition.
        Risks without a closure mechanism block migration of the affected path.
  - [x] Record current lifecycle release triggers separately for export release,
        backing release, and workflow release.
  - [x] Record current copy/reference behavior: Python dict projection,
        CUDA IPC mapping, CPU memfd mmap/private mapping, region-backed direct
        write, temporary-payload fallback, and binding copy/fill plans.
  - [x] Import the non-obsolete `0116` TODOs as retained realization behavior:
        reservation bytes, acquire validation, TTL/idle retirement, metrics,
        status/debug visibility, and transform-required fail-closed behavior.
  - [x] Pin `0055` operation semantics for prefetch and async realization:
        `Operation[T]`, deterministic idempotency, scoped cancel, `NO_LEASE`,
        and degraded/timeout behavior.
  - [x] Pin `0115` mounted-source semantics: successful trusted source resolve
        produces `msa1:` artifact identity and no source-handle-only execution
        bypass.
  - [x] Pin `0117` target-set semantics: `same_selection` and
        `per_part_selection`, staged values, publish barriers, and group-aware
        acquire.
  - [x] Add or update tests that pin current behavior before refactoring.

- [x] Phase 2: Control-Plane Authority Cleanup
  - [x] Inventory every SDK direct Global Store import/channel/RPC used by
        realization, publication, layout provisioning, key resolution, operation
        observation, and group admission.
  - [x] Add daemon API coverage or daemon-client wrappers for layout
        provisioning and representation publication helpers currently using
        Global Store directly.
  - [x] Migrate SDK publication/layout helpers to daemon-mediated APIs.
  - [x] Add static or unit guardrails that fail on direct Global Store access in
        SDK realization paths.
  - [x] Keep runtime orchestration Global Store startup/connect logic out of this
        ban; the ban is for SDK artifact metadata and realization authority.

- [x] Phase 3: Canonical Selection Resolver
  - [x] Introduce `ResolvedArtifactSelection` and a single resolver in the SDK.
  - [x] Move artifact id/key resolution, canonical index fetch, artifact profile
        and authority scope, view/subset preparation, view index bytes, view id,
        view subset hash, logical layout hash, and generation hint handling into
        the resolver.
  - [x] Admit `msa1:` mounted-source artifacts through the same resolver while
        preserving same-daemon, non-routable authority.
  - [x] Migrate `tensor_dict`, `tensor_dict_into`, `bind`, `bind_into`,
        `prefetch`, retained/runtime paths, `Plan` prefetch-set lowering,
        inplace refill, owned-binding refill, and local-ready source views to use
        the resolver.
  - [x] Add a guardrail test that fails when SDK realization paths build
        `ArtifactSelection` outside the resolver.
  - [x] Add tests for canonical, subset, view, mapped-source, `msa1:`, and
        group-member source selection identity.

- [x] Phase 4: Target, Strategy, Representation, Lifecycle, And Resource Plans
  - [x] Define internal DTOs for `RealizationTargetPlan`,
        `RealizationStrategyPlan`, `RepresentationAdmissionPlan`, and
        `RealizationLifecyclePlan`.
  - [x] Define internal `RealizationResourceEnvelope` fields shared by every
        target: backing, export, projection, owner, release policy,
        mutability contract, release strictness, and cost model.
  - [x] Define the common risk-relevant admission fields:
        `authority_evidence`, `source_selection_digest`, `target_layout_digest`,
        `copy_plan_digest`, `fallback_policy`, `release_strictness`,
        `export_lifetime_kind`, and `mutability_contract`. For process-visible
        CUDA IPC and CPU memfd exports, `export_lifetime_kind` must be
        token-backed; mint failure is a hard error.
  - [x] Represent existing targets as target kinds: `tensor_dict`,
        `caller_tensors`, `binding_owned`, `binding_adopted`,
        `retained_replica`, `runtime_attachment`, and `target_set`.
  - [x] Separate source selection digest from target layout digest, mapped view
        id, binding layout id, and copy-plan digest.
  - [x] Move source policy, P2P/disk preference, verification, retry, deadline,
        region-backed fallback, wait-for-shared-disk, lease/export policy, and
        collective policy into `RealizationStrategyPlan`.
  - [x] Carry `0108` lane allocation and residual fallback accounting, or the
        lowered daemon/core equivalent, when tensor-aware strategy is involved.
  - [x] Move representation/layout/schema/member compatibility checks into
        `RepresentationAdmissionPlan`.
  - [x] Model publishability, retained claims, current/staged binding values,
        borrowed caller tensors, TensorDict payload ownership, and ephemeral
        TensorDict as lifecycle capabilities.
  - [x] Map daemon `BodyBacking*`, `HandleLeaseRegistry`,
        `SessionLifecycleManager`, `LifecycleKernel`, and core
        `ExecutionCommitReport` concepts into the envelope instead of adding
        path-specific resource managers.
  - [x] Reject executable plans that are missing risk-relevant fields required
        by their target kind or strategy. Missing fields fail admission rather
        than selecting a broad fallback.

- [x] Phase 5: ArtifactRealizationSpec, Handle Facade, And Projection Lifetime
  - [x] Introduce `ArtifactRealizationSpec` constructors for TensorDict,
        caller tensors, binding, adopted binding, retained replica, retained
        binding, mounted-source realization, model runtime, publication, and
        target set.
  - [x] Introduce `Artifact.realize(spec=..., ctx=...)` for completed
        realization and `Artifact.realize_async(...)` only where operation
        semantics are required.
  - [x] Introduce `ArtifactRealizationHandle` projections/actions:
        `tensor_dict()`, `binding()`, `attach(...)`, `prefetch_handoff()`,
        `publish_replica(...)`, `promote(...)`, and `report`.
  - [x] Implement TensorDict projection ownership so projected tensors cannot
        outlive the payload lease/export owner unsafely.
  - [x] Implement shared projection-owner adapters so raw tensor escape, handle
        close, binding close, runtime detach, and retained acquire close all use
        the envelope release contract.
  - [x] Route runtime detach, retained acquire close, and publication
        projection close through `RealizationReleaseContract`.
  - [x] Extend the release-contract lifecycle matrix across raw tensor escape,
        export release, backing release, caller-target cleanup, runtime workflow
        cleanup, retained acquire close, and publication projection close.
  - [x] Enforce token-backed export release policies. If CUDA IPC or CPU memfd
        handle-lease minting fails, fail before returning tensors. Do not add a
        PID-bound export fallback.
  - [x] Classify CPU TensorDict as read-only/read-mostly private mapping and
        reject TensorDict write semantics.
  - [x] Keep `Artifact.tensor_dict(...)`, `tensor_dict_into(...)`, `bind(...)`,
        and `bind_into(...)` as ergonomic public entrypoints, but rewrite them
        to call target-state realization directly. Delete their old independent
        implementation paths.
  - [x] Keep `Artifact.prefetch(...)` as `Operation[T]` and lower it through
        async realization without changing public wait/cancel/status semantics.
  - [x] Keep the `0039` SDK target surface aligned so convenience methods are
        documented and tested as ergonomic artifact entrypoints backed by
        realization handles.
  - [x] Ensure unsupported handle actions fail with clear
        `FAILED_PRECONDITION` messages.

- [x] Phase 6: Unified Report And Diagnostics
  - [x] Define `ArtifactRealizationReport`.
  - [x] Map current TensorDict `MaterializationDiagnostics` fields into the
        report.
  - [x] Map binding materialization diagnostics, execution diagnostics,
        source-bound planner diagnostics, binding value ids, and publication
        eligibility into the report for owned/adopted binding realization.
  - [x] Map retained replica handoff facts into the report without replacing
        the public `Operation` continuation contract.
  - [x] Map retained binding reservation/acquire facts into the report without
        replacing the public `Operation` continuation contract.
  - [x] Map runtime attachment facts into the report for direct serving load,
        reload, local-ready restore/finalize, and retained acquire without
        replacing runtime attachment ownership or lifecycle semantics.
  - [x] Map runtime-owned replica publication projection facts into the report
        without replacing the existing publish/retire lifecycle operations.
  - [x] Add source selection digest, target layout digest, copy-plan digest,
        operation id/backend, artifact profile/authority scope, and
        publishability facts.
  - [x] Add resource-envelope fields for TensorDict, caller tensor, owned
        binding, adopted binding, and retained-replica SDK paths: backing kind,
        export kind, projection kind, owner kind, release policy, release
        strictness, export lifetime kind, mutability contract, retained bytes,
        and temporary-replica bytes.
  - [x] Extend resource-envelope fields to retained binding SDK paths.
  - [x] Extend resource-envelope fields to runtime attachment paths.
  - [x] Extend resource-envelope fields to runtime-owned replica publication
        paths.
  - [x] Extend resource-envelope fields to retained binding set target-set
        paths.
  - [x] Add cost fields: direct-write bytes, copy bytes, copy count, mmap bytes,
        CUDA IPC open count, CPU memfd fd count, and fallback reason buckets.
  - [x] Include `ExecutionCommitReport` facts where available: lane allocation,
        committed ranges, residual fallback ranges, actual executor path, and
        reject buckets.
  - [x] Add risk-closure labels to report diagnostics for authority, identity,
        lifecycle, lease strength, mutability, hidden movement cost, async
        continuation, and target-set/group behavior. These labels should be
        derived from the plan, not hand-written per target.
  - [x] Emit consistent profile events for all realization targets.
    - [x] Emit report-shaped `artifact.realize` profile events for synchronous
          facade targets.
    - [x] Emit the same report-shaped events for retained replica, retained
          binding, target-set, runtime attachment, model-runtime, and publication
          paths.
  - [x] Update tests to assert comparable report fields across TensorDict,
        binding, retained handoff, runtime attach, and target-set paths.
    - [x] Assert shared identity, operation backend, envelope, publishability,
          risk-label, and admission fields across those report families.

- [x] Phase 7: TP As Target-Set Realization
  - SDK/report-plane items below are checked when the shared DTO/report shape is
        present and covered by SDK tests. Daemon execution and real-CUDA
        vLLM-shaped TP target-set startup now use the same shape.
  - [x] Define target-set plan shape with group selection plan, shared
        group/version-set context, shared strategy, and member-local targets.
  - [x] Support `same_selection` and `per_part_selection` as first-class target-set
        selection modes.
  - [x] Map TP rank/member, target layout, device UUID, runtime profile, semantic
        placement digests, and per-member source selection into member target
        plans.
  - [x] Map same-node source coordination, collective-first loading, group
        barriers, staged values, publish barrier, and acquire claims into
        strategy/lifecycle plans.
  - [x] Produce one group realization report with per-member diagnostics.
  - [x] Prevent TP code from adding a TP-only materialization path.

- [x] Phase 8: Daemon And Core Controller Convergence
  - [x] Add shared daemon-controller planning structures that mirror the SDK
        realization plan where appropriate.
  - [x] Lower daemon controller resource decisions through existing
        `BodyBackingManager`, `HandleLeaseRegistry`, `SessionLifecycleManager`,
        `LifecycleKernel`, and core strategy/report structures rather than
        adding target-specific resource managers.
    - [x] Map controller resource envelopes to body-backing intent,
          resource-authority concepts, and token-backed process-visible export
          admission.
    - [x] Route `MaterializeReplica` handle export selection through the
          admitted controller resource envelope.
    - [x] Gate CUDA IPC/export-lease execution branches on admitted controller
          export kind and resource-authority facts.
    - [x] Gate assembly operation-lease and publication lifecycle-redemption
          branches on admitted controller export kind and authority facts.
    - [x] Gate caller-target direct-write branches on admitted controller
          export kind and caller-allocation authority facts.
    - [x] Gate binding refill and retained serving prefetch branches on
          admitted controller export kind and resource-authority facts.
  - [x] Route existing RPC handlers through common target-plan,
        representation-admission, strategy, and lifecycle helpers.
    - [x] Route `MaterializeReplica` through common controller realization plan
          construction.
    - [x] Route `MaterializeIntoTarget` and `MaterializeIntoMappedTarget`
          through common controller realization plan construction.
    - [x] Route retained serving prefetch and retained binding acquire through
          common controller realization plan construction.
    - [x] Route publication through common controller realization plan
          construction.
    - [x] Route binding controller flows through the same plan model.
    - [x] Route assembly controller flows through the same plan model.
  - [x] Keep protocol messages stable until the shared controller path is
        proven. The controller convergence work added shared plan/linkage
        helpers and tests without introducing another protocol shape.
  - [x] Start proto cleanup now that SDK and daemon controllers share one plan
        model: reserve redundant `MaterializeReplicaResponse.view_index_json`,
        make the daemon write realized view indexes through
        `view_index_bytes`, and keep only active model surfaces such as
        `EnsureCanonicalLayout`, group realization options, and
        `CommitRegisteredArtifactResponse.view_index_json`.

- [x] Phase 9: Delete Split-Brain Paths And Enforce Guardrails
  - [x] Remove or narrow old path-specific selection helpers.
    - [x] Delete `Artifact._resolve_owner_source_selection(...)`, route
          owned-binding refill/swap through `_resolve_realization_selection(...)`,
          rename `_materialize.py` selection lowering to
          `_resolve_materialization_selection(...)`, and add guardrails against
          the removed helper names.
  - [x] Remove path-specific fallback behavior that is not represented in
        `RealizationStrategyPlan`.
    - [x] Remove Runtime-side compatibility hydrate fallback that rewrote
          `engine_request_id` hydrates through a locally cached
          `PublishManifest`; hydrate source selection now stays explicit in the
          plan handed to daemon/node-agent execution.
    - [x] Delete pure-transform serving publication string-arg fallback
          (`tc_serving_*`) and require typed `TransformSpec.publication_spec`
          for transform registration publication intent.
    - [x] Delete the redundant daemon materialization response alias
          `MaterializeReplicaResponse.view_index_json`; core/daemon now emits
          realized view indexes only through `view_index_bytes`, and
          daemon protocol documentation now names only the canonical field.
    - [x] Delete unused disk-path `DaemonCtl.materialize_by_artifact_id(...)`
          semantics; the canonical method name now means artifact-selection
          materialization through `MaterializeReplica`.
    - [x] Remove retired `_v2` SDK/DaemonCtl materialize/import aliases and
          trace/error labels. In-repo SDK, node-agent, and tests now call
          `materialize_artifact(...)`, `materialize_by_artifact_id(...)`,
          `materialize_into_target(...)`, `import_artifact_from_path(...)`, and
          `import_artifact_from_path_stream(...)` directly, with a guardrail
          against reintroducing the retired alias names.
    - [x] Remove the retired daemon loaded-replica status `V2` names.
          `GetLoadedReplicasV2`, `GetLoadedReplicasV2Request`,
          `GetLoadedReplicasV2Response`, and C++ helper aliases were renamed to
          the canonical `GetLoadedReplicas*` surface, with generated Python/C++
          stubs refreshed and a guardrail blocking the old names in daemon
          proto/service/state code.
    - [x] Delete daemon `RegisterVramRegion` / `UnregisterVramRegion` RPCs and
          proto objects; VRAM region registration now uses the unified
          `RegisterRegion(memory_kind=VRAM)` / `UnregisterRegion` daemon
          surface, while SDK `register_vram_region(...)` remains a thin helper.
    - [x] Delete unused daemon proto shells `RegisterRequest`,
          `RegisterResponse`, and `DiskFallbackHint`; there is no service RPC or
          client path that still consumes these side objects.
    - [x] Remove SDK canonical-index fallback from daemon materialization
          decode; missing `MaterializeReplicaResponse.canonical_index_bytes`
          now fails closed instead of reusing a pre-RPC hint.
    - [x] Rename daemon controller `resolve_retrieval_policy_compat(...)` to
          `resolve_retrieval_policy(...)` now that it is the shared controller
          policy resolver, not a compatibility adapter.
    - [x] Delete the adopted-binding publication-token fallback from
          `MaterializeIntoTargetResponse`; client binding registration now uses
          only the canonical `CreateBindingResponse` publication token, binding
          swap/commit uses `CommitBindingArtifactResponse`, and SDK code reads
          current proto fields directly.
    - [x] Delete retired target-publication token authority:
          `TargetPublicationScope` and
          `CAPABILITY_AUDIENCE_TARGET_PUBLICATION` are no longer generated
          proto APIs, and daemon target publication accepts only
          binding-current-value publication authority.
    - [x] Delete the daemon routed-authority front-door compatibility alias
          for `target_publication_token`; old wire labels now fail closed
          instead of resolving to the binding-current-value publication token
          kind.
    - [x] Delete core `ReplicaConfig` placement inference: `device_type` is now
          authoritative through shared
          `resolve_replica_config_device_key(...)`; `local_device_id` no longer
          selects GPU placement by sign, CPU configs with a nonnegative device
          id fail closed, and ordinal-less GPU keys are not silently normalized
          to GPU0.
    - [x] Route runtime endpoint source-selection fallback through unified
          artifact realization reports: `RuntimeWorkerView` now projects source
          selection from `ArtifactRealizationReport.materialization_diagnostics`,
          `execution_commit`, and finally
          `RealizationStrategyPlan`/`RealizationResourceEnvelope` fallback
          fields instead of requiring a runtime-path-specific diagnostics
          object.
    - [x] Route daemon byte-artifact put-if-absent join/conflict through shared
          authority preflight before source/target lowering. `HomeBatch*` and
          local `BatchPutIfAbsentFromRegion` now use the same claim decision,
          avoiding duplicate deterministic backing staging while preserving
          invisible-claim rebind and conflict semantics.
    - [x] Delete mounted-source absolute-path compatibility fallback.
          `public_disk_source` now exposes only trusted root policies;
          `unmatched_path_mode` is reserved in config proto, daemon options no
          longer carry allow-absolute fallback state, and unmatched absolute
          mounted-source paths fail closed outside configured trusted roots.
    - [x] Replace daemon canonical-index disk fallback with explicit source
          authority. Local import, mounted-source, and GS-unavailable disk paths
          read canonical indexes from disk by plan; GS-managed artifacts read
          from engine metadata and fail closed if that metadata is unavailable.
          This now covers target materialization, owned-binding refill/piece
          planning, and `MaterializeReplica`; no path retains an
          engine-metadata-failure-to-disk retry.
  - [x] Remove path-specific cleanup and cost-report behavior that is not
        represented in `RealizationResourceEnvelope`.
    - [x] Caller-tensor CUDA region-backed direct-write vs fallback-copy cost
          and cleanup now flow from `GetIntoResult` into
          `RealizationResourceEnvelope`, including precise direct-write bytes,
          temporary copy bytes, fallback buckets, and release policy.
    - [x] Adopted binding and low-level client-owned binding region
          registration now share `target_region_lifecycle.py`, a target-region
          resource envelope, and a release contract; `_bind_into`,
          `_bind_into_mapped`, and
          `Store.create_binding(ownership="client")` no longer carry separate
          partial-registration cleanup loops.
    - [x] Adopted and mapped `bind_into` now release registered caller target
          regions when post-materialization `CreateBinding` fails or returns
          malformed current-value metadata. Rollback closes any newly-created
          binding through a logged best-effort close helper instead of silently
          suppressing close failures.
    - [x] `MaterializationPipeline.get_into` target-region registration now
          uses the same lifecycle helper for partial-registration rollback and
          stale-region retry/failure invalidation while preserving successful
          region-cache ownership for reuse.
    - [x] `InplaceSlot` region refresh and close now release stale target
          regions through `release_target_region_ids_for_realization(...)`
          instead of carrying its own unregister/cache cleanup loop.
    - [x] Remove broad silent cleanup from API realization and registration
          paths. `tensorcast/api` no longer contains
          `contextlib.suppress(Exception)`; best-effort cleanup failures are
          logged with context, TensorDict owner attachment fails closed, and
          target-region registration uses only offset-aware CUDA IPC export.
    - [x] Delete the core/pybind non-offset CUDA IPC export path.
          `get_cuda_memory_handle(...)` is no longer declared in
          `core/checkpoint`, exported from `tensorcast._C`, declared in
          `_C.pyi`, or wrapped by `tensorcast._c_ext`;
          `get_cuda_memory_handle_with_offset(...)` now fails closed if CUDA
          address-range lookup cannot prove the backing allocation base and
          offset.
    - [x] Remove broad silent cleanup from touched Global Store production
          control-plane paths. Cluster-runtime close cleanup is logged, schema
          resource lookup narrows expected missing-resource errors, instance
          heartbeat capability updates fail closed, and the shared guardrail
          covers these files alongside the API realization paths.
    - [x] Remove broad silent cleanup from serving retained binding,
          local-ready, recipe-cache, and runtime attachment lifecycle paths.
          Lease rollback and handle close cleanup logs failures, descriptor
          canonical-index fallback narrows expected descriptor shape errors,
          placement payload failures are no longer silently dropped, and
          deferred recipe cache write failures are logged.
  - [x] Remove duplicate diagnostics construction once reports cover all paths.
    - [x] Centralize daemon controller realization span diagnostics through
          `attach_controller_realization_plan_span_attrs(...)`.
    - [x] Centralize serving runtime source-selection projection in
          `RuntimeWorkerView` and stop storing a duplicated derived
          `source_selection` value in `_state_seed(...)` diagnostics.
    - [x] Delete the runtime diagnostics `source_selection` side channel;
          `RuntimeWorkerView` now derives source selection only from
          materialization diagnostics, execution diagnostics, or realization
          report facts.
    - [x] Remove `legacy_diagnostics` from `ArtifactRealizationReport`; store
          materialization facts under `materialization_diagnostics` and guard
          against reintroducing the legacy field.
    - [x] Move SDK binding materialization source labels, response
          materialization diagnostics, execution diagnostics, and source-bound
          plan diagnostics into `realization_kernel.py`; delete the
          path-specific owned-binding diagnostics test file and assert the
          shared helper in `test_realization_kernel.py`.
    - [x] Delete the TensorDict-only profile event
          `artifact.tensor_dict_with_diagnostics`; TensorDict public entrypoints
          now emit only the unified report-shaped `artifact.realize` event.
  - [x] Remove runtime directory compatibility shims from
        `TensorCastSignals`; directory listing and instance routing now flow
        only through `TensorCastDirectory`.
  - [x] Keep guardrails for direct selection construction, direct Global Store
        access, unmanaged source-handle bypasses, second operation-continuation
        models, and TP-only materialization bypasses.
  - [x] Update README/design references after code convergence lands.

# Tasks

- [x] Add SDK realization package/module ownership for the kernel, distinct from
      the current binding-only `tensorcast/api/store/realization_plan.py` copy/fill
      helper.
- [x] Add `RealizationResourceEnvelope` DTOs and adapters so TensorDict,
      caller-tensor direct write, caller-tensor copy fallback, owned binding,
      adopted binding, retained prefetch, runtime attachment, and target-set
      member realization all expose the same backing/export/projection/owner/
      release/cost fields.
- [x] Add an adapter matrix from existing implementation objects
      (`MaterializationPayload`, region-backed `MaterializeIntoTarget`,
      `OwnedBindingSlot`, retained acquire state, runtime attachment state,
      group realization state) into `RealizationResourceEnvelope`.
- [x] Add a risk-closure matrix fixture or generated table that ties every
      design risk to the concrete admission field, envelope adapter, report
      assertion, and deletion guardrail that closes it.
- [x] Add plan-admission tests that reject execution when risk-relevant fields
      are absent: source authority, target layout digest for mapped targets,
      fallback policy for optional fallback, release strictness for exports, and
      mutability contract for CPU TensorDict.
- [x] Add export-lease failure tests that force CUDA IPC and CPU memfd
      handle-lease mint failures and assert hard realization errors before any
      tensor projection is returned.
- [x] Add daemon-mediated layout provisioning and representation publication
      helpers, then remove direct `GlobalStoreCompositeStub` usage from SDK
      realization/publication helpers.
- [x] Add a guardrail test for forbidden direct Global Store access in SDK
      artifact metadata, realization, publication, operation, and group
      authority paths.
- [x] Add focused unit tests for `ResolvedArtifactSelection`, including
      canonical id/key, subset, view, mapped source view, generation hint,
      artifact profile, authority scope, and digest stability.
- [x] Migrate `artifact.py`, `materialization.py`, `plan.py`,
      `inplace_slot.py`, and `owned_binding_slot.py` selection construction to
      the shared resolver, then add a guardrail for direct
      `build_artifact_selection(...)` calls outside resolver/adapters.
- [x] Add integration tests covering TensorDict and Binding source-selection
      equivalence while asserting separate target layout and copy-plan digests
      for mapped/adopted targets.
- [x] Add `msa1:` mounted-source tests for same-daemon realization, rejected
      durable GS routing/key activation, explicit promotion to `mi2:`, and
      local-ready pending-verification admission.
- [x] Add TensorDict projection lifetime tests proving tensors returned through
      ergonomic artifact entrypoints cannot outlive the payload lease/export
      owner unsafely and daemon materialized payloads are released after
      projection/handle close.
- [x] Add lifecycle release tests that separately assert export release, backing
      release, and workflow release for TensorDict, temporary copy fallback,
      owned binding, retained acquire, and runtime attachment.
- [x] Add handle-lease failure tests proving CUDA IPC and CPU memfd exports fail
      closed when token-backed lease minting fails.
- [x] Add CPU TensorDict mutability tests that document read-only/read-mostly
      private mapping behavior and reject TensorDict write semantics.
- [x] Add `get_into` cost tests proving region-backed direct write reports
      direct-write bytes and temporary-payload fallback reports source export,
      copy bytes, temp bytes, unload, and fallback reason.
  - [x] Cover caller-tensor facade direct-write and temporary-copy cost fields.
- [x] Add prefetch and async realization tests proving `Operation[T]` status,
      wait, cancel, deterministic idempotency, `NO_LEASE`, degraded, and timeout
      behavior are preserved.
- [x] Add retained prefetch/report tests with source selection digest, target
      layout digest, copy-plan digest, operation id/backend, and lifecycle
      capability fields.
- [x] Add retained acquire capability, TTL, idle-retire, and status/debug tests
      from the retired `0116` plan where still relevant.
- [x] Add strategy/report tests for `0108` lane allocation, committed ranges,
      residual fallback ranges, executor path, and reject buckets when
      tensor-aware strategy is involved.
- [x] Add TP target-set planning tests with real CUDA coverage in this
      environment, using fake CUDA only for narrow unit cases that do not need
      IPC/device behavior.
- [x] Add TP target-set tests for `same_selection` and `per_part_selection`,
      staged values, publish barriers, group acquire claims, and group reports
      with per-member diagnostics.
- [x] Add daemon-controller tests for common target-plan lowering.
- [x] Update vLLM integration tests or scenario fixtures once runtime attach
      lowers through the kernel.
- [x] Update docs that mention serving-specific target names after replacement
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
6. switch TensorDict/binding/caller-tensor ergonomic entrypoints to the facade;
7. switch retained, local-ready, runtime attach, and `msa1:` mounted-source
   paths;
8. switch TP/group realization to target-set planning;
9. converge daemon/core controllers, then delete duplicate logic and keep
   guardrails.

Backout should happen at coherent semantic boundaries, not by preserving old and
new implementations indefinitely. Once a target-state path is proven, keep the
ergonomic artifact method if it is useful, but delete or rewrite its old
independent implementation instead of maintaining compatibility code.

# Risks & Tracking

- [x] Selection resolver becomes too broad.
      Track by keeping artifact identity, view/subset, and index facts in the
      resolver while target layout remains in target planning.
- [x] SDK direct Global Store access survives behind helper APIs.
      Track by blocking Phase 3 until publication, layout provisioning, key,
      operation, and group authority calls have daemon-mediated replacements and
      guardrail coverage.
- [x] `PublicDiskSourceHandle` becomes a permanent source authority.
      Track by requiring every successful trusted mounted-source path to carry
      an `msa1:` identity, source format facts, generation, and daemon policy
      evidence into `ResolvedArtifactSelection`.
- [x] Mapped target layout is confused with source selection.
      Track by asserting separate source selection digest, target layout digest,
      and copy-plan digest for mapped/adopted cases.
- [x] TensorDict accidentally inherits binding lifecycle.
      Track by capability tests proving TensorDict handles cannot publish,
      swap, or retain beyond their lease.
- [x] TensorDict projections release daemon payloads too early or leak them.
      Track by tests covering tensors returned through ergonomic artifact
      entrypoints, explicit handle close, projection close, and daemon release
      calls.
- [x] Resource lifecycle remains path-specific under a unified API.
      Track by requiring every existing path to emit a
      `RealizationResourceEnvelope` before execution and by deleting cleanup or
      cost-report code that cannot be explained through envelope fields.
- [x] Handle-lease mint failure silently weakens export lifetime.
      Track by deterministic-release tests that force CUDA IPC and CPU memfd
      mint failures and assert hard errors before tensor projection.
- [x] CPU TensorDict mutability stays ambiguous.
      Track by classifying CPU TensorDict as read-only/read-mostly private
      mapping and by rejecting TensorDict write semantics.
- [x] `get_into` hides expensive fallback copies.
      Track by strategy/report tests that compare region-backed direct write
      with temporary-payload fallback and assert direct-write bytes, copy bytes,
      temporary bytes, unload, and fallback reason fields.
- [x] Prefetch grows a second continuation model.
      Track by preserving `Operation[T]` for public async realization and
      prefetch status/wait/cancel/idempotency semantics.
- [x] Binding paths bypass strategy planning.
      Track by requiring every daemon materialization call to receive a
      `RealizationStrategyPlan` or its lowered equivalent.
- [x] Tensor-aware strategy loses lane/residual visibility.
      Track by requiring `ExecutionCommitReport` fields in reports whenever the
      lower strategy plane uses mixed execution or fallback.
- [x] TP grows special-case orchestration.
      Track by requiring TP additions to be target-set fields, strategy policy,
      or lifecycle state.
- [x] RPC cleanup is attempted too early.
      Track by forbidding protocol unification until SDK and daemon-controller
      planning are already shared.
- [x] Target-state behavior regresses while compatibility code is deleted.
      Track by scenario tests for vLLM startup, reload, publication, retained
      acquire, local-ready promotion, and shutdown retirement, with real CUDA
      coverage for IPC/runtime-sensitive cases.
