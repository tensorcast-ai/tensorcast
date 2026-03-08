---
slug: core-backed-body-handles-and-backing-policy
title: Plan - Core-Backed Body Handles and Backing Policy
status: completed
areas: ["core", "daemon", "sdk", "tests", "docs"]
created: 2026-03-08
last_updated: 2026-03-08
related_code:
  - docs/designs/0089-core-backed-body-handles-and-backing-policy.md
  - core/store/runtime/ingestion/artifact_lowering_plan.h
  - core/store/runtime/ingestion/materialization_facade.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/store_engine.h
  - core/store/store_engine.cc
  - daemon/service/byte_artifact_body_handle.h
  - daemon/service/byte_artifact_body_handle.cc
  - daemon/service/byte_artifact_body_store.h
  - daemon/service/byte_artifact_body_store.cc
  - daemon/service/controllers/byte_artifact_controller.cc
  - daemon/service/payload_transport_broker.h
  - daemon/service/payload_transport_broker.cc
  - daemon/service/controllers/transport_controller.cc
links:
  design: ../designs/0089-core-backed-body-handles-and-backing-policy.md
---

# Objective

Land the post-`0088` retained-body architecture without introducing a second body-private dataplane.

The target outcome is:

- one artifact model,
- one selection model,
- one shared ByteSpace dataplane,
- one shared capability-resolution layer,
- one memory/export ledger vocabulary,
- one authority-local retained-backing index for `byte_artifact`.

Concretely, this plan makes retained bodies:

- core-backed by default,
- descriptor-validated instead of reread-validated,
- policy-driven through authority-local lowering hints derived from `StorePolicy` and route role,
- aligned with existing stable retention, local export, communicator export, and transfer infrastructure,
- expressed as one profile over the shared ByteSpace dataplane rather than as a body-specific runtime.

Additional scope rule after review:

- `0089` is a clean containment design for retained byte bodies,
- it does not attempt a repo-wide rename of generic seams in the same patch,
- but it must not let body-specific concepts solidify into a second long-term subsystem.

# Latest Status

Updated 2026-03-08:

- already completed in the `0088` baseline:
  - `ArtifactLoweringPlan` and `execute_artifact_lowering_plan(...)` exist,
  - byte-artifact get and put flows lower through the shared core executor,
  - `BodyHandle` is core-backed via `ReplicaHandle`,
  - `payload_ref` can be issued from and chunk-read from core-backed bodies.
- not yet solved:
  - remote `payload_ref` still falls back to chunk RPC instead of resolving to a communicator-backed shared remote
    source,
  - stable-retention policy is still driven through existing core hooks rather than a full `StorePolicy` / retention
    registry seam,
  - future publish or retention capabilities do not yet terminate in the same shared resolution layer,
  - explicit orphan-cleanup verification for expiry and failed forwarding still needs dedicated tests.

Architectural corrections adopted from review:

- `BodyBackingIntent` is an internal lowering hint only, not a second policy surface.
- `BodyStore` is only an authority-local index for visibility / TTL / fence / join metadata.
- `BodyHandle` must stay thin; descriptor and observation belong to `BodyStoreEntry` or equivalent authority views.
- `payload_ref` is capability plus fallback, not the remote dataplane.
- `BodyDescriptor` and deterministic `physical_artifact_id` are treated as scoped names for seams that must remain
  structurally generic.

Implemented in this change set:

- the shared lowering result now returns verified retained-body content metadata,
- `BodyBackingManager` now owns retained-body staging and controller-local staging helpers are removed,
- `BodyStore` now stores descriptor + handle + authority metadata and derives join truth from descriptors,
- common-path invariant validation is descriptor-driven,
- `payload_ref` issuance consumes descriptor metadata instead of rereading bodies,
- `BodyHandle` has been kept thin again as a core-backed execution reference,
- deterministic physical backing identity is guarded so join cleanup does not retire the canonical retained backing,
- a minimal shared `inspect_replica_backing(...)` seam now exposes core-backed residency and export observation,
- `ResolvedBodyCapability` is now explicit for `payload_ref` resolution,
- local body-backed `payload_ref` now resolves to reusable body capability and can skip redundant restaging,
- capability resolution and retained-body retirement now emit explicit metrics for resolution mode, access class or intent,
  and retirement reason.

Remaining follow-ups after this cut:

- the capability-resolution seam is now explicit locally, but remote `payload_ref` fetch still uses chunk-RPC fallback
  rather than a communicator-backed shared remote-source resolution order,
- richer repo-wide genericization of verified-content descriptors and physical backing identity is intentionally deferred,
- stable-retention admission is wired through existing core hooks but not yet fully unified with `StorePolicy`-driven
  TTL / retention-registry policy surfaces,
- future publish or retention capabilities have not yet been routed through the same shared capability-resolution layer.

# Current State & Grounding

Current baseline in code:

- canonical internal execution IR already exists:
  - `core/store/runtime/ingestion/artifact_lowering_plan.h`
- the core executor already consumes that IR:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `core/store/store_engine.cc`
- `BodyHandle` is already core-backed:
  - `daemon/service/byte_artifact_body_handle.h`
  - `daemon/service/byte_artifact_body_handle.cc`
- byte-artifact put/get paths already lower through the shared executor:
  - `daemon/service/controllers/byte_artifact_controller.cc`
- `payload_ref` already binds to body handles instead of always binding to raw payload strings:
  - `daemon/service/payload_transport_broker.cc`
  - `daemon/service/controllers/transport_controller.cc`

Original gaps that motivated this plan included:

- controller-local retained-body lowering and CPU-default staging,
- common-path reread validation through `BodyHandle::compute_sha256_hex()`,
- lack of a separate verified `BodyDescriptor` in `BodyStore`,
- remote `payload_ref` transfer being primarily a chunk-RPC path instead of a shared remote-source path,
- body policy being underspecified relative to existing core concepts:
  - `MemoryLocation`,
  - stable lease,
  - export pin,
  - CPU memfd lease,
  - CUDA IPC,
  - communicator export.

Hard constraints inherited from `0039` / `0055` / `0078` / `0088`:

- no second value model,
- no second selection model,
- no second executor in daemon/service,
- no new copy engine,
- no new export runtime,
- no Global Store blob truth,
- no return to daemon heap payload strings as the primary retained-body contract.

Additional long-term constraints:

- treat the shared substrate as a profile-agnostic ByteSpace dataplane, not as a tensor-dict-private dataplane,
- make verified content metadata a shared executor output instead of a body-private side channel,
- make protocol capabilities converge to one shared capability-resolution layer before copy execution begins,
- keep `StorePolicy` as the only durability and placement declaration,
- keep `BodyBackingManager` as an internal helper under the `byte_artifact` profile runtime rather than a new top-level
  owner.

# Phases & Milestones

- [x] Phase 1: Freeze the `0088` shared-execution baseline
  - [x] Milestone 1.1: treat `ArtifactLoweringPlan` and `execute_artifact_lowering_plan(...)` as the canonical retained
    body staging seam.
  - [x] Milestone 1.2: document all controller-local body staging helpers as transitional only.
  - [x] Milestone 1.3: forbid any new primary retained-body path that stores ordinary payloads as daemon heap bytes.
  - [x] Milestone 1.4: define the shared substrate explicitly as a ByteSpace dataplane reused by all profiles.

- [x] Phase 2: Introduce the right long-term body abstractions
  - [x] Milestone 2.1: add `BodyDescriptor` as verified content truth.
  - [x] Milestone 2.2: add `BodyAccessClass` and `BodyBackingIntent`, with `BodyBackingIntent` explicitly scoped as an
    internal lowering hint derived from `StorePolicy` + route role + access pattern.
  - [x] Milestone 2.3: add `BodyBackingObservation` as the runtime result snapshot derived from core state.
  - [x] Milestone 2.4: add `ResolvedBodyCapability` as the shared capability-to-source or handle-resolution result,
    treating `payload_ref` as one front door rather than as the seam itself.
  - [x] Milestone 2.5: add a `BodyBackingManager` under `ByteArtifactProfileRuntime` to replace controller-local staging
    and policy glue.
  - [x] Milestone 2.6: formalize deterministic physical backing identity outside controller-local helpers and keep it
    aligned with broader backing-identity work.

- [x] Phase 3: Tighten validation and ownership boundaries
  - [x] Milestone 3.1: make the shared lowering result return verified content metadata for retained bodies.
  - [x] Milestone 3.2: make common-path invariant validation descriptor-driven.
  - [x] Milestone 3.3: ensure `BodyStore` commits `BodyDescriptor + BodyHandle` pairs as the canonical retained-body
    representation and treats join invariants as projections derived from descriptors.
  - [x] Milestone 3.4: keep `BodyStore` metadata-centric and prevent export/transfer state from moving into it.
  - [x] Milestone 3.5: keep `BodyHandle` thin and prevent `BodyHandle::Kind` or equivalent from becoming a second
    body-private backing taxonomy.
  - [x] Milestone 3.6: keep `BodyBackingManager` as a profile-runtime helper rather than a new long-lived owner layer.

- [x] Phase 4: Align retained-body policy with core memory/export contracts
  - [x] Milestone 4.1: map `BodyBackingIntent` to standard core execution inputs instead of body-private executor types.
  - [x] Milestone 4.2: add any minimal core observation APIs needed to describe:
    - residency location,
    - local export availability,
    - communicator export state,
    - stable retention state.
  - [x] Milestone 4.3: make retained CPU bodies use existing stable-memory admission hooks when policy requires stable
    retention.
  - [x] Milestone 4.4: keep local memfd and CUDA IPC export on existing handle-lease/export paths.
  - [x] Milestone 4.5: ensure communicator export remains owned by existing core APIs rather than by body-private state.
  - [x] Milestone 4.6: keep protocol-specific export or transport choices out of `BodyBackingIntent` and resolve them
    through shared capability logic instead.
  - [x] Milestone 4.7: document the body-term -> core-contract mapping explicitly and keep implementation explainable
    through that mapping.

- [x] Phase 5: Replace controller-local hard-coded staging
  - [x] Milestone 5.1: remove controller ownership of hard-coded CPU body lowering.
  - [x] Milestone 5.2: make inline ingress, `payload_ref` ingress, and source-layout ingress all stage bodies through
    one manager.
  - [x] Milestone 5.3: keep forwarder bodies transient and prevent transient staging from becoming home retained truth.
  - [x] Milestone 5.4: avoid restaging when an existing core-backed body can already supply a shared source.

- [x] Phase 6: Converge protocol capabilities toward shared resolution
  - [x] Milestone 6.1: make `payload_ref` issuance consume descriptor metadata instead of rereading bodies.
  - [x] Milestone 6.2: keep local chunk serving on `BodyHandle` or loader paths without materializing whole payload
    strings.
  - [x] Milestone 6.3: add one shared capability-resolution order for the `0089` cut:
    - local body handle reuse first when a matching retained backing already exists,
    - local loader path next,
    - chunk-RPC fallback last for remote `payload_ref`.
    communicator-backed remote source resolution remains a follow-up after this cut.
  - [x] Milestone 6.4: document and instrument chunk-RPC as fallback rather than as the canonical long-term path.

- [x] Phase 7: Lifecycle, budgets, and observability
  - [x] Milestone 7.1: separate retained-body ownership from stable retention, local export, and remote export in code
    and telemetry.
  - [x] Milestone 7.2: keep per-entry size and prune bookkeeping in `BodyStoreEntry` authority metadata without
    replacing core memory accounting.
  - [x] Milestone 7.3: emit cut-scope metrics for:
    - access class and intent,
    - transfer resolution mode,
    - retirement reason.
    observation residency and export state remain available through authority observation rather than a second metric
    taxonomy in this cut.
  - [x] Milestone 7.4: verify no orphaned core replicas remain after expiry, conflict, or failed forwarding.

- [x] Phase 8: Hard cut and cleanup
  - [x] Milestone 8.1: remove obsolete controller-local staging helpers and duplicate retained-body policy logic.
  - [x] Milestone 8.2: demote remaining transport-specific behavior to explicit fallback semantics instead of a second
    body-private transport assumption.
  - [x] Milestone 8.3: complete the `0089` cut without preserving a second retained-body semantics line in the primary
    path.

# Post-0089 Follow-ups

- route remote `payload_ref` resolution to communicator-backed shared remote sources before chunk-RPC fallback,
- converge future publish and retention capabilities onto the same shared capability-resolution seam,
- unify stable-retention admission more directly with `StorePolicy` and retention-registry policy surfaces,
- continue repo-wide genericization of verified-content descriptors and physical backing identity outside the `0089`
  cut.

# Tasks

- Core runtime
  - keep `ArtifactLoweringPlan` as the single retained-body staging IR,
  - expose verified retained-body content metadata as part of the shared lowering result or a shared executor-owned
    projection,
  - expose minimal observation hooks needed for body code to describe actual admitted result,
  - preserve core ownership of:
    - local loader opening,
    - retirement,
    - stable retention admission,
    - local export,
    - communicator export.
  - target files:
    - `core/store/runtime/ingestion/artifact_lowering_plan.h`
    - `core/store/runtime/ingestion/materialization_facade.h`
    - `core/store/runtime/ingestion/materialization_facade.cc`
    - `core/store/store_engine.h`
    - `core/store/store_engine.cc`
    - `core/store/runtime/replica/replica_runtime.cc`
    - `core/store/runtime/ingestion_events.h`

- Daemon retained-body lifecycle
  - introduce `BodyDescriptor`,
  - define `BodyAccessClass`,
  - define `BodyBackingIntent` as an internal lowering hint only,
  - define `BodyBackingObservation`,
  - define `ResolvedBodyCapability`,
  - add `BodyBackingManager` under the `byte_artifact` profile runtime,
  - move policy and staging ownership out of `ByteArtifactController`,
  - keep `BodyStore` metadata-centric and authority-local,
  - keep `BodyHandle` thin and move descriptor/observation ownership into `BodyStoreEntry` or equivalent authority
    views.
  - target files:
    - `daemon/service/body_backing_manager.h`
    - `daemon/service/body_backing_manager.cc`
    - `daemon/service/byte_artifact_body_handle.h`
    - `daemon/service/byte_artifact_body_handle.cc`
    - `daemon/service/byte_artifact_body_store.h`
    - `daemon/service/byte_artifact_body_store.cc`
    - `daemon/service/controllers/byte_artifact_controller.cc`
    - `daemon/service/artifact_profile_registry.cc`

- Payload transport and transfer convergence
  - keep `payload_ref` bound to body handles or shared sources,
  - remove hot-path dependence on full body rereads,
  - make all protocol capabilities resolve through shared loaders, handle leases, or remote sources before chunk
    fallback,
  - preserve `FetchPayloadRefChunk` only as a bounded fallback where still needed.
  - target files:
    - `daemon/service/payload_transport_broker.h`
    - `daemon/service/payload_transport_broker.cc`
    - `daemon/service/controllers/transport_controller.cc`

- Shared-memory and export integration
  - reuse CPU memfd and stable-lease semantics from `0049` and `0082`,
  - keep CUDA IPC and region-backed writes on existing paths,
  - keep communicator export on existing core APIs,
  - ensure retained-body policy decisions are explainable through existing memory and export state,
  - keep concrete export or transport protocol choice out of retained-body truth.

- Docs
  - keep `0088` as the shared-dataplane baseline,
  - make revised `0089` the canonical retained-body follow-on,
  - explicitly state that `BodyBackingIntent` is not a second `StorePolicy`,
  - add a body-term -> core-contract mapping table,
  - explicitly narrow `payload_ref` to capability + fallback semantics,
  - document hard-cut target and post-cut cleanup expectations clearly.

# Acceptance Checks

- `BodyDescriptor` plus `BodyHandle` become the canonical retained-body representation, with descriptor truth owned by
  `BodyStoreEntry` / authority view rather than by the handle itself.
- the shared executor returns verified content metadata needed to build `BodyDescriptor`.
- `BodyDescriptor` is the common-path validation source; common-path body rereads are removed.
- `BodyAccessClass`, `BodyBackingIntent`, `BodyBackingObservation`, and `ResolvedBodyCapability` are explicit and
  centrally owned, with `BodyBackingIntent` scoped as an internal lowering hint.
- `BodyStore` stores descriptor plus handle and remains metadata-centric.
- `BodyHandle` remains a thin execution reference and does not become the owner of descriptor, policy, or export state.
- `BatchPutIfAbsentFromRegion` no longer hard-codes CPU retained-body staging in controller-local helpers.
- local export, stable retention, and remote communicator export continue to use existing core APIs and terminology.
- protocol-capability resolution prefers shared loaders, local handle paths, or shared remote sources and treats chunk RPC
  as fallback.
- body expiry, conflict cleanup, and forwarder cleanup retire staged replicas through existing core lifecycle APIs.
- no new daemon-local copy, export, or transfer runtime is introduced.
- `BodyBackingManager` remains subordinate to the profile runtime instead of becoming a new long-lived owner layer.
- no long-term compatibility shim preserves a second retained-body semantics line after cutover cleanup.

Repository searches expected to converge:

- `rg -n "build_body_lowering_plan|stage_loader_to_body" daemon/service/controllers/byte_artifact_controller.cc`
  - expected result: no controller-local retained-backing owner remains
- `rg -n "compute_sha256_hex\\(" daemon/service/artifact_profile_registry.cc daemon/service/controllers/byte_artifact_controller.cc`
  - expected result: no common-path body reread validation remains
- `rg -n "descriptor\\(|observation\\(" daemon/service/byte_artifact_body_handle.h daemon/service/byte_artifact_body_handle.cc`
  - expected result: no long-term authority-truth ownership is pushed into `BodyHandle`
- `rg -n "std::shared_ptr<const std::string>|payload = .*std::string" daemon/service/byte_artifact_body_store.* daemon/service/byte_artifact_body_handle.*`
  - expected result: no primary retained-body heap-byte contract remains
- `rg -n "BodyBackingTarget|CPU_SHARED|GPU_RESIDENT|INLINE_FALLBACK" daemon/service docs/designs/0089-core-backed-body-handles-and-backing-policy.md`
  - expected result: old body-private target taxonomy is removed or explicitly demoted from the design
- `rg -n "FetchPayloadRefChunk|chunk_rpc" daemon/service/payload_transport_broker.cc docs/designs/0089-core-backed-body-handles-and-backing-policy.md`
  - expected result: chunk path is fallback-oriented, not described as the canonical long-term path

# Test / Rollout / Backout

Tests:

- core and daemon C++ tests for:
  - descriptor validation,
  - retained CPU body observation and stable-retention admission,
  - retained GPU body observation,
  - `payload_ref` issuance without reread,
  - remote-source resolution ordering,
  - forwarder cleanup and idempotent retirement.

Suggested commands:

- `bazel test //daemon:grpc_service_impl_cpu_memfd_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:grpc_service_impl_batch_redirect_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

Rollout stance:

- favor the end-state model over compatibility-first landing,
- land the semantic cut,
- clean duplicate code immediately after the cut,
- do not treat the work as complete while duplicate retained-body semantics still exist.

Backout:

- revert the in-progress implementation if correctness or lifecycle regressions appear,
- do not preserve a long-term feature flag that keeps the duplicate retained-body design alive.
