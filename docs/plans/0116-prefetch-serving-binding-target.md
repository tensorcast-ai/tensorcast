---
slug: prefetch-serving-binding-target
title: Prefetch Serving Binding Target Implementation Plan
status: draft
areas: ["daemon", "sdk", "proto", "serving", "tests"]
related_code:
  - docs/designs/0116-prefetch-serving-binding-target.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/publication/v1/publication.proto
  - proto/tensorcast/config/v1/daemon_config.proto
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/types.py
  - tensorcast/node_agent/executor.py
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/state/binding_registry.*
  - daemon/state/daemon_kernel.cc
  - daemon/state/handle_lease_registry.*
links:
  design: ../designs/0116-prefetch-serving-binding-target.md
---

# Objective

Implement `ServingBindingTarget` as an explicit `Artifact.prefetch(...)` target
while preserving ordinary prefetch semantics. The implementation must provide a
retained daemon-owned serving binding resource, safe worker acquire with fresh
leases, and simple GPU memory retention.

# Latest Implementation Status

Updated 2026-05-09.

Implemented in this change:

- SDK/runtime contract types for topology, member, source refs, source reuse
  decisions, resolved layout, target/set target, retention policy, reservation
  capability, and typed serving prefetch results.
- Proto contract for serving binding targets, explicit
  `ServingBindingSourceRef`, source reuse decisions, retention policy, typed
  prefetch results, partitioned-set member failure diagnostics,
  `PrefetchServingBinding`, and `AcquireBindingValue`.
- Plan payload fields and node-agent/local-plan routing for explicit serving
  prefetch targets without changing ordinary `prefetch_set` semantics.
- Daemon config shape for serving prefetch enablement/default retention/cache
  root and same-daemon acquire gating.
- Daemon config plumbing into service options, with explicit disabled-feature
  RPC failures before the materialization/acquire executor is implemented.
- Daemon RPC stubs that fail closed with `UNIMPLEMENTED` when the feature is
  enabled but materialization/acquire execution is not yet wired.
- Daemon request validation now runs before those `UNIMPLEMENTED` boundaries:
  serving prefetch rejects missing/incomplete resolved layout before allocation,
  rejects runtime target layouts that are not daemon `TargetLayout` wire
  format, and acquire validates full binding value identity, capability
  metadata, daemon/session/device/member/layout/schema/build digests, expiry,
  and caller pid when a caller pid is required.
- SDK fail-closed validation for `serving_transform_required` and
  `unsupported` before daemon allocation can be requested.
- SDK compatibility tests covering ordinary prefetch signature compatibility,
  source-aware target/set serialization, direct serving-member-copy eligibility,
  source-reuse admission planning, typed result serialization, partitioned-set
  partial diagnostics, and retention validation.
- Binding registry retained-resource state split:
  `pid_bound` vs `daemon_retained`, retained refs vs attachment refs, explicit
  retire/free separation, creator-pid exit behavior, and unacquired/idle/
  materialization timeout sweep primitives.
- Binding retention is wired into the daemon background scheduler, and
  binding-value attachment refs support keepalive plus idle-TTL arming on
  release.
- `AcquireBindingValue` now validates retained ready binding identity and mints
  a fresh external CUDA lease for the caller pid, returning the IPC handle,
  target index bytes, payload descriptors, reservation bytes, and current value.
- RPC acquire tests now assert stale daemon-session/local-ready requests fail
  before a fresh lease is minted.
- `AcquireBindingValue` is gated to local loopback/UDS peers before honoring the
  explicit caller pid; PID-level transport credentials remain enforced by the
  local handle UDS plane where gRPC does not expose peer pid.
- Single-member `PrefetchServingBinding` now reuses `CreateOwnedBinding` source
  materialization, converts the binding to daemon-retained `READY_LOCAL`, drops
  the bootstrap export lease, records serving member/build/layout/schema
  metadata, and returns `PrefetchServingBindingResult`.
- Partitioned serving binding sets now run per-member materialization
  synchronously, return typed per-member failures, and retire already-created
  siblings on member failure.
- Daemon fake-GPU RPC integration now covers serving prefetch materialization
  into daemon-owned memory followed by same-daemon `AcquireBindingValue`.
- CPU metadata implementation for the resolved serving binding spec cache:
  canonical key/spec hashing, `serving_binding_specs/v1` directory layout,
  per-key file lock, tmp write plus atomic publish, blob hash/size validation,
  unsafe path rejection, producer/runtime/schema validation, same-key
  different-spec rejection, group index read/write, and cache readback
  validation.
- Exact-match resolved spec cache read API for steady-state reuse: callers pass
  the expected spec and cache hits are rejected when the stored spec differs
  from the worker/runtime expectation.
- CPU cache tests now cover cold-path publication of a readable resolved spec,
  exact steady-state reuse, and fail-closed worker mismatch handling.
- Partitioned target tests now cover distinct per-member device/layout specs
  sharing one topology ref.
- TensorCast SDK store docs mention explicit serving prefetch targets and the
  feature-gated/fail-closed state of daemon materialization.

Still not implemented:

- Runtime-owned cold-start trace/compile persistence into the resolved spec
  cache. TensorCast has the cache format, exact-match reuse, and fail-closed
  validation path; the runtime trace producer is outside daemon-only scope.
- `serving_published_ready` promotion and topology-scoped reshard execution for
  `serving_transform_required`.
- End-to-end GPU process integration test that exercises a real worker acquire
  and tensor read.

Next execution plan:

1. Wire the resolved spec cache into runtime production:
   - cold runtime trace/compile publishes the exact resolved spec entry after
     resolved-layout validation;
   - steady materialization uses `read_matching_resolved_spec_cache_entry()` and
     fails before allocation on mismatched member/runtime specs.
2. Add end-to-end fake/real GPU integration coverage for
   `PrefetchServingBinding` plus `AcquireBindingValue`.
3. Defer internal-vLLM integration tasks until TensorCast daemon/API atomic
   capabilities pass their own tests.

Latest verification:

- `bash tools/build_proto_python.sh`
- `source .venv/bin/activate && ruff format tensorcast/api/store/serving_binding_spec_cache.py tensorcast/api/store/__init__.py tests/python/api/test_serving_binding_spec_cache.py tests/python/api/test_prefetch_serving_binding_target.py`
- `source .venv/bin/activate && ruff check .`
- `source .venv/bin/activate && pytest tests/python/api/test_serving_binding_spec_cache.py tests/python/api/test_prefetch_serving_binding_target.py tests/python/api/test_operation_semantics.py` (39 passed on 2026-05-11)
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:binding_registry_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_operation_rpc_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

# Current State & Grounding

- Ordinary prefetch is already operation-shaped and `NO_LEASE`:
  - `tensorcast/api/store/artifact.py`
  - `tensorcast/node_agent/executor.py`
  - `proto/tensorcast/plan/v1/plan.proto`
- The daemon already supports binding allocation, current value metadata,
  local-ready values, promotion, and IPC lease-backed tensor restore:
  - `daemon/service/controllers/owned_binding_service.cc`
  - `daemon/state/binding_registry.*`
  - `daemon/state/handle_lease_registry.*`
  - `tensorcast/api/store/owned_binding_slot.py`
- The current binding registry ties owner pid exit and export refs too tightly
  for process-external serving preload:
  - `daemon/state/binding_registry.cc`
  - `daemon/state/daemon_kernel.cc`
- `internal-vllm` already has same-binding local-ready and published-ready
  runtime state concepts, but external prefetch/acquire needs TensorCast-side
  primitives before it should be wired in:
  - `/data/workspace/internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py`
  - `/data/workspace/internal-vllm/vllm/v1/worker/gpu_worker.py`

# Phases & Milestones

- [x] Phase 0: Contract hardening and 0111 alignment
  - [x] Use the 0111 topology abstraction note: topology-sensitive serving
        designs use `ServingTopologyRef` and `ServingBindingMemberRef`, while
        framework labels such as TP or PP stay in adapters/diagnostics.
  - [x] Define which topology facts are `schema_topology_digest`,
        `admission_topology_digest`, or runtime-ephemeral diagnostics.
  - [x] Define same-daemon acquire as the first implementation boundary for
        `serving_local_ready`.
  - [x] Define `BindingReservationCapability` as the authority carrier for
        retained serving binding acquire.
  - [x] Align serving prefetch source identity with existing TensorCast
        `ArtifactSelection`, ByteSpace, `RepresentationTransformContract`,
        `ServingArtifactManifest`, and `PublishedModelVersion` semantics rather
        than introducing a parallel source model.
  - [x] Define direct-serving-source eligibility:
        `serving_direct_member_copy` is allowed only for compatible
        representation/topology/member/layout, while topology/layout mismatch
        becomes `serving_transform_required`.
  - [x] Define typed operation result carriers packed into
        `OperationStatus.result` / `GetOperationResponse.snapshot`.
  - [x] Define daemon config fields for feature enablement, retention defaults,
        materialization timeout, and resolved spec cache root.

- [x] Phase 1: Contract and types
  - [x] Add SDK/runtime types for `ServingTopologyRef`,
        `ServingBindingMemberRef`, `ServingBindingTarget`,
        `ServingBindingSetTarget`, `PrefetchRetentionPolicy`,
        `BindingReservationCapability`, `PrefetchedServingBinding`, and
        `PrefetchedServingBindingSet`.
  - [x] Add SDK/runtime types for `ServingBindingSourceRef`,
        `ServingBindingSourceMemberRef`, and
        `ServingBindingSourceReuseDecision`.
  - [x] Add a resolved per-member serving binding layout shape carrying layout,
        target index, copy plan / mapped tensor specs, schema hash, topology,
        member, source representation, reuse decision, and build digest.
  - [x] Define resolved spec cache keys and values, including source schema
        hash, source kind, artifact-selection digest, source representation
        identity, source topology/member identity when applicable, reuse
        decision, schema/admission topology digests, target member identity,
        serving build digest, and `spec_digest`.
  - [x] Define the on-disk resolved spec cache layout:
        `serving_binding_specs/v1/keys`, `groups`, `specs`, `locks`, and
        `tmp`.
  - [x] Define canonical JSON hashing for `cache_key_digest` and
        `spec_digest`, including required fields and excluded volatile fields.
  - [x] Add cache manifest validation for schema version, embedded key match,
        blob hash, blob size, producer version, and supported runtime.
  - [x] Add a group target/result shape for partitioned serving binding
        prefetch.
  - [x] Add typed partitioned-set partial-result and per-member failure
        diagnostics.
  - [x] Add proto fields or operation payloads for serving binding target,
        requested readiness, retention policy, and result metadata.
  - [x] Keep ordinary `Artifact.prefetch(device=...)` return type and behavior
        unchanged.
  - [x] Add status/result serialization for `BindingValueRef`,
        daemon/session identity, device UUID, member, reservation capability,
        readiness, and `expires_at_ms`.

- [x] Phase 2: Daemon retained resource lifecycle
  - [x] Extend binding registry state to distinguish creator, retained
        resource, and attachment refs.
  - [x] Add retained binding identity, reservation capability id,
        `control_lifetime`, timestamps, deadlines, and retired reason fields.
  - [x] Integrate retention with `LifecycleKernel`, `BindingRegistry`, and
        handle-lease/capability tracking rather than adding a separate cleanup
        system.
  - [x] Update pid-exit handling so creator exit does not free retained
        local/published-ready binding when policy allows daemon-retained
        lifetime.
  - [x] Add retention sweeper for unacquired TTL, idle TTL, and materialization
        timeout.
  - [x] Ensure active attachments prevent free but do not prevent retire.
  - [x] Implement binding-value-scoped pin/keepalive without overloading
        artifact replica placement pins.

- [ ] Phase 3: Serving prefetch execution
  - [x] Lower `Artifact.prefetch(target=ServingBindingTarget(...))` to a
        daemon operation.
  - [x] Require an explicit resolved layout from the runtime adapter or engine
        parent before GPU reservation. Defer executor-generated resolution to a
        future phase.
  - [x] Use cached resolved specs for steady-state prefetch when the cache key
        matches exactly.
  - [x] Add a source/target compatibility planner that emits
        `checkpoint_to_serving`, `serving_direct_member_copy`,
        `serving_transform_required`, or `unsupported` before GPU allocation.
  - [x] Treat invalid, missing, or unsupported cache entries as pre-allocation
        cache misses; never reserve GPU memory from an unverified spec.
  - [x] Support `serving_direct_member_copy` through ordinary artifact/view
        materialization and P2P only when source and target serving
        representation/topology/member/layout are compatible.
  - [x] Fail `serving_transform_required` before allocation until a future
        topology-scoped reshard executor consumes
        `RepresentationTransformContract`.
  - [ ] Build a first-cold-start path that traces/compiles runtime load and
        writes the resolved per-member spec cache entry.
  - [x] Write cache entries through a per-key lock, temporary directory, blob
        verification, and atomic rename.
  - [x] Reject serving prefetch before allocation if the target layout/index,
        schema hash, or member/device assignment cannot be resolved.
  - [x] Require the runtime-provided serving binding layout to parse as the
        daemon `TargetLayout` wire format before executor allocation.
  - [x] Build the serving binding layout and target index through existing
        binding-native builder code during materialization.
  - [x] Materialize source data into daemon-owned GPU allocation.
  - [x] Run framework finalize, semantic validation, schema validation, and
        binding invariant validation before reporting serving readiness.
  - [x] Freeze to `serving_local_ready`; optionally start async promotion to
        `serving_published_ready`.

- [x] Phase 4: Worker acquire
  - [x] Add `AcquireBindingValue` or equivalent acquire path.
  - [x] Require full `BindingValueRef`; keep `local_serving_ref` only as a
        lookup/diagnostic hint and reject acquire requests that rely on it as
        authority.
  - [x] Validate reservation capability, daemon id, daemon session id, device
        UUID, member, expected layout/schema/build digests, state, seal
        generation, and expiry.
  - [x] Validate local peer credentials and caller pid when the transport
        exposes them.
  - [x] Mint a fresh external CUDA lease for the authenticated local caller.
  - [x] Return IPC handle, payload descriptors, target index bytes, and
        reservation bytes.
  - [x] Release attachment refs through tensor/lease lifetime, not owner close.

- [x] Phase 5: Plan and node-agent integration
  - [x] Add plan action support for serving binding target without changing
        `prefetch_set` readiness floor.
  - [x] Add serving binding set handling for partitioned serving: per-member
        specs, group session id, member barrier, and sibling cleanup.
  - [x] Add optional group cache index handling that maps a topology-level key
        to per-member resolved spec entries.
  - [x] Ensure idempotency keys include target kind, runtime, layout/build
        digests, topology/member, device, and readiness.
  - [x] Surface partial-result and per-member failure diagnostics for serving
        binding sets.
  - [x] Lower generic member index/count into `CollectiveLoadGroup` only for
        execution strategies that need collective hints.

- [ ] Phase 6: Internal-vLLM consumption
  - [ ] Consume `PrefetchedServingBinding` from engine parent or manual launch.
  - [ ] Map vLLM TP/PP/DP placement into `ServingTopologyRef` and
        `ServingBindingMemberRef` in the adapter layer.
  - [ ] Inject reservation capability, reservation bytes, and binding identity
        before worker startup memory snapshot checks.
  - [ ] Recompute or validate expected spec/layout/schema/build digests during
        worker startup and fail fatal/fail-closed on mismatch.
  - [ ] Acquire in `load_model()` and attach returned tensors.
  - [ ] Treat `serving_reserved`, `serving_local_ready`, and
        `serving_published_ready` as distinct runtime states.
  - [ ] Keep fail-closed behavior when reservation or identity is missing.

# Tasks

- [x] Add unit tests for SDK type validation and ordinary prefetch compatibility.
- [x] Add SDK serialization tests for `ServingTopologyRef`,
      `ServingBindingMemberRef`, `ServingBindingSourceRef`, reuse decisions,
      binding sets, and typed operation results.
- [x] Add daemon tests for retained binding pid-exit behavior.
- [x] Add daemon tests for unacquired TTL, idle TTL, and materialization timeout.
- [x] Add acquire tests for stale `binding_value_id`, stale `seal_generation`,
      layout mismatch, schema mismatch, build digest mismatch, retired state,
      daemon session mismatch, device/member mismatch, caller identity mismatch,
      expired/tampered capability, and `local_serving_ref`-only requests.
- [x] Add tests that serving prefetch fails before allocation when the per-member
      layout spec cannot be resolved.
- [x] Add tests that serving-to-serving direct P2P is admitted only for matching
      representation/topology/member/layout and that mismatched topology returns
      `serving_transform_required` before allocation.
- [x] Add CPU cold-path cache publication test that writes a readable resolved
      spec cache entry.
- [x] Add CPU steady-state test that reuses an exact cached resolved spec and
      rejects mismatched specs before allocation.
- [x] Add cache format tests for canonical key hashing, spec hashing, blob hash
      validation, unsupported schema version, partial temp write cleanup, and
      concurrent writer idempotency.
- [x] Add cache conflict tests for unsupported producer/runtime metadata and
      same-key different-spec rejection.
- [x] Add worker validation tests that cached spec mismatch is
      fatal/fail-closed.
- [ ] Add GPU integration test for prefetch serving binding, process exit,
      worker acquire, tensor read, worker exit, and memory release.
- [x] Add daemon-session mismatch acquire test showing local-ready acquire fails
      after the expected daemon session changes.
- [x] Add feature-disabled tests for prefetch and acquire config gates.
- [x] Add partitioned-set failure cleanup test once group prefetch support lands.
- [x] Add partitioned-set test that verifies per-member layout/device specs are
      distinct and share one topology ref.
- [x] Add partitioned-set typed partial diagnostics tests.
- [x] Add partitioned-set cache-index test that verifies group lookup does not
      hide per-member cache mismatches.
- [x] Add daemon/API fake-GPU integration coverage for prefetch serving binding
      materialization and same-daemon acquire.
- [x] Update TensorCast SDK docs after API names are finalized.
- [ ] Update internal-vLLM docs and tests only after TensorCast atomic
      capabilities pass their own validation.

# Test / Rollout / Backout

## Test Plan

- SDK:
  - ordinary prefetch returns `PrefetchedReplica`;
  - serving target prefetch returns `PrefetchedServingBinding`;
  - serving set prefetch returns `PrefetchedServingBindingSet`;
  - topology/member models reject missing identity fields;
  - retention policy validates non-negative durations;
  - idempotent operation keys reject mismatched target payloads.
- Daemon:
  - creator pid exit does not free retained ready binding;
  - acquirer pid exit releases only that attachment;
  - TTL retire blocks new acquire;
  - active refs block free;
  - zero-ref retired bindings free GPU memory;
  - acquire rejects stale daemon sessions, wrong member/device, and invalid
    reservation capabilities;
  - local-ready acquire is same-daemon only.
- Integration:
  - manual prefetch then process-external acquire succeeds;
  - second prefetch with the same resolved spec key materializes directly from
    cached layout/copy plan;
  - corrupted cache blob fails before GPU allocation;
  - mismatched cached spec fails before worker attach;
  - prefetch expires when not acquired;
  - acquired binding survives retire until tensors are dropped;
  - local-ready cannot be used as ordinary published serving state.
  - partitioned-set prefetch requires all member specs to resolve before
    serviceable success is reported.

## Rollout

- Keep serving binding target behind a typed daemon config feature flag.
- Default retention policy should be conservative and bounded.
- Enable manual prefetch first, then engine-parent orchestration.
- Enable partitioned binding sets only after single-member lifecycle,
  capability validation, and memory release evidence is stable.
- Keep executor-generated layout resolution disabled until explicit
  resolved-layout mode is validated.

## Backout

- Disable the serving target feature flag.
- Leave ordinary `prefetch` and `prefetch_set` untouched.
- Retire any retained serving bindings through daemon cleanup.
- Keep acquire RPC disabled if fresh lease or caller identity validation is not
  available.

# Risks & Tracking

- [ ] Ordinary prefetch compatibility regression.
- [ ] GPU memory retained after abandoned prefetch.
- [ ] Fresh lease release tied to wrong process or wrong tensor lifetime.
- [ ] Local-ready value misused as durable artifact-backed serving.
- [ ] Runtime-specific topology names leak into daemon lifecycle APIs.
- [ ] Resolved spec cache key misses a layout-affecting input.
- [ ] Parent injects reservation capability after worker startup memory
      snapshot.
- [ ] Partitioned-set partial success leaves sibling allocations resident.
