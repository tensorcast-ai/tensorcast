---
slug: prefetch-serving-binding-target
title: Prefetch Serving Binding Target Follow-up Plan
status: retired
areas: ["daemon", "sdk", "proto", "serving", "tests"]
related_code:
  - docs/designs/0116-prefetch-serving-binding-target.md
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/designs/0121-unified-artifact-realization-kernel.md
  - docs/plans/0121-unified-artifact-realization-kernel.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/operation/v1/operation.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/config/v1/daemon_config.proto
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/owned_binding_slot.py
  - tensorcast/api/store/serving_binding_reference_consumer.py
  - tensorcast/api/store/serving_binding_spec_cache.py
  - tensorcast/types.py
  - daemon/service/controllers/materialization_controller.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/state/binding_registry.*
  - daemon/state/handle_lease_registry.*
links:
  design: ../designs/0116-prefetch-serving-binding-target.md
  superseded_by:
    - ../designs/0120-artifact-centered-model-runtime-realization.md
    - ../designs/0121-unified-artifact-realization-kernel.md
    - ../plans/0121-unified-artifact-realization-kernel.md
---

# Objective

This plan is retired as an active execution plan. The initial
`ServingBindingTarget` implementation has landed, and the remaining work is now
owned by `0121` as retained artifact realization target/lifecycle/report work.

The content below is retained as historical grounding and a checklist source.
Do not execute it as a standalone serving-preload plan. Move any still-relevant
item into `0121` before implementation.

The shipped baseline already supports daemon-retained `serving_local_ready`
bindings, process-external `AcquireBindingValue`, resolved spec cache
validation, TensorCast reference consumer coverage, internal-vLLM basic
consumption, and real CUDA E2E validation.

# Current Baseline

Updated 2026-05-11.

- `Artifact.prefetch(target=ServingBindingTarget(...))` materializes a
  daemon-retained local serving binding without changing ordinary prefetch
  semantics.
- `AcquireBindingValue` validates binding identity, daemon/session/device/member
  identity, layout/schema/build digests, reservation capability, state, expiry,
  and caller pid before minting a fresh external CUDA handle lease.
- Tensor restore owns the fresh external handle lease lifecycle; callers should
  not pass acquire handle tokens to `ReleasePlacementLease`.
- `AcquireBindingValueResponse.lease_token` is `bytes`, matching the binary
  capability token also carried in `MemCopyHandle.lease_token`.
- Resolved serving spec cache entries are exact-match, hash-validated, and
  fail-closed on runtime/member/layout/schema mismatch.
- TensorCast has a reference serving consumer and real CUDA E2E that covers:
  public disk source resolve, retained prefetch, worker-process acquire, CUDA
  IPC restore/read, worker exit, parent reacquire, tensor-lifetime release, and
  idle TTL expiry.
- internal-vLLM has the basic integration path for consuming prefetched binding
  metadata and acquiring during model load.

# Retired Follow-up TODO Source

## P0: Stabilize The Public Example

- [ ] Convert `examples/serving_binding_consumer/` into the canonical
      TensorCast-side serving binding example.
- [ ] Document the parent-to-worker handoff payload:
      `ServingBindingTarget`, `PrefetchedServingBinding`, expected digests, and
      reservation capability.
- [ ] Document the lease rule explicitly: acquire returns an external tensor
      handle lease, and tensor restore / tensor lifetime releases it.
- [ ] Add the example to docs navigation from `docs/README.md` or the Store API
      README.

## P1: Harden Consumer-Side E2E

- [ ] Add an internal-vLLM real GPU E2E that consumes a
      `PrefetchedServingBinding` from parent orchestration, reaches worker
      `load_model()`, acquires tensors, and exits cleanly.
- [ ] Add internal-vLLM fail-closed E2E coverage for wrong daemon session,
      wrong member, wrong device UUID, wrong layout/schema/build digest, missing
      reservation capability, and late injection after memory snapshot.
- [ ] Keep TensorCast reference consumer E2E as the consumer-agnostic contract
      test, with internal-vLLM E2E covering runtime adapter behavior only.

## P2: Add Operational Visibility

- [ ] Add daemon metrics for retained serving binding count, active attachment
      refs, acquire attempts, acquire validation failures by reason, retire
      reasons, idle TTL expiry, unacquired TTL expiry, and freed bindings.
- [ ] Add structured lifecycle logs for prefetch success, acquire success,
      acquire rejection, idle retire, and final free.
- [ ] Add a read-only status/debug RPC or CLI surface for retained serving
      bindings keyed by local serving ref or binding value ref.

## P3: CI And Test Tiering

- [ ] Mark the real CUDA TensorCast E2E as a GPU-only integration test tier in
      test documentation.
- [ ] Add a nightly or GPU-runner command group that includes:
      `bazel build //daemon:tensorcast_daemon`, daemon fake-GPU RPC tests,
      TensorCast real CUDA E2E, and internal-vLLM real GPU E2E.
- [ ] Keep ordinary Python API compatibility tests in the default non-GPU tier.

## P4: Runtime-Owned Resolved Spec Production

- [ ] Promote the reference consumer cache helpers into a runtime-adapter helper
      API where useful.
- [ ] Let runtime cold-start trace/compile publish exact resolved spec cache
      entries after validation.
- [ ] Ensure every layout-affecting runtime input participates in the cache key
      or spec digest.
- [ ] Keep cache mismatch fatal before GPU allocation.

## P5: Published-Ready And Transform Work

- [ ] Design `serving_published_ready` as a durable serving publication state,
      separate from same-daemon `serving_local_ready`.
- [ ] Define Global Store metadata, capability audience, and lifecycle semantics
      for published serving bindings.
- [ ] Design topology-scoped reshard execution for
      `serving_transform_required`.
- [ ] Keep transform execution out of the local-ready acquire path until the
      reshard executor has explicit rollback, timeout, progress, and sibling
      cleanup semantics.

# Verification Baseline

Run these after changing the current serving binding contract or daemon acquire
path:

```bash
bash tools/build_proto_python.sh

bazel build //daemon:tensorcast_daemon \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error

bazel test //proto/... \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error

bazel test //daemon:grpc_service_impl_operation_rpc_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error

source .venv/bin/activate
pytest tests/python/api/test_serving_binding_reference_consumer.py \
  tests/python/api/test_serving_binding_spec_cache.py \
  tests/python/api/test_prefetch_serving_binding_target.py \
  tests/python/api/test_operation_semantics.py -q

pytest tests/python/daemon/test_prefetch_serving_binding_real_cuda_e2e.py -q
```

The real CUDA test requires a real CUDA backend, a built daemon binary, and CUDA
IPC support on the host. It should skip or move to the GPU integration tier when
those prerequisites are absent.

# Rollout

- Keep serving binding target behind the daemon config feature gate.
- Treat `serving_local_ready` as same-daemon only.
- Treat `serving_published_ready` and transform execution as separate follow-up
  capabilities.
- Keep default retention bounded and conservative.
- Prefer fail-closed validation over fallback materialization when runtime,
  topology, member, layout, schema, or capability identity does not match.

# Backout

- Disable the serving prefetch feature flag.
- Keep ordinary `prefetch` and `prefetch_set` untouched.
- Retire retained serving bindings through daemon cleanup.
- Disable acquire if fresh external handle lease minting or identity validation
  regresses.
