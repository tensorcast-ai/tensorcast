---
title: Region-Backed tensor_dict_into (Plan)
links:
  design: ../designs/0042-region-backed-tensor-dict-into.md
---

# Current State

`tensor_dict_into` materializes a daemon-owned replica, exports a single IPC handle, and performs client-side copies. The v2 materialization path assumes `mem_handle` on completion, so there is no RPC that writes directly into client regions. Region-backed registration exists but is only used for LIP and transport staging.

# Latest Status

Phase 1–3 implementation is complete (proto + daemon + core + SDK + docs). Added basic
daemon validation tests; remaining Phase 4 tests (region bounds, tensor name mismatch,
device_uuid mismatch, poison paths, SDK tests) are still pending.

# Phases & Milestones

- [x] Phase 1: RPC + proto surface
- [x] Add `MaterializeIntoTarget` RPC and `MaterializeIntoTargetRequest/Response`,
      plus `TargetLayout` / `TargetTensorOffset` in `proto/tensorcast/daemon/v2/store_daemon.proto`
- [x] Define `logical_layout_hash` as a logical-only optional field in `TargetLayout`;
      document `selection_hash` as Phase 5 (view/subset) only
- [x] Wire `MaterializeIntoTarget` in `daemon/grpc_service_impl.h/.cc`
- [x] Add `supports_region_backed_get_into` to materialize capabilities
- [x] Document Phase 1 constraints: `artifact_id` required, `disk_fallback` hint-only,
      `tensor_names` / `view_subset_hash` reserved (no subset support)
- [x] Regenerate stubs with `bash tools/build_proto_python.sh`

- [x] Phase 2: Daemon materialize-into path (artifact_id only, canonical only, no verification)
- [x] Implement `MaterializeIntoTarget` handler in `daemon/service/controllers/materialization_controller.cc`
- [x] Add `StoreEngine::materialize_into_target` + `MaterializationFacade` wiring so the
      daemon delegates to core rather than owning dataplane logic
- [x] Add target layout validation (COALESCED only, single storage, `index_kind=CANONICAL`)
- [x] Require `artifact_id`; reject key-based requests and `disk_fallback` without `artifact_id`
- [x] Reject `tensor_names` / `view_subset_hash` and any non-full layout (Phase 1)
- [x] Require `device_uuid`; resolve UUID to device ordinal and validate `storage.device_id`
- [x] Reject `view` / `view_id` for region-backed targets in Phase 1
- [x] Acquire/release region refs via `IpcRegionRegistry` and map IPC handles
- [x] Extend `IpcRegionRegistry` to track region state (active/poisoned) and reject
      poisoned regions; clear poison on `UnregisterVramRegion`
- [x] Add external GPU target path in `core/store/materialization/runtime/pipeline/*`
- [x] Skip verification for external targets; ignore `enable_verification` and emit a
      "verification_skipped" metric
- [x] Emit `tc_store_materialize_into_target_total{result,reason,source}` in
      `daemon/service/controllers/materialization_controller.cc`
- [x] Add SegmentPlan cache keyed by `logical_layout_hash` (or `index_multihash`) + `generation`
- [x] Compute `logical_total_size = max(offset + size)` from canonical index and require
      `storage_length == logical_total_size`
- [x] Treat gRPC cancellation as pre-transfer only; once pump starts, ignore cancellation
- [x] On transfer failure after start, mark region poisoned and return `DATA_LOSS`
      (retryable=false); use `FAILED_PRECONDITION` for poisoned region

- [x] Phase 3: SDK region-backed get_into
- [x] Add `MaterializeIntoTarget` method to `tensorcast/daemon_ctl.py`
- [x] Build `TargetLayout` from target tensors in `tensorcast/api/store/materialization.py`
- [x] Scope `region_backed_mode` to into APIs only; `get` / `get_view` always use the
      daemon-owned replica path
- [x] Enforce contiguous + single-storage + region coverage; require `artifact_id` in Phase 1
- [x] Populate `device_uuid` from the resolved CUDA device and treat it as authoritative
- [x] Require full layout: `tensor_names` empty or full set; reject subsets when
      `region_backed_mode=require` and fallback in `auto`
- [x] Validate target dtype/shape/stride against canonical index without relying on `mem_handle`
- [x] Compute `logical_total_size` and require `storage_length` to match
- [x] Call `MaterializeIntoTarget` RPC and return a dedicated into result type (no
      `mem_handle`, no `UnloadReplica`)
- [x] Make `get_into_async.cancel()` return `False` for in-flight region-backed transfers;
      optionally fall back pre-transfer when `region_backed_mode=auto`
- [x] Evict poisoned regions from the SDK registry/cache on `DATA_LOSS` / `FAILED_PRECONDITION`
- [x] Mark region-backed failures as non-retryable in SDK error surfaces
- [x] Add `region_backed_mode` enum to `proto/tensorcast/config/v1/client_config.proto`
- [x] Wire `region_backed_mode` through client config loader and `client_runtime.client_defaults()`
- [x] Extend `apply_client_load_defaults_if_present` in `tensorcast/api/_runtime.py` and
      `GetArtifactOptions` in `tensorcast/api/_config.py` to respect config defaults
- [x] Regenerate stubs after config proto updates (`bash tools/build_proto_python.sh`)
- [x] Gate on `supports_region_backed_get_into` capability
- [x] Extend `StoreCapabilities` in `tensorcast/api/store/types.py` and session init in
      `tensorcast/api/store/runtime.py` to call `GetMaterializeCapabilities`
- [x] Add SDK counters for fallback reasons and verification skips in
      `tensorcast/api/_metrics.py` and emit them from `tensorcast/api/store/materialization.py`

- [ ] Phase 4: Tests and docs
- [x] Add daemon unit tests for `MaterializeIntoTarget` validation (missing artifact_id,
      disk_fallback empty, subset/view, missing device_uuid)
- [ ] Add daemon unit tests for `MaterializeIntoTarget` region bounds
- [x] Add daemon tests for `view` / `view_id` rejection
- [ ] Add daemon tests for tensor name mismatch
- [ ] Add daemon tests for `device_uuid` mismatch and `disk_fallback` without `artifact_id`
- [ ] Add daemon tests for subset rejection (`tensor_names` / `view_subset_hash`)
- [ ] Add SDK tests for layout building and fallback behavior
- [x] Update `docs/internals/tensor_dict_into_dataflow.md`
- [x] Update module README.md files (`daemon/README.md`, `tensorcast/api/README.md`)

- [ ] Phase 5 (follow-up): Key path, view layouts, verification
- [ ] Implement key-based `MaterializeIntoTarget`
- [ ] Add `INDEX_KIND_VIEW` support and view-index preflight if needed
- [ ] Introduce `selection_hash` for view/subset identity and update SegmentPlan cache
      to key on `logical_layout_hash + selection_hash + generation`
- [ ] Split `GetArtifactOptions` into `GetIntoOptions` / `TargetSpec` so into-specific
      options do not affect `get` / `get_view`
- [ ] Implement external-target verification (SegmentPlan hashing)
- [ ] Add range-based GPU→external target copy helper to enable local-replica fast path

# Acceptance Checks

- `MaterializeIntoTarget` completes without allocating daemon VRAM replicas.
- Responses return no `mem_handle`; IPC handle count equals number of regions.
- Phase 1 rejects key-based requests and `INDEX_KIND_VIEW` layouts.
- Phase 1 rejects non-full layouts (`tensor_names` / `view_subset_hash`).
- Phase 1 requires `device_uuid` and validates against `storage.device_id`.
- External-target verification is skipped and reported via metrics.
- `get` / `get_view` remain unchanged; `region_backed_mode` only affects into paths.
- Region-backed transfers are non-cancelable once started; failures poison the region
  and return non-retryable `DATA_LOSS` / `FAILED_PRECONDITION`.
- Fallback path remains unchanged when region-backed requirements are not met.

# Test Plan

- `bazel test //daemon:materialize_into_target_test` (new)
- `bazel test //daemon:grpc_service_impl_registration_test`
- `bazel test //daemon:materialize_into_target_validation_test` (new)
- `bazel test //daemon:materialize_into_target_poison_test` (new)
- `uv run pytest tests/python/api/test_materialize_into_target.py` (new)
- `uv run pytest tests/python/api/test_get_into_cancel.py` (new)
- `uv run pytest tests/python/test_store_region_registration.py`
- `uv run pytest tests/python/api/test_client_defaults.py` (if adding config defaults)

# Rollout / Backout

- Rollout: deploy daemon and SDK together, enable `region_backed_mode=auto` to start
  using `MaterializeIntoTarget`.
- Backout: disable via config or revert SDK to use legacy `get_into` copies.

# Risk Tracking

- Risk: partial writes on retry when `region_backed_mode=require`.
- Risk: poisoned regions reduce reuse and require explicit re-registration.
- Risk: non-contiguous targets silently mis-copied if validation is incomplete.
- Risk: region TTL expiry mid-transfer without refresh.
- Risk: no external-target verification in Phase 1 could mask corruption.
- Risk: device_id/UUID mismatches across processes if validation is lax.

# Owner Checklist

- Confirm proto regeneration and RPC wiring.
- Validate metrics and tracing in both daemon and SDK.
- Update module READMEs if behavior or API surface changes.
- Document Phase 1 limits (artifact_id only, canonical only, verification skipped).
