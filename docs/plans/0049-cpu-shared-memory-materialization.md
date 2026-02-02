---
slug: cpu-shared-memory-materialization
title: CPU Shared Memory Materialization (Plan)
links:
  design: ../designs/0049-cpu-shared-memory-materialization.md
areas:
  - core
  - daemon
  - sdk
related_code:
  - core/store/replica/unified_memory_authority.h
  - core/store/replica/unified_memory_authority.cc
  - core/store/materialization/runtime/pipeline/handle_stage.cc
  - daemon/service/controllers/materialization_controller.cc
  - daemon/state/session_lifecycle.h
  - daemon/state/ref_tracker.h
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/api/_materialize.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/artifact.py
  - tensorcast/csrc/checkpoint_py.cc
  - core/checkpoint/checkpoint.cc
---

# Objective

Implement CPU tensor_dict materialization with zero-copy CPU tensors by exporting a local CPU handle from UMA and mapping
it in the SDK using `memfd_create` + local FD handoff (UDS + SCM_RIGHTS).

Lifecycle objective (required for correctness): the daemon returns a `lease_token` capability and the SDK binds it to the
lifetime of the returned `torch.Tensor` objects (RAII). When the last tensor is destroyed, the client releases the lease
token; only then may the daemon drop export pins / stable leases and reclaim.

Shared-DRAM enablement and pool sizing must be decided at daemon startup and fail fast when required resources are
unavailable.

# Current State and Grounding

- **Proto + wire contract are implemented**:
  - `MemCopyHandle` contains `bytes lease_token` and uses a `oneof` handle (`cuda_ipc_handle` or `cpu_memfd`)
    (`proto/tensorcast/daemon/v2/store_daemon.proto`).
  - `GetServerConfigResponse` advertises `local_handle_socket_path` and `cpu_shared_memory_enabled` so the SDK can use
    the local handle plane without ad-hoc env vars.
  - `MaterializeByKeyRequest.target_device_type` allows CPU targets for key-based loads.
- **Daemon local handle plane + handle leases are implemented**:
  - `daemon/state/local_handle_server.{h,cc}` serves a daemon-owned Unix domain socket for:
    - `GetCpuMemfdFd(lease_token)` FD handoff via `SCM_RIGHTS`
    - `ReleaseHandle(lease_token)` best-effort lease retirement
  - `daemon/state/handle_lease_registry.{h,cc}` mints unguessable `lease_token`s, integrates with
    `SessionLifecycleManager::create_ttl_use_lease(...)`, and refcounts CPU export state per `store::loading::ReplicaKey`.
  - Peer checks are enforced via `SO_PEERCRED` (UID must match daemon euid).
- **UMA CPU shared-memory backing is implemented**:
  - When `engine.cpu_shared_memory.enabled` is true, UMA backs CPU allocations with `memfd_create` + `MAP_SHARED`
    (`core/store/replica/unified_memory_authority.cc`).
  - For memfd-backed arenas, UMA forbids the anonymous `MAP_FIXED|MAP_ANONYMOUS` “write-span remap” fallback to preserve
    cross-process sharing semantics.
- **CPU export correctness is budgeted and pinned**:
  - CPU memfd exports acquire UMA stable leases for the exported chunk ranges and hold UMA export pins for the lifetime
    of the handle lease (`engine.memory_tiers.stable_bytes` is the admission budget).
- **SDK CPU tensor_dict is implemented and lease-aware**:
  - `tensorcast/api/_materialize.py` branches on `MemCopyHandle` handle type and maps CPU memfd after readiness.
  - `Artifact.tensor_dict(...)` no longer eagerly calls `UnloadReplica` on success; the C++ extension releases
    `lease_token` when the last tensor view is destroyed (RAII).
- **Hardening deltas discovered during integration** (now implemented in code):
  - Avoid double-dropping `RefTracker` refs when `UnloadReplica` runs alongside `UseLease` finalizers.
  - Bound LocalHandle I/O so `ReleaseHandle` cannot hang tensor destruction paths indefinitely.

# Consistency and Reuse Constraints (Implementation Rules)

- **No new ad-hoc env vars**: all knobs go through `proto/tensorcast/config/v1/daemon_config.proto` and are surfaced to
  clients via additive fields on `GetServerConfigResponse`.
- **One lifetime model**: “borrowed view” APIs (returning tensors backed by daemon memory) must be lease-based; “owned
  copy” APIs (`get_into`, `tensor_dict_into`) remain copy-then-release.
- **Replica identity is `store::loading::ReplicaKey`** across modules:
  - `RefTracker`, unload, and any new handle-lease registries must use the exact same key (including `device.uuid` and
    `view_id` when present).
  - Avoid reconstructing keys from `{artifact_id, device_id}`; pass `ReplicaKey` through finalizers and cleanup.
- **Reuse existing daemon subsystems** instead of introducing parallel ones:
  - Use `SessionLifecycleManager` + `PidMonitor` + `BackgroundScheduler` (`TaskKind::kSessionLifecycle`) for PID/TTL
    cleanup.
  - Reuse `RefTracker` as the eviction “do not reclaim while borrowed” gate (`daemon/state/sweep_tasks.h`).
- **Local-only data plane**: FD handoff and lease release happen over UDS (no gRPC FD hacks); enforce socket filesystem
  permissions (`0600`) and validate peers with `SO_PEERCRED`.
- **Bounded blocking**: any “release lease” path invoked from tensor destruction must be best-effort and bounded
  (timeouts, no unbounded waits), and must not require Python `__del__` hooks.

# Ownership Map (keep layering intact)

- **core/store (UMA + materialization pipeline)**:
  - Owns memfd-backed CPU allocations and any invariants about “what memory is safe to export”.
  - Emits enough metadata in `store::loading::ReplicaHandle` (via `core/store/materialization/runtime/pipeline/handle_stage.cc`)
    for the daemon to describe CPU handles without reaching into unrelated internals.
- **daemon (control plane + local IPC)**:
  - Owns `lease_token` minting, token→replica bookkeeping, and LocalHandle UDS for FD handoff + `ReleaseHandle`.
  - Reuses `SessionsService`/`SessionLifecycleManager`/`RefTracker` so eviction and PID-exit cleanup stay consistent.
- **sdk (Python + C++ extension)**:
  - Owns handle-type branching (CUDA IPC vs CPU memfd), readiness (`ConfirmReplica`), and binding `lease_token` to tensor
    lifetimes via the C++ storage owner.

# Phases and Milestones

- [x] Phase 1: Lease-token lifecycle + local handle plane (foundation)
  - [x] Milestone 1: Extend v2 proto:
    - Add `bytes lease_token` to `MemCopyHandle` (applies to both CUDA IPC and CPU memfd).
    - Add `string local_handle_socket_path` to `GetServerConfigResponse` so SDK can discover the local handle plane.
    - Convert `MemCopyHandle` to a `oneof` handle (`cuda_ipc_handle` vs `cpu_memfd`) so SDK can branch cleanly.
  - [x] Milestone 2: Implement daemon LocalHandle service (UDS):
    - Add `lifecycle.handle_leases` to `proto/tensorcast/config/v1/daemon_config.proto` (local socket path + TTL/auth knobs).
    - Bind/listen on the configured local handle socket path with secure permissions (`0600`).
      - If a stale socket file exists at startup, safely `unlink` it only if it is a socket and is owned by the daemon user; otherwise fail closed.
    - Add `ReleaseHandle(lease_token)` which drops daemon refs/leases (equivalent to `UnloadReplica` semantics but keyed
      by token instead of replica_uuid).
    - Enforce local trust boundary:
      - Require `SO_PEERCRED` UID matches daemon euid (and place the socket in a daemon-owned directory).
      - Do not rely on gRPC-provided pid for security; optionally bind `lease_token -> uid` on first successful LocalHandle
        request.
    - Integrate with daemon lifecycle:
      - Start/stop the LocalHandle server alongside existing sweepers (`daemon/service/grpc_service_impl.cc` start/stop paths).
      - Reuse the unified `SessionLifecycleTask` for TTL/PID cleanup of handle leases (avoid adding a second periodic
        sweeper).
    - Integrate PID-exit cleanup via `PidMonitor` and TTL sweep via `BackgroundScheduler`.
    - Implementation consistency:
      - Run accept/recvmsg/sendmsg on `common::AsyncRuntime::blocking_executor()` (or equivalent) to avoid blocking shared CPU executors.
      - Use `SOCK_SEQPACKET` (preferred) or a length-prefixed stream protocol to avoid ad-hoc framing bugs.
  - [x] Milestone 3: Update SDK + C++ extension:
    - Remove eager `UnloadReplica` in borrowed-view tensor-returning paths (at minimum `Artifact.tensor_dict()` and
      `MaterializationPipeline.get_async()`).
    - Extend the existing CUDA IPC tensor owner in `core/checkpoint::restore_tensors` to optionally also release
      `lease_token` over the LocalHandle socket when the last tensor is destroyed (best-effort, bounded).
    - Thread the LocalHandle socket path from `GetServerConfigResponse` through `StoreRuntimeContext` and into the C++
      owner (avoid per-call RPCs and avoid global env vars).
    - Ensure any FD received via `SCM_RIGHTS` is `CLOEXEC` in the receiver (`MSG_CMSG_CLOEXEC` or `FD_CLOEXEC`) to avoid FD leaks across exec.

- [x] Phase 2: Config + protocol + local FD handoff (CPU handle plane)
  - [x] Milestone 1: Add CPU shared memory enablement config and implement fail-fast startup validation:
    - Add `engine.cpu_shared_memory.enabled` to `proto/tensorcast/config/v1/daemon_config.proto` (memfd export enablement).
    - `memfd_create`/`ftruncate` probe (no silent fallback).
    - Validate stable pool sizing against cgroup limits when present (fail closed on finite `memory.max`).
    - Source `GetServerConfigResponse.cpu_shared_memory_enabled` from daemon config + runtime validation.
  - [x] Milestone 2: Extend v2 proto:
    - Add `CpuMemfdHandle` under `MemCopyHandle` (size/offset; FD exchange uses `lease_token`).
    - Add `target_device_type` to `MaterializeByKeyRequest`.
    - Add capability field to `GetServerConfigResponse` (`cpu_shared_memory_enabled`).
  - [x] Milestone 3: Implement daemon LocalHandle FD exchange:
    - Add `GetCpuMemfdFd(lease_token)` via `SCM_RIGHTS`.
    - Add handle-token registry + per-replica export refcount semantics, mirroring the proven patterns in
      `daemon/state/ipc_region_registry.{h,cc}` (token minting, TTL, refcount, idempotent release).

- [x] Phase 3: UMA shareable CPU backing + export locking + stable budget semantics
  - [x] Milestone 1: Replace CpuArena::allocate_region with shareable memfd-backed mapping and record backing metadata
    needed for export.
  - [x] Milestone 2: Make shareable CPU arena semantics safe:
    - Use UMA stable-lease acquisition for exported CPU chunk ranges as the admission control for `stable_bytes`; fail
      with `RESOURCE_EXHAUSTED` when exhausted.
    - Avoid double-counting: hold stable leases once per unique export coverage and refcount across lease tokens.
    - Remove/forbid anonymous `MAP_FIXED` remap fallback in write_span for shareable mappings.
    - Align eviction/preemptible advice with shareable mappings (`MADV_DONTNEED`/`MADV_PAGEOUT` only when not exported;
      avoid `MADV_FREE`).
  - [x] Milestone 3: Ensure exported CPU chunks are pinned + stable for the lifetime of the handle lease.

- [x] Phase 4: SDK CPU materialization + C++ extension
  - [x] Milestone 1: Add `restore_tensors_from_cpu_fd_with_lease` in core/checkpoint and expose it in tensorcast._C.
  - [x] Milestone 2: Update SDK materialization to:
    - Request CPU targets (`target_device_type=CPU`) and accept `device="cpu"`.
    - `memfd`: exchange token for FD via local handle socket and map with `MAP_PRIVATE`.
    - Keep readiness invariant: only perform FD exchange + `mmap` after `ConfirmReplica` has succeeded.
    - Update `DaemonCtl.confirm_replica_loaded` to pass the requested target device type for logging/consistency.

- [x] Phase 5: Tests and docs
  - [x] Milestone 1: Add C++ tests (include an end-to-end smoke test) for:
    - **E2E (daemon + engine + LocalHandle + memfd)**: `daemon/service/grpc_service_impl_cpu_memfd_e2e_test.cc`
      - Setup a minimal disk artifact using `core/testing/common.{h,cc}`:
        - Create `tensor.data_0` with deterministic content (`create_dummy_file(path, size, start_char)`).
        - Write `tensor_index.json` + `artifact_descriptor.json` via
          `write_rfc0007_descriptor_for_standard_artifact_dir(artifact_dir)`.
      - Start `StoreEngine` with a real `memory_tier_config` (stable budget is required for exports) and offline mode:
        - Set `StoreEngineOptions.memory_tier_config = MemoryTierConfig{stable_bytes=..., enable_preemptible=...}`.
        - Use `opts.global_store_address.clear()` (matches existing daemon tests).
      - Start `StoreDaemonServiceImpl` with:
        - `storage_path` rooted at the test tmpdir (matches shared-root enforcement).
        - CPU shared-memory enabled (new config/option introduced in this RFC).
        - LocalHandle enabled with a **short** filesystem socket path (avoid `sockaddr_un` path length limits).
      - RPC sequence (mirrors SDK contract):
        - Call `MaterializeReplica(disk_path, target_device_type=CPU, wait_for_completion=true, pid=getpid(), replica_uuid=...)`.
          - Assert `resp.status == ALLOCATED` (handle returned before ready).
          - Assert `resp.mem_handle` selects `cpu_memfd` and includes non-empty `lease_token`.
          - Assert `cuda_ipc_handle` is not set for CPU responses.
        - Call `ConfirmReplica(replica_uuid, target_device_type=CPU)` and assert success.
      - LocalHandle FD exchange:
        - Connect to `local_handle_socket_path` and request `GetCpuMemfdFd(lease_token)` (UDS + `SCM_RIGHTS`).
        - `recvmsg(..., MSG_CMSG_CLOEXEC)` and assert the received FD has `FD_CLOEXEC`.
        - Assert memfd seals via `fcntl(F_GET_SEALS)` include `F_SEAL_GROW|F_SEAL_SHRINK`.
      - Data correctness:
        - `mmap` the FD with `MAP_PRIVATE` and compare bytes against the source file content.
        - Optional hardening check: mutate a byte in the private mapping, unmap, re-map, and assert the underlying
          bytes are unchanged (validates `MAP_PRIVATE` copy-on-write behavior).
      - Lease release + cleanup:
        - Call `ReleaseHandle(lease_token)` over LocalHandle and assert success.
        - Call `ReleaseHandle(lease_token)` again and assert idempotent behavior (OK or NOT_FOUND; pick one and codify).
        - After release, `GetCpuMemfdFd(lease_token)` must fail (NOT_FOUND/FAILED_PRECONDITION).
    - **Stable budget gating**: `daemon/service/grpc_service_impl_cpu_memfd_stable_budget_test.cc`
      - Configure `stable_bytes` smaller than the requested export coverage and assert `MaterializeReplica(...CPU...)`
        fails with `RESOURCE_EXHAUSTED` (and returns no usable `cpu_memfd` handle).
      - This test is the acceptance check that CPU exports are admission-controlled by `engine.memory_tiers.stable_bytes`.
    - **TTL/crash safety**: `daemon/state/local_handle_lease_ttl_expiry_test.cc`
      - Configure a short handle-lease TTL via `lifecycle.handle_leases` and assert that after expiry:
        - `GetCpuMemfdFd(lease_token)` fails (NOT_FOUND/DEADLINE_EXCEEDED).
        - A fresh materialization issues a new token that succeeds (ensures old exports don’t linger forever).
    - **Basic LocalHandle correctness**: `daemon/state/local_handle_unknown_token_test.cc`
      - Unknown/random `lease_token` must be rejected (NOT_FOUND) and must not crash the daemon.
  - [x] Milestone 2: Add Python tests for Artifact.tensor_dict(device="cpu").
  - [x] Milestone 3: Update docs/architecture/api/materialization-flow.md, tensorcast/api/README.md,
    daemon/README.md, core/store/README.md to reflect CPU tensor_dict support.

- [x] Phase 6: Production hardening (security, semantics, operability)
  - [x] Milestone 1: Enforce local-only issuance for lease-bearing handles.
    - CPU memfd and CUDA IPC are host-local; reject minting `lease_token` for non-local gRPC peers (or require mTLS +
      authenticated client identity), so remote callers cannot pin stable bytes they cannot release.
    - Add token mint rate limiting (`lifecycle.handle_leases.max_mints_per_second`) as a best-effort guardrail.
    - Align this policy with `examples/config/store_daemon_config.yaml` and deployment docs.
  - [x] Milestone 2: Decide and document handle-lease TTL semantics for long-lived tensors.
    - Default is **no TTL** (ttl unset/0s): local same-UID clients rely on explicit `ReleaseHandle` + PID-exit cleanup.
    - If enabled (ttl > 0), TTL is treated as crash/bug backstop and may retire leaked leases; avoid small TTLs for
      long-lived tensor use cases (future: add `TouchLease` renewal if needed).
  - [x] Milestone 3: Add FD-pressure controls.
    - Add proactive guardrails (warn/fail) when approaching `RLIMIT_NOFILE`.
    - Track/export metrics for active leases + CPU exports (`tc_handle_leases_active_gauge`, `tc_handle_cpu_exports_active_gauge`).
  - [x] Milestone 4: Stress/soak validation.
    - Multi-client fan-out: multiple lease tokens per replica remain valid until the last release (`daemon/service/grpc_service_impl_cpu_memfd_e2e_test.cc`).
    - Long-lived tensors + GC-delayed destruction: lease release is bound to tensor lifetime (`tests/python/test_cpu_memfd_lease_raii.py`).
    - Stable budget exhaustion and recovery when leases are released/expired (`daemon/service/grpc_service_impl_cpu_memfd_stable_budget_test.cc`).

# Tasks

- [x] Update `proto/tensorcast/config/v1/daemon_config.proto` (`lifecycle.handle_leases` + `engine.cpu_shared_memory`) and rebuild config stubs.
- [x] Update `proto/tensorcast/daemon/v2/store_daemon.proto`:
  - Add `lease_token` to `MemCopyHandle`
  - Add `GetServerConfigResponse.local_handle_socket_path` + `GetServerConfigResponse.cpu_shared_memory_enabled`
  - Add `CpuMemfdHandle`
  - Add `MaterializeByKeyRequest.target_device_type`
  - Run `bash tools/build_proto_python.sh`.
- [x] Update daemon `GetServerConfig` plumbing end-to-end:
  - Populate new fields in `daemon/service/controllers/status_controller.h`.
  - Extend the typed Python model `tensorcast/types.py:ServerConfig` and mapping in `tensorcast/daemon_ctl.py:get_server_config`.
  - Ensure `tensorcast/api/store/runtime.py` caches the extended `ServerConfig` so materialization can reuse it.
- [x] Implement LocalHandle service (UDS):
  - Add a dedicated daemon module (e.g., `daemon/state/local_handle_server.{h,cc}`) with a narrow interface; avoid mixing UDS
    framing/parsing into `grpc_service_impl.cc`.
  - `GetCpuMemfdFd(lease_token)` via `SCM_RIGHTS`
  - `ReleaseHandle(lease_token)` to drop daemon refs/leases
  - Integrate peer credential checks via `SO_PEERCRED` (UID required; PID best-effort) and enforce uid==daemon user.
  - Ensure all accepted/received FDs are `CLOEXEC` (`accept4(..., SOCK_CLOEXEC)` and `recvmsg(..., MSG_CMSG_CLOEXEC)` where available).
  - Handle stale socket files safely (unlink only when path is a socket owned by daemon user).
  - Start/stop the LocalHandle server with daemon lifecycle (`daemon/service/grpc_service_impl.cc`) and ensure shutdown is
    idempotent.
- [x] Implement handle leases on top of `daemon/state/session_lifecycle.h`:
  - Add a focused token registry module (e.g., `daemon/state/handle_lease_registry.{h,cc}`) modeled after
    `daemon/state/ipc_region_registry.{h,cc}` for minting/unminting tokens and tracking refcounts/TTLs.
  - Key cleanup by full `store::loading::ReplicaKey` (includes view_id + device.uuid), not `{artifact_id, device_id}`.
  - Fix existing lifecycle finalizers that reconstruct `ReplicaKey` without uuid so they cannot diverge from `RefTracker`.
    - Concretely: `create_use_lease` / `create_ttl_use_lease` should capture the exact `ReplicaKey` used in
      `RefTracker::add_ref` (from `register_session_and_refs`) for `drop_ref` / unload decisions.
  - Extend lifecycle support to CPU and attach additional finalizers for CPU export cleanup.
  - Maintain an unguessable `lease_token` (bytes) mapped to an internal `LeaseId` for `release_lease(...)`.
- [x] Implement shareable CPU arena backing in `core/store/replica/unified_memory_authority.cc` (memfd-only) and remove
  anonymous remap fallback for shareable mappings.
- [x] Add CPU export handle extraction to `core/store/materialization/runtime/pipeline/handle_stage.cc`.
- [x] Update `daemon/service/controllers/materialization_controller.cc` to:
  - Accept CPU targets for replica and by-key paths.
  - For `MaterializeByKeyRequest`, ignore `device_id` when `target_device_type=CPU` (no GPU ordinal validation).
  - Skip the GPU-only LIP fast path when target_device_type is CPU.
  - Populate `MemCopyHandle.cpu_memfd`.
  - Populate `MemCopyHandle.lease_token`.
  - Pin/unpin exports via UMA stable leases and keep per-replica export records refcounted across leases.
- [x] Add `restore_tensors_from_cpu_fd_with_lease` to `core/checkpoint/checkpoint.cc` and bind in
  `tensorcast/csrc/checkpoint_py.cc`.
- [x] In the C++ extension lease owner, apply `MADV_DONTFORK` on client mappings (or document “not fork-safe”) to avoid
  forked children retaining stale mappings after the parent releases `lease_token`.
- [x] Update `tensorcast/api/_materialize.py` to support handle-type branching and lease-aware restores:
  - Stop assuming `mem_handle.cuda_ipc_handle` is always present when `wait_for_completion=true` (it will be empty for
    CPU memfd).
  - Branch on `MemCopyHandle` oneof:
    - GPU: import CUDA IPC + call lease-aware restore so the C++ owner releases `lease_token` on last-tensor destruction.
    - CPU: exchange `lease_token` for an FD via LocalHandle (`SCM_RIGHTS`), then call
      `restore_tensors_from_cpu_fd_with_lease`.
  - Source `local_handle_socket_path` from the cached `ServerConfig` in `StoreRuntimeContext` (no per-call GetServerConfig).
- [x] Relax CPU device checks in `tensorcast/api/_device.py` and `tensorcast/api/store/materialization.py`.
  - Keep existing guards for CUDA-only APIs (region-backed get_into, registration, etc.); only allow CPU for the
    new CPU materialization path (`Artifact.tensor_dict(device=\"cpu\")` and related).
- [x] Update `tensorcast/api/store/artifact.py` / `tensorcast/api/store/materialization.py` to remove eager unload for
  `tensor_dict()` and rely on lease-token release instead.
  - Call sites today: `Artifact.tensor_dict(...): finally: pipeline._release_materialized(...)` and
    `MaterializationPipeline.get_async(): finally: _release_materialized(...)`.
- [x] Update `examples/config/store_daemon_config.yaml` to document `engine.cpu_shared_memory` and the local handle
  socket path under `lifecycle.handle_leases`.
- [x] Add C++ Bazel tests for RFC-0049 acceptance:
  - `//daemon:grpc_service_impl_cpu_memfd_e2e_test` (end-to-end disk->CPU memfd->LocalHandle->mmap->ReleaseHandle).
  - `//daemon:grpc_service_impl_cpu_memfd_stable_budget_test` (stable_bytes admission control).
  - `//daemon:local_handle_lease_ttl_expiry_test` (TTL cleanup / crash-safety).
  - `//daemon:local_handle_unknown_token_test` (unknown token rejection).
  - Prefer `--test_env=TENSORCAST_CUDA_BACKEND=fake` so tests are GPU-independent.

# Test, Rollout, Backout

- Tests
  - C++:
    - `bazel test //core/common:daemon_config_io_test`
    - `bazel test //core/store/replica:unified_memory_authority_test`
    - `bazel test //daemon:session_lifecycle_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - `bazel test //daemon:grpc_peer_utils_test`
    - `bazel test //daemon:grpc_service_impl_cpu_memfd_e2e_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - `bazel test //daemon:grpc_service_impl_cpu_memfd_stable_budget_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - `bazel test //daemon:grpc_service_impl_cpu_memfd_fd_pressure_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - `bazel test //daemon:local_handle_lease_ttl_expiry_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
    - `bazel test //daemon:local_handle_unknown_token_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - Python:
    - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python`
- Rollout
  - Land daemon plumbing first (proto additive fields + `GetServerConfig` + LocalHandle server), but keep behavior
    unchanged until the SDK/extension is ready to release `lease_token`s.
  - Land lease-token lifecycle first (GPU path) behind additive fields; old clients ignore lease_token and continue
    calling `UnloadReplica` (wire-compatible), but zero-copy lifetime safety requires updated clients that honor
    `lease_token`.
  - Land config + daemon fail-fast checks (no behavior change when cpu_shared_memory is disabled).
  - Land memfd export + SDK CPU path after daemon support so CPU requests are not rejected.
- Backout
  - Disable `engine.cpu_shared_memory` and revert CPU handle population/SDK CPU path; leave GPU behavior untouched.

# Risks and Tracking

- [x] Ensure token/FD cleanup on handle lease release + TTL expiry + PID exit to avoid leaks.
- [x] Validate stable lease behavior under memory pressure to prevent reading reclaimed pages.
- [x] Validate local IPC auth (`SO_PEERCRED`) and ensure clients cannot fetch arbitrary exports.
- [x] Track FD pressure (per-export memfd, per-client FDs) and fail with clear errors before hitting `RLIMIT_NOFILE`.
- [x] Ensure SCM_RIGHTS receives do not leak FDs across exec (`FD_CLOEXEC`), and ensure the local socket path cleanup does not delete non-socket files.
- [x] Ensure `lease_token` issuance is restricted to local callers (or otherwise protected) to avoid remote pin/DoS.
- [x] Rate-limit `lease_token` issuance (`max_mints_per_second`) to reduce impact of buggy/abusive clients.
- [x] Decide TTL semantics for long-lived tensors (absolute TTL vs renewal vs disable).
- [x] Confirm CPU device selection does not regress GPU default behavior.
- [x] Confirm hardened/container deployments allow `memfd_create` (seccomp) and that startup fails with a clear error
  when blocked.
- [x] Avoid LocalHandle hangs in tensor destruction paths (bounded I/O and best-effort release).
- [x] Avoid double-dropping PID refs when `UnloadReplica` races with lease finalizers.

# Status (2026-01-15)

- RFC-0049 implemented end-to-end (daemon LocalHandle + handle leases + memfd-backed UMA + SDK restore).
- Unified runtime config handling per `docs/designs/0004-unified-runtime-config.md`: daemon starts from `--config` only (no inline config flags); tests write YAML/JSON config files.
- Hardening/semantics:
  - Lease tokens are minted only for loopback gRPC peers; CPU shared-memory materialization is rejected for non-loopback callers, and non-loopback `pid` values are ignored for refs/leases.
  - `lifecycle.handle_leases.max_mints_per_second` provides a best-effort global mint rate limit (0 => unlimited).
  - `lifecycle.handle_leases.ttl: 0s` disables TTL expiry (leases rely on explicit release + PID exit); TTL is disabled when unset, and enabled only when `ttl > 0`.
  - FD pressure guardrails: CPU export lease creation fails fast with `RESOURCE_EXHAUSTED` when `RLIMIT_NOFILE` headroom is low.
  - Metrics: `tc_handle_leases_active_gauge`, `tc_handle_cpu_exports_active_gauge`.
  - Stable tier policy: CPU export stable leases are admitted by `stable_bytes` and may preempt stable cache entries (export > cache).
- Verified: `bash tools/build_proto_python.sh`, `bazel test //daemon/... --test_env=TENSORCAST_CUDA_BACKEND=fake`, `bazel test //core/... --test_env=TENSORCAST_CUDA_BACKEND=fake --test_tag_filters=-rdma`, `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python`, `BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext`.
