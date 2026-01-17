---
slug: daemon-structure-refactor
title: Daemon Structure Refactor (Plan)
links:
  design: ../designs/0050-daemon-structure-refactor.md
areas:
  - daemon
related_code:
  - daemon/app/server_main.cc
  - daemon/service/grpc_service_impl.{h,cc}
  - daemon/service/controllers/*
  - daemon/ha/worker_lifecycle_manager.{h,cc}
  - daemon/state/session_lifecycle.{h,cc}
  - daemon/state/background_scheduler.h
  - daemon/state/local_handle_server.{h,cc}
---

# Objective

Make `./daemon` structurally easier to evolve long-term by:
- extracting a single composition root (lifetime + wiring),
- keeping gRPC/UDS/HA as thin adapters with stable injected deps,
- enforcing one-way dependencies and reducing header/utility duplication,
while preserving behavior and API compatibility.

# Current State & Grounding

Note: this section reflects the pre-refactor layout for historical context.

- **Construction/ownership is inside the gRPC service**:
  - `StoreDaemonServiceImpl` constructs scheduler/lifecycle/local-handle/persistence/controllers in its ctor
    (`daemon/grpc_service_impl.h`).
- **HA depends on the gRPC service concrete type**:
  - `WorkerLifecycleManager` includes `daemon/grpc_service_impl.h` and stores a `StoreDaemonServiceImpl*`
    (`daemon/worker_lifecycle_manager.h`).
- **Lifecycle is a large implementation-heavy header**:
  - `PidMonitor` + `SessionLifecycleManager` are implemented in `daemon/session_lifecycle.h`.
- **Duplicated cross-cutting utility**:
  - Disk path normalization logic exists in both `daemon/grpc_service_impl.cc` and
    `daemon/service/controllers/materialization_controller.cc`.
- **Existing daemon tests are already structured around the current boundaries**:
  - gRPC service/controller behavior: many `daemon/grpc_service_impl_*_test.cc` targets in `daemon/BUILD`.
  - HA safety + sync: `//daemon:worker_lifecycle_manager_sync_test` exists (`daemon/BUILD`).
  - lifecycle correctness: `//daemon:session_lifecycle_test` and `//daemon:session_lifecycle_pid_unwatch_test`
    exist (`daemon/BUILD`).

# Phases & Milestones

- [x] Phase 1: Consolidate duplicated utilities (no wiring changes)
  - [x] Milestone 1: Add `daemon/util/path_utils.{h,cc}` and switch all callers to it.
  - [x] Milestone 2: Add/adjust unit tests if path validation behavior is covered; otherwise keep refactor-only.

- [x] Phase 2: Extract daemon kernel + composition root (lifetime + start/stop)
  - [x] Milestone 1: Introduce `DaemonKernel` + `DaemonApp` that own long-lived daemon state and expose narrow ports.
  - [x] Milestone 2: Move all side effects (threads/scheduler/UDS listen/HA loops) out of constructors into explicit `start()`.
  - [x] Milestone 3: Refactor `StoreDaemonServiceImpl` into a pure gRPC adapter that only routes to controllers.

- [x] Phase 3: Decouple HA from gRPC service implementation (ports + Bazel)
  - [x] Milestone 1: Introduce `WorkerIdentityStore`, `RetireGates`, `ShutdownSignal`, and bundle them as `WorkerLifecyclePorts`.
  - [x] Milestone 2: Update `WorkerLifecycleManager` to depend only on ports (no `grpc_service_impl.h` include or service pointer).
  - [x] Milestone 3: Update Bazel deps so there is no build-graph path HA → service (enforced by `bazel query somepath(...)`).

- [x] Phase 4: Split lifecycle implementation out of the mega-header
  - [x] Milestone 1: Move `PidMonitor` implementation into `daemon/state/pid_monitor.{h,cc}`.
  - [x] Milestone 2: Move `SessionLifecycleManager` implementation into `daemon/state/session_lifecycle.{h,cc}`.
  - [x] Milestone 3: Keep public API narrow; reduce includes in headers (include-what-you-use).

- [x] Phase 5: Directory layout cleanup + docs
  - [x] Milestone 1: Move files into stable module roots (`app/`, `service/`, `ha/`, `state/`, `util/`), update Bazel.
  - [x] Milestone 2: Update `daemon/README.md` “Directory Layout” section to match the new layout.

# Tasks

## General refactor rules

- Keep behavior identical:
  - No changes to RPC semantics, error codes, deadlines, or policy logic; only wiring/moves/duplication removal.
- Keep diffs reviewable:
  - Prefer “move-then-clean” over mixing file moves with logic edits.
- Keep Bazel deps explicit:
  - Fix missing headers by adding precise `deps` (no reliance on transitive includes).

## Phase-specific tasks

- Phase 1:
  - Replace both callers of `normalize_disk_path(...)` with `daemon/util/path_utils`.
  - Add a small unit test if there is an existing narrow test harness; otherwise rely on existing integration tests that
    cover disk path behavior (e.g., `//daemon:resolve_artifact_from_disk_test`).
- Phase 2:
  - Create `daemon/app/daemon_app.{h,cc}` (or similar) that builds the daemon kernel and adapters.
  - Introduce daemon-internal “shared facts” as stable state modules owned by the kernel:
    - `WorkerIdentityStore` (worker_id/node_id/is_registered),
    - `ShutdownSignal` (begin_shutdown + is_shutting_down),
    - `RetireGates` (ref/use/pin/lock queries for retire gating).
  - Move background task start/stop out of `StoreDaemonServiceImpl`:
    - scheduler start/stop,
    - PID monitor start/stop,
    - local handle server start/stop.
  - Ensure `StoreDaemonServiceImpl` is a pure gRPC adapter:
    - constructor is side-effect free,
    - accepts already-built controllers and stable state/ports,
    - does not own or `new` long-lived subsystems.
  - Update unit/integration tests that currently construct `StoreDaemonServiceImpl` directly:
    - tests must not rely on constructor side effects (background threads, UDS listen),
    - prefer introducing a small shared test harness (e.g., `daemon/testing/*`) to build a ready-to-call gRPC service
      adapter from `DaemonKernel` without repeating wiring in every `grpc_service_impl_*_test.cc`.
- Phase 3:
  - Introduce `daemon/ha/worker_lifecycle_ports.h` (or similar) that exposes only what HA needs:
    - identity updates via `WorkerIdentityStore`,
    - retire gating reads via `RetireGates`,
    - shutdown reads via `ShutdownSignal`,
    - async execution via `common::AsyncRuntime` (non-owning).
  - Update `WorkerLifecycleManager` to:
    - drop `StoreDaemonServiceImpl*` from its public API,
    - stop calling `service_->set_global_store_client(...)` / `service_->set_worker_registered(...)`,
    - read shutdown via `ShutdownSignal` and publish identity via `WorkerIdentityStore`.
  - Wire shared dependencies (e.g., Global Store client) in the composition root rather than “through the service”.
- Phase 4:
  - Split `daemon/session_lifecycle.h`:
    - isolate OS-facing pidfd/epoll details behind `PidMonitor` in a `.cc`,
    - keep `SessionLifecycleManager` API stable (or introduce a narrow compatibility header temporarily during Phase 5).
- Phase 5:
  - Move files into module roots and update `daemon/BUILD` or create subpackage `BUILD` files.
  - Move `*_test.cc` files alongside their owning module (service/ha/state/util/app) and update Bazel targets accordingly.
    Prefer `bazel test //daemon/...` in docs/CI to avoid label churn during package splits.
  - Update `daemon/README.md` and ensure the new directory layout description matches the code.

# Test / Rollout / Backout

## Tests: Relocation & Refactor Strategy

The daemon has extensive colocated C++ tests under `daemon/*_test.cc`. This refactor changes boundaries and ownership, so
tests need two kinds of updates:

- **Refactor**: update test fixtures to construct the new composition root / ports instead of reaching through
  `StoreDaemonServiceImpl`.
- **Relocation** (Phase 5): move tests into the new Bazel subpackages so tests remain colocated with the code they cover.

Principles:
- Keep diffs reviewable: prefer “move test with its module” over “move all tests at once”.
- Prefer stable invocation: use `bazel test //daemon/...` instead of hard-coding labels that will change during package
  splits.
- Enforce boundaries via tests too: HA tests must not include service headers; use ports or fakes.

Phase-by-phase guidance:
- Phase 2: introduce a small shared test harness to build a `StoreDaemonServiceImpl` adapter (and/or `DaemonKernel`)
  without constructor side effects; migrate the `grpc_service_impl_*_test.cc` suite to it incrementally.
- Phase 3: refactor `worker_lifecycle_manager_sync_test.cc` to construct `WorkerLifecycleManager` via
  `WorkerLifecyclePorts` (identity, retire gates, shutdown, async runtime) and remove any dependency on
  `daemon/grpc_service_impl.h`.
- Phase 4: update lifecycle tests to include `daemon/state/{pid_monitor,session_lifecycle}.h` (post-split) and keep
  OS-level pid monitoring as its own integration test where needed.

Proposed Phase 5 test file mapping (steady-state):
- `daemon/app/`
  - `daemon_shutdown_drain_test.cc`
- `daemon/service/`
  - `grpc_service_impl_*_test.cc`
  - `materialize_into_target_validation_test.cc`
  - `resolve_artifact_from_disk_test.cc`
- `daemon/ha/`
  - `worker_lifecycle_manager_sync_test.cc`
- `daemon/state/`
  - `ipc_region_registry_test.cc`
  - `lip_metadata_utils_test.cc`
  - `local_handle_lease_ttl_expiry_test.cc`
  - `local_handle_unknown_token_test.cc`
  - `persistence_manager_test.cc`
  - `pid_monitor_unwatch_integration_test.cc`
  - `session_lifecycle_*_test.cc`
  - `store_policy_resolver_test.cc`
  - `transport_lock_manager_test.cc`
  - `eviction_task_behavior_test.cc`
- `daemon/util/`
  - `grpc_peer_utils_test.cc`
  - `status_utils_test.cc`

## Test (required executable checks)

Run these after each phase:

- Build:
  - `bazel build //daemon:tensorcast_daemon`
- Daemon unit/integration tests:
  - `bazel test //daemon/... --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`

Validation matrix (recommended fast checks per phase, in addition to `bazel test //daemon/...`):
- Phase 1 (path utils): `bazel test //daemon:resolve_artifact_from_disk_test //daemon:grpc_service_impl_disk_index_test --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Phase 2 (composition root): `bazel test //daemon:grpc_service_impl_parity_test //daemon:grpc_service_impl_registration_test --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Phase 3 (HA decoupling): `bazel test //daemon:worker_lifecycle_manager_sync_test --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Phase 4 (lifecycle split): `bazel test //daemon:session_lifecycle_test //daemon:session_lifecycle_pid_unwatch_test --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`

Note: after Phase 5 package splits, these specific labels will move (e.g., `//daemon/service:*_test`,
`//daemon/state:*_test`). Prefer `bazel test //daemon/...` during/after Phase 5 to avoid label churn.

## Structural validation (layering / invariants)

These checks validate that the *refactor goals* are actually achieved (not just “tests pass”):

- HA no longer depends on the gRPC service implementation:
  - `rg -n '#include \"daemon/service/grpc_service_impl.h\"' daemon/ha/worker_lifecycle_manager.{h,cc} && exit 1 || true`
  - `rg -n 'StoreDaemonServiceImpl' daemon/ha/worker_lifecycle_manager.{h,cc} && exit 1 || true`
  - `bazel query 'somepath(//daemon:worker_lifecycle, //daemon:grpc_service_impl)'` (should return empty)
- Include-level enforcement is enabled for HA targets (preferred hardening):
  - Enable Bazel `layering_check` (e.g., `features = [\"layering_check\"]`) on `daemon/ha` libraries once subpackages exist.
- Service does not depend on HA either (avoid accidental cycles during refactor):
  - `bazel query 'somepath(//daemon:grpc_service_impl, //daemon:worker_lifecycle)'` (should return empty)
- `StoreDaemonServiceImpl` is no longer a composition root:
  - `rg -n 'make_unique<BackgroundScheduler>|make_shared<SessionLifecycleManager>|make_unique<LocalHandleServer>|make_unique<PersistenceManager>' daemon/service/grpc_service_impl.{h,cc} && exit 1 || true`
  - `rg -n 'start_sweepers\\(|pid_monitor_->start\\(|local_handle_server_->start\\(' daemon/service/grpc_service_impl.{h,cc} && exit 1 || true`
- Directory moves did not introduce forbidden alias headers as permanent API:
  - `rg -n \"^#include \\\"daemon/(state/session_lifecycle|ha/worker_lifecycle_manager)\\.h\\\"\" daemon | cat` (review include surface after Phase 5)

## Rollout

This is a refactor-only change; rollout is “ship as usual” gated by passing the test matrix above. Avoid landing
multiple phases in one PR unless diff size stays reviewable.

## Backout

- Backout is a straight revert of the refactor commits (no data migrations).
- If a directory move causes downstream include breakage, temporarily revert only the move commit and re-land with
  compatibility shims as a short-lived follow-up (avoid permanent alias headers).

# Risks & Tracking

- **Hidden dependency cycles**: mitigated by introducing module targets and fixing deps explicitly in Bazel.
- **Header include churn**: mitigated by phase 4 being isolated and backed by `bazel test //daemon/...`.
- **Behavior regression risk** (should be low): mitigated by no logic edits and by exercising existing daemon tests.

# Acceptance Criteria (plan-level)

- `StoreDaemonServiceImpl` is constructible without starting background threads/work; background work starts only from the
  composition root.
- `WorkerLifecycleManager` does not include `daemon/service/grpc_service_impl.h` and does not require the concrete gRPC service
  type in its public API.
- There is no Bazel build-graph path HA → service (enforced by query/visibility).
- `daemon/state/session_lifecycle.h` no longer contains OS-level implementation details (pidfd/epoll loops) in the header.
- All tests in the validation matrix pass under `TENSORCAST_CUDA_BACKEND=fake`.

# Open Questions

- Bazel enforcement: introduce `daemon/ha` and `daemon/service` subpackages + `visibility` in Phase 3 to make HA decoupling
  durable; defer broader directory moves and package splits to Phase 5.
- CI checks: codify “no HA↔service deps” with both `bazel query somepath(...)` (build-graph) and either `layering_check` or
  a small include-check script (header-level) so accidental reach-through fails fast.

# Appendix: Proposed file mapping (Phase 5)

This mapping is intended to keep responsibilities obvious and support one-way dependencies.

- `daemon/app/`
  - `server_main.cc`
  - `daemon_app.{h,cc}` (new)
- `daemon/service/`
  - `grpc_service_impl.{h,cc}`
  - `service/controllers/*` (already present)
- `daemon/ha/`
  - `worker_lifecycle_manager.{h,cc}`
  - `worker_lifecycle_ports.h` (new)
- `daemon/state/`
  - `background_scheduler.h`
  - `session_lifecycle.{h,cc}` (new split out of `daemon/session_lifecycle.h`)
  - `pid_monitor.{h,cc}` (new split out of `daemon/session_lifecycle.h`)
  - `shutdown_signal.h` (new)
  - `worker_identity_store.h` (new)
  - `retire_gates.h` (new)
  - `sweep_tasks.h`
  - `replica_session_manager.h`
  - `sessions_service.h`
  - `ref_tracker.h`
  - `transport_lock_manager.h`
  - `ipc_region_registry.{h,cc}`
  - `handle_lease_registry.{h,cc}`
  - `local_handle_server.{h,cc}`
  - `persistence_manager.{h,cc}`
  - `registration_manager.h`
  - `store_policy_resolver.{h,cc}`
  - `lip_manager.{h,cc}`
  - `lip_bridge.{h,cc}`
  - `lip_metadata_utils.{h,cc}`
  - `verification_tracker.h`
- `daemon/util/`
  - `path_utils.{h,cc}` (new)
  - `deadline_utils.h`
  - `device_resolver.h`
  - `grpc_peer_utils.{h,cc}`
  - `rpc_context.h`
  - `status_utils.h`
  - `types.h`
- `daemon/observability/`
  - `grpc_metrics.h`
  - `grpc_span.h`
  - `otel_metrics.{h,cc}` (new)
