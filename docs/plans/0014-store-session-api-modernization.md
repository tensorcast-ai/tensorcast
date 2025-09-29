---
slug: 0014-store-session-api-modernization
title: Store Session API Modernization Plan
links:
  design: ../designs/0014-store-session-api-modernization.md
---

# Objective

Deliver the Store-centric artifact session API described in Design 0014: ship a reusable `Store` session object with synchronous/async verbs (`register`, `put`, `get`, `get_into`), centralize lease/keepalive policy, and preserve compatibility by shimming the legacy module-level helpers until consumers migrate.

# Draft Execution Insights

- Existing helpers in `tensorcast/api/_register.py` and `_loader.py` already implement most daemon/Global Store flows but assume a singleton runtime context (`startup.require_initialized()`). We can refactor these flows into composable primitives consumed by the new `Store` without rewriting RPC wiring.
- Asynchronous paths (`get_artifact_async`, `load_dict_async`) currently return bespoke `LoadHandle` objects. Converging on a shared `ArtifactFuture` backed by `concurrent.futures.Future` simplifies cancellation semantics but requires a lightweight executor scoped per `Store`.
- Lease keepalive threads in `RegisteredArtifact` / `RegisteredLease` rely on ad hoc threading; encapsulating them inside the Store with lifecycle hooks avoids per-call duplication and lets us guarantee clean cancellation on future completion.
- Global Store lookups (canonical index fetch) currently deserialize JSON; we can reuse that logic but cache results inside the Store using a TTL keyed by artifact id to minimize repeated RPCs.

# Clarifications & Additions (Review)

## ArtifactFuture semantics: confirm callback, readiness gating, concurrency

- Result shape and confirm: `ArtifactFuture.result()` returns the final payload (`dict[str, torch.Tensor]`) and internally performs the daemon-side confirmation to ensure materialization completed (mirrors legacy `confirm_fn` behavior). The future also exposes an idempotent `confirm()` that callers may invoke explicitly to block until ready. `confirm()` is safe to call multiple times and returns immediately if already confirmed.
- Readiness gating: Tensors are not exposed until confirmation completes; legacy semantics where `state_dict` access is blocked until `wait()`/`result()` are preserved. For eager notification without blocking on full materialization, `ArtifactFuture.add_done_callback(...)` is available.
- Thread safety: `ArtifactFuture.done()`, `ArtifactFuture.result(timeout: float | None)`, `ArtifactFuture.cancel()` and `ArtifactFuture.confirm()` are thread-safe and may be called concurrently from multiple threads. `result()` returns the same payload instance to all callers. `cancel()` is best-effort; it may race with completion and will return `False` if cancellation could not be effected.

## Canonical Index v2: storage_offset support and compatibility

- Python’s canonical-index representation is v2-only: `[offset, size, shape, stride, dtype, storage_offset]`. v1 (5-field) parsing is removed from the SDK.
- Global Store API already returns canonical index bytes plus `schema_version`; we will verify end-to-end that v2 indices are persisted and returned with `schema_version = "v2"`. No daemon RPC field additions are required because the canonical index travels as an opaque payload.
- Codegen pipeline: run `bash tools/build_proto_python.sh` after proto changes (if any); ensure Buf format/lint targets pass. Update any import sites to the regenerated Python modules under `tensorcast.proto.*`.
- Python parser and loaders now require 6 fields per entry and raise `IndexParseError` otherwise.

## Store fork-safety and thread-safety policy

- Fork-safety: We will register `os.register_at_fork` hooks in `Store`. In the child process, the Store will mark its gRPC channel pool invalid and lazily rebuild channels on first use, avoiding inherited FDs from the parent. As an immediate mitigation, calling `Store.close()` before `fork()` is supported and documented. Current `DaemonCtl` does not install at-fork hooks; this work is tracked under the Store lifecycle milestone.
- Thread-safety: A single `Store` instance is safe for concurrent use across threads. All public verbs (`register`, `put`, `get`, `get_into`) may be invoked concurrently. Internal state (lease scheduler, channel pool) is guarded appropriately to avoid races. Returned `ArtifactFuture` objects follow the concurrency guarantees above.

## Cancellation semantics and retries

- Zero-fill ordering on cancel: We will ensure that during large transfers (including LIP), when a cancellation occurs, zero-fill for any partially written PAD ranges is scheduled on the same CUDA stream as the copy operations, with stream-ordered events guaranteeing write-after-cancel ordering. This prevents visibility of torn regions and avoids DMA stalls by maintaining stream correctness. For CPU paths, zero-fill will be performed before exposing buffers to readers. Implementation will be integrated into the C++ loader path.
- Retry guidance: Surface `ArtifactError` with a `retryable: bool` flag for caller policy. Example with exponential backoff and jitter:

```python
import random, time

max_retries = 5
base_delay = 0.05

for attempt in range(max_retries):
    try:
        fut = store.get_async(artifact_id, options=opts)
        tensors = fut.result(timeout=opts.timeout_s)
        # consume tensors...
        fut.confirm()
        break
    except ArtifactError as err:
        if not getattr(err, "retryable", False) or attempt == max_retries - 1:
            raise
        delay = base_delay * (2 ** attempt) * (1.0 + random.random() * 0.2)
        time.sleep(delay)
        continue
```

# Phases & Milestones

- [ ] Phase 0: Baseline & Guardrails
  - [ ] Milestone 0.1: Capture current public API surface and doc touch points (`README.md`, `docs/internals/model-loading.md`, examples/) to ensure parity tracking
  - [ ] Milestone 0.2: Add targeted tracing around `tensorcast.api.register_artifact` and `get_artifact_*` to confirm daemon/global store RPC contracts before refactor (temporary instrumentation removed once new Store is live)
- [ ] Phase 1: Session & Future Infrastructure
  - [x] Milestone 1.1: Introduce `tensorcast/api/store.py` with type aliases (`TensorDict`, `ArtifactStatusCode`, `RetryPolicy`, etc.) matching the design
  - [x] Milestone 1.2: Implement `ArtifactFuture` wrapper + single-threaded completion executor (reuse async-copy manager hooks from Design 0005) with cancellation propagation skeletons
  - [x] Milestone 1.3: Build `Store` constructor that resolves `DaemonCtl` connections via `tensorcast.daemon_ctl.get_daemon_client` and performs one-time capability discovery + telemetry registration
  - [x] Milestone 1.4: Port lease/keepalive management from `RegisteredArtifact` / `RegisteredLease` into Store-scoped helpers with deterministic shutdown
- [ ] Phase 2: Registration & Put Verbs
  - [x] Milestone 2.1: Refactor `_register._register_artifact_core` to a pure helper consumed by `Store.register`/`register_async`
  - [x] Milestone 2.2: Implement `Store.put`/`put_async`, consolidating coalesced VRAM flows and returning the new `RegisteredArtifact` dataclass with canonical index materialization
  - [ ] Milestone 2.3: Integrate retry policy evaluation + `ArtifactError` mapping for all registration failure paths
  - [ ] Milestone 2.4: Add metrics counters for registration success/failure/latency tagged by daemon endpoint
- [ ] Phase 3: Retrieval & In-Place Verbs
  - [x] Milestone 3.1: Extract canonical-index fetch + replica selection code from `_loader.get_artifact_*` into reusable Store helpers with fallback policy evaluation
  - [x] Milestone 3.2: Implement `Store.get`/`get_async` returning `dict[str, torch.Tensor]` with automatic lease keepalive and future completion/cancellation semantics
  - [x] Milestone 3.3: Implement `Store.get_into`/`get_into_async`, including strict tensor layout validation and zero-copy CUDA IPC path selection
  - [x] Milestone 3.4: Wire fallback options (disk, P2P) and verification toggles through the Store API; emit telemetry for fallback decisions
- [ ] Phase 4: Legacy API Shims & Observability
  - [ ] Milestone 4.1: Introduce a lazily constructed process-level Store used by `register_artifact`, `get_artifact_sync`, etc., with `DeprecationWarning` emission
  - [ ] Milestone 4.2: Update examples, docs, and quickstarts to promote the new `Store` entry point; provide a migration section in `docs/internals/model-loading.md`
  - [ ] Milestone 4.3: Add user-facing metrics/traces (Prometheus + OTEL) for Store verbs and keepalive lifecycle
  - [ ] Milestone 4.4: Remove temporary tracing from Phase 0 and ensure legacy helpers rely solely on Store shims
- [ ] Phase 5: Validation, Rollout, & Backout Readiness
  - [ ] Milestone 5.1: Extend Python tests to cover sync/async variants, cancellation, and fallback permutations (`tests/python/test_register_*`, new `test_store_session_api.py`)
  - [ ] Milestone 5.2: Run integration suites against fake CUDA and staged daemons (`uv run pytest tests/python/...`, `bazel test //daemon:session_lifecycle_test`)
  - [ ] Milestone 5.3: Document rollout steps (feature flag/env var gating) and backout procedure in `docs/architecture/architecture-overview.md` + ops runbook
  - [ ] Milestone 5.4: Tag release checklist ensuring global store schemas, daemon binaries, and SDK wheels ship together

# Task Breakdown

## SDK / Python

- [x] Create `tensorcast/api/store.py` with type definitions, `StoreOptions`, `FallbackOptions`, `RetryPolicy`, `ArtifactError`, `RegisteredArtifact`
- [x] Implement `Store` lifecycle (channel pool, capability probe, lease scheduler, metrics hooks)
- [x] Refactor `_register.py` helpers into internal functions consumed by `Store.register`/`put`; delete duplicated keepalive threads
- [x] Refactor `_loader.py` to expose reusable replica-materialization helpers invoked by `Store.get`/`get_into`
- [x] Implement `ArtifactFuture` using `concurrent.futures.Future` + Store executor; add cancellation propagation to daemon RPCs (Abort/Revoke)
- [x] Wire cancellation propagation for `Store.get_async`/`get_into_async` using daemon unload hooks
- [ ] Update `tensorcast/api/__init__.py` to export `Store` and route legacy helpers through a cached Store instance (respect `tensorcast.client_runtime.daemon_target_default()`)
- [ ] Introduce env/config knob to opt into immediate Store usage in downstream apps; default legacy helpers to shim path
- [ ] Update module docs (`docs/internals/model-loading.md`, `docs/internals/save_dict_flow.md`, repository `README.md`) to describe Store object workflows and migration timeline
- [ ] Refresh examples under `examples/` to instantiate `Store`

## Daemon / Core / Global Store (verification + compatibility)

- [ ] Confirm existing daemon RPCs satisfy capability probe needs; add `GetCapabilities` RPC if missing and update client stubs (Bazel `bazel test //proto/...` after regen)
- [ ] Validate lease TTL and keepalive expectations against daemon (`//daemon:session_lifecycle_test`) with new Store-managed timers
- [ ] Ensure Global Store canonical index fetch path handles increased caching (adjust TTL hints if needed)
- [ ] Update daemon observability to consume new client-provided metadata (e.g., fallback reasons in headers) if the SDK emits them

## Tooling & Telemetry

- [ ] Add OTEL spans and Prometheus counters for Store verbs (`tensorcast/observability/otel.py`, metrics exporters)
- [ ] Update CLI tooling (`tensorcast/cli_utils`) to print Store session status for debugging
- [ ] Provide Grafana dashboard snippets capturing Store client metrics (attach to ops runbook)

## Testing & QA

- [ ] Add unit tests for `ArtifactFuture` cancellation/timeout behavior
- [ ] Add Store-level tests using fake daemon fixtures covering register/put/get/get_into flows (sync + async)
- [ ] Update existing lease tests (`tests/python/test_register_lease_in_place_helper.py`, `test_register_vram_leased_and_dvmp_stream.py`) to assert new return types while maintaining coverage
- [ ] Expand integration coverage for fallback paths (disk-first, P2P) using `FallbackOptions`
- [ ] Add regression tests ensuring deprecated helpers raise warnings but still function

# Test / Rollout / Backout Strategy

- Unit tests: `uv run pytest tests/python/test_store_session_api.py`, existing registration/materialization suites, new cancellation tests
- Integration: `uv run pytest tests/python/test_register_vram_leased_and_dvmp_stream.py`, `uv run pytest tests/python/test_register_lease_in_place_helper.py`, gRPC contract checks via `bazel test //daemon:session_lifecycle_test --define=use_fake_cuda=true`
- Performance smoke: run current throughput benchmarks with Store shims enabled to validate ≤2% regression (baseline captured in Phase 0)
- Rollout: ship SDK with both Store object and legacy shims; gate warning enforcement behind env var `TENSORCAST_STORE_SESSION_REQUIRED=1` toggled per release; communicate migration plan to downstream teams
- Backout: retain branch with legacy helpers untouched; if regressions surface, disable Store shim via env flag and re-enable in hotfix once addressed; no schema migrations required

# Dependencies & Coordination

- Coordinate with Daemon team on capability probe RPC and keepalive expectations (`daemon/grpc_service_impl.cc` owners)
- Align with Global Store maintainers on canonical-index caching hints and telemetry
- Partner with Observability team to integrate new metrics into dashboards before GA
- Communicate migration timeline to SDK consumers (PyTorch integration, model-serving frameworks) via release notes and internal channels

# Risks & Mitigations

- **Risk:** Lease keepalive regressions if Store executor backlog grows.
  - *Mitigation:* Dedicated keepalive scheduler thread with bounded queue; integration tests measuring lease TTL margins.
- **Risk:** Increased latency due to added abstraction layers.
  - *Mitigation:* Benchmark before/after across register/get flows; profile and optimize hot paths before GA.
- **Risk:** Legacy helper consumers depend on implicit global startup state.
  - *Mitigation:* Maintain compatibility shim, document migration, add runtime warning pointing to `Store` constructor.
- **Risk:** Canonical index caching inconsistency.
  - *Mitigation:* Respect Global Store invalidation hints; add TTL-based cache with explicit eviction on failures.
- **Risk:** Async cancellation surface diverges from design.
  - *Mitigation:* Add thorough tests, align with Design 0005 executor semantics, review with async subsystem owners.

# Acceptance Criteria

- Store API available, documented, and covered by unit/integration tests for all verbs (sync + async)
- Legacy helpers delegate through Store, emit migration warnings, and pass existing test suites unchanged
- Observability (metrics + traces) exposes success/failure/latency metrics per verb and lease lifecycle
- Documentation updated across README, internals, and design cross-links; examples demonstrate new workflow
- Release notes drafted with rollout/backout instructions and communicated to downstream teams
