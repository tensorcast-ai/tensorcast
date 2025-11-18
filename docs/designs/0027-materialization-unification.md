---
slug: 0027-materialization-unification
title: Materialization Control/Data-plane Unification
areas:
  - core
related_code:
  - core/store/materialization
  - core/store/docs
  - core/store/README.md
---

# Summary

`core/store/materialization` now owns both orchestration (`MaterializationService`, `MaterializeOrchestrator`, replica registration helpers, UMA-aware planning) and the streaming dataplane. Prior to this work the two halves lived under `core/store/loading` and `core/store/loader`, forcing StoreEngine to include dataplane headers directly. This design tracks the migration into a dedicated module with explicit contracts, registry interfaces, and dependency guardrails so control components consume factories instead of instantiating loaders directly.

The historical sections below retain the original directory references for traceability, but the implementation has fully moved into `core/store/materialization/**` and the legacy trees have been deleted.

## Latest Landing (2025-02)

- `MaterializationRequest` centralizes validation for canonical identifiers, variant hints, and device targeting so callers get structured errors instead of repeating checks (`core/store/loading/materialization/materialization_request.{h,cc}`, tests in `//core/store/materialization/contracts:materialization_request_test`).
- `MaterializationService` now owns replica reuse, COPY_ONLY, LOAD_ONLY, and AUTO orchestration behind a `MaterializationDeps` bundle (replica registry, pinned pool, disk ingest callback, optional `run_auto` lambda, and optional `compute_view_hash`). Unit tests exercise each branch via injected fakes (`core/store/loading/materialization/materialization_service.{h,cc}`, `core/store/loading/materialization/materialization_service_test.cc`).
- `MaterializationBackend` abstracts the ingestion surface that control helpers need (`core/store/loading/materialization/materialization_backend.h:14-36`), and `MaterializeOrchestrator` consumes it instead of calling the StoreEngine directly, keeping gRPC/global-store logic isolated (`core/store/loading/materialization/materialize_orchestrator.cc:18-140`).
- `StoreEngine::materialize_replica()` now builds the deps bundle once per call and hands it to `MaterializationService`, while the `run_auto` callback spins up `MaterializeOrchestrator` only when the Global Store client is connected (`core/store/store_engine.cc:1150-1280`). Documentation in `core/store/README.md:103-190` and `docs/architecture/p2p-transfer-strategies.md` was updated to describe the new service boundary.

## Latest Landing (2025-11)

- Deleted the remaining `core/store/loading/**` shim headers and removed the `//core/store/loader` Bazel alias package. Every consumer now depends directly on `//core/store/materialization/{contracts,control,planning,dataplane}:*` with no forwarding layers.
- Updated `daemon`, `core`, `checkpoint`, `examples`, and `core/store/replica` BUILD targets to point at the new dataplane packages so rebuilds no longer drag the entire loader tree when only control code changes.
- Refreshed module documentation (`core/store/materialization/README.md`, `core/store/materialization/dataplane/README.md`, `docs/architecture/p2p-transfer-strategies.md`) plus plan/design 0027 to describe the final structure and remove references to the legacy directories.
- Added validation guidance so Bazel tests (`bazel test --define=use_fake_cuda=true //core/store/materialization/control:materialization_service_test`, `bazel test //core/store:store_engine_test`, `bazel test //daemon:grpc_service_impl_registration_test --define=use_fake_cuda=true`) and targeted Python smoke (`uv run pytest tests/python/test_store_session_api.py -k materialize`) gate future changes.

# Goals / Non-Goals

## Goals

- Introduce a shared contracts layer that contains replica specs, view descriptors, and loader registry interfaces.
- Split orchestration (control + planning) from the dataplane so Bazel boundaries match runtime responsibilities.
- Provide factories (`IArtifactLoaderRegistry`, `ArtifactSourceRouter`) so `MaterializationService` and peers never include `disk_loader.h`/`p2p_loader.h`.
- Preserve current behavior for AUTO, LOAD_ONLY, and COPY_ONLY materialization flows, including Global Store registration and UMA accounting.
- Document the structure so `core/store/README.md` and `AGENTS.md` describe a coherent story for future contributors and agents.

## Non-Goals

- Changing artifact formats, UMA semantics, or the StoreEngine public API.
- Rewriting loader algorithms or verification logic beyond moving files behind new targets.
- Modifying Global Store RPC contracts or schema (`schema.sql` remains untouched).
- Adding new transports or chunk-repair policies beyond wiring existing code to the new interfaces.

# Architecture & Interfaces

## Current Coupling

> Historical context: this section captures the pre-unification layout. The final implementation now lives entirely under `core/store/materialization/**`.

- `MaterializationService` now consumes injected callbacks via `MaterializationDeps`, but it still lives under `core/store/loading/materialization/materialization_service.h:24-104` and is instantiated inline for every `StoreEngine::materialize_replica()` invocation (`core/store/store_engine.cc:1150-1280`). Without a shared contracts/control package there is no stable seam for registry/router implementations or for tests that need to exercise the decision tree without spinning up the full engine.
- `MaterializeOrchestrator` (`core/store/loading/materialization/materialize_orchestrator.cc:18-140`) uses the new `MaterializationBackend` interface for ingestion, yet source selection and fallback logic are still hard-coded. There is no `ArtifactSourceRouter` or loader registry, so orchestrator code continues to construct `DiskSource`/`ReplicaTarget` structs directly instead of relying on contracts.
- The chunk-aware strategy (`core/store/loading/chunk_aware_loading_strategy.cc:318-429`) still constructs `DiskLoader` instances just to obtain `SeekableSource` handles, tightly binding UMA planning to disk transport internals.
- `loading_spec.h` re-exports `loader::ViewSpec` and `loader::ViewPlan` (`core/store/loading/loading_spec.h:25-140`), entangling consumer APIs with view planner implementation headers.

These edges make the layering described in `0026-loader-subsystem-refactor` unenforceable, because control code still points into dataplane directories.

## Target Layout

The new root becomes `core/store/materialization/` with the following packages (each with its own `BUILD` and README section):

```mermaid
flowchart TD
    A[StoreEngine API] --> B[materialization/control]
    B --> C[materialization/planning]
    B --> D[Materialization Contracts]
    D --> E[materialization/dataplane/contracts]
    E --> F[materialization/dataplane/runtime]
    F --> G[Sources|Sinks|Loaders|View|Metadata]
    B --> H[ArtifactSourceRouter]
    H --> G
```

### Contracts Layer (`core/store/materialization/contracts/`)

- Moves `loading_spec.h`, `materialization_request.h`, and related types out of `core/store/loading`.
- Pulls `ViewSpec`, `ViewPlan`, and view identifiers into `contracts/view/` so consumer APIs never include dataplane headers.
- Introduces two new interfaces:

```cpp
class IArtifactLoaderRegistry {
 public:
  virtual ~IArtifactLoaderRegistry() = default;
  virtual absl::StatusOr<std::unique_ptr<IArtifactLoader>> CreateLoader(
      const ArtifactSource& source,
      const MaterializeHints& hints) = 0;
};

class ArtifactSourceRouter {
 public:
  virtual ~ArtifactSourceRouter() = default;
  virtual absl::StatusOr<ArtifactSource> SelectSource(
      const MaterializationRequest& request,
      components::IGlobalStoreClient& gs_client) = 0;
};
```

- The contracts target (`//core/store/materialization/contracts:*`) exposes thin, header-only deps consumed by control and dataplane packages.

### Control Layer (`core/store/materialization/control/`)

- Hosts `MaterializationService`, `MaterializeOrchestrator`, `ReplicaRegistrationHelper`, and adapters to the StoreEngine API.
- Depends on:
  - `contracts` for specs/interfaces.
  - `components` (DeviceManager, ReplicaRegistry, GlobalStoreClient).
  - `planning` for chunk repair strategies.
  - The loader registry interface only (no direct includes of `disk_loader.h` or `p2p_loader.h`).
- Execution flow:
  1. Build `MaterializationRequest` from StoreEngine arguments.
  2. Ask `ArtifactSourceRouter` for the preferred `ArtifactSource`.
  3. Request a loader via `IArtifactLoaderRegistry`.
  4. Pass the resulting `SeekableSource` to `ReplicaLoadController::load_async_from_source`.
- Control modules record state transitions, register replicas with Global Store, and handle fallback sequencing (AUTO → P2P → disk) without reaching into dataplane headers.

_Bridge state:_ the current implementation wires these pieces together via `MaterializationDeps` (std::function callbacks for disk ingestion, AUTO orchestration, and optional view hashing) that are created inside `StoreEngine::make_materialization_deps()` (`core/store/store_engine.cc:1150-1180`). `MaterializationBackend` lives alongside the legacy headers (`core/store/loading/materialization/materialization_backend.h:14-36`) and is implemented by `StoreEngine`, so moving the files under `core/store/materialization/control/` preserves today’s behavior while giving us a long-lived injection layer instead of per-call lambdas.

### Planning Layer (`core/store/materialization/planning/`)

- Contains `ChunkAwareLoadingStrategy`, future chunk repair policies, and UMA coordination helpers.
- Exposes `plan_and_execute(...)` APIs that accept an abstract `ChunkTransferAdapter`, decoupling UMA manipulations from transport creation.
- Relies only on `contracts`, UMA headers, and `planning/interfaces.h`.

### Dataplane (`core/store/materialization/dataplane/`)

- Relocates the reorganized loader hierarchy proposed in `0026`:
  - `contracts/` (current loader/source/sink/buffer_pool headers).
  - `runtime/` (pump, streaming adapters).
  - `sources/`, `sinks/`, `loaders/`, `view/`, `metadata/`.
- Adds `registry/` that implements `IArtifactLoaderRegistry` by wiring available loader factories (Disk, P2P, Inline) and exposing them via Bazel target `loader_registry_lib`.
- Provides `source_router_lib`, which implements `ArtifactSourceRouter` using Global Store transport RPCs and disk hints from `MaterializeHints`.
- Control modules depend on `registry/` rather than individual loader targets.

### Bazel Boundaries

- `//core/store/materialization/contracts:*` — header-only, visibility `//visibility:public`.
- `//core/store/materialization/control:*` — depends on contracts, planning, registry targets, and store components.
- `//core/store/materialization/planning:*` — depends on contracts + UMA libs.
- `//core/store/materialization/dataplane/*` — internal targets; only `registry_lib` and `source_router_lib` are exported.
- Legacy `//core/store/loader` aliases were removed once all consumers migrated; dataplane targets are now referenced directly.

## Migration Plan & File Mapping

The following tables enumerate every file under `core/store/loading/` and `core/store/loader/`, the destination package in the new hierarchy, and whether it needs structural changes while moving. This mapping gives implementers an exact checklist and clarifies which bundles can be migrated in parallel.

### Control, Contracts, and Planning

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loading/materialization/materialization_service.{h,cc}` | `core/store/materialization/control/materialization_service.{h,cc}` | Pure move; trim includes to contracts + registry headers. |
| `core/store/loading/materialization/materialization_service_test.cc` | `core/store/materialization/control/materialization_service_test.cc` | Update fixtures to use registry/router fakes. |
| `core/store/loading/materialization/materialize_orchestrator.{h,cc}` | `core/store/materialization/control/materialize_orchestrator.{h,cc}` | Path change only; target now depends on planning. |
| `core/store/loading/replica_registration_helper.{h,cc}` | `core/store/materialization/control/replica_registration_helper.{h,cc}` | Path change only. |
| `core/store/loading/materialization/materialization_request.{h,cc}` | `core/store/materialization/contracts/materialization_request.{h,cc}` | Types move verbatim; unit test follows. |
| `core/store/loading/materialization/materialization_request_test.cc` | `core/store/materialization/contracts/materialization_request_test.cc` | Update Bazel deps only. |
| `core/store/loading/loading_spec.h` | `core/store/materialization/contracts/view/{view_spec.h,view_plan.h,view_id.h}` | File splits into three headers for specs, plan descriptors, and identifiers; no `.cc`. |
| `core/store/loading/chunk_aware_loading_strategy.{h,cc}` | `core/store/materialization/planning/chunk_aware_strategy.{h,cc}` | Logic moves as-is; helper structs pulled into new `planning/chunk_transfer_adapter.h`. |

### Dataplane Contracts

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/loader.h` | `core/store/materialization/dataplane/contracts/loader.h` | Header-only move; namespace updates only. |
| `core/store/loader/source.h` | `core/store/materialization/dataplane/contracts/source.h` | Same move; depends on contracts view headers. |
| `core/store/loader/sink.h` | `core/store/materialization/dataplane/contracts/sink.h` | Same move. |
| `core/store/loader/buffer_pool.h` | `core/store/materialization/dataplane/contracts/buffer_pool.h` | Same move; consumed by runtime. |
| `core/store/loader/inline_buffer_loader.h` | `core/store/materialization/dataplane/contracts/inline_buffer_loader.h` | Keep inline helpers; include new loader contract. |

### Dataplane Runtime & Buffering

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/pump.{h,cc}` | `core/store/materialization/dataplane/runtime/pump.{h,cc}` | Path change only. |
| `core/store/loader/pump_test.cc` | `core/store/materialization/dataplane/runtime/tests/pump_test.cc` | Update includes. |
| `core/store/loader/pump_direct_test.cc` | `core/store/materialization/dataplane/runtime/tests/pump_direct_test.cc` | Update includes. |
| `core/store/loader/pump_async_test.cc` | `core/store/materialization/dataplane/runtime/tests/pump_async_test.cc` | Update includes. |
| `core/store/loader/streaming_buffer_adapter.{h,cc}` | `core/store/materialization/dataplane/runtime/streaming_buffer_adapter.{h,cc}` | Path change only. |
| `core/store/loader/artifact_streaming_buffer_test.cc` | `core/store/materialization/dataplane/runtime/tests/artifact_streaming_buffer_test.cc` | Update includes. |
| `core/store/loader/artifact_gpu_auto_release_test.cc` | `core/store/materialization/dataplane/runtime/tests/artifact_gpu_auto_release_test.cc` | Update includes. |

### Dataplane Sources & Routing

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/file_partition_source.{h,cc}` | `core/store/materialization/dataplane/sources/file_partition_source.{h,cc}` | Path change only. |
| `core/store/loader/file_partition_source_test.cc` | `core/store/materialization/dataplane/sources/tests/file_partition_source_test.cc` | Update includes. |
| `core/store/loader/multi_safetensors_source.{h,cc}` | `core/store/materialization/dataplane/sources/multi_safetensors_source.{h,cc}` | Path change only. |
| `core/store/loader/multi_safetensors_source_test.cc` | `core/store/materialization/dataplane/sources/tests/multi_safetensors_source_test.cc` | Update includes. |
| `core/store/loader/safetensors_source.{h,cc}` | `core/store/materialization/dataplane/sources/safetensors_source.{h,cc}` | Path change only. |
| `core/store/loader/safetensors_source_test.cc` | `core/store/materialization/dataplane/sources/tests/safetensors_source_test.cc` | Update includes. |
| `core/store/loader/segment_plan_source.{h,cc}` | `core/store/materialization/dataplane/sources/segment_plan_source.{h,cc}` | Path change only. |
| `core/store/loader/view_plan_source.{h,cc}` | `core/store/materialization/dataplane/view/view_plan_source.{h,cc}` | Lives with view modules because it consumes `ViewSpec`. |
| `core/store/loader/view_plan_source_test.cc` | `core/store/materialization/dataplane/view/tests/view_plan_source_test.cc` | Update includes. |
| `core/store/loader/remote_key_source.{h,cc}` | `core/store/materialization/dataplane/sources/remote_key_source.{h,cc}` | Path change only. |
| `core/store/loader/mux_seekable_source.{h,cc}` | `core/store/materialization/dataplane/sources/mux_seekable_source.{h,cc}` | Path change only. |
| `core/store/loader/mux_seekable_source_test.cc` | `core/store/materialization/dataplane/sources/tests/mux_seekable_source_test.cc` | Update includes. |
| `core/store/loader/source_hash.{h,cc}` | `core/store/materialization/dataplane/metadata/source_hash.{h,cc}` | Classified under metadata; hooks into registry wiring. |

### Dataplane Sinks & Loaders

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/cpu_va_sink.{h,cc}` | `core/store/materialization/dataplane/sinks/cpu_va_sink.{h,cc}` | Path change only. |
| `core/store/loader/gpu_memory_sink.{h,cc}` | `core/store/materialization/dataplane/sinks/gpu_memory_sink.{h,cc}` | Path change only. |
| `core/store/loader/gpu_memory_sink_test.cc` | `core/store/materialization/dataplane/sinks/tests/gpu_memory_sink_test.cc` | Update includes. |
| `core/store/loader/disk_loader.{h,cc}` | `core/store/materialization/dataplane/loaders/disk_loader.{h,cc}` | Path change only; exported via loader registry. |
| `core/store/loader/disk_cpu_load_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_cpu_load_test.cc` | Update includes. |
| `core/store/loader/disk_cpu_gpu_transfer_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_cpu_gpu_transfer_test.cc` | Update includes. |
| `core/store/loader/disk_loader_streaming_buffer_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_loader_streaming_buffer_test.cc` | Update includes. |
| `core/store/loader/disk_loader_safetensors_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_loader_safetensors_test.cc` | Update includes. |
| `core/store/loader/disk_error_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_error_test.cc` | Update includes. |
| `core/store/loader/disk_mmap_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_mmap_test.cc` | Update includes. |
| `core/store/loader/disk_multi_gpu_test.cc` | `core/store/materialization/dataplane/loaders/tests/disk_multi_gpu_test.cc` | Update includes. |
| `core/store/loader/p2p_loader.{h,cc}` | `core/store/materialization/dataplane/loaders/p2p_loader.{h,cc}` | Path change only. |
| `core/store/loader/p2p_loader_tcp_test.cc` | `core/store/materialization/dataplane/loaders/tests/p2p_loader_tcp_test.cc` | Update includes. |
| `core/store/loader/inline_buffer_loader.h` | `core/store/materialization/dataplane/contracts/inline_buffer_loader.h` | Listed above under contracts; loader registry references it for inline path. |

### Dataplane View Execution

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/view_planner.{h,cc}` | `core/store/materialization/dataplane/view/view_planner.{h,cc}` | Drops embedded `ViewSpec` definitions (now in contracts). |
| `core/store/loader/view_planner_test.cc` | `core/store/materialization/dataplane/view/tests/view_planner_test.cc` | Update includes. |
| `core/store/loader/view_ingest_executor.{h,cc}` | `core/store/materialization/dataplane/view/view_ingest_executor.{h,cc}` | Path change only. |
| `core/store/loader/view_ingest_executor_test.cc` | `core/store/materialization/dataplane/view/tests/view_ingest_executor_test.cc` | Update includes. |
| `core/store/loader/view_transform_executor.{h,cc}` | `core/store/materialization/dataplane/view/view_transform_executor.{h,cc}` | Path change only. |
| `core/store/loader/view_transform_executor_test.cc` | `core/store/materialization/dataplane/view/tests/view_transform_executor_test.cc` | Update includes. |

### Dataplane Metadata & Verification

| Current Path | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| `core/store/loader/canonical_index.{h,cc}` | `core/store/materialization/dataplane/metadata/canonical_index.{h,cc}` | Path change only. |
| `core/store/loader/index_reader.{h,cc}` | `core/store/materialization/dataplane/metadata/index_reader.{h,cc}` | Path change only. |
| `core/store/loader/index_reader_test.cc` | `core/store/materialization/dataplane/metadata/tests/index_reader_test.cc` | Update includes. |
| `core/store/loader/disk_dir_hash.{h,cc}` | `core/store/materialization/dataplane/metadata/disk_dir_hash.{h,cc}` | Path change only. |
| `core/store/loader/safetensors_util.{h,cc}` | `core/store/materialization/dataplane/metadata/safetensors_util.{h,cc}` | Path change only. |
| `core/store/loader/verification_utils.{h,cc}` | `core/store/materialization/dataplane/verification/verification_utils.{h,cc}` | Path change only. |
| `core/store/loader/verification_utils_test.cc` | `core/store/materialization/dataplane/verification/tests/verification_utils_test.cc` | Update includes. |
| `core/store/loader/artifact_verification_test.cc` | `core/store/materialization/dataplane/verification/tests/artifact_verification_test.cc` | Update includes. |
| `core/store/loader/artifact_verifier_direct_test.cc` | `core/store/materialization/dataplane/verification/tests/artifact_verifier_direct_test.cc` | Update includes. |

### Registry & Router Implementations

| Item | Destination Path | Notes / Required Changes |
| --- | --- | --- |
| Loader registry implementation (new) | `core/store/materialization/dataplane/registry/loader_registry.{h,cc}` | Wraps disk/p2p/inline loader factories behind `IArtifactLoaderRegistry`. |
| Loader registry tests (new) | `core/store/materialization/dataplane/registry/loader_registry_test.cc` | Validates AUTO → P2P → disk fallback sequencing. |
| Source router implementation (new) | `core/store/materialization/dataplane/registry/source_router.{h,cc}` | Implements `ArtifactSourceRouter` using Global Store metadata. |
| Source router tests (new) | `core/store/materialization/dataplane/registry/source_router_test.cc` | Covers routing hints and backoffs with fake GS responses. |

Only `loading_spec.h` and `chunk_aware_loading_strategy` require structural changes; every other file is a straight move with include updates. Implementers can therefore size work accurately and split the migration by table section (e.g., “runtime & buffering” vs. “metadata & verification”) without re-triaging the old directories.

## Execution Plan

The execution keeps StoreEngine runnable after every step. Each phase lands independently and includes doc updates (`core/store/README.md`, `core/store/docs/architecture.md`, and `docs/designs/0026-loader-subsystem-refactor.md` cross-links).

### Phase 0 — Contracts Scaffolding

- Create `core/store/materialization/{contracts,planning,control,dataplane}` directories with stub `BUILD` files plus READMEs that outline dependencies.
- Lift `MaterializationRequest` and view descriptors into `contracts/` while leaving the original headers as forwarding wrappers that `#include` the new locations and emit `[[deprecated("...")]]` annotations.
- Introduce `IArtifactLoaderRegistry`/`ArtifactSourceRouter` interfaces plus basic test doubles in `contracts/tests/registry_fakes.h` so control logic can compile against the new contracts before runtime pieces arrive.
- Add Bazel aliases so existing targets keep building: `//core/store/loading:loading_spec` depends on the new contracts target while the rest of the tree stays untouched.

### Phase 1 — Control Layer Migration

- Move `materialization_service`, `materialize_orchestrator`, and supporting helpers to `materialization/control/`.
- Replace direct includes of `disk_loader.h`/`p2p_loader.h` with a temporary adapter that implements `IArtifactLoaderRegistry` by delegating to the existing loader factories (`core/store/loading/loader_adapter.{h,cc}`).
- Update `//core/store/materialization/control:materialization_service_test` (formerly `//core/store:materialization_service_test`) to consume the contracts fakes and to assert that control code only touches the registry interfaces (verified via IWYU or `bazel query`).
- Wire StoreEngine to depend on `//core/store/materialization/control:materialization_service_lib` so the rest of the migration does not require StoreEngine changes.

### Phase 2 — Dataplane Relocation

- Move runtime, buffering, metadata, verification, and view execution files following the mapping tables above; keep BUILD visibility private except for `registry_lib`/`source_router_lib`.
- During the move we temporarily kept forwarding headers in `core/store/loader/` that simply included the new locations and carried TODOs referencing this design; they were removed in the final cleanup once all deps flipped.
- Maintain parity by running the `//core/store/materialization/dataplane:...` suites after every batch of moves so behavior stays identical to the legacy loader tests.
- Update `docs/core/store/README.md` diagrams to refer to the new dataplane namespaces before flipping consumers.

### Phase 3 — Registry and Router Implementations

- Implement the concrete `loader_registry.{h,cc}` that chooses disk vs. P2P vs. inline loaders using the existing heuristics in `MaterializationService`.
- Build `source_router.{h,cc}` with dependency on `components::IGlobalStoreClient` and add integration tests that mock GS responses as well as UMA hints.
- Swap `materialization_service.cc` to request loaders exclusively via the new registry target and delete the temporary adapter added in Phase 1.
- Ensure fallback order (AUTO → P2P → disk) and retry timers live in the control layer while loader instantiation stays encapsulated inside the registry.

### Phase 4 — Cleanup and Hardening

- Remove forwarding headers under `core/store/loading/` and `core/store/loader/` once all Bazel targets point at the new locations (validated via `bazel query 'rdeps(//core/...materialization/...)'`).
- Delete deprecated aliases and tighten visibility so only `//core/store/materialization/control` can access the registry target.
- Refresh `AGENTS.md`, `core/store/docs/architecture.md`, and cross-links from design `0026` with the final state diagram and guardrails.
- Capture regression coverage by promoting the new registry and router tests to run in CI pipelines that already exercise `//core/store:store_engine_test`.

## Documentation & Guidance

- Update `core/store/README.md` to describe the new module layout and clarify how StoreEngine interacts with materialization.
- Extend `core/store/docs/architecture.md` to replace the existing “Data Loading Layer” section with the new control/dataplane description, referencing the contracts layer instead of concrete loaders.
- `core/store/AGENTS.md` (if present) gains a short ruleset documenting the new targets and dependency restrictions.

## Testing & Validation

- **Unit tests:** Bridge coverage moved with the new packages.
  - `bazel test //core/store/materialization/contracts:materialization_request_test`
  - `bazel test --define=use_fake_cuda=true //core/store/materialization/control:materialization_service_test`
  - `bazel test //core/store/materialization/planning:chunk_aware_strategy_test`
  - `bazel test //core/store/materialization/dataplane/...`
- **Integration tests:** Run StoreEngine suites with fake CUDA enabled to verify ABI compatibility and UMA behavior:
  - `bazel test //core/store:store_engine_test --define=use_fake_cuda=true`
  - `bazel test //core/store:store_engine_materialization_test --define=use_fake_cuda=true`
- **Python regressions:** Ensure the user-facing APIs still stream artifacts correctly by exercising the high-level client tests:
  - `uv run pytest tests/python/test_safetensors_loading.py`
  - `uv run pytest tests/python/test_store_view_api.py`
- **Dependency guardrails:** Once the control target exists, use `bazel query --noimplicit_deps 'deps(//core/store/materialization/control:materialization_service_lib)'` to assert that dataplane runtimes never leak into the control target. During the bridge state (before we deleted the alias) we also ran `bazel query --noimplicit_deps 'deps(//core/store:materialization_service)'` to ensure callers flipped to the new layering.
- **Telemetry verification:** Add structured logging around registry routing decisions (selected transport, fallback reason, loader factory type) and confirm via integration tests that the logs appear with correct tags, ensuring future debugging has enough signal.

## Rollout & Backout

- **Feature flags:** The temporary adapter introduced in Phase 1 acts as a soft flag; reverting to the old layout simply re-points StoreEngine to the forwarding targets while registry/router implementations stabilize.
- **Incremental landing:** Land each table section (control, runtime, metadata, verification) as an isolated change-set reviewed by owners of the touched code. This reduces rebase pressure and confines risk.
- **Backout plan:** Because forwarding headers remain until Phase 4, backouts only require reverting the latest phase-specific change without undoing directory moves that already landed. Should the registry introduce regressions, switch `MaterializationService` back to the forwarding adapter while debugging occurs.
- **Monitoring:** After rollout, watch existing StoreEngine dashboards for load latency and chunk repair rates. The registry logs described above provide additional breadcrumbs if anomalies appear.

# Schema Changes

None.

# Trade-offs & Risks

- **File churn**: Moving both `core/store/loading/*` and `core/store/loader/*` risks large rebases. Mitigation: ship forwarding headers (`core/store/loading/*.h` includes `materialization/contracts/...`) and alias Bazel targets until downstreams migrate.
- **Transient duplication**: During migration some types (e.g., `ReplicaHandle`) may exist in both the old and new namespaces. Maintain TODOs and delete the legacy copies once StoreEngine and Replica modules switch to the contracts layer.
- **Build graph complexity**: Introducing registry interfaces adds targets, but the reduced dependency fan-out (control no longer seeing dataplane) yields smaller rebuilds overall.
- **Agent guidance drift**: Repo-wide AGENTS instructions must be updated alongside code to prevent future modifications from bypassing the contracts layer.

# Compatibility & Acceptance Criteria

- StoreEngine surfaces (`materialize_replica`, `ingest_from_*`) remain API and ABI compatible; no changes visible to SDK clients.
- `bazel test //core/store/materialization/...` plus `bazel test //core/store:store_engine_*` pass in both fake- and real-CUDA modes.
- Control packages only depend on `materialization/contracts`, `planning`, and dataplane registry targets (verified via `bazel query --noimplicit_deps 'deps(//core/store/materialization/control:materialization_service_lib)'` inspection).
- Documentation updates land with the code change (`core/store/README.md`, `core/store/docs/architecture.md`, `docs/designs/0026-loader-subsystem-refactor.md` cross-link to this document).
- Legacy headers under `core/store/loading` and `core/store/loader` forward to the new locations with deprecation comments until all includes are migrated.

# References

- `core/store/loading/materialization/materialization_service.h:24-104`
- `core/store/loading/materialization/materialization_request.cc:12-86`
- `core/store/loading/materialization/materialize_orchestrator.cc:18-140`
- `core/store/loading/chunk_aware_loading_strategy.cc:18-429`
- `core/store/loading/loading_spec.h:25-140`
- `core/store/store_engine.cc:1150-1280`
- `docs/designs/0026-loader-subsystem-refactor.md`
- `core/store/docs/architecture.md:120-235`
