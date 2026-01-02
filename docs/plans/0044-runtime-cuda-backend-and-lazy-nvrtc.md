---
slug: 0044-runtime-cuda-backend-and-lazy-nvrtc
title: Runtime CUDA Backend Selection and Lazy NVRTC Loading (Plan)
links:
  design: ../designs/0044-runtime-cuda-backend-and-lazy-nvrtc.md
areas: ["core", "daemon", "sdk", "build"]
related_code:
  - core/common/cuda_api.h
  - core/common/cuda_backend.cc
  - core/common/cuda_backend_real.cc
  - core/common/cuda_backend_fake.cc
  - core/common/cuda_backend_selection_test.cc
  - core/common/cuda_driver_api.cc
  - core/common/lazy_nvrtc.cc
  - core/common/dynamic_library.cc
  - core/common/artifact_hash_gpu.cc
  - core/common/BUILD
  - tensorcast/csrc/checkpoint_py.cc
  - setup.py
---

# Objective

Implement Design 0044: replace compile-time FakeCuda/RealCuda splitting with a runtime-selected backend (`real` default, `fake` test-only), and refactor NVRTC usage to be lazily loaded (PyTorch-style) so binaries can still load in environments without an NVIDIA driver and/or without NVRTC present (while still linking `libcudart.so`).

# Current State & Grounding

- Legacy build-time selection (removed):
  - Bazel select in `core/common/BUILD` used to choose `cuda_fake.cc` vs `cuda_real.cc` and define `USE_FAKE_CUDA`.
- Header-level type splitting:
  - `core/common/cuda_api.h:14` switches between CUDA headers and custom fake CUDA type definitions.
- Conditionals in behavior:
  - Checkpoint restore logic: `core/checkpoint/checkpoint.cc:462`.
  - View executors: `core/store/materialization/dataplane/view/view_transform_executor.cc:56`, `core/store/materialization/dataplane/view/view_ingest_executor.cc:58`.
  - CUDA error helpers: `core/common/error_handling.h:37`.
  - Python extension compile-time flag export/warn: `tensorcast/csrc/checkpoint_py.cc:919`.
- GPU hashing hard-depends on NVRTC and a second driver loader:
  - `core/common/artifact_hash_gpu.cc:28`.
- Legacy driver entrypoint loading (removed):
  - VMM loader in `core/common/cuda_real.cc` used `dlopen/dlsym`.
  - Separate loader in `core/common/artifact_hash_gpu.cc` duplicated the driver path.

# Files and entry points (target end state)

New/changed core modules (proposed names; all `snake_case`):
- `core/common/cuda_backend.h` / `core/common/cuda_backend.cc`: backend interface, backend selection, and dispatch façade.
- `core/common/cuda_backend_real.cc`: real backend implementation (cudart + driver entrypoints).
- `core/common/cuda_backend_fake.cc`: fake backend implementation (CPU simulation).
- `core/common/cuda_driver_api.h` / `core/common/cuda_driver_api.cc`: centralized driver entrypoints (VMM + module APIs).
- `core/common/lazy_nvrtc.h` / `core/common/lazy_nvrtc.cc`: NVRTC lazy loader (PyTorch-style).

Updated call sites (remove compile-time `#ifdef USE_FAKE_CUDA`):
- `core/common/cuda_api.h`
- `core/common/error_handling.h`
- `core/common/artifact_hash_gpu.cc`
- `core/checkpoint/checkpoint.cc`
- `core/store/materialization/dataplane/view/view_transform_executor.cc`
- `core/store/materialization/dataplane/view/view_ingest_executor.cc`
- `tensorcast/csrc/checkpoint_py.cc`

# Phases & Milestones

- [x] Phase 1: Backend interface + selector (no behavior change yet)
  - [x] Milestone 1.1: Add backend selector with strict parsing for `TENSORCAST_CUDA_BACKEND` (`real`/`fake` only).
  - [x] Milestone 1.2: Add strict test-only gate for fake: require one of `TEST_SRCDIR`, `TEST_TMPDIR`, or `PYTEST_CURRENT_TEST`.
  - [x] Milestone 1.3: Add one-time `LOG(ERROR)` banner when fake is selected (match Design 0043 banner format).
  - [x] Milestone 1.4: Add focused unit test(s) for selector parsing and gating (injectable env reader so tests can simulate “non-test” contexts even under Bazel).

- [x] Phase 2: Convert `tensorcast::cuda` façade to dispatch
  - [x] Milestone 2.1: Introduce `CudaBackend` interface and move all current `tensorcast::cuda::*` logic into backend implementations (avoid ODR conflicts once both are linked).
  - [x] Milestone 2.2: Update `core/common/cuda_api.h` to always include CUDA headers (`cuda.h`, `cuda_runtime_api.h`) and delete custom fake type definitions.
  - [x] Milestone 2.3: Delete compile-time `SC_RETURN_IF_FAKE_CUDA_UNSUPPORTED` and replace with runtime backend behavior (`absl::UnimplementedError`).
  - [x] Milestone 2.4: Update `core/common/error_handling.h:37` to remove `#ifdef USE_FAKE_CUDA` and rely on CUDA runtime string helpers (or a backend-independent mapping).

- [x] Phase 3: Centralize driver entrypoints
  - [x] Milestone 3.1: Implement `DriverApi` using `cudaGetDriverEntryPoint` (CUDA 12.6 baseline; API available since CUDA 12.4) and resolve the required symbol set from Design 0044.
  - [x] Milestone 3.1b: Define driver symbols as an X-macro catalog (PyTorch-style) and generate the `DriverApi` table + resolution code from the catalog to prevent drift.
  - [x] Milestone 3.2: Replace `core/common/cuda_real.cc:73` VMM `dlopen/dlsym` with `DriverApi`.
  - [x] Milestone 3.3: Replace the driver loader in `core/common/artifact_hash_gpu.cc:45` with `DriverApi`.
  - [x] Milestone 3.4: Add a small driver-loader unit test that runs in fake mode and ensures driver entrypoints are not touched when `TENSORCAST_CUDA_BACKEND=fake`.

- [x] Phase 4: Lazy NVRTC loader (PyTorch-inspired)
  - [x] Milestone 4.1: Implement `LazyNvrtc` that loads `libnvrtc.so.12` (from `CUDA_VERSION >= 12060`) and resolves required NVRTC symbols on demand; cache pointers after first resolution (PyTorch LazyNVRTC pattern).
  - [x] Milestone 4.1b: Define NVRTC symbols as an X-macro catalog and generate stub wrappers (or a table + accessors) from the catalog to avoid hand-written boilerplate.
  - [x] Milestone 4.1c: Introduce a small `DynamicLibrary` helper in `core/common/` and ensure all `dlopen/dlsym` calls are isolated there (mirrors PyTorch `at::DynamicLibrary` usage).
  - [x] Milestone 4.2: Refactor `core/common/artifact_hash_gpu.cc` so NVRTC is optional at runtime: on failure, fall back to CPU hashing with a clear log message.
  - [x] Milestone 4.2b: Remove the hard link to NVRTC (`@nvrtc//:nvrtc`) so produced binaries/extensions do not list `libnvrtc.so` in ELF `NEEDED` (headers remain for types; runtime loading via `DynamicLibrary`).
  - [x] Milestone 4.3: Ensure `TENSORCAST_CUDA_BACKEND=fake` never attempts NVRTC or driver module loads (guard early).

- [x] Phase 5: Remove build-time fake/real switches
  - [x] Milestone 5.1: Update `core/common/BUILD:4` to remove `config_setting(use_fake_cuda)` and build both backends into the same target (no `USE_FAKE_CUDA` define).
  - [x] Milestone 5.1b: Remove or rename other `use_fake_cuda` config_settings across the build graph (they currently exist in multiple packages and are not all CUDA-backend selection):
    - [x] `core/checkpoint/BUILD:6`
    - [x] `core/communicator/BUILD:7`
    - [x] `core/store/BUILD:6`
    - [x] `core/store/replica/BUILD:6`
    - [x] `core/store/materialization/benchmarks/BUILD:5`
    - [x] `core/store/materialization/dataplane/BUILD:369`
  - [x] Milestone 5.1c: Decouple communicator IBV mocking from CUDA backend selection:
    - [x] Replace `core/communicator/misc/ibv_wrap.h:11` `#ifdef USE_FAKE_CUDA` with a dedicated mock/real switch that does not reuse `USE_FAKE_CUDA`.
    - [x] Update `core/communicator/BUILD` so IBV mock selection is independent of `TENSORCAST_CUDA_BACKEND`.
  - [x] Milestone 5.2: Remove `USE_FAKE_CUDA` and `--use-fake-cuda` build toggles from `setup.py` (and remove `-DUSE_FAKE_CUDA` compile args for the Python extension).
  - [x] Milestone 5.3: Update remaining `#ifdef USE_FAKE_CUDA` sites to runtime selection:
    - [x] `core/checkpoint/checkpoint.cc:462`
    - [x] `core/store/materialization/dataplane/view/view_transform_executor.cc:56`
    - [x] `core/store/materialization/dataplane/view/view_ingest_executor.cc:58`
    - [x] `tensorcast/csrc/checkpoint_py.cc:919`
    - [x] `core/common/artifact_hash_gpu.cc:28`
  - [x] Milestone 5.4: Delete the `USE_FAKE_CUDA` macro entirely and ensure no call sites remain.

- [x] Phase 6: Tests + docs updates
  - [x] Milestone 6.1: Replace `--define use_fake_cuda=true` in test invocations with `--test_env=TENSORCAST_CUDA_BACKEND=fake` (keep explicit; no defaulting to fake).
  - [x] Milestone 6.2: Update Python test harness/docs to use `TENSORCAST_CUDA_BACKEND=fake uv run pytest ...`.
  - [x] Milestone 6.3: Update documentation that mentions `USE_FAKE_CUDA`:
    - [x] `README.md`
    - [x] `docs/development/testing.md`
    - [x] `AGENTS.md`
    - [x] Any additional docs discovered via `rg "USE_FAKE_CUDA|use_fake_cuda"`
  - [x] Milestone 6.4: Update `tensorcast/_build_config.py` validation to reflect runtime backend selection (and preserve strictness / test-only behavior).

# Migration inventory (grounded grep results)

These known docs currently instruct `--define=use_fake_cuda=true` and should be updated to `--test_env=TENSORCAST_CUDA_BACKEND=fake` (or removed if not applicable):
- `AGENTS.md:141`
- `AGENTS.md:156`
- `AGENTS.md:157`
- `README.md:107`
- `docs/development/testing.md:30`
- `docs/development/testing.md:38`
- `docs/development/testing.md:79`
- `docs/development/testing.md:80`
- `core/communicator/README.md:714`
- `docs/architecture/p2p-transfer-strategies.md:66`
- `docs/architecture/p2p-transfer-strategies.md:71`
- `docs/architecture/p2p-transfer-strategies.md:75`
- `docs/deployment/store-daemon.md:127`
- `docs/deployment/store-session-release-checklist.md:24`
- `docs/designs/0023-uma-single-ledger-memory.md:95`
- `docs/designs/0033-store-runtime-unification.md:146`
- `docs/designs/0036-01-materialization-pipeline-v2.md:209`
- `docs/designs/0036-04-disk-artifact-variant.md:266`
- `docs/designs/0038-daemon-only-disk-materialization.md:248`
- `docs/plans/0036-01-materialization-pipeline-v2.md:69`
- `docs/plans/0036-01-materialization-pipeline-v2.md:70`
- `docs/plans/0036-04-disk-artifact-variant.md:53`
- `docs/plans/0036-04-disk-artifact-variant.md:60`
- `docs/plans/0043-unified-pinned-memory-authority.md:108`
- `docs/plans/0043-unified-pinned-memory-authority.md:109`

These known docs currently instruct `USE_FAKE_CUDA=1` builds and should be updated/removed once runtime selection is in place:
- `docs/architecture/p2p-transfer-strategies.md:62`

Docs sweep checklist (before removing `USE_FAKE_CUDA`):
- `rg "USE_FAKE_CUDA=1|USE_FAKE_CUDA|use_fake_cuda|CUDA Toolkit 12\\.[0-9]\\+|CUDA_VERSION >= 12" docs/` and update any stale guidance to the runtime backend model + CUDA 12.6+ baseline.

# Acceptance Checks

- `TENSORCAST_CUDA_BACKEND` accepts only `real`/`fake`; invalid values fail fast.
- Default backend is `real` when env var is unset.
- Core binaries always link `libcudart.so` (this design removes hard links to NVRTC/driver modules, not the CUDA runtime).
- `TENSORCAST_CUDA_BACKEND=fake`:
  - Succeeds only in test environments (Bazel/PyTest heuristics).
  - Emits an unmistakable `LOG(ERROR)` banner once per process.
  - Does not attempt to load NVIDIA driver (`libcuda`) or NVRTC libraries (even though `libcudart.so` is present).
- No `#ifdef USE_FAKE_CUDA` remains in the codebase (selection or type splitting).
- NVRTC is loaded lazily; missing NVRTC falls back to CPU hashing with a clear log.
- Produced binaries do not have a hard link to `libnvrtc.so` (e.g., `ldd <binary> | rg nvrtc` is empty).

Dependency verification (concrete):
- Daemon:
  - `bazel build //daemon:tensorcast_daemon`
  - `readelf -d bazel-bin/daemon/tensorcast_daemon | rg NEEDED | rg -i nvrtc` (expect no output)
- Python extension:
  - `BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext`
  - `readelf -d build/lib*/tensorcast/_C*.so | rg NEEDED | rg -i nvrtc` (expect no output)

# Test Plan

- C++ (fake backend):
  - `bazel test //core/common:cuda_api_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
  - `bazel test //core/common:cuda_backend_selection_test`
  - Representative store/daemon tests that previously used `--define use_fake_cuda=true` should be migrated to `--test_env=TENSORCAST_CUDA_BACKEND=fake`.
- C++ (real backend sanity, if GPU/driver available):
  - Run one representative target without `TENSORCAST_CUDA_BACKEND` to confirm default `real`.
- Python:
  - `source .venv/bin/activate`
  - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/` (no-driver / no-NVRTC test mode; still requires `libcudart.so`)

Additional smoke coverage added during implementation:
- `core/common/cuda_backend_selection_test.cc` verifies fake-mode selection and asserts that `DriverApi` loading is never attempted under `TENSORCAST_CUDA_BACKEND=fake`.
- Optional follow-up: add a small `//core/common:artifact_hash_lib` smoke test if we want explicit NVRTC load-attempt tracking.

# Rollout / Backout

- Rollout:
  - Land backend dispatch while keeping behavior identical under `real`.
  - Migrate tests to set `TENSORCAST_CUDA_BACKEND=fake` before removing `USE_FAKE_CUDA`.
  - Remove build-time select only after all CI paths are updated.
- Backout:
  - If runtime dispatch introduces regressions, temporarily keep `real` backend only (ignore the env var) while preserving the new internal structure; re-enable fake once stabilized.

# Risks & Tracking

- Risk: Linking both implementations causes symbol/ODR conflicts.
  - Mitigation: move implementations behind classes/function tables and keep `tensorcast::cuda` as the only exported symbol set.
- Risk: Fake backend accidentally used in production.
  - Mitigation: strict test-environment gating + loud logs + fail fast on violation.
- Risk: NVRTC/driver dynamic loading diverges from expected CUDA packaging.
  - Mitigation: try a small, well-defined library name set (CUDA 12.x => `libnvrtc.so.12`), and surface clear fallback-to-CPU behavior.

# Open Questions

- Should we explicitly support `TENSORCAST_CUDA_BACKEND=fake` for non-test local runs (e.g., `bazel run`), or keep the “test-only” gate absolute?
- Should IBV mocking be transitioned to dynamic loading as well (to remove compile-time dependency on `infiniband/verbs.h`), or is a dedicated build define acceptable for communicator tests?
