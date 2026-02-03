---
slug: 0070-mapped-binding-requirements
title: Plan - Mapped Binding (Map Abiding) Requirements
links:
  design: ../designs/0070-mapped-binding-requirements.md
areas: ["sdk", "materialization", "binding", "inplace"]
related_code:
  - docs/designs/0063-binding-first-inplace-updates.md
  - docs/designs/0061-slot-based-inplace-binding-and-swap.md
  - docs/designs/0039-artifact-first-sdk.md
  - docs/internals/byte-range-mapping-and-execution.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/materialization.py
  - tensorcast/daemon_ctl.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - daemon/service/controllers/materialization_controller.cc
  - core/store/materialization/dataplane/sources/byte_range_map_builder.{h,cc}
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.{h,cc}
---

# Objective

Implement mapped binding per `docs/designs/0070-mapped-binding-requirements.md`:

- Allow `Artifact.bind_into(..., mapping=copy_plan)` to fill user-owned CUDA tensors using a traced copy plan
  (src name + slice -> dst name + slice).
- Store the mapping with the resulting `Binding` and reuse it on `Binding.swap(...)` for pointer-stable reloads.
- Execute the plan inside the TensorCast materialization pipeline (no Python copy loops).

# Status (2026-02-03)

- In progress: proto + daemon/core execution + SDK bind/swap wiring are implemented; docs updated.
- Pending: test coverage (Python + Bazel) and proto lint.

# Current State & Grounding

- SDK `Binding`/`InplaceSlot` swap is name-aligned and delegates to a single `MaterializeIntoTarget` RPC:
  - `tensorcast/api/store/inplace_slot.py`
  - `tensorcast/daemon_ctl.py`
  - `proto/tensorcast/daemon/v2/store_daemon.proto` (`MaterializeIntoTargetRequest`)
- SDK `Artifact.bind_into(...)` requires the target tensor names to exactly match the artifact selection names:
  - `tensorcast/api/store/artifact.py`
- The daemon `MaterializeIntoTarget` surface has no way to express a rename/split/merge plan; it materializes the
  requested selection directly into the provided `TargetLayout`:
  - `daemon/service/controllers/materialization_controller.cc`
- Daemon `MaterializeIntoTarget` validates `target_layout` tensor names against the artifact canonical index
  (and rejects “unknown tensor name”), so the existing RPC cannot be reused for `dst_name != ckpt_name` mappings:
  - `daemon/service/controllers/materialization_controller.cc`
- TensorCast already has a canonical byte-range mapping/execution engine (`ByteRangeMap` / `ByteRangeProgram`) used
  by view planning and materialize-into-target, including strided execution:
  - `docs/internals/byte-range-mapping-and-execution.md`
  - `core/store/materialization/dataplane/sources/byte_range_map_builder.{h,cc}`
  - `core/store/materialization/dataplane/sources/byte_range_mapped_source.{h,cc}`
- View semantics already support `narrow` and `transpose` ops, but mapped binding v1 intends narrow-only:
  - `tensorcast/api/_view_ops.py`
  - `proto/tensorcast/common/v1/common.proto` (`ViewSpec`, `NarrowOp`, `TransposeOp`)

# Consistency and Reuse Constraints (Implementation Rules)

- **No Python copy loops**: the copy plan must execute in the daemon/core materialization pipeline, not via
  `artifact.tensor_dict(...)` + `torch.copy_` loops in Python.
- **Preserve Binding safety**: keep the existing `retire -> overwrite -> (optional) publish` ordering and dirty-state
  semantics from `Binding`/`InplaceSlot` (mapped binding is a different fill strategy, not a different lifecycle).
- **Additive-only**: behavior is unchanged when `mapping=None`; mapped binding requires explicit opt-in via
  `bind_into(..., mapping=...)`.
- **Deterministic + validated**: v1 should reject dst overlaps and define dst coverage rules up front; avoid “partially
  updated weights” footguns.
- **Reuse ByteRangeMap**: compile the copy plan into byte ranges and use existing byte-range execution machinery.
- **No new ad-hoc config**: do not add environment variables; any required compatibility signaling should use
  UNIMPLEMENTED handling or additive proto fields.

# Ownership Map

- **core/store**: mapping compilation (tensor-range -> byte-range) and execution primitives (ByteRangeMap/Program).
- **daemon**: RPC surface + preflight validation + invoking the execution pipeline into external targets.
- **sdk**: user-facing API (`bind_into(..., mapping=...)`), early validation, and storing/reusing mapping for `swap()`.

# Phases & Milestones

- [x] Phase 1: Nail down v1 semantics (so implementation is not ambiguous)
  - [x] Milestone 1.1: API surface decision: use `bind_into(..., mapping=...)` (no separate method)
  - [x] Milestone 1.2: Specify v1 dst semantics: coverage requirements + overlap rules + deterministic ordering
    - [x] Update `docs/designs/0070-mapped-binding-requirements.md` to make these semantics normative (not “open questions”)
  - [x] Milestone 1.3: Specify v1 view-translation constraints (support narrow-only views; reject transpose/permutation views for mapped binding)
    - [x] Update `docs/designs/0070-mapped-binding-requirements.md` with an explicit “allowed view ops” rule
  - [x] Milestone 1.4: Specify publish gating for mapped bindings (start strict; likely “identity/full-coverage only” in v1)
    - [x] Update `docs/designs/0070-mapped-binding-requirements.md` with the exact v1 publish rule

- [ ] Phase 2: Daemon execution path for mapped copy plans
  - [x] Milestone 2.1: Add an additive proto surface for mapped materialization (new RPC recommended) + message types for the copy plan
  - [ ] Milestone 2.2: Run codegen + proto lint
    - [x] `bash tools/build_proto_python.sh`
    - [ ] `bazel test //proto/... --test_output=errors`
  - [x] Milestone 2.3: Implement daemon-side preflight:
    - [x] Resolve artifact canonical index and (optional) view metadata
    - [x] Validate mapping bounds, dtype/shape compatibility, and dst overlap/coverage policy
    - [x] Fail without writing any bytes on preflight error
  - [x] Milestone 2.4: Compile + execute:
    - [x] Compile mapping into a byte-range program (dst logical ByteSpace -> src canonical ByteSpace)
    - [x] Execute using existing byte-range machinery and return errors with entry context
    - [x] Propagate `operation_id` and record basic metrics (bytes copied, time)

- [x] Phase 3: SDK integration (bind + swap)
  - [x] Milestone 3.1: Build a stable destination layout:
    - [x] Register VRAM regions for dst tensors (reuse `bind_into` region registration rules)
    - [x] Define a stable logical dst ByteSpace (deterministic tensor order and logical offsets)
  - [x] Milestone 3.2: Implement `Artifact.bind_into(..., mapping=...)`:
    - [x] Relax the current “names must match selection” constraint when `mapping` is provided
    - [x] Call the mapped daemon RPC and construct a `Binding`/`InplaceSlot` backed by the dst tensors
  - [x] Milestone 3.3: Persist mapping + dst layout identity on the binding:
    - [x] Store enough data to rerun the exact same mapped fill on `swap()` (mapping + dst layout hash/order)
    - [x] Ensure `Binding.swap(...)` uses the stored mapping without caller resupplying it
  - [x] Milestone 3.4: Swap safety:
    - [x] Validate new artifact indices against the stored mapping before overwrite (dtype/shape/range checks)
    - [x] On mismatch, fail swap without partial overwrite

- [ ] Phase 4: Tests, observability, and rollout
  - [ ] Milestone 4.1: Python acceptance tests for correctness, pointer stability, and failure semantics (use fake CUDA backend where applicable)
  - [ ] Milestone 4.2: Add daemon/core unit tests for mapping compilation and overlap/gap validation (Bazel)
  - [x] Milestone 4.3: Document the new API in `tensorcast/api/store/README.md` with a vLLM-oriented example

# Tasks

- SDK
  - Define stable Python types for the copy plan (`Range`, `CopyPlanEntry`, `CopyPlan`) and a canonical JSON serialization
    (stable ordering, version tag).
  - Extend `Artifact.bind_into(...)` to accept `mapping=...` and to relax the current “names must match selection” constraint
    when mapping is provided.
  - Persist the mapping with the binding (so `swap()` reuses it without the caller resupplying it).
  - Add client-side validation:
    - target tensors are CUDA, writable, and on a single device
    - mapping references existing `dst_name`s
    - per-entry dtype/shape/slice compatibility and bounds
    - overlap/gap policy for dst slices (reject overlaps in v1 unless explicitly defined)
  - Ensure error messages include entry index and `(ckpt_name, ckpt_range) -> (dst_name, dst_range)`.
  - Add a `DaemonCtl` wrapper for the new mapped RPC:
    - `tensorcast/daemon_ctl.py`

- Daemon / core
  - Add proto message(s) for a mapped copy plan and a mapped materialization RPC (new RPC preferred to avoid changing
    `MaterializeIntoTarget` name-aligned semantics).
  - Implement preflight validation and mapping compilation in the daemon/controller layer; avoid “write then fail”
    partial updates.
  - Compile the copy plan into a byte-range program using existing byte-range machinery; prefer a single compiled plan per
    bind/swap.
  - Execute with bounded concurrency and clear progress reporting; plumb `operation_id` through to logs/metrics.

- Compatibility
  - Make mapped binding purely additive: existing `bind`/`bind_into`/`swap` behavior is unchanged when `mapping=None`.
  - SDK behavior on older daemons:
    - treat gRPC `UNIMPLEMENTED` as “mapped binding unsupported” and raise a clear, non-retryable error
    - avoid silent fallback to Python copy loops

# Test / Rollout / Backout

- Acceptance checks (must pass)
  - [ ] `bind_into(..., mapping=...)` supports rename + split/merge within v1 constraints
  - [ ] Pointer stability: `data_ptr()` of dst tensors is unchanged across `swap()`
  - [ ] Correctness matches a Python baseline (same mapping applied with torch slicing/copy)
  - [ ] Swap mismatch detection: wrong dtype/shape fails without partial overwrite
  - [ ] Failure semantics: overwrite failure marks binding dirty; publish failure keeps bytes but remains local-only
  - [ ] Publish gating matches the documented v1 rule

- Tests
  - Run Python tests with fake CUDA backend:
    - `TENSORCAST_CUDA_BACKEND=fake uv run pytest tests/python/...`
  - Add a focused suite for mapped binding:
    - split mapping (one src -> many dst)
    - merge mapping (many src -> one dst)
    - scalar fill
    - swap mismatch detection (dtype/shape)
    - dirty semantics on overwrite failure
    - publish gating behavior (allowed vs rejected cases)
  - Prefer a dedicated test module for readability (example):
    - `tests/python/test_mapped_binding.py`
  - Add daemon/core tests for mapping compilation:
    - `bazel test //core/store/... --test_output=errors`
    - `bazel test //daemon/... --test_output=errors --test_env=TENSORCAST_CUDA_BACKEND=fake`

- Rollout
  - Introduce the mapped API as additive; keep existing name-aligned bind/swap as the default path.
  - Land daemon support first (new RPC/additive fields), then SDK wiring, then docs/tests.

- Backout
  - If daemon-side mapping is unstable, keep the feature behind explicit opt-in (still `mapping=`) and fail fast on
    any detected ambiguity (overlap/partial coverage) rather than “best effort”.
  - If needed, keep the SDK API but disable mapped RPC usage via a runtime capability check (temporary mitigation).

# Risks & Tracking

- **Strided slicing complexity**: dim=1 slices imply strided reads/writes; mitigate by compiling into byte-range programs and starting with narrow-only views.
- **Publish semantics**: publishing a mapped layout is only safe when the resulting bytes are routable under the artifact selection; start with strict gating.
- **Overlap/gap semantics**: must be explicit and validated; v1 should reject overlaps and require full dst coverage if publish is requested.
- **Performance regressions**: poorly compiled plans can produce many small runs; mitigate via normalization/merging in the byte-range compiler and perf tests on vLLM traces.
