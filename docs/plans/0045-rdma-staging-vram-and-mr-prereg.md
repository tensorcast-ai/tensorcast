---
slug: rdma-staging-vram-and-mr-prereg
title: RDMA Staging Backends (GPU VRAM) and MR Preregistration (Plan)
areas: ["core", "daemon"]
related_code:
  - core/communicator/engine/engine.{h,cc}
  - core/communicator/engine/host_pinned_gpu_stager.h
  - core/communicator/engine/host_pinned_cpu_stager.{h,cc}
  - core/communicator/engine/gpu_vram_rdma_stager.{h,cc} # new (RDMA-only GPU VRAM staging)
  - core/communicator/engine/mr_cache.{h,cc}
  - core/communicator/misc/ibv_wrap.{h,cc}
  - core/communicator/transport/{partition_tensor,net_dev}.{h,cc}
  - daemon/server_main.cc
  - proto/tensorcast/communicator/v1/communicator_config.proto
links:
  design: ../designs/0045-rdma-staging-vram-and-mr-prereg.md
---

# Objective

Land P0/P1 end-to-end:

- **P0**: make `rdma_preregister` effective for GPU host-pinned staging (HostPinnedGpuStager; formerly `GpuNetStager`) by using slab-base MR normalization, and remove pointer-contract ambiguity by renaming `StageLease::host_ptr()` to `StageLease::exposed_ptr()`.
- **P1**: add a GPU VRAM staged-RDMA backend (bounded VRAM bounce pool + D2D staging) with unified config. `STAGED_RDMA_BACKEND_GPU_VRAM` is a forced mode: initialization must fail fast if unsupported, and runtime must not fall back.

# Current State & Grounding

- Staged RDMA uses `MemoryStager::stage()` and registers MRs via `MrCache`:
  - `core/communicator/engine/engine.cc`
- `rdma_preregister` preregisters host-pinned slabs once per NIC/PD:
  - `core/communicator/engine/engine.cc`
- Slab-base MR reuse is now applied for both `HostPinnedCpuStager` and `HostPinnedGpuStager` via stager-provided slab lookup:
  - `core/communicator/engine/engine.cc`
- `HostPinnedGpuStager` stages GPU tensors via `cudaMemcpyDeviceToHost` into `PinnedBufferPool` slices:
  - `core/communicator/engine/host_pinned_gpu_stager.h`
- Direct RDMA already stores a device pointer in `StageLease::exposed_ptr()` (formerly `host_ptr()`). This plan renames the API to match reality and make GPU VRAM staged RDMA safe to reason about.

## Constraints and assumptions

- No ad-hoc env vars; config must be protobuf-backed and follow unified runtime config (`docs/designs/0004-unified-runtime-config.md`).
- After any `.proto` edit, regenerate code via `bash tools/build_proto_python.sh`.
- Prefer bounded waiting; do not introduce blocking waits on shared worker threads (see repository concurrency guidelines).

# Configuration Compatibility Strategy

- This work is additive to `tensorcast.communicator.v1.CommunicatorConfig` and does not change defaults:
  - `rdma.staging_backend=STAGED_RDMA_BACKEND_UNSPECIFIED` behaves as STAGED_RDMA_BACKEND_HOST_PINNED.
  - `rdma.staging_backend=STAGED_RDMA_BACKEND_GPU_VRAM` is explicit and forced: communicator initialization fails if VRAM pool allocation or `ibv_reg_mr` preregistration fails for any required NIC/PD.
- Existing configs continue to parse; new keys are optional.

# Phases & Milestones

- [x] Phase 1 (P0): Make preregistration effective + clarify naming/contracts
  - [x] Milestone: rename stagers for clarity:
    - `DRAMStager` → `HostPinnedCpuStager`
    - `GpuNetStager` → `HostPinnedGpuStager`
  - [x] Milestone: rename `StageLease::host_ptr()` → `StageLease::exposed_ptr()` and update all call sites / docs.
  - [x] Milestone: add a stager-provided slab lookup hook (optional) so MR base normalization does not depend on `dynamic_cast`.
  - [x] Milestone: staged-RDMA MR registration uses slab bases for both host-pinned CPU and host-pinned GPU staging.
  - [x] Milestone: add targeted unit coverage proving slab reuse is taken when prereg is enabled.

- [x] Phase 2 (P1): GPU VRAM staged-RDMA backend
  - [x] Milestone: add typed config for staged-RDMA backend selection and VRAM pool sizing.
  - [x] Milestone: implement `GpuVramStagingPool` + `GpuVramRdmaStager` and wire into RDMA staged fallback.
  - [x] Milestone: add init-time validation and preregistration for VRAM pools using `ibv_reg_mr` (fail-fast; no runtime fallback).

# Acceptance Checks

- Default config produces no behavior change (STAGED_RDMA_BACKEND_HOST_PINNED staged RDMA).
- P0:
  - When `pinned_memory.classes[name=comm_gpu].rdma_preregister=true`, GPU staged RDMA uses slab-base MRs (no per-slice churn) and emits consistent MR registration metrics.
- P1:
  - With `rdma.staging_backend=STAGED_RDMA_BACKEND_GPU_VRAM`, communicator initialization fails if `ibv_reg_mr` preregistration for VRAM pools is unsupported/unavailable.
  - On supported systems, staged RDMA for GPU tensors avoids D2H staging and remains bounded by configured VRAM pool.
  - Runtime does not fall back to STAGED_RDMA_BACKEND_HOST_PINNED (forced mode).

# Tasks

## Phase 1 (P0)

- Rename stagers for clarity:
  - `DRAMStager` → `HostPinnedCpuStager`
  - `GpuNetStager` → `HostPinnedGpuStager`
- Rename `StageLease::host_ptr()` → `StageLease::exposed_ptr()` and update:
  - RDMA response serialization (`ProtoReadResponseExSeg.addr`)
  - StageLeaseRegistry-related logs/docs
  - `core/communicator/README.md` to avoid “host pointer” wording for RDMA.
- Add `MemoryStager` optional slab lookup (e.g., `mr_slab_for_ptr(void*) -> optional<Slab>`).
- Implement slab lookup for:
  - `HostPinnedCpuStager` (pinned pool slabs)
  - `HostPinnedGpuStager` (pinned pool slabs)
- Update `MakeStageFunction(...)` staged path:
  - normalize `(mr_base, mr_bytes)` using stager-provided slab lookup.
  - ensure preregistration (`MrCache`) is hit for slab-bases, not per-slice ptrs.
- Add tests:
  - a unit test that stages via `HostPinnedGpuStager` and verifies MR-cache keying uses slab base (e.g., preregister one slab and assert no new registrations occur for slice pointers).
- Update docs:
  - `core/communicator/README.md` to clarify preregistration behavior for GPU staging.

## Phase 2 (P1)

- Extend `proto/tensorcast/communicator/v1/communicator_config.proto`:
  - `RdmaConfig::StagedRdmaBackend staging_backend`
  - `uint64 vram_pool_bytes_per_gpu`
  - `uint64 vram_slice_bytes`
  - regenerate code (`bash tools/build_proto_python.sh`)
- Update config normalization (if needed) to treat missing/unspecified values as STAGED_RDMA_BACKEND_HOST_PINNED without changing behavior.
- Implement VRAM pool:
  - per-GPU fixed-size pool, slice size derived from `rdma.vram_slice_bytes`.
  - non-blocking acquire and immediate `ResourceExhausted` on failure (no runtime fallback).
- Implement VRAM staged-RDMA staging:
  - D2D copy into acquired VRAM slice.
  - rely on preregistered `ibv_reg_mr` results for the VRAM pool (no per-slice registration churn).
  - release on `RDMA_READ_DONE_EX` via `StageLease` lifecycle.
- Implement init-time validation (forced mode):
  - allocate VRAM pools
  - preregister VRAM pool MRs for every NIC/PD via `ibv_reg_mr`
  - fail communicator initialization if any preregistration step fails
- Add tests:
  - logic/unit tests that validate backend selection, credit release, and no MTCP usage of VRAM stager.
  - optional integration tests that require real CUDA + verbs; skip otherwise.
- Update docs:
  - `core/communicator/README.md` to document staged backend configuration and prerequisites.

# Test / Rollout / Backout

- Tests (C++):
  - After proto edits: `bash tools/build_proto_python.sh` and `bazel test //proto/... --test_output=errors`
  - Run relevant communicator tests (most can use fake CUDA): `bazel test //core/communicator/... --define=use_fake_cuda=true`
  - Add/maintain a targeted P0 regression test (fake CUDA): `bazel test //core/communicator/engine:staging_flow_controller_test --define=use_fake_cuda=true`
  - Run STAGED_RDMA_BACKEND_GPU_VRAM staged-RDMA integration tests only on real CUDA + verbs machines; skip otherwise.
- Rollout:
  - Default remains STAGED_RDMA_BACKEND_HOST_PINNED; STAGED_RDMA_BACKEND_GPU_VRAM staging is opt-in and fail-fast (no runtime fallback).
- Backout:
  - Set `rdma.staging_backend=STAGED_RDMA_BACKEND_HOST_PINNED` (or leave unspecified).
  - Revert the paired changeset if necessary; wire protocol remains unchanged.

# Risks & Tracking

- GPU VRAM pool sizing errors can cause load failures under concurrency; mitigation: bounded retries + clear metrics + configurable pool size.
- `STAGED_RDMA_BACKEND_GPU_VRAM` is forced and fails fast. This is intended; mitigation is operational (change config to STAGED_RDMA_BACKEND_HOST_PINNED) and improved diagnostics/metrics during init.

# Status

- Implementation complete; Phase 1 and Phase 2 milestones checked off.
- Proto regeneration: `bash tools/build_proto_python.sh`.
- Tests run:
  - `bazel test //core/communicator:stager_mr_normalization_test --define=use_fake_cuda=true`
  - `bazel test //core/communicator:gpu_vram_rdma_stager_test --define=use_fake_cuda=true`
  - `bazel test //proto/... --test_output=errors`
