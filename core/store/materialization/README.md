<!-- Copyright (c) 2025, TensorCast -->

# Materialization Module

This package consolidates the contracts, control-plane, planning helpers, and
dataplane runtimes that back `StoreEngine::materialize_replica()`. The
subdirectories map 1:1 to runtime responsibilities:

- `contracts/` — Header-only specs (replica keys, hints, view descriptors) and
  interfaces (`MaterializationRequest`, `ArtifactSourceRouter`,
  `IArtifactLoaderRegistry`) consumed by both control and dataplane code.
- `control/` — Orchestration services, Global Store coordination helpers, and
  adapters that wire StoreEngine components to the contracts layer.
- `planning/` — UMA aware planning helpers (chunk-aware loading strategy) that
  guide ingestion behavior without importing dataplane loaders directly.
- `dataplane/` — Loader runtime, sources/sinks, registry implementations, and
  streaming infrastructure that execute the actual materialization.

The refactor tracks design 0027 (Materialization Control/Data-plane
Unification). Each subdirectory exposes a Bazel package so layering can be
enforced (`contracts` ← `control` ← `planning` ← `dataplane`). The legacy
`core/store/loading` and `core/store/loader` trees have been retired, so all
callers include and depend on the materialization packages directly—there are
no remaining `//core/store:*` compatibility aliases.

## Benchmarks

- Safetensors load strategy benchmark harness:
  - Target: `//core/store/materialization/benchmarks:safetensors_load_strategy_benchmark`
  - Runbook: `docs/internals/safetensors-load-strategy-benchmark.zh.md`
  - Notes:
    - Strategy A supports A4 NCCL collectives via `--enable_collectives=true` (real CUDA only, single-process multi-GPU).
    - Strategy B is disk-only (no NCCL) and merges adjacent segments in `commit`; the runner logs `segments_pre_merge/segments_merged/src_runs` to show planner effectiveness.
    - Baselines include disk `--disk_io_mode=auto|direct|buffered`, NCCL baselines, and `gpu_sched_waits/gpu_sched_wait_sec` self-proof fields.
    - The benchmark logs a `/proc/meminfo` summary for cache-state recording.
