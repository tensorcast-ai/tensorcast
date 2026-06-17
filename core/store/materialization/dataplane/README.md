<!-- Copyright (c) 2025, TensorCast Team -->

# Materialization Dataplane

Home for loader runtime code: sources/sinks, pump implementations, disk/P2P
loaders, view planners/executors, metadata helpers, and the loader registry.
Targets are organized by concern (contracts, runtime, sources, sinks, view,
metadata, verification, registry) so control packages only depend on the
registry/router exports. The legacy `core/store/loader` aliases have been
removed, making these targets the single source of truth for dataplane code.

## Key Knobs (I/O + H2D)

- `FilePartitionSource::Options::direct_io_mode` (`auto|direct|buffered`)
  - `auto` selects `O_DIRECT` for large payloads (currently `> 5GiB`) and will fall back to buffered I/O if `open(O_DIRECT)` fails with a common “unsupported” errno (see `direct_io_fallback_errno()` / `direct_io_fallback_reason()`).
- `GpuMemorySink::Options::{gpu_sched_enabled,gpu_sched_limit_bytes,gpu_sched_limit_copies}`
  - Limits per-GPU in-flight H2D bytes/copies for `write_at_async()` to reduce transient oversubscription; stats are readable via `get_gpu_scheduler_stats(device_id)`.
- Byte-range mapping (`engine.byte_mapping.*`)
  - `enable_strided_execution`: toggles strided run detection + block-cache execution in the `ByteRangeCompiler`.
  - `enable_direct_write_at`: allows mapped direct-write when programs contain no strided runs.
  - `program_cache_entries`: LRU size for compiled `ByteRangeProgram`s.
  - `strided_*`: thresholds for strided detection (min ranges/row length/amplification) and block sizing.

## View Execution (SelectionPlan)

- `ViewPlanSource` compiles the `SelectionPlan` `ByteRangeMap` into a `ByteRangeProgram` and executes it via `ByteRangeMappedSource`.
- Strided `narrow(axis=1)`-style runs use row-block coalesced reads plus host packing, with PAD=0 semantics preserved.
- Strided execution is auto-gated by run length, row width, and amplification bounds; `VLOG(1)` summarizes read/pack/caching totals per instance.
