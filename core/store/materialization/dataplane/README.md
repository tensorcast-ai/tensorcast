<!-- Copyright (c) 2025 -->

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
