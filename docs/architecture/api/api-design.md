---
title: API Design
description: Public SDK surface and contracts
---

# API Design

This document describes the public SDK surface for artifact registration and
materialization. It is the entry point for application developers and SDK
integrators.

## Store And Entry Points

TensorCast exposes a process-wide Store session and module-level helpers that
bind to it.

- `tensorcast.init(...)` establishes or connects to a Store Daemon and sets the
  daemon address for the process.
- `tensorcast.store()` returns the process Store. It is created lazily when
  first accessed.
- Module-level helpers (`tensorcast.register`, `tensorcast.put`,
  `tensorcast.register_view`, `tensorcast.artifact`, `tensorcast.from_disk`) are
  thin wrappers around the process Store.

Store construction is supported for advanced use:

- `tensorcast.api.store.Store(daemon_endpoint, opts=StoreOptions)`

## Registration And Upload APIs

Primary registration verbs:

- `Store.register(...)` registers existing GPU memory using lease-in-place (LIP)
  when supported.
- `Store.put(...)` uploads tensors into daemon-owned stable DRAM.
- `Store.register_view(...)` registers a slice or transpose view and allows the
  daemon to rebuild the canonical artifact from a partial upload.

All registration verbs accept an optional `policy` and an optional
`RegisterArtifactOptions`:

- `policy: StorePolicy | str | None` is the preferred surface.
- `options.policy` is an advanced escape hatch.
- If both are provided they must normalize to the same policy.

## Artifact Handles And Materialization

Retrieval is centered on Artifact handles, not on `Store.get` or `Store.get_into`.

- `tensorcast.artifact(...)` and `Store.artifact(...)` return an `Artifact` handle
  that exposes metadata and materialization helpers.
- `Artifact.tensor_dict(...)` and `Artifact.tensor_dict_into(...)` provide the
  primary read surface.
- `tensorcast.from_disk(path)` creates a handle backed by a disk path and
  configures disk-first fallback.

This design keeps the public surface small while allowing the SDK to evolve
materialization internals without breaking callers.

## Fallback Selection

Materialization behavior is controlled by `FallbackOptions` and
`GetArtifactOptions`:

- `FallbackOptions.prefer` selects `auto`, `local`, `p2p`, or `disk`.
- `disk_path` pins a specific disk location for fallbacks.
- `replica_uuid` hints the daemon to reuse a prefetched replica.
- `verify_checksums` controls descriptor validation on disk reads.

## StorePolicy And Persistence Hooks

`StorePolicy` is the single durability and placement declaration for `register`
and `put`. It supports:

- Profiles (`cache`, `durable`, `ha`, `cold`, `warm`, `pinned`).
- Explicit tiers with `must`, `should`, `may`.
- `overflow_policy` and `layout` overrides.

When policy includes shared disk or remote stable tiers, the SDK triggers
`StartPersistence` after registration and stores the returned
`persistence_task_id` on `RegisteredArtifact`.

`Store.query_persistence_status(...)` exposes daemon task state by task id or
artifact id.

## Region APIs

For region-backed registration and quiesced cleanup:

- `Store.register_vram_region(...)` and `Store.unregister_vram_region(...)` manage
  reusable CUDA IPC regions.
- `Store.deregister_artifact(...)` quiesces and drains active exports, then
  revokes the lease and performs best-effort Global Store cleanup.

## Contracts And Invariants

- The SDK owns the process Store and its runtime caches.
- All public methods raise `ArtifactError` with structured status codes.
- Registration and materialization errors map to retryable or non-retryable
  categories; the SDK applies bounded retries for transient errors.
- Policy resolution is authoritative on the daemon; the SDK only validates the
  policy shape.

## Code Map

- Public store facade: `../../../tensorcast/api/store/__init__.py`
- Store runtime: `../../../tensorcast/api/store/runtime.py`
- Registration pipeline: `../../../tensorcast/api/store/registration.py`
- Materialization pipeline: `../../../tensorcast/api/store/materialization.py`
- Policy model: `../../../tensorcast/api/_config.py`
