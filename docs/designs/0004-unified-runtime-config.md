---
id: design-0004-unified-runtime-config
slug: 0004-unified-runtime-config
title: Unified Runtime Configuration System (Design)
status: proposed
areas: ["daemon", "global_store", "sdk", "core"]
related_code:
  - proto/tensorcast/config/v1/*.proto
  - core/communicator/config_io.*
  - tensorcast/global_store/config/settings.py
  - tensorcast/client_config_loader.py
  - tensorcast/daemon_config.py
created: 2025-09-09
last_updated: 2025-09-09
---

# Summary

Standardize all runtime configuration across TensorCast behind a single, strong‑typed configuration file per process. Protobuf is the authoritative schema; YAML/JSON are operator‑friendly carriers that load into the same Protobuf messages. Processes accept one flag only, `--config=/path/to/file.{yaml,json}`, and no environment variables or ad‑hoc flags influence runtime behavior. Unknown keys are rejected. Configuration changes take effect on restart; no hot reload is supported.

This design integrates and supersedes prior communicator‑only configuration unification by embedding `tensorcast.communicator.v1.CommunicatorConfig` as a subsection under the daemon configuration.

# Goals (Why)

- Single entry point: exactly one `--config` per process; no other sources.
- Strong typing: Protobuf schemas for all runtime config with centralized defaulting and validation.
- Unified format: YAML/JSON load into the same Protobuf messages.
- Zero environment variables: move OTel, logging, and other knobs into explicit config.
- Predictable behavior: reject unknown fields and eliminate conflicting priority rules.
- Cross‑language parity: shared schema and consistent loaders for C++ and Python.

# Non‑Goals

- Hot reload or SIGHUP‑based reconfiguration.
- TLS certificate rotation without restart.
- Merging multiple sources or supporting precedence chains (no ENV/flags stacking).

# Scope & Interfaces

Authoritative package namespace: `tensorcast.config.v1`. Separate top‑level messages per process type; no process union types:

- `DaemonConfig` (C++ Store Daemon)
  - `server`: `listen` (and optional `p2p_listen`) as `SocketAddress`, `num_threads`, `storage_path`, `grpc` server args, TLS.
  - `lifecycle`: eviction toggles and intervals, PID scanning, sweeper intervals, TTLs, VRAM fraction.
  - `high_availability`: `global_store_endpoints`, heartbeat/periodic sync/retry.
  - `communicator`: `tensorcast.communicator.v1.CommunicatorConfig` (reused schema).
  - `engine`: `*_bytes`, streaming and DVMP sizing, `pinned_allocation_timeout`, `p2p_fallback_disk_dir`.
  - `observability`: OTel (lang‑agnostic), logging (enum level, sinks), tracing; `otel_cxx` holds C++‑specific toggles.
  - `compatibility`: targeted compatibility switches (e.g., `confirm_requires_disk_path`, `verification_timeout_status`).
  - `checkpoint.streaming`: `num_buffers`, `io_chunk_bytes`, `pinned_pool_bytes`.
  - `debug.cuda`: guardrails for same‑process IPC fallback.

- `GlobalStoreConfig` (Python Global Store)
  - `database`, `server` (`listen`, worker pool, `grpc`), `worker_policy`, optional `web_ui`, `observability`.

- `ClientConfig` (Python client/CLI)
  - `daemon.target` (`SocketAddress`), `bin_path`, `python_interpreter`, `storage.default_root`, client defaults and observability.

Common types live in `proto/tensorcast/config/v1/common.proto` (e.g., `SocketAddress`, `GrpcServer`, `Logging`, `Observability`, `ConfigMeta`).

# Loader Semantics

- Parse path (C++ and Python): YAML → in‑memory tree → JSON → `google::protobuf::util::JsonStringToMessage`; JSON loads directly.
- Unknown keys: hard fail (`ignore_unknown_fields=false`). YAML tree is validated for unknown keys before Protobuf conversion for precise errors.
- Defaults: centralized `normalize_defaults(...)` functions set numeric and duration defaults only; do not override explicit booleans. Use `optional` fields in `.proto` where presence is required.
- Units: `*_bytes` accept plain integers; loaders may support `KB/MB/GB` suffixes. Durations use `google.protobuf.Duration` or loaders accept `ms/s/m` with strict parsing.
- Single flag: processes accept only `--config=/path/to/file`; if missing, the process exits with a clear error and sample path.

# Invariants & Error Model

- Single source of truth: only the configuration file influences runtime behavior for covered areas; ENV and ad‑hoc flags are not read.
- Determinism: all defaults are applied in one place; behavior does not depend on process environment.
- Fail‑fast: unknown fields, type mismatches, and invalid units/durations cause startup failure.
- Cross‑language equivalence: the same file yields identical Protobuf messages in C++ and Python.

# Schema Outline & Conventions

- Authoritative files (do not duplicate field listings here):
  - `proto/tensorcast/config/v1/daemon_config.proto`
  - `proto/tensorcast/config/v1/global_store_config.proto`
  - `proto/tensorcast/config/v1/client_config.proto`
  - `proto/tensorcast/config/v1/common.proto`
- Conventions:
  - Duration fields end with `*_interval`, `*_timeout`, or `*_ttl` and use `google.protobuf.Duration`.
  - Byte sizes end with `*_bytes`; loader may accept humanized suffixes.
  - Enums define `*_UNSPECIFIED = 0` for safe defaults.
  - Deletions retain `reserved` numbers/names for forward/backward compatibility.

# Alternatives & Rationale

- Continue with ENV/flags: rejected due to fragmentation, hidden precedence, and drift between components and languages.
- Hybrid (config + ENV overrides): rejected to keep behavior predictable and debuggable; forbidding overrides avoids surprises in production.
- JSON‑only without Protobuf: rejected; Protobuf provides strong typing, enums, and presence semantics across languages.
- Hot reload: deferred; requires explicit dynamic‑safe field catalog and re‑init semantics, which is out of scope.

# Compatibility & Migration

- Communicator: integrate RFC‑0013 outcomes by embedding `tensorcast.communicator.v1.CommunicatorConfig` under `DaemonConfig.communicator`; remove `--comm_config_path` and related flags/ENV.
- Flags/ENV mapping: legacy flags and environment variables are mapped into explicit fields (e.g., daemon lifecycle intervals, VRAM fraction, OTel/logging, checkpoint streaming, P2P fallback dir). After migration they are ignored by processes.
- Restart boundary: all fields are startup‑only; changes require a process restart. No partial runtime mutation is supported.
- Schema discipline: new runtime behavior enters via `.proto` first; code reads from Protobuf messages only. Deprecated fields are removed with `reserved` guards.

# Risks & Mitigations

- No hot reload reduces operational flexibility.
  - Mitigation: keep config files small and modular per process; document safe restart procedures.
- Misconfiguration causes startup failures.
  - Mitigation: strict validation with precise error messages; example configs under `examples/config/`.
- Cross‑language drift.
  - Mitigation: Protobuf as the sole schema; reuse the same parse/normalize logic; CI type checks and unit tests on loaders.
- Duplicate toggles across sections.
  - Mitigation: consolidate ownership (e.g., RDMA enable lives under `communicator` only; C++‑specific OTel options under `observability.otel_cxx`).

# Acceptance Criteria

- Daemon and Global Store start with only `--config` and reject unknown keys in the file.
- Protobuf schemas exist for `DaemonConfig`, `GlobalStoreConfig`, `ClientConfig` under `tensorcast.config.v1` with common types extracted.
- Loaders in C++ and Python apply the same defaults and unit/duration parsing.
- Environment variables and ad‑hoc flags that previously affected runtime behavior are removed or ignored in favor of config fields.
- Example configurations are available under `examples/config/` for each process.
- Documentation for daemon, global store, and client reflects the single‑file configuration model.

# References

- Communicator config unification (prior art): `tensorcast.communicator.v1.CommunicatorConfig` and `core/communicator/config_io.*`.
- Protobuf sources: `proto/tensorcast/config/v1/*.proto`.
- Python loaders: `tensorcast/global_store/config/settings.py`, `tensorcast/client_config_loader.py`.

