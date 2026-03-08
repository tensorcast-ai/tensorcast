---
slug: 0004-unified-runtime-config
title: Unified Runtime Configuration System (Design)
related_code:
  - proto/tensorcast/config/v1/*.proto
  - core/communicator/config_io.*
  - core/common/config/daemon_config_io.cc
  - tensorcast/global_store/config/settings.py
  - tensorcast/client_config_loader.py
  - tensorcast/daemon_config.py
  - tensorcast/common/config/normalize.py
created: 2025-09-09
last_updated: 2026-03-05
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
  - `engine`: `*_bytes`, streaming and CPU VS sizing, `streaming_buffer_chunks`.
  - `pinned_memory`: daemon-wide pinned budget, class pools, and allocation timeout.
  - `observability`: OTel (lang‑agnostic), logging (enum level, sinks), tracing; `otel_cxx` holds C++‑specific toggles.
  - `compatibility`: targeted compatibility switches (e.g., `confirm_requires_disk_path`, `verification_timeout_status`).
  - `debug.cuda`: guardrails for same‑process IPC fallback.

- `GlobalStoreConfig` (Python Global Store)
  - `database`, `server` (`listen`, worker pool, `grpc`), `worker_policy`, `observability`.

- `NodeAgentConfig` (Python Node Agent)
  - `server` (`listen`, `grpc`), `daemon_target`, `identity` (daemon_id/instance_id/engine),
    `global_store` endpoints + heartbeat, `engine_adapter` (target TTL + plugins), `observability`.

- `ClientConfig` (Python client/CLI)
  - `daemon.target` (`SocketAddress`), `bin_path`, `python_interpreter`, `storage.default_root`, client defaults and observability.

Common types live in `proto/tensorcast/config/v1/common.proto` (e.g., `SocketAddress`, `GrpcServer`, `Logging`, `Observability`, `ConfigMeta`).

# Loader Semantics

- Parse path (C++ and Python): YAML → in‑memory tree → JSON → `google::protobuf::util::JsonStringToMessage`; JSON loads directly.
- Unknown keys: hard fail (`ignore_unknown_fields=false`). YAML tree is validated for unknown keys before Protobuf conversion for precise errors.
- Defaults: centralized `normalize_defaults(...)` functions set numeric and duration defaults only; do not override explicit booleans. Use `optional` fields in `.proto` where presence is required.
- Units: `*_bytes` accept plain integers; loaders may support `KB/MB/GB` suffixes. Durations use `google.protobuf.Duration` or loaders accept `ms/s/m` with strict parsing.
- Single flag: processes accept only `--config=/path/to/file`; if missing, the process exits with a clear error and sample path.
- CLI/SDK overlays: the orchestrator may materialize an effective config by applying CLI overlays (including convenience flags and `--set`) to a base config, but the daemon still receives a single config file and enforces the same schema/validation.

## Normalization & Aliases (Cross‑Language)

To improve operator ergonomics while keeping Protobuf as the single source of truth, loaders normalize a small set of user‑friendly values into canonical enum names and duration/size formats before Protobuf parsing.

- Enum aliases
  - Accepted friendly values (case‑insensitive):
    - Observability.OTelProtocol: `grpc`, `http/protobuf`
    - Observability.LogLevel: `debug`, `info`, `warn`, `warning`, `error`
  - Canonicalization:
    - C++: `core/common/config/daemon_config_io.cc::normalize_enum_aliases()` maps aliases to enum names (e.g., `O_TEL_PROTOCOL_GRPC`).
    - Python: `tensorcast/common/config/normalize.py` performs descriptor‑driven, generic enum normalization and is used by
      - `tensorcast/global_store/config/settings.py`
      - `tensorcast/client_config_loader.py`
  - Strictness: after normalization, parsing still rejects unknown keys and invalid enum values.

- Durations
  - Protobuf canonical JSON: strings like `"120s"`, `"0.5s"` are accepted everywhere.
  - C++ convenience: the daemon loader accepts `ms/s/m/h` inputs (e.g., `2m`) and rewrites them to canonical Protobuf JSON (`120s`).
  - Python (current): relies on canonical Protobuf JSON duration strings. Inputs like `2m` are not yet normalized; use `120s` for portability.
  - Follow‑up: extend the shared Python normalizer to add duration parsing for full parity with C++.

- Byte sizes
  - C++ daemon loader normalizes humanized sizes (e.g., `256MB`, `8GB`) into bytes for `*_bytes` fields.
  - Python Global Store currently has no byte‑size fields; if introduced, reuse a shared helper for parity.

Tests cover normalization of enum aliases in both Global Store and Client loaders (`tests/python/test_config_enum_normalization.py`).

# Invariants & Error Model

- Single source of truth: only the final configuration file influences runtime behavior for covered areas; ENV and ad‑hoc flags are not read by processes.
- Determinism: all defaults are applied in one place; behavior does not depend on process environment.
- Fail‑fast: unknown fields, type mismatches, and invalid units/durations cause startup failure.
- Cross‑language equivalence: the same file yields identical Protobuf messages in C++ and Python.
- Distributed namespace profiles (for example byte artifact routing invariants such as shard count/hash version/lease
  staleness policy) must be modeled as typed config fields and remain cluster-consistent; incompatible rolling changes
  must have an explicit cutover strategy instead of mixed semantics.

# Schema Outline & Conventions

- Authoritative files (do not duplicate field listings here):
  - `proto/tensorcast/config/v1/daemon_config.proto`
  - `proto/tensorcast/config/v1/global_store_config.proto`
  - `proto/tensorcast/config/v1/node_agent_config.proto`
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
- Flags/ENV mapping: legacy flags and environment variables are mapped into explicit fields (e.g., daemon lifecycle intervals, VRAM fraction, OTel/logging). After migration they are ignored by processes.
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
- Node Agent starts with only `--config` and rejects unknown keys in the file.
- Protobuf schemas exist for `DaemonConfig`, `GlobalStoreConfig`, `ClientConfig` under `tensorcast.config.v1` with common types extracted.
- Loaders in C++ and Python apply the same defaults and equivalent normalization rules:
  - Enum aliases (e.g., `grpc`, `info`) are accepted and canonicalized.
  - Durations use canonical Protobuf strings; C++ additionally accepts `ms/s/m/h` shorthand.
- Environment variables and ad‑hoc flags that previously affected runtime behavior are removed or ignored in favor of config fields.
- Example configurations in `examples/config/store_daemon_config.yaml` and `examples/config/global_store_config.yaml` stay in sync with config changes (add/remove fields or default updates).
- Documentation for daemon, global store, and client reflects the single‑file configuration model.

# References

- Communicator config unification (prior art): `tensorcast.communicator.v1.CommunicatorConfig` and `core/communicator/config_io.*`.
- Protobuf sources: `proto/tensorcast/config/v1/*.proto`.
- Python loaders: `tensorcast/global_store/config/settings.py`, `tensorcast/client_config_loader.py`.
