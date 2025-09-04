# 0013-Communicator Config Unification (Proto + YAML)

## Overview

- Problem: Communicator configuration is defined twice (C++ header structs and protobuf), defaults are scattered, and initialization relies on flags/ENV. This causes drift, conversions, and operational friction.
- Goals:
  - Single source of truth: protobuf schema for communicator config.
  - First-class YAML init: ops-friendly, file-based, minimal knobs.
  - Remove communicator-related ENV/flags to reduce divergence.
  - Keep code simple: typed config end-to-end (C++/Python).
  - Align namespaces: change proto package to `tensorcast.communicator`.
- Non‑goals (for this RFC): Migrating the entire StoreDaemonConfig to proto. We will keep existing Python Pydantic model for daemon settings for now and focus on communicator.

## Current Architecture Analysis

- Duplicate definitions
  - Protobuf: `proto/communicator_config.proto` (Python generated at `tensorcast/proto/communicator_config_pb2.py`).
  - C++: `core/communicator/engine/communicator_config.h` (hand-rolled structs).
- Usage
  - Engine consumes `CommunicatorConfig` (C++ struct) directly: ```1:120:core/communicator/engine/engine.h```
  - CommunicationManager builds/forwards config: ```1:220:core/store/components/communication_manager.cc```
  - Python binding manually maps `dict -> C++ structs`: ```200:1200:tensorcast/csrc/store_engine_py.cc```
  - Daemon flags (to be removed for communicator): ```1:240:daemon/server_main.cc```
- Pain points
  - Drift risk: fields exist in C++ struct but not in proto (e.g., RDMA QP tuning, TCP TOS/connect timeout).
  - Inconsistent defaults: spread across comments and code; difficult to reason about.
  - Operational complexity: flags/ENV vs files; no single artifact to review/version.

## Proposed Solution

1) Protobuf as the single schema
- Update `proto/communicator_config.proto`:
  - Change package to `package tensorcast.communicator;` to match C++ namespace.
  - Add missing fields present in C++ structs:
    - `RdmaConfig`: `traffic_class`, `qp_timeout`, `qp_retry` (int32).
    - `TransportConfig`: `tcp_tos`, `connect_timeout_sec` (int32).
  - Keep existing structures: `StagerConfig`, `PoolConfig`, `TransportConfig`, `AffinityConfig`, `SimpleNumaConfig`, `CommunicatorConfig`.
  - Defaults remain implemented in code via normalization helpers (proto3 has no defaults).

2) Generate C++ protobuf and use it end‑to‑end
- Bazel: add a C++ proto target for communicator config (e.g., `cpp_proto_library(name = "communicator_config_cc", protos = [":communicator_config_proto"])`).
- Delete `core/communicator/engine/communicator_config.h` entirely.
- Include `proto/communicator_config.pb.h` directly where needed and use `tensorcast::communicator::*` types.
- Engine/manager consume the protobuf message directly, storing a copy of the proto as `config_`.

3) YAML/JSON config ingestion (C++)
- Add `core/communicator/config_io.{h,cc}`:
  - `absl::StatusOr<tensorcast::communicator::CommunicatorConfig> LoadCommunicatorConfigFromFile(const std::string& path);`
    - Detect `.yaml/.yml` or `.json`.
    - YAML: parse via `yaml-cpp` to a generic tree, serialize to JSON text (using `nlohmann::json`), then `google::protobuf::util::JsonStringToMessage(...)` to fill the proto.
    - JSON: directly `JsonStringToMessage`.
  - `void NormalizeDefaults(tensorcast::communicator::CommunicatorConfig* cfg);` centralizes defaults.
- Bazel deps: add `yaml-cpp`, `nlohmann_json`, and Protobuf util to the new library.

4) Operational model: file‑only configuration for communicator
- Remove all communicator-related flags/ENV in the daemon and lower layers.
- Daemon bootstrap supports exactly one config artifact input (file-based). Two options:
  - Preferred: reuse the existing top-level `StoreDaemonConfig` YAML (Pydantic). The Python launcher reads it, and passes only the communicator sub-config to C++ as a file path or serialized bytes. This keeps daemon settings under Python for now while communicator uses proto.
  - Alternate: allow C++ daemon to read a dedicated communicator YAML file if launched standalone (for tests/dev), also using the same schema and normalizer.
- Result: no per-field flags/ENV for communicator; a single YAML governs it.

5) Python support (optional quality of life)
- Add `tensorcast/communicator/config_io.py` to help users validate/edit communicator YAML in Python:
  - `from_yaml(path) -> CommunicatorConfig` (using `google.protobuf.json_format` behind the scenes).
  - `to_yaml(proto) -> str` for debugging/export.
- Pybind extension: add convenient constructors
  - `CommunicationManager.from_yaml(listen_addr, port, path)` → internally calls the C++ file loader + normalizer.
  - `CommunicationManager.from_proto_json(listen_addr, port, json_str)` to support direct programmatic use.

6) Namespaces and packages
- With `package tensorcast.communicator;` the generated C++ namespace becomes `tensorcast::communicator`, aligning with engine code.
- Python still imports `tensorcast/proto/communicator_config_pb2.py` (module file path unchanged), messages will have the correct `.DESCRIPTOR.package` for clarity.

### Configuration Flow (Mermaid)

```mermaid
flowchart LR
  subgraph Ops
    A[YAML/JSON file]
  end
  subgraph C++
    B[config_io: yaml-cpp / nlohmann]
    C[Proto util: JsonStringToMessage]
    D[CommunicatorConfig (proto)]
    E[NormalizeDefaults]
    F[Engine / CommManager]
  end
  A --> B --> C --> D --> E --> F
```

## Implementation Plan

### Phases and Milestones

1. Schema alignment (P0)
- Change proto package to `tensorcast.communicator`.
- Add missing fields (`RdmaConfig.traffic_class/qp_timeout/qp_retry`, `TransportConfig.tcp_tos/connect_timeout_sec`).
- Regenerate Python and add C++ proto targets.

2. Adapter + loader (P1)
- Replace `core/communicator/engine/communicator_config.h` with pb aliases + `NormalizeDefaults`.
- Add `core/communicator/config_io.{h,cc}` and BUILD rules; vendor/enable `yaml-cpp`.

3. Engine integration (P2)
- Update engine and manager to store/consume proto message.
- Add initializer from YAML path for tests; update unit tests to use pb or YAML where it improves clarity.

4. Daemon bootstrap change (P3)
- Remove communicator-related flags/ENV from daemon; accept a config file (provided by Python launcher) and wire communicator from file only.
- Keep non-communicator flags as-is (daemon-wide migration to proto/YAML is out-of-scope here and covered in a future RFC).

5. Python QoL and deprecation (P4)
- Add Python helper (`tensorcast/communicator/config_io.py`) and pybind shortcuts.
- Deprecate dict-based communicator config constructors in bindings; migrate tests.

6. Cleanup (P5)
- Remove any residual struct references in C++/Python.
- Documentation updates, examples, and operational playbooks.

### File Modification Table

| File | Action | Notes |
|------|--------|-------|
| `proto/communicator_config.proto` | Update | Change package; add missing fields; comments updated to reflect defaults.
| `proto/BUILD` | Update | Add C++ proto targets for communicator config.
| `tools/build_proto_python.sh` | Verify | Ensure regeneration covers communicator; no changes expected beyond package.
| `core/communicator/engine/communicator_config.h` | Replace | Thin adapter over generated pb + helpers; remove duplicate structs.
| `core/communicator/config_io.h, .cc` | Add | YAML/JSON loader + `NormalizeDefaults`.
| `core/communicator/BUILD` | Update | New targets and deps (`yaml-cpp`, `nlohmann_json`, proto util).
| `core/communicator/engine/engine.h/.cc` | Update | Store `CommunicatorConfig` (proto) and use fields directly; remove env/flag fallbacks.
| `core/store/components/communication_manager.h/.cc` | Update | Ensure signatures use the proto type; keep convenience overloads.
| `daemon/server_main.cc` | Update | Remove communicator flags/ENV; read communicator config from file via loader.
| `tensorcast/csrc/store_engine_py.cc` | Update | Switch dict → proto-based path under the hood; add `from_yaml` utility.
| `tensorcast/communicator/config_io.py` | Add | Python helpers for YAML/proto conversions (optional but recommended).
| `AGENTS.md` | Update | Reference unified config and regeneration step.

### API Changes

- C++
  - Internal: `CommunicatorConfig` now refers to the proto message; include path changes to `communicator_config.pb.h`.
  - New: `LoadCommunicatorConfigFromFile(path)`, `NormalizeDefaults(CommunicatorConfig*)`.
  - Removed: direct use of C++ struct fields from the old header (migrate call sites).
- Python
  - New: `CommunicationManager.from_yaml(listen_addr, port, path)`.
  - Deprecated: dict-style communicator config construction in pybind (to be removed after migration).
- Daemon
  - Removed: communicator-related flags/ENV (e.g., `--enable_p2p_engine`, `--enable_rdma` for communicator). Bootstrap uses file-only config for communicator.

## Trade-offs and Alternatives

- Alternative: Keep C++ structs and generate them from proto (codegen). Rejected for complexity; direct use of proto is simpler and consistent.
- Alternative: Use JSON only (no YAML). Rejected for ops ergonomics; YAML is standard for infra teams.
- Keeping a single `--config` bootstrap flag vs zero flags: we will allow a single bootstrap config path flag initially (or default search paths) to avoid hardcoding locations; all communicator knobs live in the file.

## Testing Strategy

- Unit Tests
  - `config_io` parsing: YAML and JSON parity; default normalization; error cases (unknown fields, wrong types).
  - Engine construction with proto: RDMA enabled/disabled paths, TCP staging behavior.
- Integration Tests (Bazel + Fake CUDA)
  - End-to-end TCP and (optional) RDMA flows configured solely via YAML.
  - NUMA mapping examples (`simple_numa.nodes`) and staging pool sizing.
- Python Tests
  - YAML → proto conversion using `json_format`; pybind `from_yaml` constructor.
- Back-compat verification
  - Tests formerly constructing C++ structs migrate to proto equivalents or builders.

## Rollout Plan

1. Land schema and build changes; regenerate protobufs (Python/C++).
2. Introduce adapter and loader; keep old struct includes temporarily for transitional compilation (type alias path).
3. Switch engine and manager to proto; update C++ tests; pass with Fake CUDA.
4. Remove communicator flags/ENV in daemon; wire YAML loading; update examples and docs.
5. Add Python helpers; migrate Python tests; deprecate dict constructor.
6. Cleanup and remove dead code.

## Progress Tracking

| Phase | Task | Status | Notes |
|------:|------|:------:|-------|
| P0 | Proto package + fields | ✅ Done | Package `tensorcast.communicator`; added RDMA + transport fields |
| P1 | Adapter + loader | ✅ Done | Thin adapter header; `config_io` with YAML/JSON + defaults |
| P2 | Engine integration | ✅ Done | Engine/manager now use proto; removed ENV reads |
| P3 | Daemon bootstrap | ✅ Done | File-only via `--comm_config_path`; flags removed |
| P4 | Python helpers | ✅ Done | Pybind `from_yaml`; Python `tensorcast/communicator/config_io.py` |
| P5 | Cleanup | ✅ Done | Removed C++ struct duplicates; docs updated |

## Success Criteria

- Single schema: No duplicated communicator config structs in C++.
- File-first ops: A minimal YAML fully initializes communicator; no ENV/flags required.
- Build health: All C++/Python tests pass with Fake CUDA (and CUDA when available).
- Maintainability: New fields added once (proto) and reflected across C++/Python by regeneration.

## Code References

- Proto schema (to be updated): `proto/communicator_config.proto`
- C++ engine uses config: `core/communicator/engine/engine.h`, `core/communicator/engine/engine.cc`
- Manager wires config: `core/store/components/communication_manager.h/.cc`
- Daemon entry: `daemon/server_main.cc`
- Python binding touch points: `tensorcast/csrc/store_engine_py.cc`

## Open Questions

- Should we introduce default search paths for the communicator YAML (e.g., `/etc/tensorcast/communicator.yaml`, `$XDG_CONFIG_HOME/tensorcast/communicator.yaml`) to avoid a bootstrap `--config` flag entirely?
- For the top-level StoreDaemon config, when we later migrate to proto, do we embed CommunicatorConfig directly or reference it (e.g., Any/oneof)? This is deferred but should consider cross-language ergonomics.
