---
title: Communicator Interface Usage (Aligned)
sidebar_label: Communicator Interface Usage
---

# Communicator Interface Usage (Aligned)

Status: implementation-aligned baseline + target alignment contract (Phase-1/2 implemented).
Date: 2026-02-09.

This document uses two explicit layers:
- Layer A: current implemented contract (as-is behavior in store/daemon paths).
- Layer B: target alignment contract (for topology/routing integration).

Use Layer A as the runtime source of truth. Use Layer B for migration planning.
For a raw callsite inventory snapshot, see `communicator-interface-usage-temp.md`.

## 0) Scope and non-goals

Scope:
- Document communicator APIs currently exercised by product paths.
- Map those APIs to concrete store/daemon callsites.
- Define a non-breaking migration contract toward topology/routing.

Non-goals:
- Claiming full endpoint-coverage routing is already active in product reads.
- Introducing multi-hop routing in MVP.
- Changing engine-level registration/read semantics during alignment.

## 1) Layer A — current implemented contract (as-is)

### 1.1 Communicator API surface used in product paths

| API | Current behavior | Primary callsites |
| --- | --- | --- |
| `engine::Communicator::Communicator(config[, pools])` | Construct communicator with typed config and optional pinned pools. | `core/store/components/communication_manager.cc`, `daemon/app/server_main.cc` |
| `engine::Communicator::init(local_ip, local_port[, conn_count])` | Bind listener and start TCP/RDMA internals. | `core/store/components/communication_manager.cc` |
| `engine::Communicator::listening_port()` | Return actual bound port after `init`. | `core/store/components/communication_manager.cc`, `core/store/runtime/context/runtime_context.cc`, `daemon/app/server_main.cc` |
| `engine::Communicator::register_tensor_ex(...)` | Register exported CPU/GPU ranges for P2P access. | `core/store/replica/memory_export_registry.cc`, `daemon/state/lip_manager.cc`, `core/store/components/communication_manager.cc` |
| `engine::Communicator::unregister_tensor(...)` | Unregister exported ranges; semantically idempotent. | `core/store/replica/memory_export_registry.cc`, `daemon/state/lip_manager.cc` |
| `engine::Communicator::read_tensor(...)` | P2P read by `ip/port + tensor_key` (returns `future_read_result_t`). | `core/store/materialization/dataplane/sources/remote_key_source.cc` |
| `engine::Communicator::is_rdma_enabled()` | Gate direct-RDMA/staging behavior. | `core/store/replica/memory_export_registry.cc`, `core/store/components/communication_manager.cc`, `core/store/materialization/dataplane/sources/remote_key_source.cc`, `daemon/state/lip_manager.cc` |
| `routing::RoutingContext::get_communicator(src_endpoint, dst_endpoint)` | Routed communicator lookup for endpoint-aware reads (optional path with strict direct fallback). | `core/store/materialization/dataplane/sources/remote_key_source.cc` |

Notes:
- `unregister_tensor` idempotency is implemented in `core/communicator/engine/engine.cc` (missing key returns OK).
- `close_connection(...)` is not used in store/daemon product paths (only tests and routing adapter scaffolding).
- `set_dram_lease_provider(...)` / `set_residency_provider(...)` currently have no product callsites.
- RDMA NIC selection (`RdmaContext::get_best_dev`) now degrades safely on sparse HCA sets: no visible RDMA device returns `nullptr` with warnings, and GPU affinity selection falls back to max PCI-prefix candidates when `max GID + max prefix` intersection is empty.
- RDMA handshake failure signaling is now strict and backward-safe: server-side connect failures always send `ENGINE_OP_RDMA_CONNECT_FAILED`, and client-side response handling validates payload size so malformed/legacy failure payloads are treated as connect-failed instead of crashing in payload decode.

### 1.2 Initialization and wiring

- Store runtime initializes communicator through `CommunicationManager` unless an external manager is injected:
  - `core/store/runtime/context/runtime_context.cc`
  - `core/store/components/communication_manager.cc`
- Daemon startup builds pinned staging pools and initializes communicator with daemon config:
  - `daemon/app/server_main.cc`
  - `core/store/components/communication_manager.cc`

### 1.3 Data-plane flows currently active

1) Publisher export registration:
- `MemoryExportRegistry::export_chunks(...)` registers per-range keys through `register_tensor_ex(...)`.
- CPU and GPU options differ by RDMA availability.
- File: `core/store/replica/memory_export_registry.cc`

2) Publisher cleanup:
- `MemoryExportRegistry::unexport_chunks(...)` iterates keys and calls `unregister_tensor(...)`.
- File: `core/store/replica/memory_export_registry.cc`

3) Consumer P2P read:
- `MaterializationFacade` requests transport from GS and populates `P2PSource` from `remote_replica` (`node_address`, `node_port`, `endpoint_id`, `remote_memory_keys`, `buffer_sizes`), then injects local `endpoint_id` and optional `routing_context`.
- `P2PLoader` builds `RemoteKeySource` with `ip/port` + endpoint metadata + key vectors.
- `RemoteKeySource` uses routed-first read when `local_endpoint_id`/`remote_endpoint_id` + `routing_context` are all present; on any routed failure it immediately falls back to direct `comm_engine->read_tensor(...)` and keeps direct mode for the rest of this source instance.
- Files:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `core/store/materialization/dataplane/loaders/p2p_loader.cc`
  - `core/store/materialization/dataplane/sources/remote_key_source.cc`

4) Daemon LIP export:
- LIP staged export registers/unregisters GPU keys via communicator.
- File: `daemon/state/lip_manager.cc`

### 1.4 Current metadata contract used by read path

Active read path currently consumes:
- `P2PSource.ip`
- `P2PSource.port`
- `P2PSource.local_endpoint_id` (optional)
- `P2PSource.remote_endpoint_id` (optional)
- `P2PSource.routing_context` (optional)
- `P2PSource.memory_keys`
- `P2PSource.buf_sizes`
- `P2PSource.size_bytes`

Current structures and conversion path:
- `P2PSource` carries `local_endpoint_id` / `remote_endpoint_id` and optional `routing_context`: `core/store/communication_types.h`
- `RemoteReplicaInfo` carries `endpoint_id` + `node_address/node_port` and key vectors:
  - `core/store/components/global_store_client.h`
  - `core/store/components/global_store_client.cc`

Identity note:
- Daemon node identity defaults to hostname-based derivation (`derive_node_id`) and is propagated into control-plane registration.
- Global Store client memory export path also falls back to hostname when node id is unset.

### 1.5 Topology/routing state today (implemented and partially wired)

Modules exist:
- `core/communicator/topology/topology.h`
- `core/communicator/routing/routing_context.h`
- `core/communicator/routing/types.h`

Current code constraints:
- `RoutingContext::set_topology(...)` is one-shot immutable.
- `RoutingContext::set_endpoint_bindings(...)` is one-shot immutable.
- `RoutingContext::update_endpoint_binding(...)` returns `FAILED_PRECONDITION`.
- `RouteChannel` supports direct 1-hop only; multi-hop read returns `UNIMPLEMENTED`.
- When a direct topology link is missing for cross-node traffic, `RoutingContext` now attempts a rail-matched NIC fallback:
  - infer source/destination device pool hints (`gpu<id>` / `cpu<id>`) from endpoint ids or binding metadata,
  - resolve local NIC by source rail + GPU/CPU pool affinity from topology,
  - choose a destination-node binding with network address via weighted rail-aware scoring (`preferred rail`, `destination rail`, topology NIC presence, GPU/CPU pool affinity, endpoint-id/NIC heuristics),
  - build a single-hop channel anchored on the selected local NIC link.
- This fallback is additive and keeps the same strict direct read fallback in `RemoteKeySource` if routed lookup/read fails.
- Local-fabric adapter matrix:
  - `NvlinkAdapter` is wired for protocol selection and executes through `engine::Communicator::read_tensor_local(...)` when the source tensor is present in the same process.
  - `PcieAdapter` is wired for protocol selection and executes via current engine-backed path.
  - same-node protocol selection is deterministic: NVLINK first (`prefer_nvlink` + adapter available), then PCIE (`prefer_pcie` + adapter available), else `AUTO`.

Important:
- Store read paths now call `RoutingContext::get_communicator(...)` in routed-first mode when endpoint metadata and context are available.
- Direct `ip/port` read remains the compatibility baseline and is always the fallback path.
- Daemon-side product paths are still direct `engine::Communicator` flows for export/register operations.

## 2) Layer B — target alignment contract (not fully wired yet)

### 2.1 Target identity/binding model

- Introduce per-device `endpoint_id` as additive identity.
- Recommended format: `<node_id>/dev/<dev_type>/<dev_id>`.
- Target binding shape:
  - `{ endpoint_id, node_id, ip, port, dev_type, dev_id }`
- Once integrated, GS should distribute these bindings as routing inputs.

### 2.2 Target read-path precedence

After integration, read path should follow:
1) If `endpoint_id` is present and binding lookup succeeds:
   - resolve route via `RoutingContext::get_communicator(...)`
   - call routed `read_tensor(ReadRequest)`
2) Otherwise:
   - fallback to current direct `engine::Communicator::read_tensor(...)` with `ip/port`

Fallback is mandatory until endpoint coverage is complete.

### 2.3 Target registration compatibility

- Keep `register_tensor_ex(...)` / `unregister_tensor(...)` as stable base APIs.
- Preserve `unregister_tensor` idempotent behavior.
- Routing/topology integration should adapt around existing registration semantics.

### 2.4 Target rollout phases

1) ✅ Add `endpoint_id` to metadata while preserving `ip/port`.
2) ✅ Add optional routing read path with strict fallback.
3) ⏳ Enable routing by default only for endpoints with valid bindings.
4) Optionally deprecate `ip/port` only after full coverage and validation.

## 3) Change-control checklist

When behavior changes, keep this document aligned with:
- `core/store/materialization/**` for consumer read wiring.
- `core/store/replica/**` and `daemon/state/**` for registration/unregistration.
- `core/communicator/routing/**` for routing capability and limits.
