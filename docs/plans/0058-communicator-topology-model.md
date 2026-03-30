---
slug: communicator-topology-model
title: Plan - Communicator Topology Model and Routing
links:
  design: ../designs/0058-communicator-topology-model.md
areas:
  - core
  - daemon
  - proto
related_code:
  - core/communicator/README.md
  - core/communicator/engine/engine.h
  - core/communicator/engine/channel.h
  - core/communicator/transport/rdma_transport.h
  - core/communicator/transport/mtcp_transport.h
  - core/communicator/docs/comm-read-tensor.md
  - docs/architecture/p2p-transfer-strategies.md
  - proto/tensorcast/communicator/v1/communicator_config.proto
---

# Objective

Introduce a searchable topology model (Pool / Endpoint / Link) and
path-level abstractions (Connection / Channel / Communicator) on top of the
current Communicator implementation, so multipath, topology-aware routing, and
failover share one unified data model and policy entry point while keeping the
current read/write path backward-compatible and rollback-safe.

# Current State & Grounding

- The communication layer centers on `communicator::engine::Communicator`.
  Peer-level connection abstraction is `engine::Channel`, which manages
  TCP/MTCP/RDMA connections and handshake state
  (`core/communicator/engine/engine.h`,
  `core/communicator/engine/channel.h`).
- Data-plane transfer is currently chosen directly by Communicator (RDMA or
  MTCP). Path selection has no explicit topology graph, and there is no
  multi-hop or multipath abstraction
  (`core/communicator/README.md`,
  `core/communicator/docs/comm-read-tensor.md`).
- P2P source selection is coordinated by Global Store, while data transfer is
  still executed by Communicator. There is currently no topology-level routing
  or automatic failover
  (`docs/architecture/p2p-transfer-strategies.md`).
- Existing `CommunicatorConfig` only has lightweight mappings such as
  `simple_numa`; it lacks a topology model that can represent switching
  structures (`proto/tensorcast/communicator/v1/communicator_config.proto`).

# Phases & Milestones

- [ ] Phase 1: Topology model and config skeleton
  - [ ] Milestone 1.1: Add topology data structures (`Pool` / `Endpoint` /
    `Link`) and graph builder (recommended under
    `core/communicator/topology/`), with minimal visualization/debug export.
  - [ ] Milestone 1.2: Extend `CommunicatorConfig` with topology description
    (or equivalent standalone topology config section), and provide
    compatibility mapping from `simple_numa`; follow
    `docs/designs/0004-unified-runtime-config.md`.
  - [ ] Milestone 1.3: Add topology validation and unit tests
    (connectivity, bandwidth/latency fields, Switch Endpoint legality).

- [ ] Phase 2: Execution-layer mapping and compatibility bridge
  - [ ] Milestone 2.1: Add `Connection` and path-level `Channel`
    (to avoid conflict with `engine::Channel`, use `RouteChannel` in a new
    namespace) and complete mapping to existing transport objects.
  - [ ] Milestone 2.2: Build a `Communicator` aggregator (src-dst dimension)
    plus connection-reuse cache, with multi-hop support.
  - [ ] Milestone 2.3: Add baseline stats and health collection
    (Link/Connection level) for later routing decisions.

- [ ] Phase 3: Path search, multipath, and failover
  - [ ] Milestone 3.1: Implement k-shortest or shortest-path search
    (mixed scoring across BW/latency/stats) and output multiple candidate
    channels.
  - [ ] Milestone 3.2: Build `mct_map` feedback loop to support
    message-size-aware path selection and striping.
  - [ ] Milestone 3.3: Add automatic failover on connection/handshake failures
    (switch to alternate channel or default direct path).

- [ ] Phase 4: Product integration and regression validation
  - [ ] Milestone 4.1: Introduce routing selection entry in P2P read path
    (default fallback to existing logic), then gradually integrate with
    `P2PLoader` / `RemoteKeySource`.
  - [ ] Milestone 4.2: Add integration tests and stress regression
    (multi-GPU / multi-NIC / switched-topology simulation), and finish doc
    updates.

## Phase 2 Mapping Decisions (Supplement)

**Objective**: map topology reachability to existing communication execution
without changing current data-plane behavior, while explicitly including NVLINK
as the local GPU-to-GPU transfer path.

**Runtime mapping source**
- Runtime endpoint binding metadata (IP/Port/NIC/GPU UUID, etc.) uses
  **Global Store** as the source of truth.
- The routing layer only consumes node/device metadata from GS and does not
  introduce extra hardware discovery or local probing.

**Transport adapters and integration point**
- Add **NvlinkAdapter** as the only execution entry for NVLINK transport
  (routing/middleware selects it, and execution is completed by
  NvlinkAdapter).
- Existing RDMA/MTCP/TCP execution remains in `engine::Communicator`, reused
  through **EngineAdapter**.

**Mapping rules (minimum viable)**
- Same-node GPU-to-GPU: prefer NVLINK (`NvlinkAdapter`), fallback to existing
  path on failure.
- Cross-node communication: continue using RDMA/MTCP/TCP (`EngineAdapter`).
- Switch Endpoint participates only in path search and does not directly create
  `Connection` objects.

# Tasks

- Update `core/communicator/README.md` with topology and multipath semantics.
- Update `docs/architecture/p2p-transfer-strategies.md` to describe routing and
  failover placement in the data plane.
- If `.proto` is modified, run `bash tools/build_proto_python.sh` and
  `bazel test //proto/...`.
- Add benchmark or debug tooling to export topology graphs and
  path-selection logs.

# Test / Rollout / Backout

- **Unit tests**: topology construction, path search, Switch Endpoint
  constraints, and connection-health updates.
- **Integration tests**: reuse current `tcp_engine_test` / RDMA-related tests,
  plus new multipath and failover scenarios.
- **Regression validation**: run performance comparison on
  `read_tensor` critical path and confirm no regression.
- **Rollout strategy**: add a config switch, default off; first run in
  shadow mode with path-selection logs.
- **Backout strategy**: disable topology-routing config and return to the
  previous direct-channel selection.

# Risks & Tracking

- **Complexity increase**: keep semantic boundaries clear between topology model
  and existing `engine::Channel`.
- **Config compatibility**: topology config must stay compatible with
  `simple_numa` to avoid deployment breakage.
- **Performance uncertainty**: path search and stats updates require bounded
  frequency and caching.

# Owner Checklist

- [ ] Bidirectional links between design and plan are in place.
- [ ] Config changes follow unified config design
  (`docs/designs/0004-unified-runtime-config.md`).
- [ ] Doc updates cover core module README and architecture guides.
- [ ] Benchmark/regression metrics cover the critical path.
