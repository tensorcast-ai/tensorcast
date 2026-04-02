---
slug: strategy-guided-topology-plane
title: Strategy-Guided Topology Plane Plan
status: proposed
areas: ["core", "daemon", "proto", "docs", "tests"]
created: 2026-04-02
last_updated: 2026-04-02
related_code:
  - docs/designs/0109-strategy-guided-topology-plane.md
  - core/communicator/topology/discovery/host_topology_builder.cc
  - core/communicator/routing/routing_context.cc
  - core/store/components/communication_manager.h
  - core/store/components/communication_manager.cc
  - core/store/communication_types.h
  - core/store/materialization/dataplane/loaders/p2p_loader.cc
  - core/store/materialization/dataplane/sources/remote_key_source.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/materialization/runtime/pipeline/source_adapter.cc
  - daemon/app/server_main.cc
  - proto/tensorcast/config/v1/daemon_config.proto
  - tools/testing/topology_guided_routing_2node_h800_smoke.sh
  - core/store/materialization/dataplane/sources/tests/remote_key_source_routing_fallback_test.cc
  - core/communicator/routing/routing_context_test.cc
links:
  design: ../designs/0109-strategy-guided-topology-plane.md
---

# Objective

Land a product-usable strategy-guided topology plane that:

- bootstraps a live topology runtime into store and daemon paths,
- lowers `0108` semantic plans into topology-aware transfer groups,
- preserves current routed and direct fallback behavior,
- enables remote bootstrap plus same-node fanout as a first-class execution
  shape,
- keeps grouped topology strategy internal to the runtime rather than exposing a
  new public API.

# Current State & Grounding

- communicator topology, discovery, and routing exist:
  - `core/communicator/topology/topology.h`
  - `core/communicator/topology/discovery/host_topology_builder.cc`
  - `core/communicator/routing/routing_context.cc`
- product data path can already consume routing metadata when present:
  - `core/store/communication_types.h`
  - `core/store/materialization/dataplane/sources/remote_key_source.cc`
  - `core/store/materialization/runtime/pipeline/source_adapter.cc`
- `CommunicationManager` has a routing-context slot but no non-test bootstrap
  currently fills it:
  - `core/store/components/communication_manager.h`
  - `core/store/components/communication_manager.cc`
- `0108` already gives the correct semantic insertion point:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `core/store/runtime/ingestion/materialization_strategy_types.h`
- existing routing coverage already proves the target grouped shape:
  - cross-node bootstrap plus same-node GPU or memory fanout in
    `core/communicator/routing/routing_context_test.cc`
- existing smoke already validates topology-guided communication outside the
  store runtime:
  - `tools/testing/topology_guided_routing_2node_h800_smoke.sh`

Root-cause gap:

- TensorCast has route execution but not strategy-owned route grouping.
- Topology discovery exists but is not bootstrapped into product runtime.
- `ChunkAwareLoadingStrategy` is not the right seam for future work because it
  predates `0108` and is too concrete.

# Phases & Milestones

- [ ] Phase 1: Runtime Topology Bootstrap
  - [ ] Milestone 1.1: Build a product bootstrap path that constructs
    `RoutingContext` from the communicator engine plus typed topology discovery
    inputs during daemon or runtime startup.
  - [ ] Milestone 1.2: Install local endpoint bindings and remote peer bindings
    into the live context from existing directory or transport metadata.
  - [ ] Milestone 1.3: Inject the live context into `CommunicationManager` and
    ensure product `P2PSource` creation sees it by default.

- [ ] Phase 2: Strategy Contracts
  - [ ] Milestone 2.1: Add internal planner types aligned with the design:
    - `TopologyRuntimeSnapshot`
    - `TopologyStrategyInput`
    - `RouteIntent`
    - `TransferLeg`
    - `TransferGroup`
    - `TopologyGuidedPlan`
  - [ ] Milestone 2.2: Keep these types internal to common runtime; do not add
    new SDK or generic daemon request surfaces.
  - [ ] Milestone 2.3: Define one diagnostics payload shape that can explain
    guided vs degraded planning decisions.

- [ ] Phase 3: Topology-Guided Planner In `0108`
  - [ ] Milestone 3.1: Add planner invocation inside
    `MaterializationFacade` after semantic truth and source binding are known.
  - [ ] Milestone 3.2: Keep the planner purely lowering-oriented:
    it must not query Global Store directly or bypass existing source
    acquisition.
  - [ ] Milestone 3.3: Ensure the planner can emit:
    - direct remote per-target,
    - remote bootstrap then local fanout,
    - owner-file collective,
    - residual generic fallback.

- [ ] Phase 4: Remote Bootstrap Then Local Fanout
  - [ ] Milestone 4.1: Implement anchor selection for multi-target same-node
    requests.
  - [ ] Milestone 4.2: Lower one grouped remote ingress leg into existing P2P
    machinery with routed-first semantics.
  - [ ] Milestone 4.3: Lower same-node fanout legs into existing local copy or
    tensor-aware execution paths.
  - [ ] Milestone 4.4: Preserve exact residual fallback for bytes not safely
    covered by the grouped plan.

- [ ] Phase 5: Configuration and Rollout
  - [ ] Milestone 5.1: Add typed config fields under
    `engine.materialization_strategy` for topology-guided transfer mode.
  - [ ] Milestone 5.2: Support `DISABLED`, `OBSERVE_ONLY`, and
    `PREFER_GUIDED` rollout behavior.
  - [ ] Milestone 5.3: Keep topology discovery configuration under
    `CommunicatorConfig`; do not introduce env-only knobs.

- [ ] Phase 6: Observability and Safety
  - [ ] Milestone 6.1: Add runtime logs and metrics for:
    - transfer-group kind,
    - anchor target,
    - routed bytes,
    - local fanout bytes,
    - residual fallback bytes,
    - degrade reason.
  - [ ] Milestone 6.2: Add no-progress and stale-route warnings consistent with
    existing fail-fast guidance for async systems.
  - [ ] Milestone 6.3: Prove that disabling guidance expands fallback instead of
    changing semantic outputs.

- [ ] Phase 7: Integration and Regression Coverage
  - [ ] Milestone 7.1: Extend routing and P2P regression coverage from test-only
    routing contexts to product bootstrap wiring.
  - [ ] Milestone 7.2: Add common-runtime tests for grouped remote bootstrap and
    local fanout lowering.
  - [ ] Milestone 7.3: Keep communicator smoke passing on the two-node H800
    topology-guided routing workflow.

- [ ] Phase 8: Optional Scheduler Follow-Up
  - [ ] Milestone 8.1: Evaluate whether `0083` should accept optional topology
    hints after daemon-local topology-guided planning is stable.
  - [ ] Milestone 8.2: Keep Global Store scheduling separate unless measured
    evidence shows local-only planning is insufficient.

# Tasks

- Add the new design and keep this plan linked bidirectionally.
- Add internal strategy-plan types under common runtime ownership rather than in
  communicator or old planner packages.
- Keep `RemoteKeySource` and `P2PLoader` as execution targets, not policy
  owners.
- Narrow future use of `ChunkAwareLoadingStrategy`; do not expand it as the new
  canonical planner surface.
- Update developer-facing indexes when the design and plan are added.

# Test / Rollout / Backout

## Test matrix

- communicator routing unit coverage:
  - `bazel test //core/communicator:routing_context_test --test_output=errors`
- routing-aware remote source fallback coverage:
  - `bazel test //core/store/materialization/dataplane/sources/tests:remote_key_source_routing_fallback_test --test_output=errors`
- common-runtime P2P ingestion coverage:
  - `bazel test //core/store/materialization/runtime/pipeline/tests:p2p_ingestion_test --test_output=errors`
- daemon config parsing coverage after new config fields:
  - `bazel test //core/common:daemon_config_io_test --test_output=errors`
- manual or staged smoke:
  - `tools/testing/topology_guided_routing_2node_h800_smoke.sh`

## Rollout

1. Land runtime bootstrap and keep `enable_topology_guided_transfer=false`.
2. Enable `OBSERVE_ONLY` in benchmark or chaos lanes and collect diagnostics.
3. Enable `PREFER_GUIDED` only after grouped bootstrap and residual fallback
   results are digest-identical to the baseline.
4. Consider scheduler follow-up only after daemon-local planning is stable.

## Backout

- set topology-guided mode back to `DISABLED`;
- keep topology discovery and routing bootstrap code dormant but intact;
- retain routed direct-read fallback and disk fallback behavior.

# Risks and Owner Checklist

- [ ] Product bootstrap does not leave `routing_context` test-only anymore.
- [ ] No new ambient environment variables are introduced.
- [ ] `0108` remains the only strategy-plane boundary.
- [ ] Residual fallback accounting stays explicit.
- [ ] Developer docs are updated with the new design and plan links.
