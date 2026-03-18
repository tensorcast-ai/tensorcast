---
slug: daemon-served-directory-and-target-resolution
title: Daemon-Served Directory, Target Resolution, and Bounded-Staleness Target Identity
status: draft
areas: ["sdk", "daemon", "global_store", "proto", "docs"]
created: 2026-03-19
last_updated: 2026-03-19
related_code:
  - tensorcast/capability_directory.py
  - tensorcast/api/signals.py
  - tensorcast/api/runtime.py
  - daemon/state/worker_directory_cache.h
  - daemon/state/worker_directory_cache.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
related_docs:
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0060-tensor-work-queue.md
  - docs/designs/0104-artifact-realization-and-cluster-rollout.md
  - docs/designs/0004-unified-runtime-config.md
  - docs/distributed-coordination-series/01-global-optimum-vs-distributed-execution-framework.md
  - docs/distributed-coordination-series/02-mode-switching-workload-playbook-and-governance.md
links:
  plan: ../plans/0106-daemon-served-directory-and-target-resolution.md
  dependencies:
    - ./0056-programmable-framework-adv.md
    - ./0060-tensor-work-queue.md
    - ./0004-unified-runtime-config.md
---

# Summary

This design freezes one repository-wide directory contract for programmable
target resolution.

It defines:

- daemon-served worker, instance, and `NodeAgentDirectory` read surfaces,
- stable caller-facing target identity by `daemon_id` and `instance_id`,
- bounded-staleness freshness evidence and fail-closed behavior,
- and the migration boundary from SDK-direct Global Store reads to one connected
  daemon front door.

Long-term repository rule:

- external programmable callers resolve targets through the connected daemon,
- `CapabilityDirectoryClient` remains compatibility or debugging surface, not
  the normative programmable front door,
- and no child design may invent a second target-directory dialect beside this
  contract.

# Problem Statement

The repo already has several partial directory shapes:

- SDK-direct `CapabilityDirectoryClient` over Global Store,
- daemon-internal `WorkerDirectoryCache`,
- `GetWorkerStatus` as the first daemon-served signal snapshot,
- and implicit `instance_id` routing assumptions around NodeAgent execution.

Those pieces are directionally compatible, but not yet one contract.

Without a frozen directory owner, adjacent designs keep depending on different
facts:

- `0056` needs daemon-served signals and directory for one runtime front door,
- `0104` needs worker and instance target resolution without direct SDK-to-GS
  calls,
- `0060` needs queue and execution leaders to resolve through bounded-stale
  directory facts,
- and execution unification needs a real `NodeAgentDirectory` before instance
  steps can route through one spine.

# Ownership Boundary

`0106` follows the repository rule that one question has one authority root.

It therefore separates:

- membership truth,
- caller-facing target identity,
- daemon-served directory read models,
- and transport endpoints used to execute the next hop.

| Question | Canonical source or contract | Owner |
| --- | --- | --- |
| which workers or instances are registered in the cluster | Global Store worker/instance registry | Global Store services plus `0056` consumption rules |
| what stable identity callers use for programmable targets | `daemon_id`, `instance_id` | `0056` plus this design |
| what bounded-staleness read model the SDK consumes | daemon-served directory and signals projection | `0106` |
| which dialable endpoint reaches the local NodeAgent for one instance | `NodeAgentDirectory` entry | `0106` |
| whether a worker or instance is currently routable under freshness budget | daemon cache freshness evidence and fail-closed policy | `0106` |

Normative rules:

1. Global Store remains the membership truth root; daemon-served directory is a
   read model over that truth, not a replacement truth layer.
2. `daemon_id` and `instance_id` remain the stable programmable identities;
   endpoints are transport facts derived from directory.
3. `NodeAgentDirectory` answers "how to reach the instance execution host for
   this `instance_id`", not "whether this instance is semantically current" for
   any workflow.
4. worker capacity snapshots are advisory read-model facts only; they are not
   reservation truth.

# Goals / Non-Goals

## Goals

- Freeze stable target identity:
  - worker targets by `daemon_id`
  - instance targets by `instance_id`
- Freeze one daemon-served directory surface for programmable callers:
  - `ListWorkers`
  - `ListInstances`
  - `GetWorkerCapacity`
- Freeze one explicit `NodeAgentDirectory` contract for instance execution:
  - `instance_id -> {daemon_id, node_agent_endpoint}`
- Require freshness evidence:
  - `as_of_ms`
  - `staleness_ms`
  - `cache_epoch`
  - `freshness_state`
- Define fail-closed behavior when freshness budget is exceeded.
- Keep directory facts low-cardinality and advisory rather than turning Global
  Store into per-item hot truth.

## Non-Goals

- Define worker-local or cluster workflow truth.
- Define rollout barriers or rollout ownership. `0104` remains the owner there.
- Replace lifecycle truth, routing truth, or operation truth with directory
  snapshots.
- Introduce a second public discovery client beside the connected daemon front
  door.

# Architecture & Interfaces

## 1. Stable caller-facing identity

Caller-facing target identity is configuration-derived and transport-independent.

Rules:

1. worker targets are named by `daemon_id`, not by address,
2. instance targets are named by `instance_id`, not by `node_agent_endpoint`,
3. `node_agent_endpoint` is transport detail resolved by directory, not stable
   user target identity,
4. user-facing APIs must not overload one string field to mean identity,
   endpoint, and label at the same time.

Read-model rule:

- the same stable `instance_id` may resolve to a different transport endpoint
  over time without changing target identity,
- and that endpoint change is a directory refresh fact, not a new workflow or
  artifact truth domain.

## 2. Directory surfaces

Recommended public SDK shape:

```python
class TensorCastSignals:
    def list_workers(...) -> SignalSnapshot[list[Worker]]: ...
    def list_instances(...) -> SignalSnapshot[list[Instance]]: ...
    def get_worker_capacity(...) -> SignalSnapshot[WorkerCapacitySnapshot]: ...
```

Recommended daemon-owned internal model:

```python
@dataclass(frozen=True, slots=True)
class NodeAgentDirectoryEntry:
    instance_id: str
    daemon_id: str
    node_agent_endpoint: str
    engine: str | None = None
    capability_flags: int = 0
```

Rules:

1. external callers query the connected daemon only,
2. daemon caches may be bounded-staleness, but must surface freshness evidence,
3. stale-beyond-budget reads fail closed rather than silently guessing,
4. addresses and NodeAgent endpoints are resolved from directory caches and are
   never promoted to stable workflow identity.

SoT rule:

- daemon-served directory answers where a target can be reached within a
  bounded-staleness budget,
- but it does not answer artifact existence, lifecycle protection, workflow
  currentness, or rollout completion.

## 3. Freshness contract

Minimum public freshness evidence:

- `as_of_ms`
- `staleness_ms`
- `cache_epoch`
- `freshness_state`

Minimum internal watch correctness floor:

- initial snapshot barrier,
- monotonic cache epoch or equivalent version,
- resume token or equivalent replay cursor,
- fail-closed behavior when freshness cannot be re-established within budget.

## 4. Migration from SDK-direct GS reads

Current repo reality:

- `CapabilityDirectoryClient` is still useful as the semantic baseline for cache
  behavior and staleness control,
- but it is not the long-term programmable front door.

Migration rule:

1. daemon-side directory caches may reuse the same semantic contract,
2. cross-language components reuse contract, not implementation,
3. new programmable features must not add fresh direct-SDK-to-GS directory
   dependencies,
4. if a compatibility path still uses SDK-direct GS reads, it must be described
   as migration debt rather than as co-equal front-door truth.

# Trade-offs and Risks

- Freezing one directory contract adds up-front design work, but it prevents
  rollout, queue, and execution work from growing separate discovery dialects.
- Requiring fail-closed freshness may surface more transient errors, but it is
  safer than silently routing against stale membership.
- Keeping `node_agent_endpoint` out of stable target identity may feel less
  direct, but it preserves one target model across current and future transports.

# Compatibility & Acceptance Criteria

- programmable callers resolve targets through the connected daemon rather than
  direct Global Store discovery,
- `daemon_id` and `instance_id` remain the only stable caller-facing target
  identifiers,
- `NodeAgentDirectory` is explicit enough to support future instance-step
  routing without caller-supplied endpoints,
- freshness metadata is exposed on daemon-served directory and capacity reads,
- stale-beyond-budget reads fail closed,
- `CapabilityDirectoryClient` remains compatibility tooling only and is not
  extended as the normative front door for new programmable features.

# References

- `0056` for runtime, ingress, and signals ownership.
- `0060` for queue and leader-directory dependencies.
- `0104` for rollout target-resolution dependency.
- distributed coordination series for current strategic grounding and migration
  rationale.
