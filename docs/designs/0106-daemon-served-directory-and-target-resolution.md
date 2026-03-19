---
slug: daemon-served-directory-and-target-resolution
title: Daemon-Served Routing Directory and Bounded-Staleness Target Resolution
status: draft
areas: ["sdk", "daemon", "global_store", "proto", "docs"]
created: 2026-03-19
last_updated: 2026-03-19
related_code:
  - tensorcast/capability_directory.py
  - tensorcast/api/plan/plan.py
  - tensorcast/api/signals.py
  - tensorcast/api/runtime.py
  - tensorcast/global_store/models/instance.py
  - tensorcast/global_store/repositories/instance_repository.py
  - tensorcast/global_store/services/instance_service.py
  - daemon/state/worker_directory_cache.h
  - daemon/state/worker_directory_cache.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - schema.sql
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

This design freezes one repository-wide routing read-model contract for
programmable target resolution.

It defines:

- one daemon-served front door for worker and instance route discovery,
- stable caller-facing target identity by `daemon_id` and `instance_id`,
- one explicit split between stable principal identity and transport facts,
- bounded-staleness freshness evidence and fail-closed behavior,
- authority modes for both Global-Store-backed and local-only runtimes,
- one migration path from today's `signals_endpoint` and SDK-direct GS reads to
  an explicit instance execution endpoint fact,
- and the migration boundary from SDK-direct Global Store reads to one connected
  daemon front door.

Long-term repository rule:

- external programmable callers resolve route facts through the connected
  daemon,
- `CapabilityDirectoryClient` remains compatibility or debugging surface, not
  the normative programmable front door,
- `0060` and later workflow owners may define their own workflow principals, but
  they must reuse this design for `daemon_id -> route` mapping instead of
  inventing a second worker-directory dialect,
- and no child design may invent a second stable-identity-to-endpoint contract
  beside this one.

# Problem Statement

The repo already has several partial directory shapes:

- SDK-direct `CapabilityDirectoryClient` over Global Store,
- daemon-internal `WorkerDirectoryCache`,
- `GetWorkerStatus` as the first daemon-served signal snapshot,
- Global Store instance rows that publish `signals_endpoint` but not a canonical
  execution endpoint,
- Python plan or runtime instance types that still carry `worker_id` and
  `signals_endpoint` rather than one HA-safe execution route,
- and implicit `instance_id` routing assumptions around NodeAgent execution.

Those pieces are directionally compatible, but not yet one contract.

Current repo mismatches:

- `GetWorkerStatus` freshness fields prove a useful metadata shape, but today
  they are local snapshot fields, not directory-cache provenance.
- `ListActiveInstances` currently exposes `signals_endpoint`, while the
  execution-unification line actually needs a dialable execution endpoint.
- `0060` needs queue-leader routing, but queue truth and worker route mapping
  are different layers and should not collapse into one owner.
- the runtime stack supports `global_store_mode=none`, so a repository-wide
  directory contract cannot assume Global Store exists in every deployment.

Without a frozen directory owner, adjacent designs keep depending on different
facts:

- `0056` needs daemon-served signals and directory for one runtime front door,
- `0060` needs queue and execution leaders to resolve through bounded-stale
  directory facts,
- `0104` needs worker and instance target resolution without direct SDK-to-GS
  calls,
- and execution unification needs a real instance-execution directory before
  instance steps can route through one spine.

# Ownership Boundary

`0106` follows the repository rule that one question has one authority root.

It therefore separates:

- membership truth,
- caller-facing target identity,
- daemon-served routing read models,
- transport endpoints used to execute the next hop,
- and workflow-specific leader or currentness truth that must stay outside
  directory.

| Question | Canonical source or contract | Owner |
| --- | --- | --- |
| which workers or instances exist in the current authority mode | Global Store worker or instance registry in cluster mode; connected daemon local projection in local-only mode | Global Store services plus daemon local runtime |
| what stable identity callers use for programmable targets | `daemon_id`, `instance_id` | `0056` plus this design |
| what bounded-staleness route read model the SDK consumes | daemon-served routing directory | `0106` |
| which dialable execution endpoint reaches one instance | `InstanceExecutionDirectory` entry | `0106` |
| whether a worker or instance is currently routable under freshness budget | daemon cache freshness evidence and fail-closed policy | `0106` |
| `queue_id -> leader_daemon_id, queue_epoch` | queue workflow truth and fencing | `0060` |
| whether a target is current, attached, replayable, or semantically complete | workflow truth and continuation families | `0096`, `0100`, workflow owners |

Normative rules:

1. `0106` owns only stable-principal-to-route read models.
2. In cluster mode, Global Store remains the membership truth root; in
   local-only mode, the connected daemon's local runtime projection is the only
   authority that `0106` may expose.
3. `daemon_id` and `instance_id` remain the stable programmable identities;
   endpoints are transport facts derived from directory.
4. `0106` does not own workflow principals such as queue leadership epoch,
   publish currentness, or rollout completion.
5. worker capacity snapshots are advisory read-model facts only; they are not
   reservation truth.
6. directory must never be the only durable owner of state that changes future
   workflow truth, join truth, or fencing outcomes.

# Goals / Non-Goals

## Goals

- Freeze stable target identity:
  - worker targets by `daemon_id`
  - instance targets by `instance_id`
- Freeze one daemon-served routing front door for programmable callers:
  - `list_workers`
  - `list_instances`
  - `resolve_instance_execution`
- Freeze one explicit instance execution contract:
  - `instance_id -> {daemon_id, execution_host_kind, execution_endpoint}`
- Freeze one shared freshness carrier reused by adjacent advisory reads such as
  `get_worker_capacity`.
- Make `execution_endpoint` an explicit fact anchor rather than an ad hoc label
  or an overloaded `signals_endpoint`.
- Require freshness evidence:
  - `as_of_ms`
  - `staleness_ms`
  - `cache_epoch`
  - `freshness_state`
  - `authority_mode`
- Define fail-closed behavior when freshness budget is exceeded.
- Define one contract that works in both:
  - `GLOBAL_STORE_BACKED`
  - `LOCAL_ONLY`
- Keep directory facts low-cardinality and advisory rather than turning Global
  Store into per-item hot truth.

## Non-Goals

- Define worker-local or cluster workflow truth.
- Define queue leadership truth, queue epoch, or workflow fencing. `0060`
  remains the owner there.
- Define rollout barriers or rollout ownership. `0104` remains the owner there.
- Replace lifecycle truth, routing truth, or operation truth with directory
  snapshots.
- Introduce a second public discovery client beside the connected daemon front
  door.
- Introduce a second instance-scoped execution host beside the existing
  NodeAgent or Instance Agent boundary.

# Architecture & Interfaces

## 1. Stable caller-facing identity

Caller-facing target identity is configuration-derived and transport-independent.

Rules:

1. worker targets are named by `daemon_id`, not by address,
2. instance targets are named by `instance_id`, not by `execution_endpoint`,
3. `execution_endpoint` is transport detail resolved by directory, not stable
   user target identity,
4. user-facing APIs must not overload one string field to mean identity,
   endpoint, and label at the same time.

Read-model rule:

- the same stable `instance_id` may resolve to a different execution endpoint
  over time without changing target identity,
- and that endpoint change is a directory refresh fact, not a new workflow or
  artifact truth domain.

## 2. Contract family

This design standardises one contract family with two concrete directory views.

- `WorkerDirectory`
  - answers `daemon_id -> worker route facts`
- `InstanceExecutionDirectory`
  - answers `instance_id -> instance execution route facts`

`NodeAgentDirectory` is not a second top-level directory family.
It is the current concrete profile of `InstanceExecutionDirectory` where:

- `execution_host_kind = "node_agent_grpc"`

Recommended public SDK shape:

```python
@dataclass(frozen=True, slots=True)
class DirectorySnapshot(Generic[T]):
    value: T
    as_of_ms: int
    staleness_ms: int
    cache_epoch: int | None = None
    freshness_state: str = "unknown"
    authority_mode: str = "unknown"


@dataclass(frozen=True, slots=True)
class WorkerRoute:
    daemon_id: str
    worker_id: str | None = None
    daemon_address: str | None = None
    capability_flags: int = 0


@dataclass(frozen=True, slots=True)
class InstanceExecutionRoute:
    instance_id: str
    daemon_id: str
    execution_host_kind: str
    execution_endpoint: str
    engine: str | None = None
    capability_flags: int = 0


class TensorCastDirectory:
    def list_workers(...) -> DirectorySnapshot[list[WorkerRoute]]: ...
    def list_instances(...) -> DirectorySnapshot[list[InstanceExecutionRoute]]: ...
    def resolve_instance_execution(...) -> DirectorySnapshot[InstanceExecutionRoute]: ...
```

Recommended runtime split:

```python
class Runtime:
    def directory(self) -> "TensorCastDirectory": ...
    def signals(self) -> "TensorCastSignals": ...
```

Compatibility rule:

- during migration, `TensorCastSignals.list_workers(...)` and
  `TensorCastSignals.list_instances(...)` may remain compatibility delegation
  shims,
- but the long-term public home of route discovery is `Runtime.directory()`,
- while `Runtime.signals()` remains the home for health and capacity snapshots.

Rules:

1. external callers query the connected daemon only,
2. daemon caches may be bounded-staleness, but must surface freshness evidence,
3. stale-beyond-budget reads fail closed rather than silently guessing,
4. addresses and execution endpoints are resolved from directory caches and are
   never promoted to stable workflow identity,
5. `Plan.run()` and daemon ingress must converge on the same directory contract
   rather than carrying separate instance-routing heuristics.

SoT rule:

- daemon-served directory answers where a target can be reached within a
  bounded-staleness budget,
- but it does not answer artifact existence, lifecycle protection, workflow
  currentness, queue leadership, or rollout completion.

## 3. Authority modes

`0106` must be explicit about authority mode because the repository supports
both Global-Store-backed and local-only runtimes.

### 3.1 `GLOBAL_STORE_BACKED`

- membership truth comes from Global Store worker and instance registries,
- daemon-served directory is a bounded-stale read model over that truth,
- watch continuity and fail-closed freshness rules apply in full.

### 3.2 `LOCAL_ONLY`

- the connected daemon exposes only daemon-local route facts,
- `list_workers()` returns the local daemon singleton view,
- `list_instances()` and `resolve_instance_execution(...)` may only expose
  daemon-local instance execution hosts,
- local-only mode must not fabricate a cluster view or pretend to know remote
  daemons or instances.

Normative rules:

1. the response surface must expose `authority_mode`,
2. the same public API contract is reused across modes,
3. mode changes must come from unified runtime config and startup semantics from
   `0004` and `0040`, not ad hoc environment-variable branches,
4. a caller must be able to distinguish:
   - no such target in this authority mode,
   - stale directory data,
   - and authority unavailable.

## 4. Instance execution endpoint anchoring

Current grounding:

- the current instance registry exposes `signals_endpoint`,
- execution-unification needs a dialable execution endpoint instead,
- and the coordination-series docs already identify that gap explicitly.

Design rule:

- the instance registry must gain an explicit execution endpoint fact anchor.

Recommended registry or proto shape:

- `execution_endpoint`
- optional `execution_host_kind`
  - current value: `node_agent_grpc`

Current and migration rules:

1. `signals_endpoint` remains an observability or signals field only; it must
   not continue as the execution-routing field.
2. If `INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED` is set, a dialable execution
   endpoint is required for routable instance execution.
3. A short-lived migration shim may temporarily mirror the endpoint in a
   reserved label such as `labels["tc.node_agent.endpoint"]`, but that shim is
   migration debt, not the long-term contract.
4. The long-term accepted contract is an explicit field, not a label convention.

## 5. Freshness contract

Minimum public freshness evidence:

- `as_of_ms`
- `staleness_ms`
- `cache_epoch`
- `freshness_state`
- `authority_mode`

Freshness state meanings:

- `current`
  - bounded-current under the configured budget
- `degraded`
  - usable only for explicitly degraded read paths
- `stale`
  - not acceptable for routing decisions that require bounded-current evidence

Correctness floor in `GLOBAL_STORE_BACKED` mode:

- initial snapshot barrier,
- monotonic cache epoch or equivalent version,
- resume token or equivalent replay cursor,
- fail-closed behavior when freshness cannot be re-established within budget.

Correctness floor in `LOCAL_ONLY` mode:

- monotonic local cache epoch or equivalent local snapshot version,
- explicit invalidation when the local runtime loses the relevant route fact,
- fail-closed behavior when the local daemon cannot re-establish route evidence.

Grounding note:

- current `GetWorkerStatus` freshness fields are a useful wire-shape precedent,
- but they are not yet sufficient proof of directory-cache correctness and must
  not be cited as if the full directory contract were already landed.

## 6. Relationship to adjacent designs

### 6.1 `0056`

`0056` owns:

- runtime front door,
- `Runtime`,
- daemon-served consumption rule,
- and execution-spine placement.

`0106` owns:

- the exact route-directory contract consumed through that front door.

### 6.2 `0060`

`0060` may define:

- `queue_id -> leader_daemon_id, queue_epoch`

It must not redefine:

- `leader_daemon_id -> endpoint`

Queue leader endpoint resolution should reuse `WorkerDirectory`.

### 6.3 `0104`

`0104` consumes:

- worker and instance route discovery

It must not define:

- a rollout-private target directory,
- or a rollout-private stable-identity-to-endpoint contract.

### 6.4 `0096` and `0100`

Workflow truth, currentness, replay, attach, and distributed continuation stay
outside directory by design.

## 7. SDK and object-model migration

Current repo reality:

- `CapabilityDirectoryClient` is still useful as the semantic baseline for cache
  behavior and staleness control,
- `tensorcast.api.plan.plan.Instance` still carries `worker_id` and
  `signals_endpoint`,
- and `Plan.run()` still does not own instance-host routing.

Migration rules:

1. daemon-side directory caches may reuse the same semantic contract,
2. cross-language components reuse contract, not implementation,
3. new programmable features must not add fresh direct-SDK-to-GS directory
   dependencies,
4. `Runtime` should grow `directory()` as the long-term route-discovery home,
5. `Plan.Instance` should grow `daemon_id` and stop treating `signals_endpoint`
   as routing state,
6. if a compatibility path still uses SDK-direct GS reads, it must be described
   as migration debt rather than as co-equal front-door truth.

# Naming Compliance

| Proposed symbol | Kind | Required style | Result |
| --- | --- | --- | --- |
| `TensorCastDirectory` | Python class | `PascalCase` | pass |
| `DirectorySnapshot` | Python dataclass | `PascalCase` | pass |
| `WorkerRoute` | Python dataclass | `PascalCase` | pass |
| `InstanceExecutionRoute` | Python dataclass | `PascalCase` | pass |
| `list_workers` | Python method | `snake_case` | pass |
| `list_instances` | Python method | `snake_case` | pass |
| `resolve_instance_execution` | Python method | `snake_case` | pass |
| `execution_endpoint` | field | `snake_case` | pass |
| `execution_host_kind` | field | `snake_case` | pass |

# Schema Changes

This design requires an additive instance-registry schema and proto update.

Recommended changes:

- `schema.sql`
  - add `execution_endpoint TEXT` to `instances`
  - optionally add `execution_host_kind TEXT`
- `proto/tensorcast/global_store/v1/global_store.proto`
  - add `execution_endpoint` to `RegisterInstanceRequest`
  - add `execution_endpoint` to `ListActiveInstancesResponse.InstanceInfo`
  - optionally add `execution_host_kind`
- `proto/tensorcast/daemon/v2/store_daemon.proto`
  - add daemon-served directory RPCs or additive responses that expose the
    route-directory contract and freshness metadata

Non-goals for schema:

- no new Global Store hot tables,
- no workflow-state migration into directory tables,
- no queue-specific routing tables in this cut.

# Trade-offs and Risks

- Freezing one route-directory contract adds up-front design work, but it
  prevents rollout, queue, and execution work from growing separate discovery
  dialects.
- Splitting `signals_endpoint` from `execution_endpoint` adds migration work,
  but it removes a more dangerous long-term ambiguity between observability and
  routing.
- Requiring fail-closed freshness may surface more transient errors, but it is
  safer than silently routing against stale membership.
- Supporting both `GLOBAL_STORE_BACKED` and `LOCAL_ONLY` in one contract adds
  some branching in implementation, but it is strategically better than letting
  cluster and local-only modes grow incompatible public APIs.
- Renaming the deeper semantic owner from `NodeAgentDirectory` to
  `InstanceExecutionDirectory` may feel less direct, but it keeps the contract
  stable if the execution-host profile evolves beyond today's NodeAgent wire
  shape.

# Compatibility & Acceptance Criteria

- programmable callers resolve route facts through the connected daemon rather
  than direct Global Store discovery,
- `daemon_id` and `instance_id` remain the only stable caller-facing target
  identifiers,
- instance execution routing uses an explicit execution endpoint fact rather
  than overloading `signals_endpoint`,
- `0060` reuses `WorkerDirectory` for leader endpoint mapping instead of
  defining a second worker-route contract,
- freshness metadata is exposed on daemon-served directory and adjacent
  advisory reads such as worker-capacity snapshots,
- stale-beyond-budget reads fail closed,
- `LOCAL_ONLY` behavior is explicitly defined and does not pretend to be a
  degraded cluster view,
- `CapabilityDirectoryClient` remains compatibility tooling only and is not
  extended as the normative front door for new programmable features.

# References

- `0056` for runtime, ingress, and front-door ownership.
- `0060` for queue identity and leader-directory dependencies.
- `0104` for rollout target-resolution dependency.
- `0096` and `0100` for workflow and continuation boundaries outside directory.
- distributed coordination series for current strategic grounding and migration
  rationale.
