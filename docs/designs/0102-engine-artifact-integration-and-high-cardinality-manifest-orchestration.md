---
slug: 0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration
title: Engine Artifact Integration and High-Cardinality Manifest Orchestration (Design)
description: Define the engine-integration layer that maps engine-owned runtime state into canonical artifact lifecycle actions and high-cardinality manifests without coupling TensorCast framework core to KV or LLM-specific semantics.
status: draft
areas:
  - sdk
  - daemon
  - integrations
  - docs
created: 2026-03-17
last_updated: 2026-03-17
related_code:
  - tensorcast/engine_adapter/kvcache_adapter.py
  - tensorcast/node_agent/executor.py
  - tensorcast/node_agent/server.py
  - tensorcast/api/plan/plan.py
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - proto/tensorcast/plan/v1/plan.proto
related_docs:
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
  - docs/designs/0092-artifact-profiles-shared-dataplane-and-truth-layering.md
  - docs/designs/0100-distributed-authority-handoff-security-and-public-surfaces.md
  - docs/designs/0039-artifact-first-sdk.md
  - docs/designs/0055-programmable-framework.md
links:
  dependencies:
    - ./0055-programmable-framework.md
    - ./0056-programmable-framework-adv.md
    - ./0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
    - ./0092-artifact-profiles-shared-dataplane-and-truth-layering.md
    - ./0094-unified-lifecycle-kernel-and-capability-families.md
    - ./0100-distributed-authority-handoff-security-and-public-surfaces.md
---

# Summary

This design extracts engine-specific integration concerns from `0056`.

TensorCast framework core remains responsible for:

- artifact-first execution semantics,
- daemon ingress and signals,
- generic plan actions,
- and distributed orchestration entrypoints.

Engine integrations remain responsible for:

- request or runtime-state inspection inside the engine,
- engine-local mutable state and staging,
- request-context requirements,
- domain-specific aliases and helper APIs,
- and storage-backend adaptation to TensorCast's artifact and byte-artifact runtime.

The canonical integration model is therefore:

- TensorCast core stays artifact-first and engine-agnostic,
- engine integrations project runtime state into canonical artifact actions on the existing execution spine,
- and high-cardinality orchestration is expressed through neutral manifest-oriented integration carriers that lower into
  framework-owned generic set carriers rather than business-specific framework types.

Current code already points in this direction:

- the file `tensorcast/engine_adapter/kvcache_adapter.py` defines generic artifact lifecycle result types,
- NodeAgent preserves `manifest/publish/hydrate/evict_local` results,
- and plan execution already aligns around those canonical action names.

Long-term convergence rule:

- `0056` owns the programmable front door and generic set carrier,
- NodeAgent or the in-process Instance Agent boundary remains the unique instance-scoped execution host in this phase,
- and `0102` owns only engine-side projection into that spine, including the explicit bridge from engine manifests into
  `0056` generic set orchestration.

`0102` is therefore not:

- a second front door,
- a second instance-hosting model,
- or a place to define workflow, continuation, or lifecycle semantics already owned by `0096`, `0100`, and `0094`.

# Problem Statement

LLM KV motivating cases exposed a real need:

- object counts are high,
- data is fragmented,
- request context often matters at the engine boundary,
- and explicit migration or warmup workflows are operationally useful.

Earlier `0056` drafts tried to capture that directly in the framework by introducing KV-centric nouns such as:

- `KvKeySet`,
- `kvcache_flush`,
- `kvcache_prefetch`,
- and engine-specific control flows.

That shape is not durable for the repository because it:

- couples framework core to one business domain,
- duplicates the canonical artifact action model already present in NodeAgent and the SDK,
- and makes future non-KV high-cardinality integrations harder rather than easier.

The root problem is real, but the right abstraction level is not "KV in framework core".
The right abstraction level is:

- engine-owned state inspection,
- canonical artifact action projection,
- and high-cardinality manifest-oriented orchestration.

# Goals / Non-Goals

Goals

- Keep TensorCast framework core KV-semantics-free.
- Define a stable engine-integration layer that maps engine-owned runtime state into canonical artifact actions.
- Make high-cardinality orchestration manifest-first rather than business-name-first.
- Keep framework set identity rooted in canonical `SelectionIdentity` rather than integration-local digests or aliases.
- Preserve the two-layer integration split:
  - engine storage-backend plugin to TensorCast data plane,
  - and instance-bound orchestration adapter to TensorCast plan/runtime.
- Keep `engine_request_id` or equivalent engine handles outside artifact identity.
- Allow compatibility aliases for existing ecosystems without making them normative framework vocabulary.

Non-Goals

- Define byte-artifact authority, shard-home routing, or lifecycle semantics.
- Move token-to-key mapping or request tables into TensorCast core.
- Require every engine to expose the same request model.
- Redefine the framework-level plan IR from `0055` and `0056`.
- Introduce a second object model beside `Artifact`, `ArtifactSelection`, `SealedByteArtifact`, and canonical action
  results.
- Introduce a second set-identity model beside framework-owned set identity over normalized `SelectionIdentity`.
- Introduce a second instance-scoped execution host beside NodeAgent or the existing Instance Agent boundary.
- Introduce an integration-private continuation or attach protocol outside `Operation[T]` and the `0100` public family.

# Boundary Contract

## Integration position in the stack

Repository rule:

- `0056` owns front-door convergence and generic set transport,
- `0102` owns engine-side projection into canonical actions and set carriers,
- and deeper owner layers still own truth, lifecycle, workflow, and continuation.

The intended stack is:

```text
Runtime / gateway ingress
  -> PlanSpec
  -> worker execution on Store Daemons
  -> instance execution on NodeAgent / Instance Agent
  -> Engine adapter projection
  -> engine runtime state
```

Normative rules:

1. `0102` must plug into the existing execution spine rather than inventing a parallel controller path,
2. integration helpers may bootstrap or configure adapters, but they must not introduce a second instance-hosting
   contract,
3. integration-layer retries, attach, wait, or replay must reuse `Operation[T]` or another explicit `0100` public
   family surface rather than inventing alias-specific protocols,
4. engine-private nouns may appear in helper APIs, but canonical framework execution still speaks
   `manifest/publish/hydrate/evict_local`,
5. in this phase, `0102` defines terminal projection semantics only; any non-terminal attach, wait, replay, or
   currentness contract must wait for a dependency-ready public continuation path from `0096` and `0100`.

# Architecture

## Two-layer integration split

```mermaid
flowchart LR
  A["Application or scheduler"] --> B["TensorCast plan/runtime"]
  B --> C["NodeAgent or Instance Agent"]
  C --> D["Engine artifact adapter"]
  D --> E["Engine runtime state"]

  E --> F["Engine storage backend plugin"]
  F --> G["TensorCast daemon and byte-artifact runtime"]
```

The two layers have different responsibilities:

1. **Storage backend plugin**
   - engine calls TensorCast as a key-to-bytes backend,
   - engine still owns request progression, page layout, staging, and mutable open state,
   - TensorCast still owns sealed publication, routing, transport, and daemon-managed residency.
2. **Instance-bound orchestration adapter**
   - TensorCast calls into the engine through NodeAgent or the Instance Agent boundary,
   - the adapter inspects engine state and projects it into canonical `manifest/publish/hydrate/evict_local` actions,
   - the adapter never becomes a second distributed control plane.

# Canonical Semantic Model

## Artifact actions are the canonical integration vocabulary

The normative action vocabulary is:

- `manifest`
- `publish`
- `hydrate`
- `evict_local`

Current code already exposes these as the canonical result family:

- `ManifestResult`
- `PublishResult`
- `HydrateResult`
- `BatchResult`

and as the canonical adapter protocol:

- `EngineArtifactAdapter`

Even though the current module path is `tensorcast/engine_adapter/kvcache_adapter.py`, the semantic contract is already
generic artifact lifecycle projection rather than framework-owned KV semantics.

Normative rules:

1. new generic plan, NodeAgent, or proto surfaces must use the canonical action names above,
2. metrics, audit logs, and idempotency fingerprints must use the canonical action names above,
3. integration-specific aliases must not become the canonical framework vocabulary.

## Frozen projection, not repeated rescan

Long-term convergence for high-cardinality instance actions is:

- `manifest` freezes one integration-owned projection of engine state,
- source-side `publish`, worker set warmup, and target-side `hydrate` should consume that same projection or an explicit
  projection form derived from it,
- and the framework should not rely on repeated ad hoc rescans of mutable engine state as a substitute for explicit
  projection identity.

Current code still threads `engine_request_id` through instance actions.
That is current compatibility plumbing, not the long-term semantic identity contract.

Normative rules:

1. if a workflow needs stable cross-step set identity, the integration must supply an explicit projection form or a
   manifest-backed set reference rather than relying on repeated rescans by `engine_request_id`,
2. projection forms owned by `0102` are integration-side carriers only; replay, attach, currentness, and status remain
   owned by `0096` and `0100`,
3. until a dependency-ready projection form exists, cross-instance migration remains terminal action composition over the
   existing spine rather than a closed non-terminal workflow family.

## `engine_request_id` is engine context, not artifact identity

`engine_request_id` or any equivalent engine-local handle is:

- a way to locate engine state,
- a possible correlation key across instances within one engine integration,
- and an input to engine-side inspection or realization.

Current code reality:

- `engine_request_id` is carried through current `PlanSpec`, NodeAgent, and engine-adapter instance actions as a request
  parameter,
- but that does not make it the long-term semantic identity for a projected set, a stable action replay key, or a
  framework-owned workflow handle.

It is not:

- artifact identity,
- selection identity,
- or framework-level distributed truth.

Artifact identity remains:

- `artifact_id` for values,
- `ArtifactSelection` for what was selected,
- `SelectionIdentity` for canonical per-item identity when high-cardinality sets are projected into framework-generic
  orchestration,
- and framework-owned set digests over resolved `SelectionIdentity` when the bridge into `0056` generic set
  orchestration is closed.

Manifest digests may still appear as integration-side carrier metadata or checksums, but they must not silently become a
second framework set-identity model beside the `0056` generic set carrier.

## Sealed byte artifacts are the publication unit

When engine integrations publish high-cardinality byte artifacts, the canonical publication unit remains:

- `OpenByteArtifact`
- `SealedByteArtifact`
- `PutIfAbsentInvariant`

This preserves the `0087` rule:

- engine-owned mutable state stays open and engine-local,
- only sealed immutable snapshots enter TensorCast publication and routing paths.

# High-Cardinality Manifest Semantics

## `ManifestResult` is the current integration-side carrier today

The repository already has a neutral carrier for high-cardinality sets:

- `ManifestResult`

Its essential semantics are:

- `artifact_ids` are the concrete artifact identities in the set,
- `layout_id` names the engine-visible layout contract for those artifacts,
- `key_set_digest_hex` is the integration-side digest carried by today's manifest result shape,
- any ordering in `artifact_ids` is an engine convenience, not the semantic set identity.

Normative rules:

1. NodeAgent and SDK plan results may preserve neutral manifest-oriented carriers as structured integration output,
2. framework-generic set orchestration must still lower through the `0056` generic set carrier over canonical
   `SelectionIdentity`,
3. future generalization should extend or rename neutral manifest carriers rather than introducing business-specific
   framework result types,
4. set ordering is never part of framework idempotency or identity,
5. manifest-oriented carriers in `0102` are integration-owned projections that must lower into the generic set carrier
   contract owned by `0056`; they are not a second framework set model.

## Current projection gap and closeout rule

Current code reality is narrower than the long-term target:

- `ManifestResult` currently carries `artifact_ids`, `layout_id`, and `key_set_digest_hex`,
- but it does not yet carry canonical per-item identities equivalent to `SelectionIdentity`,
- so it is not yet sufficient, by itself, to prove the generic set contract required by `0056`.

Normative rules:

1. `0102` owns the bridge from today's `ManifestResult` to future generic set orchestration,
2. that bridge must be explicit and versioned; local runner, gateway ingress, and worker code must not improvise their
   own derivation rules,
3. the first dependency-ready bridge should be manifest-backed:
   - integration produces a manifest artifact or equivalent manifest form with an explicit schema version,
   - that manifest resolves to canonical per-item identities and any integration-local annotations needed for the target
     engine,
   - the owner of the manifest-backed reference, its expiry or currentness rules, and its resolution authority must be
     explicit,
   - digest mismatch, item-count mismatch, schema-version mismatch, or unresolved reference must fail closed,
4. `key_set_digest_hex` from today's `ManifestResult` must not be treated as the framework-owned generic set identity
   unless and until the bridge defines it as the exact same digest over the resolved canonical item set,
5. if a future engine needs a more opaque set handle, its owner, expiry or currentness, and resolution semantics must
   still be made explicit before it becomes a dependency-ready carrier.

## Why manifest-first is the right abstraction

Manifest-first orchestration is better than KV-first orchestration because it generalizes to:

- paged KV blobs,
- runtime-generated byte artifacts outside LLMs,
- fragment sets for other engine caches,
- and future high-cardinality artifact families that are not token- or request-centric.

It also matches current code reality better than earlier `KvKeySet`-style drafts.

# Alias Policy

Compatibility aliases are allowed, but they are not normative framework surface.

Allowed:

- integration helper methods such as `kvcache_flush(...)`,
- engine-specific SDK wrappers,
- engine-specific observability tags layered on top of canonical action names.

Not allowed in framework core:

- new plan proto message kinds named after one engine domain,
- canonical NodeAgent result carriers renamed to business-specific nouns,
- framework-level action identity derived from alias names instead of canonical names.

Practical rule:

- framework core speaks `manifest/publish/hydrate/evict_local`,
- integration packages may offer `kvcache_*` wrappers that lower to those actions.

# Integration Patterns

## Pattern 1: storage-backend integration

For engines that already expose a storage backend abstraction, the preferred integration is:

- implement the backend against TensorCast,
- keep the engine's own mapping and staging logic intact,
- and let TensorCast remain a key-to-bytes store plus routed artifact runtime.

This is the most generic pattern and should be the default integration path.

## Pattern 2: explicit migration orchestration

When an application wants explicit cross-instance migration or warmup, the preferred canonical flow is:

1. source instance `publish`
2. optional worker set warmup
3. target instance `hydrate`

This flow is expressive enough for explicit migration while preserving framework-generic vocabulary.

Rules:

1. `publish` is the source-side publication barrier,
2. worker set warmup is worker-layer data-plane warmup only and must converge on framework-owned set orchestration
   rather than a second engine-specific batch model,
3. `prefetch_many`, if exposed, remains small-set SDK sugar rather than the primary scalable abstraction,
4. `hydrate` is the target-side engine realization step,
5. `evict_local` is engine-local reclamation only.

## Target request-context requirements

Some engines require local request context on the target instance before `hydrate` can succeed.

That is an integration fact, not a framework fact.

Therefore:

- the engine adapter may require target-local context before `hydrate`,
- missing or inconsistent local context is an adapter- or engine-level precondition failure,
- and the framework does not encode engine-private request lifecycle rules into its core action vocabulary.

# SGLang Grounding

SGLang remains a motivating example because it demonstrates:

- page-fragmented high-cardinality blobs,
- a clear separation between deterministic key derivation and backend storage,
- and a real need for explicit source publish plus target hydrate flows.

But SGLang is grounding only.
This design must remain valid for future engines whose runtime objects are:

- not KV,
- not token-addressed,
- or not request-centric in the same way.

# Naming Compliance

The interfaces formalized by this design follow repository naming rules.

Classes and structs use `PascalCase`:

- `EngineArtifactAdapter`
- `ManifestResult`
- `PublishResult`
- `HydrateResult`
- `BatchResult`
- `SealedByteArtifact`
- `PutIfAbsentInvariant`

Functions and methods use `snake_case`:

- `manifest`
- `publish`
- `hydrate`
- `evict_local`
- `seal_byte_artifact`
- `compute_key_set_digest_hex`

Compatibility aliases, when exposed, also use `snake_case`:

- `kvcache_flush`
- `kvcache_prefetch`
- `kvcache_evict_local`

Constants use `ALL_CAPS`:

- `_KEY_SET_DIGEST_PREFIX`

# Schema Changes

This design does not introduce new `schema.sql` ownership.

Proto and SDK additions that follow from this design should remain generic:

- canonical artifact action names,
- canonical manifest-oriented result carriers that lower into framework-owned generic set carriers,
- and integration-layer helper APIs that lower to those canonical surfaces.

# Trade-offs and Risks

- Canonical artifact actions are less domain-friendly than business-specific aliases, so helper layers remain useful.
- The current filename `kvcache_adapter.py` can mislead readers into over-ascribing KV semantics to the core contract.
- Some engines may need richer adapter-local preconditions than others; that complexity belongs in integrations, not in
  framework core.
- Current code still carries `engine_request_id` through instance-action request shapes; the documentation must not
  overstate that compatibility input as the long-term semantic identity contract.
- Until the manifest-backed bridge closes, there is still a risk that local runner, gateway ingress, and worker code try
  to derive incompatible set identity from today's `ManifestResult`.

# Compatibility and Acceptance Criteria

- Engine integration guidance in TensorCast docs uses canonical `manifest/publish/hydrate/evict_local` semantics.
- `0102` remains a projection layer on the existing `Runtime / gateway -> PlanSpec -> NodeAgent / Instance Agent`
  execution spine rather than a second execution model.
- NodeAgent or the existing Instance Agent boundary remains the unique instance-scoped execution host in this phase.
- `engine_request_id` never becomes artifact identity or framework truth.
- `SelectionIdentity` remains the canonical per-item identity when engine manifests lower into framework-generic set
  orchestration.
- `ManifestResult` or a future neutral successor remains the current integration-side high-cardinality carrier, while
  lowering into framework-generic set orchestration remains explicit and versioned.
- manifest-backed set references define explicit schema version, owner, expiry or currentness contract, and fail-closed
  resolution behavior before they are treated as dependency-ready carriers.
- `key_set_digest_hex` does not silently become a second framework set-identity model beside the `0056` generic set
  carrier.
- in this phase, `0102` instance-action closeout remains terminal-only unless and until a dependency-ready `0096` and
  `0100` public continuation path is explicitly adopted.
- Compatibility aliases such as `kvcache_*` remain helper vocabulary only and do not become required framework proto or
  audit vocabulary.
- Explicit migration flows can be expressed using canonical `publish`, worker set warmup, and `hydrate` actions, with
  `prefetch_many` remaining optional helper vocabulary only.
