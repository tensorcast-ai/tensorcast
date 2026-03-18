---
slug: engine-artifact-integration-and-high-cardinality-manifest-orchestration
title: Plan - Engine Artifact Integration and High-Cardinality Manifest Orchestration
status: in_progress
areas: ["sdk", "proto", "integrations", "docs"]
created: 2026-03-17
last_updated: 2026-03-17
related_code:
  - docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md
  - docs/plans/0056-programmable-framework-adv.md
  - tensorcast/engine_adapter/kvcache_adapter.py
  - tensorcast/engine_adapter/README.md
  - tensorcast/engine_adapter/adapter.py
  - tensorcast/node_agent/executor.py
  - tensorcast/node_agent/server.py
  - tensorcast/api/plan/plan.py
  - proto/tensorcast/node_agent/v1/BUILD
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - proto/tensorcast/plan/v1/plan.proto
  - tests/python/test_kvcache_adapter.py
  - tests/python/api/test_plan_spec.py
  - tests/python/node_agent/test_plan_execution.py
links:
  design: ../designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md
---

# Objective

Land `0102` as the sole execution plan for engine-side artifact projection bridge closeout:

- keep framework core on canonical `manifest/publish/hydrate/evict_local`,
- keep engine request handles as adapter context rather than artifact or set identity,
- close the current `ManifestResult` projection gap with an explicit versioned bridge owned by `0102`,
- and keep alias or engine-specific helper vocabulary outside framework proto and runtime core.

This plan does not make `0102` the owner of:

- framework-owned `ArtifactSetRef` schema or generic set orchestration from `0056`,
- public continuation semantics from `0096` and `0100`,
- byte-artifact truth, lifecycle, or authority semantics from `0087`, `0092`, `0093`, and `0094`,
- or a second instance-scoped execution host beside NodeAgent or the existing Instance Agent boundary.

# Current State & Grounding

- `tensorcast/engine_adapter/kvcache_adapter.py` already defines the canonical engine-side action result family:
  `ManifestResult`, `PublishResult`, `HydrateResult`, and `BatchResult`.
- `tensorcast/engine_adapter/adapter.py` already enforces canonical adapter entrypoints `manifest`, `publish`,
  `hydrate`, and `evict_local`, while still taking `engine_request_id` as the current request locator.
- `proto/tensorcast/plan/v1/plan.proto` and `proto/tensorcast/node_agent/v1/node_agent.proto` already expose the same
  canonical action names, but their request and result carriers still thread `engine_request_id` directly.
- `tensorcast/api/plan/plan.py`, `tensorcast/node_agent/executor.py`, and `tensorcast/node_agent/server.py` already
  preserve manifest or publish or hydrate outputs end to end, so the execution spine exists today.
- current concrete gap:
  - `ManifestResult` only carries `artifact_ids`, `layout_id`, and `key_set_digest_hex`; it does not yet carry
    canonical per-item identities or a bridge output that can lower honestly into `0056` `ArtifactSetRef`.
  - `tensorcast/node_agent/executor.py` still fingerprints some paths with `engine_request_id`, which is compatible as
    request context today but must not become long-term set identity.
  - the filename `tensorcast/engine_adapter/kvcache_adapter.py` still over-signals KV semantics even though the
    contract is already generic artifact projection.
- `docs/plans/0056-programmable-framework-adv.md` already depends on `0102` owning manifest production and the bridge
  from engine projection into `ArtifactSetRef`, so keeping that work only in the `0056` plan leaves an execution-owner
  gap.

# Status Update

- [x] Owner boundary, bridge shape, versioned bridge metadata, and fail-closed rules are now explicit in docs.
- [x] `engine_request_id` is now documented as locator-only context rather than framework set identity.
- [x] `ManifestResult` now carries an additive `ManifestArtifactSetBridge` with `bridge_schema`,
      `bridge_version`, `artifact_set_ref`, and explicit resolved items for fail-closed bridge-owned lowering.
- [x] Plan proto, NodeAgent proto, SDK decoding, local plan execution, and NodeAgent worker execution now preserve the
      same bridge end to end for `manifest_backed` `ArtifactSetRef` consumption.
- [x] Instance-action idempotency keys now stay on canonical action or target or step inputs instead of embedding
      `engine_request_id` as the step fingerprint surrogate.
- [ ] Neutral module-surface cleanup beyond the package-root and README path is still pending.

# Execution Position

- [x] Stage 1: enter after `0056` has frozen the `ArtifactSetRef` substrate contract.
- [x] Stage 2: close `0102` Phase 0-2 before `0056` runtime or ingress work consumes engine-side high-cardinality
      outputs.
- [ ] Stage 3: leave adapter compatibility wrappers and naming cleanup additive so they do not block the bridge path.

# Phases & Milestones

- [x] Phase 0: Scope freeze and owner boundaries
  - [x] Milestone 0.1: freeze `0102` as the only owner of engine manifest production, engine request-context rules,
        alias policy, and engine-to-neutral-set projection.
  - [x] Milestone 0.2: keep framework core vocabulary fixed at `manifest/publish/hydrate/evict_local`.
  - [x] Milestone 0.3: record explicitly that `engine_request_id` is request context only and not artifact identity,
        `SelectionIdentity`, or framework set identity.
  - [x] Milestone 0.4: keep NodeAgent or the existing Instance Agent boundary as the only instance-scoped execution
        host in this phase.

- [x] Phase 1: Bridge closeout into `ArtifactSetRef`
  - [x] Milestone 1.1: define an explicit bridge shape such as `ManifestArtifactSetBridge` from `ManifestResult` to
        `ArtifactSetRef`.
  - [x] Milestone 1.2: carry versioned bridge metadata (`bridge_schema`, `bridge_version`) alongside the bridged
        `ArtifactSetRef`.
  - [x] Milestone 1.3: make the first dependency-ready bridge emit `ArtifactSetRef` in `manifest_backed` form rather
        than forcing local raw-manifest expansion in framework code.
  - [x] Milestone 1.4: define fail-closed behavior for schema-version mismatch, digest mismatch, item-count mismatch,
        and unresolved manifest-backed references.
  - [x] Milestone 1.5: preserve `key_set_digest_hex` as integration metadata only unless and until it is proven to be
        the exact digest over the resolved canonical item set used by `ArtifactSetRef`.

- [x] Phase 2: Transport and API alignment
  - [x] Milestone 2.1: evolve plan and NodeAgent carriers additively so they can transport the versioned bridge form
        without introducing engine-specific nouns.
  - [x] Milestone 2.2: align SDK decoding and NodeAgent serialization on one canonical bridge-preserving carrier shape.
  - [x] Milestone 2.3: keep `engine_request_id` as an adapter input where still required for engine lookup, but remove
        any implication that it is the replay key, set identity, or step fingerprint surrogate.
  - [x] Milestone 2.4: keep metrics, audit vocabulary, and idempotency names on canonical action names rather than
        alias names.

- [ ] Phase 3: Adapter surface and compatibility wrappers
  - [ ] Milestone 3.1: make the generic artifact contract discoverable from engine-adapter modules without requiring KV
        vocabulary to understand the API.
  - [ ] Milestone 3.2: keep compatibility helpers such as `kvcache_*` as wrappers only and prevent them from becoming
        required proto or runtime surface.
  - [ ] Milestone 3.3: document any adapter-local target request-context preconditions as integration failures rather
        than framework action semantics.

- [ ] Phase 4: Verification and dependent-plan unblock
  - [x] Milestone 4.1: prove one end-to-end path where engine manifest output lowers into the neutral set bridge
        without local ad hoc recomputation.
  - [x] Milestone 4.2: verify `0056` can consume the `0102` bridge without importing KV-specific nouns or second set
        identity rules.
  - [x] Milestone 4.3: update dependent docs and plans so `0056` references `0102` as the concrete owner of engine
        projection closeout.

# Tasks

- Engine adapter:
  - define an additive bridge carrier alongside or inside `ManifestResult`.
  - keep adapter entrypoints canonical and keep helper aliases outside the normative contract.
  - reduce the semantic mismatch caused by `kvcache_adapter.py`, either by introducing a neutral module surface or by
    otherwise making the generic contract the primary import and documentation path.

- Plan and NodeAgent transport:
  - extend `proto/tensorcast/plan/v1/plan.proto` and `proto/tensorcast/node_agent/v1/node_agent.proto` additively if
    new projection metadata must cross the wire.
  - keep existing `manifest/publish/hydrate/evict_local` actions and result families stable.
  - ensure transport preserves the bridge output end to end and does not force `0056` to infer canonical item identity
    from `engine_request_id` or raw `ManifestResult`.

- SDK and execution spine:
  - update `tensorcast/api/plan/plan.py`, `tensorcast/node_agent/server.py`, and
    `tensorcast/node_agent/executor.py` to preserve the explicit projection bridge end to end.
  - audit step fingerprint or idempotency inputs so request-context compatibility fields do not silently become set
    identity.

- Documentation:
  - keep `0102` focused on engine-side projection and bridge semantics.
  - keep `0056` focused on `ArtifactSetRef`, framework-generic set orchestration, and front-door convergence.
  - document the dependency boundary from `0102` into `0056`, `0096`, and `0100` precisely.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/test_kvcache_adapter.py`
  - `source .venv/bin/activate && pytest tests/python/api/test_plan_spec.py`
  - `source .venv/bin/activate && pytest tests/python/node_agent/test_plan_execution.py`
- Proto:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`

Acceptance checks:

- `ManifestResult` or its additive successor carries an explicit bridge object such as `ManifestArtifactSetBridge`
  into `ArtifactSetRef`.
- local SDK, NodeAgent serialization, and NodeAgent execution all preserve the same bridge without recomputing
  canonical item identity from local-only heuristics.
- `engine_request_id` remains an engine lookup input only and is not documented or tested as framework set identity.
- framework code can consume the bridged `ArtifactSetRef` directly instead of deriving it from raw `ManifestResult`.
- no new engine-specific nouns leak into plan proto, NodeAgent proto, metrics, or audit vocabulary.
- `0056` can consume the `0102` bridge directly instead of inventing its own identity derivation.

Rollout:

- land the bridge additively so existing `engine_request_id`-based compatibility paths continue working while the
  explicit projection form is introduced.
- keep helper aliases and current adapter call sites working until neutral imports and docs have replaced them.

Backout:

- if the versioned projection bridge proves incomplete, keep canonical terminal actions active and fall back to the
  current compatibility carrier without claiming dependency-ready generic set identity.
- revert additive transport fields or module exports without changing canonical action names.

# Risks & Tracking

- Risk: `0056` or local execution code derives canonical item identity ad hoc from today's `ManifestResult`.
  - tracking: require the bridge to be versioned, documented, and owned by `0102`.
- Risk: `engine_request_id` remains embedded in fingerprinting or replay semantics long enough to become de facto set
  identity.
  - tracking: audit step fingerprint inputs and add tests that separate request context from set identity.
- Risk: neutralization work regresses into a big-bang rename of engine-adapter surfaces.
  - tracking: keep wire and API evolution additive and retain alias wrappers.
- Risk: engine-specific helper vocabulary leaks into proto or framework docs during rollout.
  - tracking: review canonical names in proto, SDK public docs, and NodeAgent result carriers.
