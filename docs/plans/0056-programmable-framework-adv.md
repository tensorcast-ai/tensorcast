---
slug: 0056-programmable-framework-adv
title: Plan - Programmable Framework Advanced Runtime, Ingress, and Signals
status: proposed
areas: ["sdk", "daemon", "global_store", "proto", "integrations", "docs"]
created: 2026-03-04
last_updated: 2026-03-17
related_code:
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md
  - docs/designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
  - docs/designs/0055-programmable-framework.md
  - tensorcast/runtime.py
  - tensorcast/api/plan/plan.py
  - tensorcast/node_agent/executor.py
  - tensorcast/engine_adapter/adapter.py
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - proto/tensorcast/global_store/v1/global_store.proto
links:
  design: ../designs/0056-programmable-framework-adv.md
---

# Objective

Deliver the advanced programmable framework layer on top of `0055` while keeping it thin:

- daemon-run plan ingress without semantic drift from the local runner,
- front-door convergence over one execution spine rather than another execution substrate,
- daemon-served signals and directory snapshots with explicit staleness bounds,
- generic set-level orchestration for high-cardinality artifact sets,
- and strict separation between framework core and engine integration concerns.

This plan explicitly does not turn `0056` into:

- a home-authority or lifecycle design,
- an engine-specific manifest or KV design,
- a second instance-hosting model beside NodeAgent or the existing Instance Agent boundary,
- or a second public continuation protocol beside `0096` and `0100`.

# Current State and Grounding

- `0055` already defines plan and `PlanSpec` semantics, local `Plan.run()`, and canonical instance actions in today's
  code.
- `proto/tensorcast/plan/v1/plan.proto` already provides canonical worker and instance targets plus canonical action
  names; it does not yet carry a typed governance sub-object, set carrier, or cluster-scoped target.
- `proto/tensorcast/node_agent/v1/node_agent.proto` already exposes `ExecutePlanRequest(plan, dry_run)` and canonical
  artifact action results, so the repository already has a real instance-scoped execution front door.
- `0078` and `0087` already define the artifact-selection and artifact-profile baseline that `0056` must reuse.
- `0090`, `0093`, and `0094` already define routed truth, backing truth, and lifecycle-backed serving capability for
  high-cardinality artifacts.
- `0100` is the active owner of distributed public continuation, but its first end-to-end non-terminal public
  projection is still not fully closed.
- `0102` now owns engine manifest production, engine request context, and alias policy.
- `tensorcast/common/selection_identity.py` already defines the canonical `SelectionIdentity`; `0056` must reuse it
  rather than invent a second item-identity type.
- current code already exposes:
  - canonical instance actions and results through `proto/tensorcast/plan/v1/plan.proto`,
  - `ManifestResult` and related engine-side carriers through `tensorcast/engine_adapter/adapter.py`,
  - and byte-artifact routed authority state through `daemon/service/byte_artifact_runtime_state.h`.
- current concrete gap:
  - `tensorcast/api/plan/plan.py` still leaves local instance-step execution unclosed, so the real missing work is
    convergence on one execution spine rather than another semantic layer.
  - `ManifestResult` still carries `artifact_ids/layout_id/key_set_digest_hex`, not canonical item identities rooted in
    `(artifact_id, logical_layout_hash, selection_hash)`.
  - `plan.proto` still has only worker and instance targets, so the future cluster-workflow seam exists in design but
    not yet in IR transport shape.

# Phases and Milestones

- [ ] Phase 0: Scope lock and vocabulary cleanup
  - [ ] Milestone 0.1: keep `0056` limited to runtime, ingress, signals, generic set-level orchestration, and front-door convergence.
  - [ ] Milestone 0.2: freeze canonical instance action vocabulary around `manifest/publish/hydrate/evict_local`.
  - [ ] Milestone 0.3: keep NodeAgent or the existing Instance Agent boundary as the unique instance-scoped execution
        host in this phase; do not standardize a second public runtime-hosting API.
  - [ ] Milestone 0.4: keep `mint_target` in the engine-adapter boundary and keep `materialize_into` out of required
        `0056` canonical action vocabulary for this phase.
  - [ ] Milestone 0.5: make `0102` the only owner of engine manifest production, request-context rules, and
        compatibility aliases.
  - [ ] Milestone 0.6: record the current `ManifestResult` projection gap explicitly and prevent `0056` implementations
        from inventing ad hoc local recomputation of canonical item identities.
  - [ ] Milestone 0.7: record the current local instance-step execution gap explicitly so ingress work converges on one
        spine instead of creating a second execution model.

- [ ] Phase 1: Runtime and terminal ingress
  - [ ] Milestone 1.1: land process runtime (`connect/runtime`) and gateway ingress as front-door adapters over one
        execution spine.
  - [ ] Milestone 1.2: use an explicit ingress request envelope and declared public execution class rather than raw
        `PlanSpec` plus server-side heuristics.
  - [ ] Milestone 1.3: add daemon ingress `ExecutePlan` with the same deterministic step fingerprint rules as local
        execution.
  - [ ] Milestone 1.4: keep ingress in the terminal-only execution class until a dependency-ready plan-level public
        continuation projection is available.
  - [ ] Milestone 1.5: do not add plan-private attach, wait, status, or replay surfaces.
  - [ ] Milestone 1.6: make admission depend on declared execution class and dependency readiness, not on runtime
        completion estimates.

- [ ] Phase 2: Signals and low-cardinality directory cache
  - [ ] Milestone 2.1: daemon-served worker and instance listing with `as_of_ms`, `staleness_ms`, and bounded-current
        vs degraded freshness state.
  - [ ] Milestone 2.2: GS watch streams and daemon-side cache controllers for low-cardinality routing data only.
  - [ ] Milestone 2.3: expose SDK `TensorCastSignals` as a daemon-backed read surface.
  - [ ] Milestone 2.4: expose `cache_epoch` or equivalent freshness evidence publicly, while keeping replay cursors such
        as resume tokens implementation-internal.
  - [ ] Milestone 2.5: define the watch correctness floor (`initial snapshot barrier`, cache epoch, resume token, and
        staleness-breach fail-closed behavior) and test it explicitly.
  - [ ] Milestone 2.6: define one canonical governance transport shape for plan execution and one canonical metadata
        vocabulary for non-plan RPC propagation.

- [ ] Phase 3: Set-level orchestration for high cardinality
  - [ ] Milestone 3.1: define neutral `ArtifactSetRef` and stable set digest semantics in plan and SDK surfaces.
  - [ ] Milestone 3.2: reuse `SelectionIdentity` as the canonical per-item identity and avoid introducing a second
        semantic item-identity type in SDK or proto.
  - [ ] Milestone 3.3: add worker `prefetch_set` semantics and per-item partial-result reporting.
  - [ ] Milestone 3.4: keep `prefetch_many` as SDK sugar over inline small sets rather than the primary scalable
        abstraction.
  - [ ] Milestone 3.5: make inline and manifest-backed references the first dependency-ready carrier forms for large
        sets; defer more opaque carriers until their owner and resolution contract are ready.
  - [ ] Milestone 3.6: ensure referenced set resolution verifies digest and count against the resolved item set and
        fails closed on mismatch.
  - [ ] Milestone 3.7: keep Global Store out of per-item and per-set hot truth for this path.
  - [ ] Milestone 3.8: define the `prefetch_set` readiness floor explicitly as `local_replica_ready` unless and until a
        stronger readiness selector is added to the action contract.
  - [ ] Milestone 3.9: reserve a cluster-scoped transport slot in the IR so future cluster-workflow owners do not have
        to encode workflow semantics as worker-only DAG glue.

- [ ] Phase 4: Integration alignment and rollout safety
  - [ ] Milestone 4.1: align `ManifestResult` and other `0102` engine-side carriers with neutral set references without
        introducing business-specific framework nouns.
  - [ ] Milestone 4.2: verify mixed-version compatibility between local runner and ingress mode.
  - [ ] Milestone 4.3: verify no engine-specific nouns, plan-private continuation surfaces, or second instance-hosting
        API leaked into framework proto, metrics, or docs.

# Tasks

- SDK and runtime:
  - implement runtime handle and plan submission plumbing on top of one daemon endpoint.
  - ensure local fallback and ingress mode use the same step fingerprinting inputs, canonical action names, and
    execution-class contract.
  - keep runtime as a front-door adapter; do not add a second instance-hosting lifecycle or config path.
  - add neutral set-reference helpers and keep inline small-set sugar explicitly bounded.
  - reject plans that require non-terminal public continuation before side effects start when only terminal-only ingress
    is available.

- Daemon:
  - add ingress controller, signals controller, and directory cache controller.
  - keep ingress execution terminal-only until a dependency-ready plan-level public continuation closes in `0096` and
    `0100`.
  - route instance steps through the existing daemon-local NodeAgent or Instance Agent boundary only.
  - use an explicit ingress request envelope with declared execution class rather than server-side time heuristics.
  - expose watch-backed cache epochs or equivalent freshness evidence in signals-serving code paths.
  - fail closed on directory or routing decisions when watch freshness exceeds configured bounds.

- Global Store:
  - add only low-cardinality watch streams and routing attributes required by daemon caches.
  - avoid per-request caller dependence and avoid high-cardinality per-item tables for `0056`.

- Plan and proto:
  - preserve canonical instance actions `manifest/publish/hydrate/evict_local`.
  - keep `PlanSpec` as the canonical IR and add only the minimal ingress envelope required to declare public execution
    class.
  - introduce neutral set-level worker batch carriers instead of business-specific batch vocabulary.
  - reuse `SelectionIdentity` semantics for per-item identity; if a wire message is needed, make it a field-for-field
    projection only.
  - add typed governance transport for plan execution instead of relying on free-form tags alone.
  - reserve a cluster-scoped transport slot without importing workflow-owned semantics into `0056`.
  - do not add `MintTargetAction` as required framework core surface in this phase.
  - do not add plan-private continuation or attachment carriers.

- Integration:
  - keep engine manifest production and request-context semantics in `0102`.
  - map engine-side manifest carriers to neutral set references without pushing engine nouns into framework core.
  - close the concrete `ManifestResult` projection gap through `0102` rather than local runner or gateway-only
    recomputation.

- Documentation:
  - keep `0056` framework-only, front-door oriented, and set-oriented.
  - keep `0102` integration-only and alias-owning.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/test_plan.py`
  - `source .venv/bin/activate && pytest tests/python/test_runtime.py`
- C++:
  - `bazel test //daemon:session_lifecycle_test`
  - `bazel test //daemon:grpc_service_impl_registration_test`
- Proto:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`

Additional acceptance checks:

- compare deterministic fingerprints between local and ingress execution for identical plans,
- verify `prefetch_many` lowers to the same set identity as explicit `ArtifactSetRef`,
- verify `prefetch_set(device=\"dram\")` does not silently claim `local_stable_ready` in one execution mode but not the
  other,
- verify ingress admission rejects non-terminal plans before side effects begin based on declared execution class rather
  than runtime completion estimates,
- verify watch interruption beyond budget causes routing and directory reads to fail closed,
- verify signal snapshots expose bounded-current vs degraded or stale freshness evidence consistently,
- verify large-set paths do not depend on Global Store per-item truth,
- verify referenced set carriers fail closed when resolved items do not match advertised digest or count,
- verify `SelectionIdentity` remains the single canonical semantic item identity across SDK, plan, and daemon work,
- verify no plan-private attach or status surfaces were introduced.

Rollout:

- enable ingress by daemon capability flag.
- keep caller-local runner fallback until ingress parity is proven.
- gate any future long-lived ingress continuation on explicit `0096` and `0100` closeout rather than on local
  convenience.

Backout:

- disable ingress and route `Plan.run()` back to caller-local runner.
- keep watch APIs additive and leave legacy polling path active.
- keep set-level carriers additive so small-set local flows continue working if ingress is disabled.

# Risks and Tracking

- Risk: ingress introduces divergent retry or idempotency semantics.
  - tracking: deterministic step fingerprint checks between local and daemon modes.
- Risk: ingress grows load-based admission heuristics that make identical plans behave differently across gateways.
  - tracking: require explicit execution-class fields and reject heuristic-only classifiers in review.
- Risk: high-cardinality support regresses into caller-enumerated item lists or Global Store hot truth.
  - tracking: review `ArtifactSetRef` usage and explicitly reject per-item GS ownership in `0056` work.
- Risk: implementers silently bridge the current `ManifestResult` gap inside `0056`, causing local and ingress identity
  drift.
  - tracking: require any bridge from engine manifest carriers to canonical item identities to be explicit and owned by
    `0102`.
- Risk: runtime convenience grows into a second instance-hosting surface with its own config or lifecycle semantics.
  - tracking: keep NodeAgent or the existing Instance Agent boundary as the only instance-scoped execution host in this
    phase.
- Risk: engine-specific aliases leak back into framework proto or docs.
  - tracking: review canonical names in plan proto, NodeAgent, metrics, and design docs.
- Risk: implementers add a temporary plan-private continuation surface for convenience.
  - tracking: require any non-terminal public continuation to reference `0096` and `0100` explicitly.
