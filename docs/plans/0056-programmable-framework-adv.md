---
slug: 0056-programmable-framework-adv
title: Plan - Programmable Framework Advanced Runtime, Ingress, and Signals
status: in_progress
areas: ["sdk", "daemon", "global_store", "proto", "integrations", "docs"]
created: 2026-03-04
last_updated: 2026-03-19
related_code:
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0102-engine-artifact-integration-and-high-cardinality-manifest-orchestration.md
  - docs/designs/0106-daemon-served-directory-and-target-resolution.md
  - docs/designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
  - docs/designs/0055-programmable-framework.md
  - tensorcast/api/plan/artifact_set.py
  - tensorcast/runtime.py
  - tensorcast/api/runtime.py
  - tensorcast/api/signals.py
  - tensorcast/api/plan/plan.py
  - tensorcast/node_agent/executor.py
  - tensorcast/engine_adapter/adapter.py
  - daemon/service/grpc_service_impl.h
  - daemon/service/grpc_service_impl_execute_plan.cc
  - daemon/service/controllers/status_controller.h
  - daemon/service/grpc_service_impl_execute_plan_test.cc
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/node_agent/v1/node_agent.proto
  - proto/tensorcast/global_store/v1/global_store.proto
links:
  design: ../designs/0056-programmable-framework-adv.md
---

# Objective

Deliver the advanced programmable framework layer on top of `0055` while keeping it thin:

- framework-owned `ArtifactSetRef` substrate for high-cardinality artifact sets,
- daemon-run plan ingress without semantic drift from the local runner,
- front-door convergence over one execution spine rather than another execution substrate,
- daemon-served signals and directory snapshots with explicit staleness bounds,
- and strict separation between framework core and engine integration concerns.

This plan explicitly does not turn `0056` into:

- a home-authority or lifecycle design,
- an engine-specific manifest or KV design,
- a second instance-hosting model beside NodeAgent or the existing Instance Agent boundary,
- or a second public continuation protocol beside `0096` and `0100`.

# Current State and Grounding

- `0055` already defines plan and `PlanSpec` semantics, local `Plan.run()`, and canonical instance actions in today's
  code.
- `proto/tensorcast/plan/v1/plan.proto` now carries canonical worker and instance targets, typed
  `GovernanceContext`, framework-owned `ArtifactSetRef` and `PrefetchSetAction`, and a reserved cluster-scoped target
  slot (`TARGET_TYPE_CLUSTER` plus `cluster_action`).
- `proto/tensorcast/node_agent/v1/node_agent.proto` already exposes `ExecutePlanRequest(plan, dry_run)` plus canonical
  artifact and artifact-set results, so the repository already has a real instance-scoped execution front door.
- `proto/tensorcast/daemon/v2/store_daemon.proto` now carries `ExecutePlanRequest(plan, execution_class, dry_run)` and
  a terminal-result envelope, and the C++ Store Daemon now serves the first local-only `terminal_only` ingress slice
  for worker-targeted execution.
- `0078` and `0087` already define the artifact-selection and artifact-profile baseline that `0056` must reuse.
- `0090`, `0093`, and `0094` already define routed truth, backing truth, and lifecycle-backed serving capability for
  high-cardinality artifacts.
- `0096` and `0100` already close the first landed publish `attach_existing -> Operation[T]` path for
  target-backed publish; `0056` must reuse that substrate rather than reopen it.
- `0102` now owns engine manifest production, engine request context, alias policy, and the bridge from engine-side
  manifest carriers into `ArtifactSetRef`.
- the exact daemon-served directory, bounded-staleness target identity, and
  `NodeAgentDirectory` contract are now being frozen as shared prerequisite work
  in `0106`; `0056` depends on that contract but does not own it in detail.
- `tensorcast/common/selection_identity.py` already defines the canonical `SelectionIdentity`; `0056` must reuse it
  rather than invent a second item-identity type.
- current code already exposes:
  - canonical instance actions and results through `proto/tensorcast/plan/v1/plan.proto`,
  - `ManifestResult` and related engine-side carriers through `tensorcast/engine_adapter/adapter.py`,
  - and byte-artifact routed authority state through `daemon/service/byte_artifact_runtime_state.h`.
- current concrete gap:
  - `tensorcast/api/plan/plan.py` still leaves local instance-step execution unclosed, so the real missing work is
    convergence on one execution spine rather than another semantic layer.
  - runtime ingress now executes local worker-targeted plans through the real daemon RPC, but the daemon-local
    bridge to NodeAgent or an in-process Instance Agent boundary is still not implemented.
  - `TensorCastSignals.get_worker_status()` now consumes daemon-served `as_of_ms`, `staleness_ms`, `cache_epoch`, and
    `freshness_state`, but watch-backed degraded freshness evidence and directory-cache correctness are still pending.
  - typed governance transport is implemented for `PlanSpec`, but no canonical non-plan RPC propagation vocabulary has
    landed yet.
  - `ManifestResult` is still an engine-side result carrier, not a framework-owned `ArtifactSetRef`; `0056` must not
    derive canonical set identity from it directly.
  - the future cluster-workflow seam now exists in IR transport shape, but there is still no builder or daemon
    execution surface that owns cluster workflow semantics.
  - inter-daemon control-plane dispatch still does not exist as dependency-ready
    plan substrate; current peer-daemon transport helpers are data-plane-only.

# Status Update

- [x] Scope and ownership boundaries are now frozen in the `0056` design and plan.
- [x] `ArtifactSetRef` contract, digest or cardinality rules, carrier forms, resolution contract, and the
      no-raw-`ManifestResult` rule are now explicit in docs.
- [x] Proto, SDK, runtime, and NodeAgent transport now carry `ArtifactSetRef`, `PrefetchSetAction`, and
      `ArtifactSetResult` for framework-owned set orchestration.
- [x] `prefetch_many` now lowers to bounded inline `ArtifactSetRef` (`MAX_INLINE_ARTIFACT_SET_ITEMS=1024`), and
      `prefetch_set` reports per-item partial results while holding the worker readiness floor at
      `local_replica_ready`.
- [x] `manifest_backed` owner-provided resolution and explicit `0102` bridge transport now land additively through
      `ManifestArtifactSetBridge`, `ManifestResult`, `PrefetchSetAction`, and NodeAgent result transport.
- [x] Typed governance transport now lands in `CallContext.governance` and `PlanSpec.governance`.
- [x] Runtime front-door wiring now lands through `tensorcast.connect(...)`, active-runtime `tensorcast.plan(ctx)`,
      and daemon-client `ExecutePlan` request-envelope submission with declared execution class.
- [x] The IR now reserves an explicit cluster-scoped transport slot (`TARGET_TYPE_CLUSTER` plus opaque
      `cluster_action`) without importing workflow semantics into `0056`.
- [x] Runtime-backed signals now expose a direct daemon `GetWorkerStatus` snapshot surface for the connected daemon.
- [x] The broader daemon-execution slice remains scope-locked in docs as `local-only` plus `terminal_only`; the first
      worker-targeted sub-slice is now implemented, while the daemon-local instance bridge is still pending.
- [x] Real daemon-side `ExecutePlan` serving now lands for the first `local-only` plus `terminal_only` slice behind
      `gateway_ingress_enabled`, returning terminal result envelopes and failing closed on remote worker, instance, and
      cluster targets.
- [x] Signal freshness metadata is now daemon-served for `GetWorkerStatus`: the daemon stamps `as_of_ms`,
      `staleness_ms`, `cache_epoch`, and `freshness_state`, and the SDK consumes them with mixed-version fallback.
- [x] First-slice ingress verification now covers real daemon serving, remote-target fail-closed behavior, and
      `ArtifactSetRef` digest mismatch fail-closed behavior through daemon-side C++ tests.
- [ ] Governance transport is only complete for `PlanSpec`; non-plan RPC metadata propagation remains pending.
- [ ] Daemon-side instance bridging and watch-backed signals-serving controllers are still pending.

# Execution Order

- [x] Stage 1: land `0056` scope lock and `ArtifactSetRef` substrate contract first.
- [x] Stage 2: hand bridge closeout to `0102` Phase 0-2 before `0056` runtime or ingress code consumes engine-side
      high-cardinality outputs.
- [ ] Stage 3: return to `0056` for daemon-side ingress execution, set orchestration completion, and mixed local or
      ingress consistency.
- [ ] Stage 4: execute `0103` Phase 1-3 after the set substrate and bridge semantics are stable.
- [ ] Stage 5: evaluate `0103` Phase 4 coordinated-slot follow-up only if the earlier phases still leave a real gap.

# Phases and Milestones

- [x] Phase 0: Scope lock and substrate boundaries
  - [x] Milestone 0.1: keep `0056` limited to runtime, ingress, signals, `ArtifactSetRef`-backed set orchestration,
        and front-door convergence.
  - [x] Milestone 0.2: freeze canonical instance action vocabulary around `manifest/publish/hydrate/evict_local`.
  - [x] Milestone 0.3: keep NodeAgent or the existing Instance Agent boundary as the unique instance-scoped execution
        host in this phase; do not standardize a second public runtime-hosting API.
  - [x] Milestone 0.4: keep `mint_target` in the engine-adapter boundary and keep `materialize_into` out of required
        `0056` canonical action vocabulary for this phase.
  - [x] Milestone 0.5: make `0102` the only owner of engine manifest production, request-context rules, and
        compatibility aliases.
  - [x] Milestone 0.6: freeze `ArtifactSetRef` as framework-owned substrate and prevent `0056` implementations from
        inventing ad hoc local recomputation from `ManifestResult`.
  - [x] Milestone 0.7: record the current local instance-step execution gap explicitly so ingress work converges on one
        spine instead of creating a second execution model.

- [x] Phase 1: `ArtifactSetRef` substrate contract
  - [x] Milestone 1.1: define `ArtifactSetRef` wire and SDK model in plan and runtime surfaces as the only
        framework-owned set contract.
  - [x] Milestone 1.2: define stable set digest and exact cardinality semantics over canonical `SelectionIdentity`.
  - [x] Milestone 1.3: make `inline` and `manifest_backed` the first dependency-ready carrier forms and remove vague
        set-carrier wording from `0056`.
  - [x] Milestone 1.4: define inline and manifest-backed resolution rules that verify digest and count and fail closed
        on mismatch.
  - [x] Milestone 1.5: add worker `prefetch_set` as the normative generic surface with per-item partial-result
        reporting.
  - [x] Milestone 1.6: keep `prefetch_many` as bounded SDK sugar that lowers to inline `ArtifactSetRef`.
  - [x] Milestone 1.7: state explicitly that `0056` does not derive canonical set identity from raw `ManifestResult`.

- [x] Phase 2: Bridge dependency handoff to `0102`
  - [x] Milestone 2.1: make `0102` the only owner of the `ManifestResult -> ArtifactSetRef` bridge form and its
        versioned metadata.
  - [x] Milestone 2.2: block `0056` runtime or ingress implementation from consuming engine-side high-cardinality
        output until it receives explicit bridge output rather than raw `ManifestResult`.
  - [x] Milestone 2.3: identify additive `plan.proto` and `node_agent.proto` transport changes required to preserve the
        bridge output end to end.

- [ ] Phase 3: Set orchestration and ingress or runtime closeout
  - [x] Milestone 3.1: transport `ArtifactSetRef` through runtime, daemon ingress request transport, and worker
        execution without set
        identity drift.
  - [x] Milestone 3.2: land process runtime (`connect/runtime`) and gateway ingress client adapters over one
        execution spine.
  - [x] Milestone 3.3: use an explicit ingress request envelope and declared public execution class rather than raw
        `PlanSpec` plus server-side heuristics.
  - [x] Milestone 3.4: add daemon-side ingress `ExecutePlan` execution with the same deterministic step fingerprint
        rules as local execution for the first worker-targeted `ArtifactSetRef` flows.
  - [x] Milestone 3.5: define the `prefetch_set` readiness floor explicitly as `local_replica_ready` unless and until a
        stronger readiness selector is added to the action contract.
        Follow-on owner note:
        `0104` owns that stronger worker-local realization contract; `0056`
        keeps `prefetch_set` unchanged.
  - [x] Milestone 3.6: prove at least one end-to-end high-cardinality flow that runs through `ArtifactSetRef` rather
        than raw manifest outputs.
  - [x] Milestone 3.7: keep ingress in the terminal-only execution class until a dependency-ready plan-level public
        continuation projection is available.
  - [x] Milestone 3.8: do not add plan-private attach, wait, status, or replay surfaces.
  - [x] Milestone 3.9: make admission depend on declared execution class and dependency readiness, not on runtime
        completion estimates.
  - [x] Milestone 3.10: reserve a cluster-scoped transport slot in the IR so future cluster-workflow owners do not
        have to encode workflow semantics as worker-only DAG glue.
        Follow-on owner note:
        `0104` is the intended first rollout consumer of this seam, while cluster
        truth and barriers remain outside `0056`.
  - [x] Milestone 3.11: scope the first daemon-side execution slice to `local-only` plus `terminal_only` with
        explicit fail-closed behavior for remote worker targets, non-local instance targets, and cluster targets.

- [ ] Phase 4: Signals and low-cardinality directory cache
  - [ ] Milestone 4.1: close the shared directory contract from `0106`:
    - worker directory
    - instance directory
    - `NodeAgentDirectory`
    - bounded-staleness freshness evidence
  - [ ] Milestone 4.2: GS watch streams and daemon-side cache controllers for low-cardinality routing data only.
  - [x] Milestone 4.3: expose SDK `TensorCastSignals` as a daemon-backed read surface for connected-worker status.
  - [x] Milestone 4.4: expose `cache_epoch` or equivalent freshness evidence publicly, while keeping replay cursors such
        as resume tokens implementation-internal.
  - [ ] Milestone 4.5: define the watch correctness floor (`initial snapshot barrier`, cache epoch, resume token, and
        staleness-breach fail-closed behavior) and test it explicitly.
  - [x] Milestone 4.6: define one canonical governance transport shape for plan execution.
  - [ ] Milestone 4.7: define one canonical metadata vocabulary for non-plan RPC propagation.
        Follow-on owner note:
        `0104` rollout target resolution depends on these daemon-served
        directory and freshness guarantees rather than direct SDK-to-GS
        directory calls.
  - [ ] Milestone 4.8: make `NodeAgentDirectory` a hard prerequisite to any
        attempt to close daemon-side instance-step routing.

- [ ] Phase 5: Rollout safety and verification
  - [ ] Milestone 5.1: verify mixed-version compatibility between local runner and ingress mode.
  - [ ] Milestone 5.2: verify no engine-specific nouns, plan-private continuation surfaces, or second instance-hosting
        API leaked into framework proto, metrics, or docs.
  - [ ] Milestone 5.3: keep Global Store out of per-item and per-set hot truth for this path.

# Tasks

- SDK and runtime:
  - define `ArtifactSetRef` in SDK and plan-building surfaces before adding execution-path convenience wrappers.
  - add `prefetch_set` over `ArtifactSetRef` and keep inline small-set sugar explicitly bounded.
  - [x] implement runtime handle and plan submission plumbing on top of one daemon endpoint.
  - [x] ensure plan building and ingress request transport use the same step fingerprinting inputs, canonical action
    names, `ArtifactSetRef` identity rules, and execution-class contract.
  - keep runtime as a front-door adapter; do not add a second instance-hosting lifecycle or config path.
  - [x] reject plans that require non-terminal public continuation before side effects start when only terminal-only ingress
    is available.

- Daemon:
  - preserve `ArtifactSetRef` through ingress and worker execution rather than reconstructing it from integration-local
    outputs.
  - [x] add daemon ingress RPC transport (`ExecutePlanRequest`) with declared execution class.
  - [x] override and serve `StoreDaemonService.ExecutePlan` in the C++ daemon instead of relying on the generated
        default `UNIMPLEMENTED` path.
  - [ ] add one typed daemon-local bridge to NodeAgent or the in-process Instance Agent boundary under unified config.
  - [ ] add ingress controller, signals controller, and directory cache controller.
  - keep ingress execution terminal-only until a dependency-ready plan-level public continuation closes in `0096` and
    `0100`.
  - [ ] route instance steps through the existing daemon-local NodeAgent or Instance Agent boundary only.
  - [x] fail closed on remote worker targets, non-local instance targets, and cluster targets in the first daemon-side
        `ExecutePlan` implementation.
  - [x] use an explicit ingress request envelope with declared execution class rather than server-side time heuristics.
  - expose watch-backed cache epochs or equivalent freshness evidence in signals-serving code paths.
  - fail closed on directory or routing decisions when watch freshness exceeds configured bounds.

- Global Store:
  - add only low-cardinality watch streams and routing attributes required by daemon caches.
  - avoid per-request caller dependence and avoid high-cardinality per-item tables for `0056`.

- Plan and proto:
  - preserve canonical instance actions `manifest/publish/hydrate/evict_local`.
  - [x] keep `PlanSpec` as the canonical IR and add only the minimal ingress envelope required to declare public execution
    class.
  - introduce `ArtifactSetRef` and `PrefetchSetAction` as the framework-owned set contract instead of vague or
    business-specific batch vocabulary.
  - reuse `SelectionIdentity` semantics for per-item identity; if a wire message is needed, make it a field-for-field
    projection only.
  - [x] add typed governance transport for plan execution instead of relying on free-form tags alone.
  - [x] reserve a cluster-scoped transport slot without importing workflow-owned semantics into `0056`.
  - do not add `MintTargetAction` as required framework core surface in this phase.
  - do not add plan-private continuation or attachment carriers.

- Integration:
  - keep engine manifest production and request-context semantics in `0102`.
  - consume only `ArtifactSetRef` or another explicit `0102` bridge output in framework code.
  - close the concrete `ManifestResult` projection gap through `0102` rather than local runner or gateway-only
    recomputation.

- Documentation:
  - keep `0056` framework-only, front-door oriented, and set-oriented.
  - keep `0102` integration-only and alias-owning.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/api/test_plan_spec.py`
  - `source .venv/bin/activate && pytest tests/python/api/test_runtime.py`
  - `source .venv/bin/activate && pytest tests/python/node_agent/test_plan_execution.py`
- C++:
  - `bazel test //daemon:grpc_service_impl_status_test`
  - `bazel test //daemon:grpc_service_impl_execute_plan_test`
  - `bazel test //daemon:grpc_service_impl_startup_gate_test`
- Proto:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`

Latest validation:

- Passed: `source .venv/bin/activate && bash tools/build_proto_python.sh raw`
- Passed: `source .venv/bin/activate && pytest tests/python/api/test_runtime.py`
- Passed: `source .venv/bin/activate && pytest tests/python/api/test_plan_spec.py tests/python/node_agent/test_plan_execution.py`
- Passed: `bazel test //daemon:grpc_service_impl_status_test //daemon:grpc_service_impl_execute_plan_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- Blocked by pre-existing repo failures: `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  currently fails in `//proto/tensorcast/daemon/v2:daemon_proto_lint` on existing RPC naming violations unrelated to
  this change (`ImportArtifactFromPathStream`, `StartPublishTargetReplica`, `MaterializeIntoMappedTarget`).

Completed acceptance checks in this slice:

- real daemon no longer returns the generated default `UNIMPLEMENTED` for `StoreDaemonService.ExecutePlan`,
- first daemon-side `ExecutePlan` implementation rejects remote worker targets before dispatch,
- first daemon-side `ExecutePlan` implementation rejects instance targets that are not daemon-local,
- first daemon-side `ExecutePlan` implementation rejects `TARGET_TYPE_CLUSTER` rather than executing workflow-owned
  semantics privately,
- signal snapshot freshness fields for `GetWorkerStatus` now come from daemon-side evidence rather than SDK-local
  wall-clock synthesis,
- referenced set carriers fail closed when resolved items do not match advertised digest or count,
- no plan-private attach or status surfaces were introduced in this slice.

Remaining acceptance checks:

- compare deterministic fingerprints between local and ingress execution for identical plans,
- verify proto, SDK, and design all express the same framework-owned `ArtifactSetRef` contract,
- verify `prefetch_many` lowers to the same set identity as explicit `ArtifactSetRef`,
- verify `prefetch_set(device=\"dram\")` does not silently claim `local_stable_ready` in one execution mode but not the
  other,
- verify ingress admission rejects non-terminal plans before side effects begin based on declared execution class rather
  than runtime completion estimates,
- verify watch interruption beyond budget causes routing and directory reads to fail closed,
- verify signal snapshots expose bounded-current vs degraded or stale freshness evidence consistently,
- verify large-set paths do not depend on Global Store per-item truth,
- verify `SelectionIdentity` remains the single canonical semantic item identity across SDK, plan, and daemon work,
- verify `0056` does not derive canonical set identity directly from `ManifestResult`,
- verify at least one high-cardinality end-to-end flow runs through `ArtifactSetRef`,
- verify mixed local-run vs daemon-ingress outcomes stay aligned for worker-targeted `prefetch_set` success cases.

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
  - tracking: require any bridge from engine manifest carriers to `ArtifactSetRef` to be explicit and owned by
    `0102`.
- Risk: `ArtifactSetRef` remains a doc-only abstraction and never becomes the actual proto or SDK contract.
  - tracking: require plan proto, runtime builder, and acceptance checks to name the same `ArtifactSetRef` fields and
    carrier forms.
- Risk: runtime convenience grows into a second instance-hosting surface with its own config or lifecycle semantics.
  - tracking: keep NodeAgent or the existing Instance Agent boundary as the only instance-scoped execution host in this
    phase.
- Risk: engine-specific aliases leak back into framework proto or docs.
  - tracking: review canonical names in plan proto, NodeAgent, metrics, and design docs.
- Risk: implementers add a temporary plan-private continuation surface for convenience.
  - tracking: require any non-terminal public continuation to reference `0096` and `0100` explicitly.
- Risk: the plan document overstates ingress or signals completion because client-envelope progress is mistaken for real
  daemon serving.
  - tracking: require every future `[x]` against ingress or signals to cite a concrete server-side entrypoint and a
    real daemon-side test, not only proto or fake-client coverage.

# Progress Log

| Date | Stage | Status | Notes |
| --- | --- | --- | --- |
| 2026-03-17 | Phase 3 / 4 slice | Completed | Landed real daemon `ExecutePlan` serving for the first `gateway_ingress_enabled` `local-only` `terminal_only` worker-target slice; added daemon-served `GetWorkerStatus` freshness metadata (`as_of_ms`, `staleness_ms`, `cache_epoch`, `freshness_state`); updated SDK fallback handling and synced status in this plan. Evidence: `pytest tests/python/api/test_runtime.py`, `pytest tests/python/api/test_plan_spec.py tests/python/node_agent/test_plan_execution.py`, `bazel test //daemon:grpc_service_impl_status_test //daemon:grpc_service_impl_execute_plan_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`. |
