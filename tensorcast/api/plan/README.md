# TensorCast Plan API

The plan module provides a programmable orchestration surface for control-plane
actions. Plans are serialized as a versioned IR (`PlanSpec`) and executed
against worker identities (`daemon_id`) with bounded concurrency and
best-effort cancellation.

Repository rule:

- `PlanSpec` is the canonical orchestration IR,
- `tensorcast.plan(...)`, future runtime front doors, and daemon ingress should all lower to that same IR,
- and NodeAgent or the in-process Instance Agent boundary is the unique
  instance-scoped execution host in the current phase.

## Key Concepts

- **Plan**: A collection of steps targeting workers and instances.
- **PlanSpec**: A versioned proto representation of a plan for deterministic
  replay and idempotent retries (`proto/tensorcast/plan/v1/plan.proto`).
- **ArtifactSelection**: Shared selection message owned by
  `proto/tensorcast/common/v1/common.proto` and embedded in PlanSpec actions.
- **Selection identity**: Each step binds a selection fingerprint derived from
  `(artifact_id, logical_layout_hash, selection_hash)` so retries join the same operation.
  Hashes are computed via `tensorcast.common.selection_identity` from canonical
  index bytes or view index bytes plus view/subset inputs.
- **Execution spine**: `PlanSpec` lowers worker steps to Store Daemons and
  instance steps to NodeAgent or the in-process Instance Agent boundary, rather
  than creating a second execution model per entrypoint.
- **TargetSpec**: Capability-based reference to engine-owned buffers.
- **TransformSpec**: Named transform invocation with structured arguments.
- **Artifact actions**: Canonical instance actions for artifact lifecycle orchestration:
  `manifest`, `publish`, `hydrate`, `evict_local`.

## Current surface

- `tensorcast.plan(ctx)` builds a plan with a single `CallContext`.
- `CallContext.governance` carries typed plan governance transport (`lane`,
  `policy_version`, `staleness_budget_ms`) directly in `PlanSpec`.
- `PlanSpec` now also reserves a cluster-scoped transport slot in the IR
  (`TARGET_TYPE_CLUSTER` plus opaque `cluster_action`) without defining any
  workflow semantics in the framework layer.
- Worker steps: `prefetch`, `pin_device_residency`, `unpin_device_residency`.
- Instance steps: `transform_into`, `transform_register` (executed via Node Agent).
- For `PURE_TRANSFORM` serving publication on top of `transform_register`,
  prefer the repo-owned helpers
  `build_pure_transform_publication_spec(...)` or
  `build_pure_transform_transform_spec(...)`. These helpers now attach a typed
  `TransformSpec.publication_spec` carrier instead of relying on internal
  `tc_serving_*` transform args. The legacy string-arg path is still accepted
  as a compatibility fallback, but the default identity `transform_register`
  path now consumes the typed publication spec first and prepares the reserved
  serving-manifest tensor before registration.
- If you are building the plan directly, `InstanceStepBuilder.transform_register_pure_transform(...)`
  wraps that helper path and lowers to the same canonical `transform_register`
  IR while preserving typed publish intent (`layout_id`, `requirements`,
  `readiness_policy`, version keys, manifest ref, and contract family) on the
  plan wire.
- After execution, the returned `RepresentationPublishSpec` can now be passed
  directly to `Store.start_representation_publish_attempt(...)` or
  `Store.complete_representation_publish_attempt(...)` to run the typed
  `representation_publish` closeout path. If that spec points back to a
  source artifact with exactly one attached layout, the Store helper can infer
  `layout_id`; otherwise the same typed carrier can already bring explicit
  attempt inputs through the plan / node-agent path.
- `PlanResult.require_representation_publish_spec(...)` provides the typed
  extraction path for that spec. Likewise,
  `Store.start_plan_repo_owned_representation_publish_attempt(...)` /
  `Store.complete_plan_repo_owned_representation_publish_attempt(...)` can
  route the `transform_register_pure_transform(...)` result into the same
  repo-owned publish helpers without manually unpacking `artifact_result`.
- For the current single-rank canonical publish shape, the same bundle can go
  through `Store.start_canonical_representation_publish_attempt(...)` or
  `Store.complete_canonical_representation_publish_attempt(...)` to pin
  `AssemblyRequirementSetRef.canonical_full()`.
- For structural publish shapes, pair the bundle with
  `build_representation_publish_requirements(...)` or the structural Store
  helpers so `pp`/`ep` requirements come from explicit family choice plus
  deterministic structural-view lineage instead of layout hints.
- When the transform-side bundle already carries `contract_family`, the same
  result can go straight into
  `Store.start_repo_owned_representation_publish_attempt(...)` or
  `Store.complete_repo_owned_representation_publish_attempt(...)`.
- If the builder is not running through `transform_register`, the same serving
  publication semantics are also available through
  `Store.register_pure_transform_publication(...)` and
  `Store.complete_pure_transform_publication(...)` on top of already-built
  in-memory tensors. For the current canonical single-source path, that
  one-shot helper can also contribute the source artifact before sealing when
  `source_contribution_device=...` is provided. Structural `pp` / `ep` publish
  can supply `source_contribution_artifacts=(view_a, view_b, ...)` so the same
  helper derives structural view ids and submits multiple source contributions
  before sealing.
- Instance artifact steps: `manifest`, `publish`, `hydrate`, `evict_local`.
- `tensorcast.connect(daemon_address=...)` installs a runtime front door bound
  to one daemon endpoint; with an active runtime, `Plan.run()` submits the same
  `PlanSpec` through daemon ingress in `terminal_only` mode.
- Without an active runtime, `Plan.run()` keeps the existing local worker-step
  execution path; instance steps still require submitting the same `PlanSpec`
  to NodeAgent or an equivalent in-process Instance Agent boundary with an
  engine adapter.
- Node Agent executes artifact instance actions through the engine adapter
  hooks (`register_artifact_fns(...)`).
- Any future runtime or gateway ingress should remain a front-door adapter over
  the same `PlanSpec -> NodeAgent / Instance Agent` execution spine rather than
  introducing a second instance-hosting contract.
- Instance targets must be expressed as `TargetSpec` capabilities minted by the
  engine adapter on the target process.
- Worker prefetch supports both GPU and daemon-owned CPU/DRAM warm replicas:
  - GPU: `device="cuda:0"` or `0`
  - daemon-owned DRAM: `device="cpu"`, `"dram"`, or `-1`
  `pin_device_residency` remains GPU-only (`device_id >= 0`).

## Execution semantics

- `Plan.run()` executes with bounded concurrency and returns a `PlanResult`.
- The current local runner is not yet a complete second implementation of
  instance execution; instance-scoped execution converges through NodeAgent or
  the in-process Instance Agent boundary.
- Any failure marks the overall plan as failed (`PlanResult.ok = False`).
- No rollback is attempted; completed steps may have persistent side effects.
- In-flight operations receive best-effort cancellation after the first failure.
