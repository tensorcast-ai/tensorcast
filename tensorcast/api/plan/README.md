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

## Current surface (Phase-4)

- `tensorcast.plan(ctx)` builds a plan with a single `CallContext`.
- Worker steps: `prefetch`, `pin_device_residency`, `unpin_device_residency`.
- Instance steps: `transform_into`, `transform_register` (executed via Node Agent).
- Instance artifact steps: `manifest`, `publish`, `hydrate`, `evict_local`.
- `Plan.run()` executes worker steps locally today; instance steps require
  submitting the same `PlanSpec` to NodeAgent or an equivalent in-process
  Instance Agent boundary with an engine adapter.
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
