# TensorCast Plan API

The plan module provides a programmable orchestration surface for control-plane
actions. Plans are serialized as a versioned IR (`PlanSpec`) and executed
against worker identities (`daemon_id`) with bounded concurrency and
best-effort cancellation.

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
- **TargetSpec**: Capability-based reference to engine-owned buffers.
- **TransformSpec**: Named transform invocation with structured arguments.

## Current surface (Phase-4)

- `tensorcast.plan(ctx)` builds a plan with a single `CallContext`.
- Worker steps: `prefetch`, `pin_device_residency`, `unpin_device_residency`.
- Instance steps: `transform_into`, `transform_register` (executed via Node Agent).
- `Plan.run()` executes worker steps locally; instance steps require submitting
  the `PlanSpec` to a node agent with an engine adapter.
- Instance targets must be expressed as `TargetSpec` capabilities minted by the
  engine adapter on the target process.
- Worker prefetch supports both GPU and daemon-owned CPU/DRAM warm replicas:
  - GPU: `device="cuda:0"` or `0`
  - daemon-owned DRAM: `device="cpu"`, `"dram"`, or `-1`
  `pin_device_residency` remains GPU-only (`device_id >= 0`).

## Execution semantics

- `Plan.run()` executes with bounded concurrency and returns a `PlanResult`.
- Any failure marks the overall plan as failed (`PlanResult.ok = False`).
- No rollback is attempted; completed steps may have persistent side effects.
- In-flight operations receive best-effort cancellation after the first failure.
