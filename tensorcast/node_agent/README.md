# TensorCast Node Agent

The node agent is a node-local control-plane helper that executes plan steps on
behalf of a controller. Worker/daemon steps are executed locally via StoreDaemon
RPCs, while instance/engine steps are dispatched through an in-process engine
adapter (transform plugins + target resolution).

## Responsibilities

- Execute worker-scoped steps (`prefetch`, `pin_device_residency`, `unpin_device_residency`).
- Execute instance-scoped steps (`transform_into`, `transform_register`) through
  the Engine Adapter registry.
- Enforce target identity checks (`daemon_id`, optional `instance_id`).
- Provide a gRPC surface (`NodeAgentService`) for plan execution and agent info.

## Current limitations

- The agent does not attempt plan-level retries or rollbacks; it reports per-step
  status and leaves retry decisions to the caller.
- Instance steps require an Engine Adapter to be configured at startup; missing
  adapters fail instance actions with FAILED_PRECONDITION.

## Key modules

- `executor.py`: plan execution logic (worker + instance steps).
- `server.py`: gRPC servicer wiring for `NodeAgentService`.
- `__main__.py`: node agent entrypoint (`--config`).

## Configuration

- Schema: `proto/tensorcast/config/v1/node_agent_config.proto`
- Example: `examples/config/node_agent_config.yaml`
