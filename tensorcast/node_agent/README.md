# TensorCast Node Agent

The node agent is a node-local control-plane helper that executes plan steps on
behalf of a controller. Worker/daemon steps are executed locally via StoreDaemon
RPCs, while instance/engine steps are dispatched through an in-process engine
adapter (transform plugins + target resolution).

Current repository role:

- NodeAgent is the canonical instance-scoped execution host in the current
  programmable execution spine.
- Runtime front doors and daemon ingress should lower instance work onto this
  boundary or an equivalent in-process Instance Agent host with the same
  semantics.
- NodeAgent is an execution host, not a second workflow, continuation, or
  lifecycle owner.

## Responsibilities

- Execute worker-scoped steps (`prefetch`, `pin_device_residency`, `unpin_device_residency`).
- Execute instance-scoped steps (`transform_into`, `transform_register`) through
  the Engine Adapter registry.
- Execute canonical artifact instance actions (`manifest`, `publish`, `hydrate`,
  `evict_local`) through the engine adapter boundary.
- Enforce target identity checks (`daemon_id`, optional `instance_id`).
- Propagate `CallContext.deadline_ms` to daemon RPC timeouts for worker steps.
- Provide a gRPC surface (`NodeAgentService`) for plan execution and agent info.
- Resolve local `worker_id` from the connected Store Daemon status path when
  registering or heartbeating the instance into Global Store, instead of
  reading worker-directory rows directly from Global Store.

## Current limitations

- The agent does not attempt plan-level retries or rollbacks; it reports per-step
  status and leaves retry decisions to the caller.
- Instance steps require an Engine Adapter to be configured at startup; missing
  adapters fail instance actions with FAILED_PRECONDITION.
- Public continuation, attach, replay, and workflow semantics are owned outside
  NodeAgent; this host executes canonical actions and returns canonical results.

## Key modules

- `executor.py`: plan execution logic (worker + instance steps).
- `server.py`: gRPC servicer wiring for `NodeAgentService`.
- `__main__.py`: node agent entrypoint (`--config`).

## Configuration

- Schema: `proto/tensorcast/config/v1/node_agent_config.proto`
- Example: `examples/config/node_agent_config.yaml`
