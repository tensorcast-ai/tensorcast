<!-- Copyright (c) 2025, TensorCast Team -->

# Materialization Planning

Contains UMA-aware planning helpers (chunk-aware strategy, future repair
policies) that map replica metadata to dataplane actions. Planning code only
depends on:

- `//core/store/materialization/contracts`
- UMA / replica headers

and exports narrow interfaces that the control layer invokes before instantiating
loaders. Dataplane runtimes consume the resulting plans but never call back into
planning helpers.
