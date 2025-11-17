<!-- Copyright (c) 2025 -->

# Materialization Dataplane

Home for loader runtime code: sources/sinks, pump implementations, disk/P2P
loaders, view planners/executors, metadata helpers, and the loader registry.
Targets are organized by concern (contracts, runtime, sources, sinks, view,
metadata, verification, registry) so control packages only depend on the
registry/router exports. The legacy `core/store/loader` aliases have been
removed, making these targets the single source of truth for dataplane code.
