<!-- Copyright (c) 2025 -->

# Materialization Contracts

Header-only specs and interfaces shared between control and dataplane layers:

- Replica specs (`ReplicaKey`, `ReplicaHandle`, `MaterializeHints`).
- View descriptors extracted from `loading_spec` (`ViewSpec`, `ViewPlan`).
- Control contracts (`MaterializationRequest`, `ArtifactSourceRouter`,
  `IArtifactLoaderRegistry`) plus lightweight fakes for unit tests.

Dependencies are intentionally minimal: absl + store/common/device/replica
types. Downstream packages link against this Bazel target instead of including
dataplane headers directly.
