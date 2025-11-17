<!-- Copyright (c) 2025 -->

# Materialization Control Layer

Hosts orchestration logic that wires StoreEngine components to the contracts
layer: `MaterializationService`, `MaterializeOrchestrator`,
`ReplicaRegistrationHelper`, and adapters to loader registries/source routers.
Only depends on:

- `//core/store/materialization/contracts`
- Store components (`GlobalStoreClient`, `ReplicaRegistry`, etc.)
- Planning helpers (`chunk_aware_strategy`).

No direct includes of dataplane loaders are allowed beyond the temporary
adapter introduced in design 0027 while the new registries roll out.
