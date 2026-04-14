# Engine Adapter

The engine adapter bridges TensorCast plans with in-process execution engines.
It provides:

- Target minting + capability validation (`TargetSpec`)
- Transform plugin registry (execute node-local transforms)
- Convenience identity transform (`identity.v1`)
- Canonical artifact action results (`ManifestResult`, `PublishResult`,
  `HydrateResult`, `BatchResult`)
- Explicit high-cardinality bridge metadata
  (`ManifestArtifactSetBridge`) so engine manifest output can lower into
  framework-owned `ArtifactSetRef` without local raw-manifest derivation

The adapter is intended to run inside the engine process. Node agents can
receive a reference to an adapter when executing instance-scoped plan steps.

The neutral module surface for the canonical artifact contract is
`tensorcast.engine_adapter.artifact_api`.
Adapter-local request-context requirements such as a non-empty
`engine_request_id` are integration preconditions enforced by adapter code; they
do not add new framework action semantics.
