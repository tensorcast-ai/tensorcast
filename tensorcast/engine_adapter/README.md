# Engine Adapter

The engine adapter bridges TensorCast plans with in-process execution engines.
It provides:

- Target minting + capability validation (`TargetSpec`)
- Transform plugin registry (execute node-local transforms)
- Convenience identity transform (`identity.v1`)

The adapter is intended to run inside the engine process. Node agents can
receive a reference to an adapter when executing instance-scoped plan steps.
