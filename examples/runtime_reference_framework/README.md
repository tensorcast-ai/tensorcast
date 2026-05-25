# Runtime Reference Framework

This example is the smallest TensorCast Level 1 framework integration shape.
It consumes an existing durable artifact through
`Artifact.realize(... model_runtime ...)` and validates the integration with the
artifact-runtime conformance kit.

The runtime path intentionally uses only:

- `tensorcast`
- `tensorcast.artifact_runtime.host`
- `tensorcast.artifact_runtime.testing`

It does not import `tensorcast.serving`, vLLM, source catalog helpers, retained
preload helpers, `ArtifactRuntimeSession`, or low-level bind/swap/restore
functions.

Run:

```bash
python examples/runtime_reference_framework/reference_framework.py
```
