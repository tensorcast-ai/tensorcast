# Serving Runtime Reference Framework

This example is the smallest TensorCast Level 1 framework integration shape.
It consumes an existing durable serving artifact through
`ServingRuntimeSession` and validates the integration with the conformance kit.

The runtime path intentionally uses only:

- `tensorcast.serving.runtime`
- `tensorcast.serving.hosts`
- `tensorcast.serving.testing`

It does not import `tensorcast.serving.integration`, builder/admin modules,
vLLM, source catalog helpers, retained preload helpers, or low-level
bind/swap/restore functions.

Run:

```bash
python examples/serving_runtime_reference_framework/reference_framework.py
```
