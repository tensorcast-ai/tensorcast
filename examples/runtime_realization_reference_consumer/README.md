# Runtime Realization Reference Consumer

This example is a minimal TensorCast-side consumer for the runtime realization
prefetch/acquire flow. It is intentionally independent of internal-vLLM so the
daemon API can be exercised as a public reference path.

```bash
source .venv/bin/activate
python examples/runtime_realization_reference_consumer/reference_consumer.py \
  --daemon-address 127.0.0.1:8073 \
  --source-artifact-id mi2:<source-artifact> \
  --device-uuid <daemon-device-uuid>
```

The parent process writes a resolved realization target cache entry, calls the
daemon `PrefetchServingBinding` wire RPC, and launches a worker subprocess that
reconstructs a `RealizationTarget` plus `PrefetchHandoff`, calls
`AcquireBindingValue`, and releases the returned lease.
