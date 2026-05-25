# Serving Binding Reference Consumer

This example is a minimal TensorCast-side consumer for the serving binding
prefetch/acquire flow. It is intentionally independent of vllm so the
daemon API can be exercised as a public reference path.

```bash
source .venv/bin/activate
python examples/serving_binding_consumer/reference_consumer.py \
  --daemon-address 127.0.0.1:8073 \
  --source-artifact-id mi2:<source-artifact> \
  --device-uuid <daemon-device-uuid>
```

The parent process writes a resolved serving binding spec cache entry, calls
`PrefetchServingBinding`, and launches a worker subprocess that calls
`AcquireBindingValue` and releases the returned lease.
