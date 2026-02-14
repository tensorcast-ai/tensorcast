---
title: Weight Publisher
description: Publish versioned model weights to Tensorcast and trigger online reloads
sidebar_position: 6
---

# Weight Publisher

`tensorcast.tools.weight_publisher` is a small helper for the common workflow:

1) Publish a new set of weights to Tensorcast under an immutable versioned key.
2) Trigger an inference service to reload that `weight_version`.
3) Optionally wait for acknowledgement and garbage-collect old versions.

It supports two publishing modes:

- `publish(...)`: publish a CUDA tensor dict via `tensorcast.put(...)` (preferred long-term).
- `publish_from_disk(...)`: bridge for systems that already export HuggingFace
  safetensors folders to shared storage and want Tensorcast key-based loading
  without changing the training export path yet.

## Import

```python
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

# Or, equivalently:
# from tensorcast.tools import WeightPublisher, WeightPublisherConfig
```

## Key and Version Rules (Important)

- Keys are expected to be immutable version keys.
- `version` (aka `weight_version`) must be monotonic for a given `model_name`.
- Do not reuse the same key for different weights.

By default the publisher verifies that the key mapping points to the artifact it
just published (`verify_key_mapping=True`). If the key already exists and points
to a different artifact, the publisher raises an error instead of silently
proceeding.

Default key format:

```text
model:{model_name}:v{weight_version}
```

Configure via `key_template`.

## Mode A: Publish CUDA Tensors (tc.put)

Use this when your training engine can produce a CUDA tensor dict directly.

```python
import tensorcast as tc
import torch

from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

cfg = WeightPublisherConfig(
    model_name="llama7b",
    key_template="model:{model_name}:v{weight_version}",
    policy="durable",               # default
    wait_persistence=True,          # default
    keep_last=2,                    # keep rollback window
    history_path="/tmp/weights_history.json",
    trigger_reload=False,           # publish only
)

publisher = WeightPublisher(cfg)

# Tensors can be either:
# - all CUDA tensors on the same device (recommended), or
# - all CPU tensors (Tensorcast will stage them to CUDA during `put`).
tensors = {
    "transformer.wte.weight": torch.empty((10, 10), device="cuda:0"),
}

artifact_id = publisher.publish(tensors, version=123)
print("published", artifact_id)
```

Notes:
- `tensorcast.put` requires a CUDA device to be available (even if you pass CPU
  tensors). For best performance, publish CUDA tensors on a single device.
- If you do not run Tensorcast managed persistence, you may need to set
  `policy=None` (or use a non-durable policy) and/or disable `wait_persistence`.

## Mode B: Publish from a HF Safetensors Folder (tc.from_disk)

This is an incremental bridge for disk-export systems (for example, exporting
`model.safetensors.index.json` + `model-00001.safetensors` shards).

```python
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig

publisher = WeightPublisher(
    WeightPublisherConfig(
        model_name="llama7b",
        key_template="model:{model_name}:v{weight_version}",
        from_disk_verify_checksums=True,
        trigger_reload=False,
        keep_last=0,
    )
)

artifact_id = publisher.publish_from_disk("/mnt/shared/it123_hf", version=123)
print("published", artifact_id)
```

Important:
- The HF directory must be immutable per version. Do not overwrite the same
  folder for different versions.
- `publish_from_disk` requires a Tensorcast daemon that can access the folder
  path (typically via shared storage).

## Trigger Reload: Direct HTTP Endpoint

If your inference service exposes a single reload endpoint, set `reload_url`.
The publisher sends:

```json
{ "weight_version": 123, "model_overrides": null }
```

Example:

```python
cfg = WeightPublisherConfig(
    model_name="llama7b",
    reload_url="http://127.0.0.1:8000/set_model_weight",
    trigger_reload=True,
)
WeightPublisher(cfg).publish_from_disk("/mnt/shared/it123_hf", version=123)
```

## Trigger Reload: Stepcast Router (Multi-endpoint)

For Stepcast deployments that expose vLLM dev endpoints per replica, set:

- `stepcast_router`: `host:port` of the router
- `stepcast_served_model_name`: served model name registered in Stepcast

The publisher will:

1) Discover endpoints via:
   `GET http://{router}/v1/model/{served_model_name}`
2) Optionally push vLLM Tensorcast loader config via `/collective_rpc update_config`
3) Call each endpoint:
   `POST /set_model_weight?drain_timeout_s=...`
4) Optionally ack by polling `GET /weight_version`

Example:

```python
cfg = WeightPublisherConfig(
    model_name="llama7b",
    stepcast_router="stepcast-router:9200",
    stepcast_served_model_name="llama7b",
    stepcast_update_config=True,
    stepcast_ack=True,
    vllm_drain_timeout_s=300.0,
)

publisher = WeightPublisher(cfg)
publisher.publish_from_disk("/mnt/shared/it123_hf", version=123)
```

Notes:
- The Stepcast reload path assumes vLLM dev endpoints are enabled on each
  replica (e.g. `VLLM_SERVER_DEV_MODE=1`).
- vLLM Tensorcast loader settings are pushed via:
  `load_config.load_format="tensorcast"` and `model_loader_extra_config`
  (`tensorcast_key_template`, `tensorcast_model_name`, disk fallback options).

## Retention and Garbage Collection (keep_last)

When `keep_last > 0`, the publisher records `(version, artifact_id)` in
`history_path` and calls `tensorcast.deregister_artifact(...)` for versions
older than the most recent `keep_last`.

Retention semantics are intentionally:

- **Version keys are append-only**: old key mappings are kept.
- **Replica/disk residency is bounded**: old versions are deregistered from
  daemon/GS residency and managed shared-disk copies are purged.
- Result: old keys can still resolve, but old versions are expected to become
  non-materializable.

Guidelines:
- Always keep a rollback window (for example, `keep_last=2` or `keep_last=3`).
- Only GC older versions after you have positive evidence that all target
  inference replicas have applied the new version (Stepcast mode provides an
  optional ack via `/weight_version`).

## End-to-End Harness (Single Host + Distributed)

A dedicated E2E harness is available at:

- `tensorcast/tools/weight_publisher_e2e.py`

It models two independent roles:

- `publisher`: continuously publishes new weight versions through
  `WeightPublisher.publish_from_disk(...)`.
- `receiver`: continuously receives and validates versioned weights by key.

### Single-host scenario (local)

Run both roles concurrently in one process:

```bash
uv run -m tensorcast.tools.weight_publisher_e2e single-host \
  --init-mode auto \
  --global-store-mode start \
  --allow-gs-fallback \
  --start-version 1 \
  --num-versions 3 \
  --keep-last 2 \
  --publish-interval-s 2 \
  --receiver-timeout-s 120
```

This validates all required behaviors in one run:

- publisher keeps updating versions (`v1 -> v2 -> v3`)
- receiver keeps receiving and validating each version
- retention window check: with `keep_last=2`, after publishing `v3`, `v1`
  key mapping remains but `v1` must be non-materializable, while `v2`/`v3`
  remain materializable

The harness writes a summary JSON (by default under `/tmp/tensorcast_weight_publisher_e2e/<run-id>/`).

### Distributed scenario (with Global Store)

Use two nodes (or two daemons) connected to the same Global Store:

1. Start Global Store.
2. Start daemon on node A (publisher side), connect it to the Global Store.
3. Start daemon on node B (receiver side), connect it to the same Global Store.
4. Run `publisher` role on node A.
5. Run `receiver` role on node B.

Example:

```bash
# Node A (publisher)
uv run -m tensorcast.tools.weight_publisher_e2e publisher \
  --init-mode connect \
  --connect-address <NODE_A_DAEMON_ADDR> \
  --model-name wp-e2e-dist \
  --start-version 1 \
  --num-versions 3 \
  --keep-last 2 \
  --publish-interval-s 3 \
  --receiver-timeout-s 180
```

```bash
# Node B (receiver)
uv run -m tensorcast.tools.weight_publisher_e2e receiver \
  --init-mode connect \
  --connect-address <NODE_B_DAEMON_ADDR> \
  --model-name wp-e2e-dist \
  --start-version 1 \
  --num-versions 3 \
  --receiver-timeout-s 180 \
  --fallback-prefer p2p
```

Distributed checklist:

- Use the same `model_name`, `start_version`, and `num_versions` on both roles.
- Keep publisher and receiver running concurrently so receiver can observe each update.
- For retention validation, inspect publisher summary: with `keep_last=2` and
  `v1..v3`, `v1` should remain key-resolvable but become non-materializable,
  while `v2`/`v3` remain materializable.
