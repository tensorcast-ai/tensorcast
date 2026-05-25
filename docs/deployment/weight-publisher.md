---
title: Weight Publisher
description: Publish versioned model weights to Tensorcast and trigger online reloads
sidebar_position: 6
---

# Weight Publisher

`tensorcast.tools.weight_publisher` is a legacy helper for the generic
weight-version workflow:

1) Publish a new set of weights to Tensorcast under an immutable versioned key.
2) Trigger an inference service to reload that `weight_version`.
3) Optionally wait for acknowledgement and garbage-collect old versions.

Current `internal-vllm` with `load_format="tensorcast"` no longer treats
`weight_version` or `/set_model_weight` as TensorCast serving identity. That
runtime expects a published serving artifact and reloads through
`POST /reload_serving_artifact` with a `selector + policy` request. Use
`internal-vllm/tools/tensorcast_prepare_local_dir.py` or a serving-artifact
publisher to produce that request. The reload sections below apply to
non-TensorCast or legacy `/set_model_weight` deployments.

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

## Trigger Reload: Legacy Direct HTTP Endpoint

If your non-TensorCast or legacy inference service exposes a single generic
reload endpoint, set `reload_url`. The publisher sends:

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

For current vLLM TensorCast serving reload, the request shape is instead:

```json
{
  "selector": {
    "kind": "version_key",
    "value": "models/demo/serving/v123"
  },
  "policy": {
    "mode": "from_manifest"
  },
  "model_overrides": null
}
```

`WeightPublisher` does not synthesize this serving-artifact request today.

## Trigger Reload: Legacy Stepcast Router (Multi-endpoint)

For Stepcast deployments that expose vLLM dev endpoints per replica, set:

- `stepcast_router`: `host:port` of the router
- `stepcast_served_model_name`: served model name registered in Stepcast

The publisher will:

1) Discover endpoints via:
   `GET http://{router}/v1/model/{served_model_name}`
2) Call each endpoint:
   `POST /set_model_weight?drain_timeout_s=...`
3) Optionally ack by polling `GET /weight_version`

Example:

```python
cfg = WeightPublisherConfig(
    model_name="llama7b",
    stepcast_router="stepcast-router:9200",
    stepcast_served_model_name="llama7b",
    stepcast_ack=True,
    vllm_drain_timeout_s=300.0,
)

publisher = WeightPublisher(cfg)
publisher.publish_from_disk("/mnt/shared/it123_hf", version=123)
```

Notes:
- The Stepcast reload path assumes vLLM dev endpoints are enabled on each
  replica (e.g. `VLLM_SERVER_DEV_MODE=1`).
- This path no longer pushes vLLM TensorCast loader config. Current
  `internal-vllm` artifact runtime must be configured through
  `tensorcast.TensorCastRuntimeConfig` and reloaded with
  `/reload_serving_artifact`.

## Retention and Garbage Collection (keep_last)

When `keep_last > 0`, the publisher records `(version, artifact_id)` in
`history_path` and calls `tensorcast.deregister_artifact(...)` for versions
older than the most recent `keep_last`.

The publisher always performs pre-publish trimming before `put(...)`.

This bounds publish-time overlap (for example, avoids transient 3-version overlap
when `keep_last=2`) and reduces OOM risk in long-running publisher processes.
Trade-off: if publish fails after pre-trim, rollback window may be temporarily
smaller until a successful publish restores the target window.

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
  `WeightPublisher.publish(...)` (CUDA/CPU tensor dict -> local stable DRAM).
- `receiver`: continuously receives and validates versioned weights by key by
  staging tensor-parallel rank-local values through `group_realization`,
  waiting for the publish barrier, then acquiring the staged binding values.

### Single-host scenario (local)

Run both roles concurrently in one process:

```bash
source .venv/bin/activate
tensorcast-cli daemon start \
  --config examples/config/store_daemon_config_cross_host_bench.yaml \
  --global-store-mode connect \
  --global-store-address <GS_ADDR>

python ./tensorcast/tools/weight_publisher_e2e.py single-host \
  --init-mode connect \
  --connect-address 127.0.0.1:50052 \
  --start-version 1 \
  --num-versions 3 \
  --keep-last 2 \
  --payload-mode tp_ranked \
  --tp-world-size 1 \
  --publish-interval-s 2 \
  --receiver-timeout-s 120 \
  --materialize-device cuda:0

tensorcast-cli daemon stop
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
4. Run `publisher` role on node A (`--init-mode connect --connect-address 127.0.0.1:50052`).
5. Run `receiver` role on node B (`--init-mode connect --connect-address 127.0.0.1:50052`).

Example:

```bash
# Node A (publisher)
source .venv/bin/activate
tensorcast-cli daemon start \
  --config examples/config/store_daemon_config_cross_host_bench.yaml \
  --global-store-mode connect \
  --global-store-address <GS_ADDR>

python ./tensorcast/tools/weight_publisher_e2e.py publisher \
  --init-mode connect \
  --connect-address 127.0.0.1:50052 \
  --model-name wp-e2e-dist \
  --start-version 1 \
  --num-versions 6 \
  --keep-last 2 \
  --payload-mode tp_ranked \
  --tp-world-size 1 \
  --publish-interval-s 3 \
  --receiver-timeout-s 180 \
  --retention-timeout-s 90
```

```bash
# Node B (receiver)
source .venv/bin/activate
tensorcast-cli daemon start \
  --config examples/config/store_daemon_config_cross_host_bench.yaml \
  --global-store-mode connect \
  --global-store-address <GS_ADDR>

python ./tensorcast/tools/weight_publisher_e2e.py receiver \
  --init-mode connect \
  --connect-address 127.0.0.1:50052 \
  --model-name wp-e2e-dist \
  --start-version 1 \
  --num-versions 6 \
  --receiver-timeout-s 180 \
  --payload-mode tp_ranked \
  --tp-world-size 1 \
  --materialize-device cuda:0
```

```bash
# Cleanup (both nodes)
source .venv/bin/activate
tensorcast-cli daemon stop
```

Distributed checklist:

- Use the same `model_name`, `start_version`, and `num_versions` on both roles.
- Keep publisher and receiver running concurrently so receiver can observe each update.
- App SDK only connects to local daemon in this workflow (`127.0.0.1:50052`).
- Receiver materialization path in this harness is fixed to p2p-only (no disk/local fallback).
- For retention validation, inspect publisher summary: with `keep_last=2` and
  `v1..v3`, `v1` should remain key-resolvable but become non-materializable,
  while `v2`/`v3` remain materializable.
- For cluster-level replica checks, use `--hold-after-finish-s` to keep receiver
  daemons alive briefly after completion, then query GS metadata
  (`ClusterRuntimeService.BatchGetReplicaCounts`).

## Multi-host Suite (brainctl)

For staged scale-out (`2-node -> 3-node`) with `group_realization` staged
binding updates and retention checks, use:

- `examples/cross_host/cross_host_weight_publisher_runner.py` (single case runner)
- `examples/cross_host/run_multihost_weight_publisher_suite.sh` (suite entry)

Example:

```bash
source .venv/bin/activate

export TC_WP_PUBLISHER_PROC=<PUBLISHER_PROCESS_ID>
export TC_WP_RECEIVER_PROCS=<RECEIVER1_PROCESS_ID>,<RECEIVER2_PROCESS_ID>
export TC_GS_ADDR=<GS_IP>:50051
export TC_PUBLISH_INTERVAL_S=60
export TC_RECEIVER_TIMEOUT_S=95
export TC_MAX_PUBLISH_TO_APPLY_S=30
export TC_SCALE_RECEIVER_COUNTS=1,2,4,8,16,31
export TC_SCALE_NUM_VERSIONS=10
export TC_LONG_RUN_ENABLE=1
export TC_LONG_RUN_NUM_VERSIONS=20
export TC_LONG_RUN_TARGET_DURATION_S=900
export TC_PROGRESS_POLL_S=10

bash examples/cross_host/run_multihost_weight_publisher_suite.sh
```

Progressive dissemination benchmark lane:

- Keep it disabled for baseline runs. Enable it explicitly with
  `TC_WP_PROGRESSIVE_ENABLE=1` after starting Global Store with
  `worker_policy.progressive_replication.enabled=true`.
- The suite propagates daemon-side reporting knobs:
  `TC_WP_PROGRESSIVE_REPORT_INTERVAL`,
  `TC_WP_PROGRESSIVE_MIN_REPORT_DELTA_BYTES`, and
  `TC_WP_PROGRESSIVE_VERIFY_BEFORE_REPORT`.
- Progressive and baseline runs should use the same receiver counts, payload
  mode, publish interval, and timeout settings before comparing concentration or
  tail-latency metrics.
- Add the optional failure-injection lane with
  `TC_WP_FAILURE_INJECTION_ENABLE=1`. The suite then appends one extra case
  that stops a selected receiver daemon after publisher start
  (`TC_WP_FAILURE_INJECTION_TARGET=receiver:0` by default) and records whether
  the remaining receivers still complete. Tune
  `TC_WP_FAILURE_INJECTION_DELAY_S` so the selected receiver has become a useful
  progressive source in the target topology.
- Use `examples/cross_host/summarize_scaleout_suite.py --tp-dir <RUN_DIR>` to
  produce JSON and Markdown reports. The TP report groups cases into baseline,
  progressive, and failure-injection lanes and summarizes source concentration
  and publish-to-apply latency metrics.

Timeout guidance:

- `TC_RECEIVER_TIMEOUT_S` must be larger than publish interval.  
  Example: with `TC_PUBLISH_INTERVAL_S=60`, use `TC_RECEIVER_TIMEOUT_S>=95`.
- Keep end-to-end update SLA strict with `TC_MAX_PUBLISH_TO_APPLY_S` (default `30`).
- Suite prints periodic `[progress]` heartbeats (poll interval controlled by
  `TC_PROGRESS_POLL_S`) so long-running cases can be observed in real time.
