# TensorCast Bind Gap Repro (2026-03-13)

## Goal

Reproduce three things with concrete commands:

1. Faster local daemon startup for iteration.
2. TP=4 trace/view-args data and saved per-rank shards.
3. The remaining `tensor_bind` gap between vLLM and the standalone replay path.

## Key Finding

The gap is **not** in disk materialization itself.

- vLLM TP=8 diagnostics:
  - `materialize_sec` avg: `2.687s`
  - `tensor_bind_sec` avg: `0.022s`
- Standalone TP=8 replay diagnostics:
  - `materialize_sec` avg: `2.536s`
  - `tensor_bind_sec` avg: `0.910s`

So the non-reproduced part is the **client-side bind/restore stage** after the daemon
has already materialized the payload. In TensorCast this is the
`_payload_state_dict(payload)` stage:

- `tensorcast/api/store/artifact.py`
- `tensorcast/api/store/materialization.py`

In vLLM, workers enter model loading only after `init_device()` has already run:

- `vllm/v1/worker/gpu_worker.py`
- `vllm/platforms/cuda.py`

That path eagerly sets the current CUDA device and forces a tiny allocation.
The standalone replay does not naturally inherit that worker bootstrap state.

## Faster Daemon Startup

No exposed config knob was found to disable the GPU NVRTC prewarm. The daemon
does this during startup for all visible GPUs:

- `tensorcast-280/daemon/app/server_main.cc`

So the practical startup-speed plan is:

1. Reduce `CUDA_VISIBLE_DEVICES` to the GPUs you actually need.
2. Use a smaller daemon config.
3. Prestart one daemon once, then connect to it from repeated runs.

### Small local config

Use:

- `tensorcast_local_store_daemon_tp4_small.yaml`

This config reduces:

- `num_threads`
- `artifact_chunk_bytes`
- `streaming_buffer_chunks`
- `engine.memory_tiers.stable_bytes`
- `pinned_memory.classes[*].pool_bytes`
- communicator staging / TCP fanout

### Measured startup times

On this host with `CUDA_VISIBLE_DEVICES=0,1,2,3`:

- default packaged config: `37.8s`
- small config: `21.2s`

In a later warm run of the same small config, the replay script measured:

- managed daemon startup: `12.96s`

## Daemon Startup Command

This is the direct daemon command to use for local TP=4 iteration:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TENSORCAST_HOME=/data/tc/tp4-loader-repro/tensorcast_home

.venv/bin/tensorcast-cli daemon start \
  -c tools/tensorcast_local_store_daemon_tp4_small.yaml \
  --json
```

The measured run returned:

- session: `20260313-213928-68e7`
- address: `127.0.0.1:50052`

To stop it:

```bash
export TENSORCAST_HOME=/data/tc/tp4-loader-repro/tensorcast_home

.venv/bin/tensorcast-cli daemon stop \
  --session <session_id_from_start_json>
```

## TP=4 Trace Collection Command

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=WARN
export VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0
export LD_PRELOAD=/usr/local/nvidia/lib64/libcuda.so
export PAGER=cat
export PYTHONUNBUFFERED=1

.venv/bin/vllm serve \
  /mnt/step3-alignment/inference/yuchu/step4air_toy/toy001/bootstrap_hf \
  --port 18004 \
  --tensor-parallel-size 4 \
  --disable-cascade-attn \
  --reasoning-parser=step3p5 \
  --tool-call-parser=step3p5 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --load-format tensorcast \
  --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_daemon_address":"127.0.0.1:50052","tensorcast_show_daemon_logs":false,"tensorcast_debug_path":"/data/tc/tp4-loader-repro/serve_debug"}' \
  --enforce-eager
```

Artifacts from the measured run:

- view args: `/data/tc/tp4-loader-repro/serve_debug/view_args`
- trace plan: `/data/tc/tp4-loader-repro/serve_debug/trace_plan`
- log: `/data/tc/tp4-loader-repro/logs/vllm_serve_tp4_connect.log`

Measured TP=4 vLLM diagnostics:

- per-rank estimated bytes: `2.9765 GiB`
- `materialize_sec` avg: `5.135s`
- `tensor_bind_sec` avg: `0.057s`
- `total_sec` avg: `5.192s`

## Standalone Replay Command

The standalone replay script is:

- `tensorcast_tp8_from_disk_subset_view_repro.py`

Use this command for TP=4 replay and shard saving:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python tools/tensorcast_tp8_from_disk_subset_view_repro.py \
  --args-dir /data/tc/tp4-loader-repro/serve_debug/view_args \
  --tp-world-size 4 \
  --workers 4 \
  --daemon-mode managed \
  --daemon-config-path tensorcast_local_store_daemon_tp4_small.yaml \
  --worker-runtime-mode connect \
  --device-mode cuda_rank \
  --collective-disk-pipeline \
  --no-show-progress \
  --no-worker-logs \
  --save-dir /data/tc/tp4-loader-repro/outputs/tp_state_dicts \
  --save-format safetensors \
  --json-out /data/tc/tp4-loader-repro/outputs/repro_tp4_save_summary.json
```

Outputs from the measured run:

- saved shards: `/data/tc/tp4-loader-repro/outputs/tp_state_dicts`
- summary JSON: `/data/tc/tp4-loader-repro/outputs/repro_tp4_save_summary.json`

Saved files:

- `tp0.safetensors`
- `tp1.safetensors`
- `tp2.safetensors`
- `tp3.safetensors`

Each file is about `3.0G`.

Measured standalone TP=4 replay timings:

- managed daemon startup: `12.96s`
- `from_disk_sec` avg: `0.053s`
- outer `materialize_sec` avg: `7.632s`
- `save_sec` avg: `20.751s`
- worker `total_sec` avg: `30.541s`

Standalone diagnostics embedded in the JSON show:

- `materialize_sec` avg: `6.552s`
- `tensor_bind_sec` avg: `0.359s`
- `total_sec` avg: `6.911s`

## What Is Reproduced vs Not Reproduced

What is reproduced well:

- `from_disk`
- tensor selection / slicing
- daemon-side materialization
- per-rank saved shard content

What is still not reproduced tightly:

- `tensor_bind_sec`

The strongest evidence collected so far:

1. Disk materialization is already close enough to vLLM on TP=8.
2. The remaining gap is concentrated in the client bind stage.
3. vLLM worker bootstrap pre-initializes CUDA device/context before loading.
4. A naive attempt to inject the same eager device init into the standalone
   replay made the managed-daemon path unstable, so it is not a safe default.

## Practical Recommendation

For fast iteration:

1. Use TP=4.
2. Use the small daemon config.
3. Start the daemon once and reuse `connect`.
4. Use the standalone replay script to regenerate per-rank shards.

For finishing the bind-gap investigation:

Build a tiny long-lived worker harness that reproduces only:

1. `init_device()`
2. `tc.init(mode="connect", ...)`
3. `tc.from_disk(...).subset(...).view(...).tensor_dict_with_diagnostics(...)`

That is the next step if the goal is to match `tensor_bind_sec` as closely as
possible without booting the whole vLLM engine.
