---
title: Step3p5 FP8 Mounted TP8 Cold-Start Evidence
description: 2026-04-27 initial plus 2026-04-28 same-host follow-up TP8 evidence for 0113 closure, comparing TensorCast collective_first_v4 against safetensors and fastsafetensors
areas: ["core", "daemon", "benchmarks", "integrations", "serving"]
---

# Summary

This note records the mounted TP8 cold-start reruns used for the remaining
`0113` closure gate on one 8xH800 worker, including the 2026-04-28 same-host
follow-up after the `expert_dim0_concat` fast path landed.

This packet proves:

- the current source-bound contract is live on `collective_first_v4`;
- current TP8 TensorCast startup is collective-dominant and fully observable
  through typed `/weight_version` facts; and
- the broader Step3p5 performance signoff is still open because TensorCast
  ready time `326.319s` is still worse than same-host `safetensors`
  `158.172s` and `fastsafetensors` `162.179s`.

The dominant gap is not in vLLM warmup. It is in the TensorCast bootstrap
`load_model` phase itself:

- TensorCast rank-local `load_model` took about `167.526s` to `171.516s`;
- same-host `safetensors` took about `11.512s` to `12.008s`;
- same-host `fastsafetensors` took about `16.928s` to `18.527s`.

The 2026-04-28 rerun also proves that the new executor path is real but still
insufficient for signoff:

- `mapped_expert_dim0_concat_job_summary requested=168 accepted=168`;
- `collective_mapped_target prepared ... concat_jobs=258`;
- TensorCast improved from the 2026-04-27 initial packet `370.346s` to
  `326.319s`; and
- even after that improvement, TensorCast still trails the same-host
  non-TensorCast loaders by more than `160s`.

# Scope

In scope:

- mounted Step3p5 TP8 cold-start evidence on the current `collective_first_v4`
  contract;
- typed execution, hash, and identity facts from live `/weight_version`;
- same-host `safetensors` and `fastsafetensors` baselines on the same 8xH800
  worker;
- explicit delete-gate conclusion for source-bound compatibility retirement;
- explicit `0109` residual-policy conclusion for the mounted TP8 path.

Out of scope:

- claiming full `0113` performance closure;
- relaxing `owner_file_collective_shared_fs_only` or
  `owner_file_collective_allow_mixed_residual`;
- and any attempt to hide the TP8 gap behind a different measurement scope.

# Inputs And Artifacts

- Brainctl worker:
  `ws-7681b3683947089e-worker-wcdcp`
- Host:
  `dev-yuchu-lxnsr-358366-worker-0`
- Mounted model root:
  `/mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8`
- TensorCast daemon config:
  `/data/workspace/internal-vllm/vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml`
- TensorCast case:
  `/data/tc/0113-tp8-20260428-004448`
- `safetensors` case:
  `/data/tc/0113-safetensors-tp8-20260428-005206`
- `fastsafetensors` case:
  `/data/tc/0113-fastsafetensors-tp8-20260428-005520`

Important artifacts:

- TensorCast summary:
  `/data/tc/0113-tp8-20260428-004448/tensorcast-summary.json`
- TensorCast `/weight_version`:
  `/data/tc/0113-tp8-20260428-004448/weight_version.json`
- TensorCast serve log:
  `/data/tc/0113-tp8-20260428-004448/tensorcast-serve.log`
- `safetensors` summary:
  `/data/tc/0113-safetensors-tp8-20260428-005206/tensorcast-summary.json`
- `safetensors` serve log:
  `/data/tc/0113-safetensors-tp8-20260428-005206/tensorcast-serve.log`
- `fastsafetensors` summary:
  `/data/tc/0113-fastsafetensors-tp8-20260428-005520/tensorcast-summary.json`
- `fastsafetensors` serve log:
  `/data/tc/0113-fastsafetensors-tp8-20260428-005520/tensorcast-serve.log`

# Workflow

## 1. Use the current `internal-vllm` TP8 harness on one mounted 8xH800 worker

All three runs used the same worker, model root, and environment:

```bash
CUDA_HOME=/data/cuda/cuda-12.8
PATH=/data/cuda/cuda-12.8/bin:/data/workspace/internal-vllm/.venv/bin:$PATH
LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64
NCCL_DEBUG=WARN
VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0
VLLM_SERVER_DEV_MODE=1
```

The packaged TensorCast daemon config had to trust the mounted source root:

- `public_disk_source.trusted_root_policies.root_path=/data/models`
- `public_disk_source.trusted_root_policies.root_path=/mnt/step3-alignment`

Without that mounted trusted root, the TP8 TensorCast case failed closed before
startup with `disk path is outside configured trusted roots`.

## 2. Run TP8 TensorCast cold-start on the current `v4` contract

The TensorCast harness used the current packaged daemon config and the current
runtime-binding path:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate
cd /data/workspace/internal-vllm

tensorcast-cli daemon start --blocking \
  --config vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml \
  --global-store-mode start

vllm serve /mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8 \
  --port 8041 \
  --served-model-name step3p5-tensorcast-0113 \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --reasoning-parser=step3p5 \
  --tool-call-parser=step3p5 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --max-model-len 4096 \
  --load-format tensorcast \
  --gpu-memory-utilization 0.82 \
  --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_show_daemon_logs":true,"tensorcast_enable_runtime_binding":true}'
```

The daemon summary for this packet was:

| Field | Value |
| --- | --- |
| `daemon.address` | `127.0.0.1:50052` |
| `daemon.owner` | `true` |
| `global_store.mode` | `start` |
| `session_id` | `20260427-210157-1aa8` |

## 3. Run same-host TP8 baselines

The same host then ran:

```bash
vllm serve ... --load-format safetensors
vllm serve ... --load-format fastsafetensors
```

Both baselines used the same TP8 flags as TensorCast except for
`--load-format`.

# TensorCast TP8 Packet

## Contract and typed diagnostics

The live TensorCast `/weight_version` packet recorded:

| Field | Value |
| --- | --- |
| `bootstrap_source_bound_contract_version` | `4` |
| `bootstrap_source_bound_contract_path` | `collective_first_v4` |
| `bootstrap_source_bound_capability_flags` | `FIRST_CLASS_COLLECTIVE_INGRESS`, `TYPED_EXECUTION_DIAGNOSTICS`, `SINGLE_MINT_BINDING_CLOSEOUT` |
| `bootstrap_realize_collective_policy` | `collective_first` |
| `bootstrap_realize_collective_used` | `true` |
| `bootstrap_realize_dominant_executor` | `OwnerFileCollectiveExecutor` |
| `bootstrap_realize_actual_collective_committed_bytes` | `25,550,556,928` |
| `bootstrap_realize_actual_local_typed_bytes` | `936` |
| `bootstrap_realize_actual_generic_backend_bytes` | `0` |
| `bootstrap_realize_collective_failure_class` | `null` |
| `bootstrap_publish_hash_rounds` | `0` |
| `bootstrap_publish_hash_location` | `seal` |
| `bootstrap_publish_hash_backend` | `gpu` |
| `bootstrap_publish_hash_bytes` | `25,550,557,864` |
| `bootstrap_publish_hash_wall_time_ms` | `2,556` |
| `bootstrap_publish_identity_mint_strategy` | `seal_reuse` |

This proves the intended `0113` contract facts:

- mounted TP8 is on `collective_first_v4`, not a stale bridge path;
- actual runtime execution is collective-dominant;
- same-binding closeout reuses seal identity;
- and downstream summary no longer needs daemon-log parsing or
  `operation_id` decoding.

## Planner and residual-policy facts

The same packet also recorded:

| Field | Value |
| --- | --- |
| `bootstrap_realize_execution_plan_kind` | `collective_first_mixed` |
| `bootstrap_realize_planned_collective_admitted_bytes` | `9,692,217,088` |
| `bootstrap_realize_planned_non_admitted_typed_bytes` | `15,858,339,840` |
| `bootstrap_realize_planner_reject_reason_buckets` | `tensor_copy_partition_unknown`, `typed_work_not_collective_admitted` |
| `bootstrap_realize_compatibility_lowered_bytes` | `25,550,556,928` |

The closure conclusion for `0109` is:

- `owner_file_collective_shared_fs_only=true` remains the intended steady-state
  policy;
- `owner_file_collective_allow_mixed_residual=false` remains the intended
  fail-closed policy;
- the current TP8 packet does not justify widening mixed residual, because
  actual runtime still reached `actual_generic_backend_bytes=0`; and
- the remaining issue is planner/runtime performance quality, not semantic
  policy ambiguity.

# Performance Comparison

## Ready time

All three runs reached:

- `GET /health` = `200`
- `GET /v1/models` = `200`
- `POST /v1/completions` = `200`

Ready time on the same host:

| Loader | Ready wall time (s) | Result |
| --- | ---: | --- |
| TensorCast | `326.319` | slower |
| `safetensors` | `158.172` | baseline |
| `fastsafetensors` | `162.179` | baseline |

Direct deltas:

- TensorCast is `168.147s` slower than same-host `safetensors`.
- TensorCast is `164.140s` slower than same-host `fastsafetensors`.

This means the requested signoff rule "TensorCast must not be worse than the
default loader and fastsafetensors loader" is not satisfied on this TP8 packet.

## Where the gap is

The dominant gap is in model loading, not in later warmup:

| Loader | Model loading (s) | Engine warmup (s) | Ready wall time (s) |
| --- | --- | ---: | ---: |
| TensorCast | `167.526` to `171.516` | `92.82` | `326.319` |
| `safetensors` | `11.512` to `12.008` | `91.61` | `158.172` |
| `fastsafetensors` | `16.928` to `18.527` | `88.35` | `162.179` |

Additional TensorCast evidence from the live log:

- `mapped_expert_dim0_concat_job_summary` accepted all `168` requested
  synthetic expert-dim0 concat jobs;
- `collective_mapped_target prepared` reported `concat_jobs=258`,
  `concat_job_source_bytes=127,656,296,448`, and `concat_job_exec_sec=73.948`;
- rank-local `Tensorcast load_model timings` were about `167.5s` to `171.5s`;
- `publish_hash_wall_time_ms` was only about `1.3s` to `2.6s`; and
- vLLM graph capture and engine warmup stayed close to the baselines.

So the current Step3p5 TP8 blocker is not hash reuse, not typed diagnostics,
and not server startup after weights are ready. The blocker is the TensorCast
bootstrap `from_disk -> same_binding` load path itself.

# Delete-Gate Conclusion

The compatibility-retirement gate is closed by this packet and the repo state:

- `internal-vllm` already gates on `source_bound_contract_version >= 4` plus
  the three capability bits, and no longer treats `ctx.collective` as its
  source-bound primary input;
- TensorCast daemon-owned source-bound callers reject `ctx.collective` and do
  not use `operation_id` as the source-bound collective carrier;
- strict collective failures now flow through structured gRPC trailing metadata
  only; the old status-detail marker fallback has been deleted; and
- repo-local tests already cover the supported first-class request fields and
  the supported typed diagnostics path.

This means the compatibility bridge is no longer the blocker for `0113`.

# Conclusion

This committed packet closes the evidence-capture and delete-gate part of
`0113`, but it does not close the broader Step3p5 performance signoff.

Current `0113` status after this packet:

- contract closure: done;
- typed diagnostics closure: done;
- compatibility/delete gate closure: done;
- `0109` residual-policy decision: done;
- TP8 performance signoff: still open.

The remaining closure blocker is explicit and measurable: reduce the TensorCast
TP8 bootstrap load path from about `168s` to `172s` toward the same-host
baseline band of about `12s` to `19s`, or otherwise produce a narrower and
explicitly accepted signoff scope.
