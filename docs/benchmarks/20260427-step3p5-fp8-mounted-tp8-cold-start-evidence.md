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
- the Step3p5 same-host TP8 performance signoff is closed by the tensor-aware
  mapped executor: final TensorCast ready time is `136.271s` against same-host
  `safetensors` `144.155s` and `fastsafetensors` `152.174s`.

The dominant gap is not in vLLM warmup. It is in the TensorCast bootstrap
`load_model` phase itself:

- TensorCast rank-local `load_model` took about `167.526s` to `171.516s`;
- same-host `safetensors` took about `11.512s` to `12.008s`;
- same-host `fastsafetensors` took about `16.928s` to `18.527s`.

The 2026-04-28 intermediate rerun proved that the first new executor path was
real but still insufficient for signoff:

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

- claiming that the same-host result covers multi-host or RDMA topologies;
- relaxing `owner_file_collective_shared_fs_only` or
  `owner_file_collective_allow_mixed_residual`;
- any attempt to hide the historical TP8 gap behind a different measurement
  scope; and
- treating the old `326.319s` historical packet as the final performance
  result.

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

## TensorCast disk-loader profile

The same TensorCast run selected the owner-file collective path:

- `bootstrap_realize_dominant_executor=OwnerFileCollectiveExecutor`;
- `bootstrap_realize_collective_used=true`;
- `bootstrap_realize_actual_collective_committed_bytes=25,550,556,928`;
- `bootstrap_realize_planned_collective_admitted_bytes=9,692,217,088`;
- `bootstrap_realize_planned_non_admitted_typed_bytes=15,858,339,840`; and
- `bootstrap_realize_estimated_collective_dedup_saving_bytes=67,845,519,616`.

Daemon-side profile for that executor:

| Phase | Key profile facts |
| --- | --- |
| concat fast path | `258` concat jobs; `127,656,296,448` source bytes; prepared `concat_job_exec_sec=73.948`; summed job time `73.932s`, including `68.900s` read time |
| mapped residual windows | `3,001,272` segments; `428` windows; `906` chunks; `197,099,083,520` bytes read; `173,073,623,296` peer-transfer bytes |
| full mapped collective | `total=149.071s`; residual-loop components include `read=63.511s`, `sync=13.218s`, `root_d2d=1.437s`, `issue=0.869s`, `h2d=0.012s` |

The observable source bytes read through the collective path are
`127,656,296,448 + 197,099,083,520 = 324,755,379,968` bytes. The dominant
time is therefore not GPU H2D or hash work; it is root-side source reads plus
collective synchronization in the owner-file collective path.

Interpretation: the current same-host mounted path is not shaped like the
same-host `safetensors` and `fastsafetensors` baselines, where ranks perform
their own local file loads in parallel. TensorCast currently routes the
same-host TP8 bootstrap through a root-rank staged reader and peer/NCCL
distribution path. That may reduce duplicated source reads in shared-source
multi-host cases, but it loses the per-rank parallelism required to match the
same-host baseline. The remaining performance gate is therefore a disk-loader
strategy issue, not a contract, diagnostics, hash, or compatibility issue.

## No-collective strategy probe

A follow-up 8xH800 strategy probe was run on
`dev-yuchu-kk2x2-362559-worker-0`:

- case directory:
  `/data/tc/0113-tp8-8gpu-ab-20260428-064325/0113-tp8-nocollective-rerun-20260428-064445`;
- worker:
  `ws-7681b3683947089e-worker-cptmd` on H800, deleted after the run and
  verified `NotFound`;
- TensorCast extra config:
  `{"tensorcast_collective_policy":"disable_collective"}`.

Result:

| Strategy | Ready wall time (s) | Rank-local load_model (s) | Dominant executor |
| --- | ---: | ---: | --- |
| owner-file collective default | `326.319` | `167.526` to `171.516` | `OwnerFileCollectiveExecutor` |
| no-collective probe | `262.284` | `117.313` to `128.150` | `SourceOrderedMappedTargetExecutor` |

Typed diagnostics for the no-collective probe:

- `bootstrap_realize_collective_policy=disable_collective`;
- `bootstrap_realize_collective_requested=false`;
- `bootstrap_realize_collective_used=false`;
- `bootstrap_realize_dominant_executor=SourceOrderedMappedTargetExecutor`;
- `bootstrap_realize_actual_collective_committed_bytes=0`;
- `bootstrap_realize_actual_generic_backend_bytes=25,550,556,928`; and
- `bootstrap_publish_hash_wall_time_ms=1295`.

Daemon-side mapped-target profile for that probe:

- `8` `materialize_mapped_into_target source execution` records;
- `source_ordered=1`;
- `map_total_bytes=25,550,557,864` per rank;
- `map_segments=375,214` to `375,310` per rank;
- `target_storages=1`;
- `concurrency=4`; and
- `exec_sec=108.359` to `115.486`.

This confirms that disabling collective is the best measured existing
same-host strategy knob, reducing ready time by `64.035s` and load_model time
by about `43s` to `54s` versus the owner-file collective default. It still does
not satisfy the performance gate: the no-collective path remains `104.112s`
slower than same-host `safetensors` and `100.105s` slower than same-host
`fastsafetensors` on ready time. The remaining bottleneck is the
source-ordered mapped-target executor itself: it still executes hundreds of
thousands of byte-range segments per rank instead of a safetensors-like
tensor-aware local loader.

## Tensor-aware mapped executor closure

A follow-up fix was validated on the existing 8xH800 worker
`ws-7681b3683947089e-worker-k68dd`
(`dev-yuchu-4thvk-367119-worker-0`):

- case directory:
  `/data/tc/0113-tp8-rect2d-final-20260428-125500`;
- daemon binary:
  `bazel-bin/daemon/tensorcast_daemon`, sha256
  `a143085f0ae1cbbcef8f80500e3167dec4d870248f83f1aba3e429779ef286f1`;
- same-host baseline cases:
  `safetensors` at
  `/data/tc/0113-safetensors-tp8-final-20260428-125212` and
  `fastsafetensors` at
  `/data/tc/0113-fastsafetensors-tp8-final-20260428-125809`;
- probe:
  default port `8041`, default served model
  `step3p5-tensorcast-0113-local-tensor`;
- result:
  `poll.exit=0`, `GET /health=200`, `GET /v1/models=200`,
  `POST /v1/completions=200`, and `GET /weight_version=200`.

Updated same-host ready comparison:

| Loader / TensorCast strategy | Ready wall time (s) | Result |
| --- | ---: | --- |
| TensorCast original mounted packet | `326.319` | slower |
| TensorCast no-collective probe | `262.284` | slower |
| TensorCast tensor-aware mapped executor | `136.271` | pass |
| `safetensors` | `144.155` | baseline |
| `fastsafetensors` | `152.174` | baseline |

The root cause was the same-host disjoint TP shard tail. Before the fix, the
typed concat/dim0 path handled about `24.577GB`, but the remaining
`873,562,112` bytes were lowered to `368,687` generic H2D segments per rank.
That fallback alone took about `36s` to `39s`.

The fix keeps replicated overlap in the owner-file collective lane and routes
same-host disjoint typed work through a local tensor-aware mapped executor. In
this Step3p5 TP8 shape, the local executor now accepts the residual as `137`
rect2d tensor jobs:

- `local_mapped_partial_tensor_job_summary considered=486
  accepted_rect2d=137 accepted_bytes=873562112 skipped_unsupported=0`;
- `local_mapped_target timings tensor_jobs=221 base_tensor_jobs=84
  partial_tensor_jobs=137 tensor_rect2d_jobs=137`;
- `handled_range_bytes=25450649600`, `handled_overlap_bytes=25450649600`;
- `residual_segments=0`, `residual_bytes=0`; and
- `materialize_mapped_into_target execution_commit
  actual_collective_committed_bytes=99907328
  actual_local_typed_bytes=25450650536 actual_generic_backend_bytes=0`.

Rank-local `Tensorcast load_model timings` improved to `17.434s` to `19.549s`.
The remaining ready time is dominated by vLLM compile, KV-cache setup, and graph
capture, not TensorCast disk materialization.

Correctness validation for the optimized copy path:

- `//core/store/replica:collective_disk_loader_test` now includes a real-CUDA
  rect2d byte-equivalence case for `try_local_mapped_target_load`;
- the test builds a small safetensors source, executes a non-contiguous `2x3`
  rectangular tensor slice into a `4x6` target layout, and checks the full
  target buffer byte-for-byte, including untouched bytes outside the slice; and
- the test passed with `TENSORCAST_CUDA_BACKEND=real`. The broader fake-CUDA
  suite also passes; the rect2d byte-equivalence case explicitly skips fake
  CUDA because the optimized executor uses `cudaMemcpy2DAsync`.

# Same-Prompt Model Output Comparison

The strict same-prompt serving comparison currently uses the persisted TP8
completion artifacts that were captured with the common request:

```json
{"prompt":"Say hi in five words.","max_tokens":8,"temperature":0}
```

Artifacts compared:

- TensorCast:
  `/data/tc/0113-tp8-20260428-004448/completion.json`;
- `safetensors`:
  `/data/tc/0113-safetensors-tp8-final-20260428-125212/completion.json`;
- `fastsafetensors`:
  `/data/tc/0113-fastsafetensors-tp8-final-20260428-125809/completion.json`.

Full comparable output fields match. The only expected envelope differences are
`id`, `created`, and `model`, because each server request minted its own
completion id/timestamp and used a distinct served model name.

| Field | TensorCast | `safetensors` | `fastsafetensors` |
| --- | --- | --- | --- |
| `object` | `text_completion` | `text_completion` | `text_completion` |
| `choices.length` | `1` | `1` | `1` |
| `choices[0].index` | `0` | `0` | `0` |
| `choices[0].text` | `"\tCHITCHAT\nI'm going"` | `"\tCHITCHAT\nI'm going"` | `"\tCHITCHAT\nI'm going"` |
| `choices[0].chatml` | `["\tCHITCHAT\nI'm going"]` | `["\tCHITCHAT\nI'm going"]` | `["\tCHITCHAT\nI'm going"]` |
| `choices[0].finish_reason` | `length` | `length` | `length` |
| `choices[0].stop_reason` | `null` | `null` | `null` |
| `choices[0].logprobs` | `null` | `null` | `null` |
| `choices[0].token_ids` | `null` | `null` | `null` |
| `choices[0].prompt_logprobs` | `null` | `null` | `null` |
| `choices[0].prompt_token_ids` | `null` | `null` | `null` |
| `choices[0].draft_token_num` | `0` | `0` | `0` |
| `choices[0].accepted_draft_token_num` | `0` | `0` | `0` |
| `choices[0].num_accepted_tokens_per_pos` | `[]` | `[]` | `[]` |
| `usage.prompt_tokens` | `7` | `7` | `7` |
| `usage.completion_tokens` | `8` | `8` | `8` |
| `usage.total_tokens` | `15` | `15` | `15` |
| `usage.cached_tokens` | `0` | `0` | `0` |
| `usage.prompt_tokens_details` | `null` | `null` | `null` |
| `service_tier` | `null` | `null` | `null` |
| `system_fingerprint` | `null` | `null` | `null` |
| `kv_transfer_params` | `{}` | `{}` | `{}` |
| `weight_version` | `null` | `null` | `null` |
| `serving_identity` | `null` | `null` | `null` |

Normalized response equality, after excluding the expected volatile envelope
fields `id`, `created`, and `model`, is `true` for both
TensorCast-vs-`safetensors` and `fastsafetensors`-vs-`safetensors`.

The final `136.271s` TensorCast performance packet at
`/data/tc/0113-tp8-rect2d-final-20260428-125500` used a shorter smoke request
(`max_tokens=1`, `temperature=1.0`) and therefore is not substituted for the
same-prompt comparison above. A 2026-04-28 attempt to rerun that exact optimized
packet with the same prompt was blocked by 8-GPU scheduling capacity:
`codesign` failed quota check with `gpu: 133/128`, and `tensorcast_dev`
returned `no machine available`.

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

This packet now closes the evidence-capture, delete-gate, and same-host TP8
performance parts of `0113`.

Current `0113` status after this packet:

- contract closure: done;
- typed diagnostics closure: done;
- compatibility/delete gate closure: done;
- `0109` residual-policy decision: done;
- TP8 performance signoff: done for this same-host mounted packet.

The remaining follow-up is not a correctness or performance gate for this
packet: keep the tensor-aware mapped executor coverage broad enough that new
multi-axis TP shard shapes do not silently fall back to high-cardinality generic
byte segments.
