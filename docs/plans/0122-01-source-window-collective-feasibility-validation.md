---
slug: source-window-collective-feasibility-validation
title: Source-Window Collective Feasibility Validation
status: draft
areas: ["core", "daemon", "tests", "benchmarks"]
created: 2026-06-15
last_updated: 2026-06-15
related_code:
  - docs/designs/0122-tp-source-window-collective-realization.md
  - docs/plans/0122-tp-source-window-collective-realization.md
  - tools/experiments/source_window_collective_feasibility.py
links:
  design: ../designs/0122-tp-source-window-collective-realization.md
  plan: ./0122-tp-source-window-collective-realization.md
---

# Objective

Validate the basic feasibility of `SourceWindowCollectiveExecutor` before
changing the production TensorCast loader.

This validation answers two questions:

1. Does a source-window collective plan remove the current TensorCast
   local-mapped source read amplification on the real Qwen3 TP=8 models?
2. Can rank-striped source-window reads reach the read-side throughput needed
   for InstantTensor-class cold-start performance?

This is not an end-to-end TensorCast loader implementation. It does not include
GPU scatter, NCCL distribution, target layout writes, daemon coordination, or
vLLM model initialization. It establishes the read-side and planning
feasibility gate for implementing the design.

The next pre-production gates are:

1. Phase 0.25 real `RepresentationWorkPlan` dump validation. This proves the
   production summarizer can reproduce the read/rank estimates without the
   Qwen/vLLM tensor-name classifier used by this script.
2. Phase 0.5 collective and scatter microbench. This decides whether the first
   runtime implementation can use full-window all-gather or must start with
   routed or hybrid distribution.

# Tooling Added

Added:

- `tools/experiments/source_window_collective_feasibility.py`

The script is daemon-independent and read-only by default. It parses real
safetensors headers, classifies Qwen/vLLM TP loading categories, estimates the
current local-mapped per-rank read cost, estimates a TP source-window
collective plan, and optionally runs rank-striped source reads.

The optional IO smoke mode supports `posix_fadvise(DONTNEED)` so the experiment
can follow the same cold-read convention used by the loader benchmarks without
requiring root cache drops.

# Models

All experiments use TP=8.

| Model | Path | Safetensors | Payload bytes |
| --- | --- | ---: | ---: |
| Qwen3-30B-A3B-Instruct-2507 | `/mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507` | 16 | 61,064,245,248 |
| Qwen3-235B-A22B-Instruct-2507 | `/mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507` | 118 | 470,187,269,120 |

# Metadata Feasibility

The metadata estimator reconstructs the current target-slice-first local
mapped read shape using real safetensors tensor shapes and Qwen/vLLM TP naming
conventions:

- `o_proj.weight` and `down_proj.weight` are treated as dim1-sharded row-major
  reads. Current local mapped execution reads full source rows per rank.
- `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `gate_proj.weight`,
  `up_proj.weight`, `lm_head.weight`, and `embed_tokens.weight` are treated as
  dim0-sharded source-contiguous reads.
- norms, routers, and other small tensors are treated as replicated.

The estimator matches the measured TensorCast profile closely enough to be used
as a feasibility model.

| Model | Profile read/rank | Estimated read/rank | Ratio | Profile dst/rank | Estimated dst/rank | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30B | 25,145,667,584 | 25,271,496,704 | 1.005x | 7,529,590,784 | 7,655,419,904 | 1.017x |
| 235B | 195,966,557,184 | 196,853,652,480 | 1.005x | 57,973,955,584 | 58,861,050,880 | 1.015x |

Source-window collective estimate:

| Model | Current estimated read/rank | Source-window disk read/rank | Disk read vs current | Estimated saving/rank | Window count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30B | 25,271,496,704 | 7,633,030,656 | 0.302x | 17,638,466,048 | 123 |
| 235B | 196,853,652,480 | 58,773,408,640 | 0.299x | 138,080,243,840 | 942 |

Conclusion:

- The current TensorCast regression is explained by dim1 source-row rereads.
- A source-window collective plan reduces per-rank disk read volume to almost
  exactly `model_payload / TP`.
- The expected group disk read amplification is `1.0x` against safetensors
  payload bytes.

# IO Feasibility

The IO smoke mode simulates the source-window read side only:

1. Split each safetensors payload into 512MiB source windows.
2. Split each window into TP=8 rank stripes.
3. Start eight local worker processes.
4. Each worker reads its rank stripe with 16MiB `preadv` chunks.
5. Use `posix_fadvise(DONTNEED)` before the run.

This intentionally excludes daemon startup, vLLM startup, NCCL, GPU scatter,
and model initialization.

| Model | Payload read | Wall time | Max-rank time | Throughput by wall | Throughput by max rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| 30B | 61,064,245,248 | 9.96s | 9.71s | 6.13 GB/s | 6.29 GB/s |
| 235B | 470,187,269,120 | 71.50s | 71.26s | 6.58 GB/s | 6.60 GB/s |

Reference benchmark numbers:

| Model | Loader | Weight load |
| --- | --- | ---: |
| 30B | InstantTensor SSD cold | 10.43s |
| 30B | TensorCast final auto SSD cold | 26.20s |
| 235B | InstantTensor SSD cold | 70.69s |
| 235B | TensorCast final auto SSD cold | 151.59s |

Interpretation:

- 30B read-only source-window striping is already faster than InstantTensor's
  measured end-to-end weight load window. This leaves budget for TensorCast
  scatter and collective overhead if implementation overlaps read,
  distribution, and target writes correctly.
- 235B read-only source-window striping is essentially equal to the
  InstantTensor weight load time. For 235B, implementation overhead must be
  tightly controlled and overlapped; the read side is sufficient but does not
  leave much slack.
- Both runs are close to the previous SSD direct-read upper-bound result
  (`6.72 GB/s`), so the source-window read shape can reach the required
  storage-side throughput.

# Production WorkPlan And Runtime Validation

Phase 0 used safetensors headers plus a Qwen/vLLM naming classifier. That was
useful for hypothesis testing, but production TensorCast must consume the
artifact-centered `RepresentationWorkPlan` and `IntoTargetLayout`.

The production gate now runs the group source-window planner from real vLLM
TensorCast mapped-target runs. The current implementation also has a strict
runtime path: source-window selection is required, local mapped fallback is
fail-closed, and the runtime uses rank-striped source reads plus
`FullWindowAllGather`.

30B SSD command shape:

- model: `/mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507`
- TP=8, bf16, `max_model_len=1024`, `HF_HUB_OFFLINE=1`
- TensorCast daemon and global store prestarted; daemon startup excluded from
  `LOAD_DONE`
- `materialization.collective=collective_first`
- `executor_preference=source_window`
- `source_window_selection_mode=strict`
- `source_window_distribution_mode=full_window_all_gather`
- `source_window_allow_mixed_residual=false`
- `local_mapped_safetensors_io_mode=direct`

Artifacts:

- `/data/tc/qwen3-source-window-dryrun/20260615-075740-source-window-dryrun14-baseline-fix/qwen3_30b_ssd_source_window_dryrun_baseline_fix`
- `/tmp/qwen3_30b_a3b_ssd_vllm_tensorcast_source_window_dryrun14_baseline_fix.log`
- `/data/tc/qwen3-source-window-runtime/20260615-090454-source-window-strict05-30b-ssd-direct-source/qwen3_30b_ssd_source_window_strict05_direct_source`
- `/tmp/qwen3_30b_a3b_ssd_vllm_tensorcast_source_window_strict05_direct_source.log`
- `/data/tc/qwen3-235b-source-window-runtime/20260615-091102-source-window-strict-direct-235b-ssd/qwen3_235b_ssd_source_window_strict_direct`
- `/tmp/qwen3_235b_a22b_ssd_vllm_tensorcast_source_window_strict_direct.log`

30B production strict metrics:

| Metric | Value |
| --- | ---: |
| `LOAD_DONE` | 55.428s |
| max rank weight load | 23.404s |
| runtime data-plane total | 14.527s |
| runtime read time | 11.730s |
| runtime read throughput | 5.21 GB/s |
| compressed 2D spans | 148,240 |
| candidate spans | 150,936 |
| windows | 115 |
| residual bytes | 0 |
| group disk read bytes | 61,064,245,248 |
| max rank disk read bytes | 8,047,610,880 |
| local mapped physical read baseline, max rank | 25,296,662,528 |
| rank read saving | 17,249,051,648 |
| unique payload bytes | 61,064,245,248 |
| target write bytes | 61,444,685,824 |
| read amplification | 1.000x |
| scatter ops | 154,057 |
| peer transfer bytes, full-window all-gather | 427,449,716,736 |
| peer useful bytes | 53,766,066,176 |
| peer waste bytes | 373,683,650,560 |

30B strict/direct read-ahead metrics:

| Metric | Value |
| --- | ---: |
| `LOAD_DONE` | 52.814s |
| max rank weight load | 21.269s |
| runtime data-plane total | 12.458s |
| runtime read time | 11.946s |
| runtime read throughput | 5.11 GB/s |
| active pipeline slots | 2 |
| chunks | 913 |

30B persistent-reader and ConsumerRouted follow-up:

| Variant | Max rank weight load | LOAD_DONE | Runtime note |
| --- | ---: | ---: | --- |
| persistent per-rank readers | 21.432s | 53.858s | runtime total 12.356s, read 11.872s |
| explicit ConsumerRouted | 23.096s | 56.721s | peer transfer 53.7GB, peer waste 0, scatter issue 9.758s |

235B production strict metrics:

| Metric | Value |
| --- | ---: |
| `LOAD_DONE` | 154.284s |
| max rank weight load | 121.379s |
| runtime data-plane total | 103.272s |
| runtime read time | 92.025s |
| runtime read throughput | 5.11 GB/s |
| windows | 885 |
| residual bytes | 0 |
| group disk read bytes | 470,187,269,120 |
| max rank disk read bytes | 59,054,668,288 |
| local mapped physical read baseline, max rank | 196,952,218,624 |
| rank read saving | 137,897,550,336 |
| unique payload bytes | 470,187,269,120 |
| target write bytes | 471,676,936,192 |
| read amplification | 1.000x |
| scatter ops | 313,709 |
| peer transfer bytes, full-window all-gather | 3,291,310,883,840 |
| peer useful bytes | 412,904,358,912 |
| peer waste bytes | 2,878,406,524,928 |

235B strict/direct read-ahead metrics:

| Metric | Value |
| --- | ---: |
| `LOAD_DONE` | 146.933s |
| max rank weight load | 115.080s |
| runtime data-plane total | 96.484s |
| runtime read time | 94.545s |
| runtime read throughput | 4.97 GB/s |
| active pipeline slots | 2 |
| chunks | 7,074 |

235B strict/direct persistent-reader metrics:

| Metric | Value |
| --- | ---: |
| `LOAD_DONE` | 141.427s |
| max rank weight load | 109.878s |
| runtime data-plane total | 91.707s |
| runtime read time | 90.212s |
| runtime read throughput | 5.21 GB/s |
| active pipeline slots | 2 |
| chunks | 7,074 |

Planner iteration notes:

| Iteration | Result |
| --- | --- |
| naive fragment expansion | about 262s planner time; not production viable |
| guarded high-expansion fallback | about 1s planner time, but about 20.535GB/group residual |
| compressed 2D source spans plus 2D-source to 3D-target expert spans | about 1.1-1.4s planner time, guarded residual eliminated, residual reduced to about 1.208GB/group |
| local pad/fill as typed local work and strict source-window coverage-only planning | zero residual bytes for 30B and 235B strict runs |
| skip byte-range map construction in strict source-window | 30B max rank weight load improved from 52.87s to 25.53s |
| skip source-layout map and use direct source factory | 30B max rank weight load improved to 23.40s |
| 2-slot host read-ahead | 30B improved to 21.27s; 235B improved to 115.08s |
| persistent per-rank readers | 235B improved to 109.88s; 30B was neutral/noisy at 21.43s |
| persistent read-ahead pump | 30B improved to 20.72s; 235B was neutral/noisy at 109.96s |
| direct-to-pinned source-window reads | 30B improved to 18.94s; 235B improved to 94.28s with 6.29GB/s runtime read throughput |
| explicit ConsumerRouted | peer waste fell to zero, but 30B regressed to 23.10s due to pack/scatter issue cost |
| source-window CUDA device-cache/lazy scatter | 30B ConsumerRouted improved to 19.09s and all-gather `scatter_issue` fell to 1.22s; 235B lazy-device did not refresh the best because read time regressed; batched routed pack was measured and rejected |
| forced ConsumerRouted after current fixes | 30B measured 17.61s/50.07s and peer transfer 53.7GB, but `routed_pack_ops=147585` and `scatter_issue=5.59s` still erased the peer-byte saving |
| explicit LocalOnly | planner/runtime implemented for owner-only windows; correctness covered as a hybrid-building-block path, not a Qwen3 TP=8 performance lever |
| 128MiB engine slice/chunk tuning | early runs regressed versus 64MiB read-ahead; the current same-code 235B full-window control with 128MB chunks is now the best 235B TensorCast run, so chunk size must be evaluated against the current runtime state |
| HybridWindow 64MB routed gate | routed windows are reachable but sparse on Qwen3: 30B routes 1/115 windows and measures 17.60s/48.49s; 235B routes 4/885 and measures 89.10s/126.17s |
| current 235B full-window all-gather control | 235B measures 88.52s/119.79s, with runtime read time 73.23s and direct pinned reads succeeding on 28194/28312 attempts |
| daemon parsed-index cache | 30B HybridWindow 4GB-gate `prepare_source_bound_plan` mean fell from 1.197s to 0.805s; weight load improved modestly to 19.07s |
| source-window plan hash reserve | 30B plan `hash_sec` fell from about 0.294s to 0.278s; end-to-end stayed in benchmark noise |
| source-window plan-hash v2 execution-fact payload | 30B `hash_sec` fell to 0.126s and group `plan_sec` to 1.198s; fair weight load was 17.74s but `LOAD_DONE` was 51.40s, so this is a control-plane improvement rather than final convergence |
| compact binary plan-hash payload | rejected; 30B `hash_sec` regressed to 0.565s and plan build total to 1.388s |
| flat-hash planner interval buckets | rejected; 30B `metrics_sec` regressed to 0.430s and plan build total to 1.408s |
| source-window participant copy-elision | 30B improved to 17.75s max-rank weight load and 50.76s LOAD_DONE; runtime total 9.83s |
| 2D row-band scatter span coalescing | planner unit coverage added; 30B HybridWindow 64MB rerun measured 17.43s/49.04s and reduced runtime scatter ops only from about 153754 to 153714, so this is retained as a correctness cleanup, not a throughput lever |
| borrowed planner member input | rejected; `plan_sec` fell but vLLM-visible weight load regressed to 23.30s |
| source-index singleflight/preparse overlap | 30B `parse_source_index_sec` fell to 0 and wait was microseconds; fair weight load was 18.79s/49.28s, so the optimization is retained for planner latency but not counted as the best run |
| source-window one-pass window builder with reserve fix | 30B `windows_sec` fell to 0.368s, group `plan_sec` to 1.129s, and fair weight load to 17.29s/49.93s; an exact-reserve variant was rejected because it regressed `windows_sec` to 1.329s |
| canonical-index preparse future | rejected and reverted; it reported `canonical_index_table_preparsed=1`, but still waited `0.23-0.28s` in the builder and regressed fair 30B timing to 17.62s/50.69s |
| binding source-table reuse | kept as a safe cleanup; 30B HybridWindow 64MB rerun confirmed `source_table_reused=1` on all TP ranks but measured 17.83s/49.31s, so it does not change the acceptance baseline |
| stream-ordered all-gather scatter | rejected and reverted; 30B measured 17.81s/50.62s and 235B measured 89.37s/122.39s, with wait moving from `collective_sync` into `scatter_sync` |
| multi-slot source-window read-ahead | kept; host reads now run ahead across up to 8 staging slots. 30B measured 16.81s/51.10s at 4 slots and 16.87s/48.98s at 8 slots. 235B measured 87.17s/118.95s at 8 slots, with runtime total 72.24s for 470.19GB |

Conclusion:

- The production planner reproduces the important Phase 0 result without a
  tensor-name classifier on both 30B and 235B: planned group disk read is about
  the safetensors payload size, and read amplification is `1.0x`.
- The admission-baseline bug is fixed in the current code slice. The planner
  now compares against the local mapped physical read estimate derived from
  WorkPlan geometry.
- Strict source-window publication is now correctness-validated for 30B with
  the same deterministic prompt output as default. 235B TensorCast generation
  had already produced coherent output in the loader report.
- The runtime label `full_window_all_gather` now means rank-striped reads plus
  all-gather. It is good enough for correctness and first performance data,
  but not the final InstantTensor-class distribution.
- The 30B runtime data plane benefits from deeper read-ahead: 8 staging slots
  reduce daemon runtime total to about `9.40s` and host read wait to `7.25s`.
  The vLLM-visible gap is now mostly daemon planning/group orchestration and
  vLLM/NCCL/bootstrap overhead, not missing source metadata.
- Source-index singleflight plus preparse overlap removes the execution-plan
  `parse_source_index_sec` tail, but end-to-end loading remains dominated by
  artifact resolve, group plan/admission, and runtime distribution/IO.
- Window-builder copy reduction is useful but sensitive to allocation shape:
  removing a grouped-span copy only helped after preserving exponential vector
  capacity growth for consumer spans. The retained implementation reduces group
  planning to about 1.13s on 30B without changing source-window semantics.
- `LocalOnly` is implemented only for windows whose consumer spans are all on
  the owner rank. This preserves distribution semantics and gives future
  hybrid plans a local subpath without weakening group admission.
- Repeated `cudaSetDevice` calls were a real runtime tax. Caching device
  selection and delaying scatter device switches until actual copy issue
  improved the routed path materially, but routed still needs operation-count
  reduction before it should become the default.
- The current gap is no longer residual handling or source read amplification.
  The easy read-queue-depth issue is mostly addressed: 235B 8-slot runtime
  reads 470.19GB in `72.24s`, about `6.51GB/s`, close to the measured
  `6.72GB/s` SSD O_DIRECT ceiling. The remaining high-value work is group-level
  prepared execution/cache to remove repeated per-rank prepare/control cost,
  and a lower-op-count routed/hybrid distribution to reduce full-window
  all-gather peer waste without adding 100k+ pack/scatter operations. On 235B,
  full-window all-gather still transfers `3.29TB` across peers with `2.88TB`
  waste.

# Raw Artifacts

Metadata-only:

- `/data/tc/source_window_collective_feasibility_30b_metadata.json`
- `/data/tc/source_window_collective_feasibility_235b_metadata.json`

Full rank-striped IO with `posix_fadvise(DONTNEED)`:

- `/data/tc/source_window_collective_feasibility_30b_io_full_fadvise.json`
- `/data/tc/source_window_collective_feasibility_235b_io_full_fadvise.json`

Small script sanity check:

- `/data/tc/source_window_collective_feasibility_30b_io_smoke_128m.json`

# Commands

```bash
tools/experiments/source_window_collective_feasibility.py \
  /mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507 \
  --tp 8 \
  --current-profile-read-bytes-per-rank 25145667584 \
  --current-profile-dst-bytes-per-rank 7529590784 \
  --output /data/tc/source_window_collective_feasibility_30b_metadata.json \
  --pretty

tools/experiments/source_window_collective_feasibility.py \
  /mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507 \
  --tp 8 \
  --current-profile-read-bytes-per-rank 195966557184 \
  --current-profile-dst-bytes-per-rank 57973955584 \
  --output /data/tc/source_window_collective_feasibility_235b_metadata.json \
  --pretty

tools/experiments/source_window_collective_feasibility.py \
  /mnt/host0/vllm-loader-bench/qwen3-30b-a3b-instruct-2507 \
  --tp 8 \
  --io-smoke \
  --fadvise-dontneed \
  --output /data/tc/source_window_collective_feasibility_30b_io_full_fadvise.json \
  --pretty

tools/experiments/source_window_collective_feasibility.py \
  /mnt/host0/vllm-loader-bench/qwen3-235b-a22b-instruct-2507 \
  --tp 8 \
  --io-smoke \
  --fadvise-dontneed \
  --output /data/tc/source_window_collective_feasibility_235b_io_full_fadvise.json \
  --pretty
```

# Implementation Gates

The production implementation should not be considered successful merely
because it selects a new executor. It must preserve these gates:

1. WorkPlan grounding:
   - dump real Qwen3 30B/235B TP=8 TensorCast `RepresentationWorkPlan`
     summaries from the existing vLLM TensorCast path;
   - run production `summarize_source_window_collective(...)` in dry-run mode
     on those summaries;
   - reproduce the Phase 0 source-window read/rank estimates without tensor
     naming rules;
   - infer rect2d from `RepresentationWorkSourceFragment` source/destination
     geometry, not from an explicit partition enum.
2. Strategy selection:
   - Qwen3 TP=8 mapped target selects `SourceWindowCollectiveExecutor`.
   - The old local mapped path remains available as an explicit fallback or
     residual path, not as the default for this shape.
3. Read volume:
   - 30B per-rank disk read target: about `7.63GB`.
   - 235B per-rank disk read target: about `58.77GB`.
   - Reported source read amplification against safetensors payload should be
     no more than `1.2x` in the first implementation.
4. Throughput:
   - 30B SSD cold first target: approach the `10s` read-side feasibility
     bound; first acceptable implementation should be clearly better than the
     current `26.20s`.
   - 235B SSD cold first target: approach the `71s` read-side feasibility
     bound; first acceptable implementation should be clearly better than the
     current `151.59s`.
5. Correctness:
   - 30B deterministic generation remains byte-for-byte identical to the
     default and InstantTensor checks already recorded.
   - 235B TensorCast generation remains coherent and normal.
6. Diagnostics:
   - Report source-window count, group disk read bytes, rank disk read max,
     peer transfer bytes, scatter op count, target write bytes, and residual
     bytes in `ExecutionCommitReport` or equivalent diagnostics.

Current status after the strict/direct TP=8 runs:

- WorkPlan grounding, strategy selection, read-volume, correctness, and
  diagnostics gates are satisfied for the tested 30B/235B shapes.
- Throughput is improved over TensorCast local mapped but not yet accepted as
  InstantTensor-class: 30B best measured is `16.81s` versus InstantTensor
  `10.43s`; 235B is now `87.17s` versus InstantTensor `70.69s`.
- The best fair 30B `LOAD_DONE` remains `48.49s` from the HybridWindow
  64MB routed-gate run, but it routed only one window and did not improve the
  best isolated weight-load time. The current 8-slot full-window run is
  `48.98s`; the current 235B 8-slot full-window run is the best 235B TensorCast
  sample at `87.17s`/`118.95s`.
- The latest 30B source-index preparse run improved the resolved execution
  planning segment but measured `18.79s`, so the acceptance baseline remains
  the later `17.29s` window-builder reserve-fix run.
- The plan-hash v2 run reduced planner hash cost and measured `17.74s`, but
  `LOAD_DONE=51.40s`; treat it as equal to the current best weight-load band,
  not as evidence that total startup has converged.
- The window-builder reserve-fix run reduced `windows_sec` to `0.368s` and
  group `plan_sec` to `1.129s`, but `LOAD_DONE=49.93s`; this keeps improving
  control-plane cost while the remaining InstantTensor gap stays in runtime and
  vLLM-visible orchestration.
- The binding source-table reuse rerun confirms the no-view path now avoids a
  redundant parsed-table lookup (`source_table_reused=1` on all TP ranks), but
  its 30B timing was `17.83s`/`49.31s`. This is useful negative evidence:
  rank-local parser/cache cleanups are no longer enough to close the gap.
- The stream-ordered scatter experiment confirms the explicit post-all-gather
  synchronization is not the main runtime gap. Removing it completed 30B/235B
  runs but did not improve them; the wait shifted into final scatter
  synchronization, and the code was reverted.
- The same run's profile shows `store.from_disk` max `1.693s`,
  `local_ready.realize.realize_from` max `13.546s`, daemon
  `prepare_plan_sec` max `0.957s`, daemon `prepare_execution_sec` max
  `0.778s`, daemon `materialize_sec` max `11.840s`, and Python finalize max
  `0.116s`. This confirms that the next feasibility gate should target daemon
  admission/orchestration and runtime pipeline quality rather than adding a
  vLLM-side loading branch.
- The storage-span member cache/span-view follow-up removed per-work-item
  target storage span reconstruction and measured `append_sec=0.091s`,
  group `plan_sec=1.117s`, and `17.36s`/`49.87s`. This validates the cleaner
  group input shape but does not change the acceptance baseline: 30B best weight
  load is still `17.29s`.
- The `ResolvePublicDiskSource` context-cache follow-up makes mounted-source
  resolve build metadata through the shared `DiskArtifactContext` cache, with
  the cache fingerprint strengthened to include `inode`, `size`, and
  `mtime_ns`. The 30B rerun confirmed one resolve-time context miss followed by
  rank-local materialization cache hits. `store.from_disk` average improved
  from `1.638s` to `1.307s`, but max-rank weight load remained effectively
  unchanged at `17.38s`/`49.60s`; this is a correctness-preserving metadata
  reuse cleanup, not the next performance lever.
- The local-read metrics inline follow-up keeps the same source-window
  admission values while collecting local mapped physical-read intervals during
  the existing window metrics scan. The 30B rerun measured `metrics_sec=0.261s`,
  group `plan_sec=1.095s`, and `17.36s`/`48.99s`; this is another small planner
  cleanup, not a replacement for runtime data-plane work.
- The representation builder reserve experiment was negative and reverted.
  Manual group reserve measured `17.70s`/`50.83s` and simple vector/protobuf
  reserve measured `17.68s`/`49.21s`; neither reduced
  `representation_realization_plan` beyond the existing `0.39-0.40s` noise
  band. This points away from small per-rank builder allocation tweaks and back
  toward group-level orchestration/runtime work.
- The canonical-index preparse future experiment was also negative and
  reverted. It moved a small parse cost into future wait but did not reduce
  end-to-end load time, so the next control-plane work should remove duplicated
  per-rank representation/materialization planning through a group-level
  artifact realization plan.
- The next acceptance gate is runtime data-plane quality: coalesced routed or
  hybrid peer transfer, deeper async IO, and fewer per-chunk synchronization
  points. Naive ConsumerRouted is not sufficient because operation count can
  dominate saved peer bytes.

# Risks

The validation proves the read-side shape, not the whole executor. Remaining
implementation risks:

- NCCL distribution must overlap with source reads and scatter; a naive
  read-then-allgather-then-scatter pipeline may give back too much of the
  235B margin.
- Rank-striped full-window all-gather is an MVP distribution mode, not the
  final abstraction. The production plan must retain consumer-span information
  so routed or hybrid distribution can reduce peer waste without changing
  public APIs.
- Owner-read plus broadcast is not equivalent to the target all-gather IO
  shape. It is useful for debugging coordination, but it should not be used as
  the performance basis for admission or benchmark conclusions.
- Window buffers must be bounded. The initial target should use 512MiB windows
  or smaller, not materialize whole files per rank.
- Rank-striped full-window all-gather is acceptable for the first
  implementation only if memory pressure and peer bandwidth remain bounded. A
  routed or hybrid distribution mode can reduce peer bytes for pure sharded
  tensors.
- The executor must consume `RepresentationWorkPlan` and `IntoTargetLayout`
  directly. Reconstructing vLLM-specific loader rules inside production code
  would violate the artifact-centered boundary.

# Decision

Proceed with implementing `SourceWindowCollectiveExecutor`.

The feasibility gate is passed:

- the offline planner explains the current TensorCast regression;
- source-window planning removes the read amplification in the target shapes;
- rank-striped IO reaches the required SSD throughput on both 30B and 235B;
- production WorkPlan dry-run on 30B now reproduces the `1.0x` read
  amplification result without tensor-name rules;
- the remaining work is executor implementation, admission-baseline correction,
  residual handling, and 235B production WorkPlan validation, not a fundamental
  storage-side blocker.
