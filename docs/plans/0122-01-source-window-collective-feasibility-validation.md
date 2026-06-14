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

The next pre-production gate is the Phase 0.5 collective and scatter
microbench in `docs/plans/0122-tp-source-window-collective-realization.md`.
That gate decides whether the first runtime implementation can use
full-window all-gather or must start with routed or hybrid distribution.

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

1. Strategy selection:
   - Qwen3 TP=8 mapped target selects `SourceWindowCollectiveExecutor`.
   - The old local mapped path remains available as an explicit fallback or
     residual path, not as the default for this shape.
2. Read volume:
   - 30B per-rank disk read target: about `7.63GB`.
   - 235B per-rank disk read target: about `58.77GB`.
   - Reported source read amplification against safetensors payload should be
     no more than `1.2x` in the first implementation.
3. Throughput:
   - 30B SSD cold first target: approach the `10s` read-side feasibility
     bound; first acceptable implementation should be clearly better than the
     current `26.20s`.
   - 235B SSD cold first target: approach the `71s` read-side feasibility
     bound; first acceptable implementation should be clearly better than the
     current `151.59s`.
4. Correctness:
   - 30B deterministic generation remains byte-for-byte identical to the
     default and InstantTensor checks already recorded.
   - 235B TensorCast generation remains coherent and normal.
5. Diagnostics:
   - Report source-window count, group disk read bytes, rank disk read max,
     peer transfer bytes, scatter op count, target write bytes, and residual
     bytes in `ExecutionCommitReport` or equivalent diagnostics.

# Risks

The validation proves the read-side shape, not the whole executor. Remaining
implementation risks:

- NCCL distribution must overlap with source reads and scatter; a naive
  read-then-allgather-then-scatter pipeline may give back too much of the
  235B margin.
- Full-window all-gather is an MVP distribution mode, not the final
  abstraction. The production plan must retain consumer-span information so
  routed or hybrid distribution can reduce peer waste without changing public
  APIs.
- Window buffers must be bounded. The initial target should use 512MiB windows
  or smaller, not materialize whole files per rank.
- Full-window all-gather is acceptable for the first implementation only if
  memory pressure and peer bandwidth remain bounded. A routed or hybrid
  distribution mode can reduce peer bytes for pure sharded tensors.
- The executor must consume `RepresentationWorkPlan` and `IntoTargetLayout`
  directly. Reconstructing vLLM-specific loader rules inside production code
  would violate the artifact-centered boundary.

# Decision

Proceed with implementing `SourceWindowCollectiveExecutor`.

The feasibility gate is passed:

- the offline planner explains the current TensorCast regression;
- source-window planning removes the read amplification in the target shapes;
- rank-striped IO reaches the required SSD throughput on both 30B and 235B;
- the remaining work is executor implementation and integration, not a
  fundamental storage-side blocker.
