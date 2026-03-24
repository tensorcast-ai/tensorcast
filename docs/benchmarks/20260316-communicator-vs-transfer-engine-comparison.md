# Communicator vs Transfer Engine Comparison

Date: 2026-03-16  
Status: Active

## 1. Goal

This document defines a chart set that makes TensorCast communicator and
Mooncake Transfer Engine performance easy to compare without hiding the
remaining caveats.

`2026-03-22` 起，`0105` 的主目标从“仅看带宽差距”调整为“量化任务过程中的非数据传输代价”。
因此本页中的 `GB/s` 图表用于定位趋势，不再单独作为根因结论；根因分析需结合 communicator bench 新增的 amortizable/recurring profiling 字段。

The output is intentionally split into:

1. a single-NIC aligned reference chart
2. a framework-scale full-8 reference chart
3. a topology-sensitivity chart

That is necessary because not all available results are perfect
apples-to-apples.

## 2. Generated Figures

### 2.1 Single-NIC Large-Block Sweep

![Single-NIC Reference](./image/communicator_vs_te_single_nic_reference_20260316.svg)

What it is for:

1. focus the main single-NIC figure on large single-request reads that are
   closer to weight-region loading than the earlier `1 MiB x 16 x 8` stress
   case
2. keep raw IB verbs results visible as horizontal references instead of
   giving the aligned-peak comparison its own dominant panel
3. show whether communicator and Mooncake TE converge once the logical request
   becomes large enough

What it is **not**:

1. not a strict same-host-pair apples-to-apples sweep across every point
2. the `64 MiB` point comes from the earlier `0496 -> 0078` aligned pair
3. the `256 MiB` / `1 GiB` / `4 GiB` points come from a later
   `0496 -> 0550` aligned pair after the `0078` worker stopped scheduling
   reliably

Use it to answer:

1. if a single logical `read_tensor()` is already very large, does
   communicator still lag TE?
2. how close do both frameworks get to the raw single-NIC verbs references?

Current reading:

1. raw aligned IB verbs references remain about `24.46 ~ 24.50 GB/s`
2. at `32 MiB`, communicator is still clearly below TE:
   - communicator: `18.56 GB/s`
   - TE: `22.49 GB/s`
3. at `64 MiB`, communicator is still below TE:
   - communicator: `20.56 GB/s`
   - TE: `22.75 GB/s`
4. once the logical request reaches `256 MiB+`, communicator and TE are in the
   same single-NIC class:
   - `256 MiB`: communicator `23.17 GB/s`, TE `23.02 GB/s`
   - `1 GiB`: communicator `23.96 GB/s`, TE `23.69 GB/s`
   - `4 GiB`: communicator `24.15 GB/s`, TE `23.88 GB/s`
5. this supports the interpretation that the earlier large gap is dominated by
   many-small-request control granularity, not by the large direct-RDMA path

### 2.2 Full-8 Framework Throughput Reference

![Full-8 Reference](./image/communicator_vs_te_full8_ceiling_20260316.svg)

What it is for:

1. show whether each framework can approach the full-8 hardware envelope
2. keep raw fabric ceiling visible

What it is **not**:

1. not a strict apples-to-apples chart
2. Mooncake full-8 best is write-path
3. communicator full-8 best is read-path

Use it to answer:

1. can communicator now enter the same full-8 performance class as Transfer
   Engine?

Current reading:

1. raw RDMA: `1568.75 Gbps`
2. Transfer Engine full-8 best: `1482.64 Gbps`
3. communicator full-8 best: about `1233.68 Gbps`

### 2.3 Full-8 Mapping Sensitivity

![Mapping Sensitivity](./image/communicator_vs_te_mapping_norm_20260316.svg)

This is the most important topology comparison chart.

Why it uses normalized values:

1. Mooncake mapping data and communicator mapping data are not the same
   absolute workload
2. normalized values let readers compare *direction* rather than raw level

Use it to answer:

1. does communicator show the same mapping sensitivity trend as Mooncake?

Current reading:

1. both frameworks agree that breaking destination local GPU/NIC affinity is
   bad
2. communicator shows much weaker sensitivity to pure remote permutation than
   Mooncake

## 3. Comparison Data

The chart source snapshot is:

- [20260316-communicator-vs-te-comparison.json](./data/20260316-communicator-vs-te-comparison.json)

The chart generator is:

- [render_te_comparison_charts.py](/data/workspace/tensorcast-280/tools/communicator/render_te_comparison_charts.py)

## 4. Metric Definitions

For communicator-generated data, prefer the explicit `bw_GBps` / `bw_GBps_*`
fields when present. Older result files still carry the legacy field name
`bw_gbps`, which represents the same decimal `GB/s` quantity.

### 4.0 0105 Profiling Read Order (Updated)

For new communicator runs, read metrics in this order:

1. `amortizable_*`
   - one-time costs (init, allocation, warmup) that can be amortized
2. `recurring_*`
   - per-transfer costs (clear, issue, wait, verify)
3. verify breakdown
   - `recurring_avg_verify_buffer_alloc_*`
   - `recurring_avg_verify_copy_*`
   - `recurring_avg_verify_checksum_*`
4. throughput fields
   - `bw_GBps` / `bw_gbps` used as secondary guardrail

This order prevents misattributing startup or verification overhead as pure
data-plane bandwidth regression.

### 4.0.1 2026-03-22 Non-Transfer Cost Attribution (8xH800 Nodes + RDMA)

To validate the new `0105` objective, we ran an A/B pair (one host pair per case) on `8xH800` nodes
with strict direct RDMA (`64 MiB`, `threads=1`, `batch_size=1`, `qp_count=4`,
`outstanding_wr=256`):

1. verify on: `/data/tc/0105-non-transfer-profile-20260322-221354/result.json`
2. verify off: `/data/tc/0105-non-transfer-profile-nv-20260322-222325/result.json`

Host pair (verify on):

1. target: `gpu-h800-0234.host.platform.shaipower.com`
2. initiator: `gpu-h800-0214.host.platform.shaipower.com`

Host pair (verify off):

1. target: `gpu-h800-0468.host.platform.shaipower.com`
2. initiator: `gpu-h800-0449.host.platform.shaipower.com`

Note: this is not a strict same-host-pair A/B; the throughput gap is large
enough that the non-transfer-cost attribution conclusion still holds, but
future sensitivity studies should pin both runs to the same host pair.

Key numbers (verify on):

1. `bw_GBps=0.235584`
2. `amortized_per_iteration_us=72079.8`
3. `recurring_avg_iteration_total_us=284669`
4. `recurring_avg_verify_total_us=276997` (`97.30%` of recurring total)
5. verify split:
   - `recurring_avg_verify_buffer_alloc_us=30963.8` (`11.18%` of verify)
   - `recurring_avg_verify_copy_us=4651.4` (`1.68%` of verify)
   - `recurring_avg_verify_checksum_us=241378` (`87.14%` of verify)

Key numbers (verify off):

1. `bw_GBps=19.8732`
2. `amortized_per_iteration_us=1625.35`
3. `recurring_avg_iteration_total_us=3222.3`
4. `recurring_avg_wait_us=3116.3` (`96.71%` of recurring total)

Direct interpretation for `0105`:

1. this run's low throughput number is not a pure data-plane limit
2. recurring verify cost dominates the denominator when verify is enabled
3. startup/warmup is amortizable and measurable (`amortizable_*`), but not the
   primary bottleneck in this A/B pair
4. disabling verify restores the expected transfer-class throughput (`~19.87 GB/s`)

### 4.1 Single-NIC Large-Block Sweep

This chart intentionally centers the large single-request sweep:

1. bars:
   - communicator vs Mooncake TE at `32 MiB`, `64 MiB`, `256 MiB`, `1 GiB`,
     and `4 GiB`
   - all are single logical requests with `threads=1`, `batch_size=1`
2. reference lines:
   - raw verbs aligned `ib_read_bw`
   - raw verbs aligned `ib_write_bw`

Aligned pair used for raw verbs and Mooncake:

1. initiator: `gpu-h800-0496.host.platform.shaipower.com`
2. target: `gpu-h800-0078.host.platform.shaipower.com`
3. physical pair: `GPU2 <-> mlx5_2` on both hosts

Values used:

1. `ib_write_bw`: `196.00 Gbps` = `24.50 GB/s`
2. `ib_read_bw`: `195.72 Gbps` = `24.46 GB/s`
3. communicator `32 MiB`: `18.56 GB/s`
4. Mooncake TE `32 MiB`: `22.49 GB/s`
5. communicator `64 MiB`: `20.56 GB/s`
6. Mooncake TE `64 MiB`: `22.75 GB/s`
7. communicator `256 MiB`: `23.17 GB/s`
8. communicator `1 GiB`: `23.96 GB/s`
9. communicator `4 GiB`: `24.15 GB/s`
10. Mooncake TE `256 MiB`: `23.02 GB/s`
11. Mooncake TE `1 GiB`: `23.69 GB/s`
12. Mooncake TE `4 GiB`: `23.88 GB/s`

Evidence:

1. [/data/tc/single-nic-compare-20260316-161728/ib_write_bw_client_run_nopin.json](/data/tc/single-nic-compare-20260316-161728/ib_write_bw_client_run_nopin.json)
2. [/data/tc/single-nic-compare-20260316-161728/ib_read_bw_client_run_nopin.json](/data/tc/single-nic-compare-20260316-161728/ib_read_bw_client_run_nopin.json)
3. [/data/tc/single-nic-compare-20260316-161728/te_write_mlx5_2.json](/data/tc/single-nic-compare-20260316-161728/te_write_mlx5_2.json)
4. [/data/tc/single-nic-compare-20260316-161728/te_read_mlx5_2.json](/data/tc/single-nic-compare-20260316-161728/te_read_mlx5_2.json)
5. [/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json](/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json)
6. [/data/tc/single-nic-compare-20260316-161728/comm_initiator_run_aligned_mlx5_1.json](/data/tc/single-nic-compare-20260316-161728/comm_initiator_run_aligned_mlx5_1.json)
7. [/data/tc/large-single-compare2-20260316-200526/summary.json](/data/tc/large-single-compare2-20260316-200526/summary.json)
8. [/data/tc/large-single-32m-20260316-202602/summary.json](/data/tc/large-single-32m-20260316-202602/summary.json)

Important metric clarification:

1. `32 MiB`, `64 MiB`, `256 MiB`, `1 GiB`, and `4 GiB` here are the single
   logical read request sizes
2. they are not single RDMA READ WR sizes
3. communicator still splits one large request into multiple windows / segments
   before posting the actual RDMA operations

### 4.2 Communicator Full-8 Aggregate

For communicator full-8 mapping cases, the chart uses:

1. `sum(lane bytes) / max(lane wall_us)`

This is the correct data-plane wall for a lane-parallel benchmark.

We do **not** use:

1. the whole case wall

because whole-case wall includes:

1. detached worker orchestration
2. target bring-up
3. sequential lane startup

Those inflate the denominator and understate the real full-8 data-plane
throughput.

### 4.3 Communicator Single Big Read

For communicator's best-known single-artifact reference, we use the strict
direct RDMA case:

1. `64 MiB`
2. `threads=1`
3. `batch_size=1`
4. `qp_count=4`
5. `outstanding_wr=256`

Case:

- [/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json](/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json)

### 4.4 Mooncake Full-8 Best

Mooncake full-8 framework best comes from:

- `/data/workspace/Mooncake/docs/source/performance/transfer-engine-full8-rdma-best-practices-20260316.md`

Value used:

1. `1482.64 Gbps`

### 4.5 Mooncake Mapping Sensitivity

Mooncake mapping values come from:

- `/data/workspace/Mooncake/docs/source/performance/transfer-engine-gdr-mapping-20260315.md`

Values used:

1. `map8_id`
2. `map8_within_swap`
3. `map8_half_swap`
4. `map8_target_bad_local`

## 5. Caveats

These caveats must remain attached to the figures.

### 5.1 Single-NIC Caveat

1. the raw verbs reference lines come from the earlier aligned
   `0496 GPU2/mlx5_2 -> 0078 GPU2/mlx5_2` pair
2. the `32 MiB` bar comes from `0496 GPU0/mlx5_7 -> 0078 GPU0/mlx5_4`
3. the `64 MiB` communicator and TE bars come from an earlier aligned
   `0496 GPU1/mlx5_1 -> 0078 GPU1/mlx5_1` run set
4. the `256 MiB` / `1 GiB` / `4 GiB` sweep uses another aligned H800 pair
   (`0496 GPU0/mlx5_2 -> 0550 GPU0/mlx5_5`) because the `0078` worker was not
   schedulable in a stable way during this rerun window

### 5.2 Operation Direction

1. Mooncake full-8 best is a write-path best practice result
2. communicator full-8 best currently comes from read-path mapping results

Therefore:

1. the framework-scale ceiling chart is a *reference comparison*, not a strict
   apples-to-apples benchmark

### 5.3 Framework Behavior vs Framework Capability

Communicator capability and current TensorCast framework behavior are different.

Communicator capability:

1. communicator can now drive one large strict-direct `read_tensor()` to about
   `20.25 GB/s`
2. communicator can also drive `8` explicit lane-parallel transfers to about
   `150 ~ 154 GB/s` data-plane aggregate

Current TensorCast framework behavior:

1. `RemoteKeySource` binds to one `ip:port`
2. one artifact load is still single-source-session oriented
3. one artifact does not automatically stripe itself over `8` source GPUs/NICs

So when presenting the figures, say this explicitly:

1. the communicator charts show what the current communicator can do
2. they do not mean the current TensorCast artifact load path automatically
   uses all `8` NICs for one artifact

### 5.4 Mapping Conclusion Scope

Mooncake and communicator currently agree on:

1. local GPU/NIC pairing matters a lot

They do not currently agree on:

1. how strongly remote one-to-one permutation matters after both hosts already
   preserve good local pairs

That difference should be presented as:

1. a real behavioral difference between the two systems today
2. not as a charting artifact

## 6. Recommended Reader Flow

When presenting these charts to others, use this order:

1. show the single-NIC aligned reference chart first
2. then show the full-8 framework reference chart
3. then show the normalized mapping chart
4. finally explain whether the current TensorCast framework will naturally
   expose enough independent transfers to realize communicator's full-8
   capability
