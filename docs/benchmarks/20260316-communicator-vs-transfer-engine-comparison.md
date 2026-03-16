# Communicator vs Transfer Engine Comparison

Date: 2026-03-16  
Status: Active

## 1. Goal

This document defines a chart set that makes TensorCast communicator and
Mooncake Transfer Engine performance easy to compare without hiding the
remaining caveats.

The output is intentionally split into:

1. a framework-scale reference chart
2. a topology-sensitivity chart
3. a per-lane reference chart

That is necessary because not all available results are perfect
apples-to-apples.

## 2. Generated Figures

### 2.1 Full-8 Framework Throughput Reference

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

### 2.2 Full-8 Mapping Sensitivity

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

### 2.3 Per-lane Reference

![Per-lane Reference](./image/communicator_vs_te_per_lane_20260316.svg)

This chart shows how close communicator has moved to the single-lane / per-lane
reference regime.

Use it to answer:

1. can one large communicator `read_tensor()` now reach the same order of
   magnitude as per-lane full-8 traffic?

Current reading:

1. communicator single big read is now about `20.25 GB/s`
2. communicator full-8 per-lane average is about `19.28 GB/s`
3. Mooncake full-8 best per-lane average is about `23.17 GB/s`

## 3. Comparison Data

The chart source snapshot is:

- [20260316-communicator-vs-te-comparison.json](./data/20260316-communicator-vs-te-comparison.json)

The chart generator is:

- [render_te_comparison_charts.py](/data/workspace/tensorcast-280/tools/communicator/render_te_comparison_charts.py)

## 4. Metric Definitions

For communicator-generated data, prefer the explicit `bw_GBps` / `bw_GBps_*`
fields when present. Older result files still carry the legacy field name
`bw_gbps`, which represents the same decimal `GB/s` quantity.

### 4.1 Communicator Full-8 Aggregate

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

### 4.2 Communicator Single Big Read

For the single-artifact communicator chart, we use the strict direct RDMA case:

- `64 MiB`
- `threads=1`
- `batch_size=1`
- `qp_count=4`
- `outstanding_wr=256`

Case:

- [/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json](/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json)

### 4.3 Mooncake Full-8 Best

Mooncake full-8 framework best comes from:

- `/data/workspace/Mooncake/docs/source/performance/transfer-engine-full8-rdma-best-practices-20260316.md`

Value used:

1. `1482.64 Gbps`

### 4.4 Mooncake Mapping Sensitivity

Mooncake mapping values come from:

- `/data/workspace/Mooncake/docs/source/performance/transfer-engine-gdr-mapping-20260315.md`

Values used:

1. `map8_id`
2. `map8_within_swap`
3. `map8_half_swap`
4. `map8_target_bad_local`

## 5. Caveats

These caveats must remain attached to the figures.

### 5.1 Operation Direction

1. Mooncake full-8 best is a write-path best practice result
2. communicator full-8 best currently comes from read-path mapping results

Therefore:

1. the framework-scale ceiling chart is a *reference comparison*, not a strict
   apples-to-apples benchmark

### 5.2 Framework Behavior vs Framework Capability

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

### 5.3 Mapping Conclusion Scope

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

1. show the full-8 framework reference chart first
2. then show the normalized mapping chart
3. then show the per-lane chart
4. finally explain whether the current TensorCast framework will naturally
   expose enough independent transfers to realize communicator's full-8
   capability
