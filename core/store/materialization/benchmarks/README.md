# Materialization 基准（Safetensors 加载策略）

本目录包含一个独立的基准测试（benchmark）二进制，用于评估 safetensors 加载策略，以及相关微基准（磁盘、GPU peer copy、NCCL）。

## 目标

- Bazel 目标：`//core/store/materialization/benchmarks:safetensors_load_strategy_benchmark`
- 源码：`core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc`

使用 Bazel 构建/运行：

```bash
bazel build //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark

# 注意 `--` 分隔符：前面是 Bazel 参数，后面是二进制的 flags。
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- --help
```

## 输入

### Safetensors 分片（shards）

`--mode=loader` 与 `--mode=safetensors_disk_baseline` 需要提供一个包含一个或多个 `*.safetensors` 文件的目录：

- `--safetensors_dir=/path/to/shards_dir`

程序会枚举该目录下的 `*.safetensors` 并按文件名排序。

### 磁盘基线文件

`--mode=disk_baseline` 与 `--mode=disk_fragmentation` 需要提供一个文件路径：

- `--disk_bench_path=/path/to/large_file`

创建一个大文件（示例）：

```bash
fallocate -l 8G /tmp/tensorcast_disk_bench.bin
```

## 模式与测量内容

通过 `--mode=...` 选择：

  - `loader`：对比加载策略 A/B/C：
    - A：把 payload 先整块读到 GPU “文件缓冲”（可选 GPU 内 pack 到最终布局）。
    - B：先规划 segment，再 VMM reserve+map，按需把片段泵到最终地址。
    - C：基于 B 的“已知完整计划”，按 checkpoint tensor 分组批处理：对非 contiguous slice（典型是 2D 的 `axis=1` 切片）改为顺序读大块 + GPU pack，避免海量碎片化小 IO；并对同源 slice 做读一次 + D2D 复制去重。
- `safetensors_disk_baseline`：把所有 safetensors payload bytes 按顺序 disk→GPU 读取（不做 per-tensor 规划）。
- `disk_baseline`：把单个大文件按顺序 disk→GPU 读取。
- `disk_fragmentation`：按固定 stride 读取大量小段（模拟碎片化/随机访问）。
- `gpu_peer_baseline`：测一次 GPU0→GPU1 的 `cudaMemcpyPeerAsync`。
- `h2d_nccl_broadcast_baseline`：先把数据做一次 `host(pinned) -> root GPU`，再用 NCCL `broadcast` 分发到 `tp_world_size` 张 GPU。
- `safetensors_hot_host_baseline`：**显式预热 OS page cache**（先顺序读取全部 payload 一次），再测 disk(buffered)→pinned host 吞吐。
- `safetensors_hot_disk_baseline`：**显式预热 OS page cache** 后，再测 disk(buffered)→pinned host→GPU 的端到端吞吐（含 H2D）。
- `safetensors_dram_mirror_host_baseline`：显式申请一块大 DRAM 缓冲，把 payload 拷入 DRAM（不计入测量），再测 **DRAM(userspace)→pinned host bounce buffer** 的吞吐；用于和 `safetensors_hot_host_baseline`（page cache→userspace）互相印证。
- `materialize_d`：Strategy D：用 `loading-meta.json`（plan）把“最终 output layout”在 GPU 上一次性生成后，**落盘为每个 rank 一个连续文件**（以及 meta json）。
- `materialized_disk_baseline`：Strategy D：从 `materialize_d` 生成的文件中读取 `bytes(output)` 并写入 GPU（本质是 `disk_baseline` 的便捷封装，支持 `--disk_io_mode=direct` 做冷读上限）。
- `nccl_baseline`：扫 message size，输出 NCCL `broadcast` 或 `send/recv` 的吞吐。
- `nccl_launch_tax`：测每次 NCCL 调用的 launch 开销（极小 message、很多 iters）。

## 如何运行并拿到结果

所有结果都会以 `LOG(INFO)` 的形式输出到 stdout/stderr。`loader` 模式的关键结果行前缀是：

- `result strategy=...`（包含各阶段耗时、字节计数、planner 统计、pinned/VMM 占用、scheduler waits 等）

建议把输出保存下来便于对比（示例）：

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- --help 2>&1 | tee /tmp/tensorcast_bench.log
```

### Loader：用 JSON “最终状态计划” 驱动（更贴近真实加载）

`--load_plan_json_path=/path/to/plan.json` 会让 `--mode=loader` 按计划文件选择 tensor、切片与复制关系，而不是使用默认的 “row_only” TP 规则。

限制：
- 仅对 `--mode=loader` 生效。
- 不支持 `--enable_collectives`（单进程 NCCL 路径）与 `--run_both_strategies`。

计划文件关键点（示意）：
- 以 checkpoint tensor 名做主索引：`ranks[i].tensors[ckpt_name] = {shape,dtype,copies...}`
- `copies[]` 记录每次 copy 的 `dst_param`，以及 `slices[]`（axis/start/size 列表；空数组表示整 tensor copy）

```json
{
  "version": 2,
  "ranks": [
    {
      "tp_rank": 0,
      "tp_world_size": 2,
      "tensors": {
        "model.embed_tokens.weight": {
          "shape": [152064, 5120],
          "dtype": "torch.bfloat16",
          "copies": [
            { "dst_param": "embed_tokens.weight", "slices": [{ "axis": 0, "start": 0, "size": 76032 }] }
          ]
        }
      }
    }
  ]
}
```

日志字段变化：
- `selection=... tensors copies=...`：`copies` 是本 rank 的 copy 条目数（计划文件中的 copies 总数）。
- `bytes(d2d)`：策略 A 的 “GPU 内 pack”（从 GPU payload buffer 拷到最终输出布局）会产生。对于 axis=1 等会展开成“每行一个小 segment”的切片，策略 A 会将连续等宽段合并为一次 `cudaMemcpy2DAsync` 来减少拷贝调用数；如果最终输出缓冲分配失败（通常是显存不足），基准会跳过该可选步骤并打印 `strategy_a: skipping optional GPU pack ...`。
- `bytes(output)`：本 rank 最终输出布局的总字节数（按 plan 统计；策略 B 通常也等于 `bytes(disk)`）。

### Loader：只跑策略 A

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=loader \
  --strategy=a \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0
```

### Loader：只跑策略 B

策略 B 需要 pinned host buffer（`--use_pinned_host_buffer=true`）。
在多 socket / NUMA 机器上，建议加上：
- `--pinned_numa_node=-2 --pinned_numa_prefault=true`（按 `--device_id` 自动推断 NUMA node，并在 pin 前触页）

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=loader \
  --strategy=b \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --use_pinned_host_buffer=true
```

### Loader：只跑策略 C（批处理 + 去碎片，面向 plan JSON）

策略 C 也需要 pinned host buffer（`--use_pinned_host_buffer=true`），并建议配合 `--load_plan_json_path=...` 使用（它会利用 plan 的全局信息做分组与去重）。
在多 socket / NUMA 机器上，建议加上：
- `--pinned_numa_node=-2 --pinned_numa_prefault=true`（按 `--device_id` 自动推断 NUMA node，并在 pin 前触页）

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=loader \
  --strategy=c \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --use_pinned_host_buffer=true \
  --strategy_c_staging_bytes=$((1024*1024*1024)) \
  --load_plan_json_path=/path/to/loading-meta.json
```

说明：
- `--strategy_c_staging_bytes`：用于“顺序读大块→GPU staging→GPU pack”的 staging 缓冲大小；越大通常越能减少每个大 tensor 的 chunk 次数，但也会占用更多显存。

### Loader：一次跑 A 再跑 B（可选：抽样校验正确性）

`--run_both_strategies` 会在一次运行中先跑 A 再跑 B，并启用 A vs B 的抽样对拍：

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=loader \
  --run_both_strategies \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --tp_world_size=1 \
  --tp_rank=0 \
  --check_correctness_samples=50
```

说明：
- 正确性检查仅对每个 sample 中大小不超过 4 MiB 的 tensor 做字节级比较。
- TP 切片规则是 “row_only”：2D tensor 按行切分，否则按字节等分。

### Loader：策略 A + 单进程多 GPU NCCL collectives

在一个进程里用多张 GPU 走策略 A 的跨 rank NCCL 路径：

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=loader \
  --strategy=a \
  --enable_collectives=true \
  --tp_world_size=2 \
  --tp_devices=0,1 \
  --safetensors_dir=/path/to/shards_dir
```

说明：
- `--enable_collectives` 仅对 `--mode=loader --strategy=a` 生效。
- collectives 开启时不支持 `--run_both_strategies`。

### Safetensors 磁盘基线（读取全部 payload bytes）

该模式测的是“磁盘 → host bounce buffer（pinned/pageable）→ GPU buffer”的端到端吞吐（包含 H2D）。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_disk_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8
```

### Safetensors Host 基线（磁盘 → host bounce buffer，不做 H2D）

该模式测的是“磁盘 → host bounce buffer（pinned/pageable）”吞吐，不包含 GPU 拷贝。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_host_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### Safetensors Hot Host 基线（热读：预热 page cache → disk(buffered)→pinned host）

该模式会先做一次不计时的顺序读取，把模型 payload 尽量放进 OS page cache（前提是内存足够），再对第二次读取计时。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_hot_host_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### Safetensors DRAM Mirror Host 基线（DRAM→pinned host）

该模式分两步：
- 先申请一块大的用户态 DRAM 虚拟内存（`mmap`），并把全部 payload 从 safetensors 读入该 DRAM（**不计入测量**）。
- 再用相同的 pump/buffer pool 设置，把 DRAM 数据拷贝到 pinned host bounce buffer（计时），得到更接近“用户态 memcpy + 框架开销”的上限，用于对照 page cache 热读。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_dram_mirror_host_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### Safetensors O_DIRECT Host 基线（冷读上限：磁盘 O_DIRECT → pinned host）

该模式通过 `O_DIRECT` 绕过 OS page cache，测的是“磁盘（O_DIRECT）→ pinned host bounce buffer”的吞吐，可作为冷读上限参考。

注意：
- 会按 512B 对齐读取每个 `.safetensors` 文件的前缀对齐部分；每个文件末尾不足 512B 的 tail 会被跳过并统计到 `skipped_tail_bytes`（对大模型权重影响可忽略）。
- 需要 `--use_pinned_host_buffer=true`（保证 512B 对齐的 buffer）。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_o_direct_host_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### Safetensors O_DIRECT Disk 基线（冷读：磁盘 O_DIRECT → pinned host → GPU）

该模式在 `safetensors_o_direct_host_baseline` 的基础上增加 H2D，把数据搬到 GPU buffer，用于测“冷读端到端（含 H2D）”吞吐。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_o_direct_disk_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### Safetensors Hot Disk 基线（热读：预热 page cache → disk(buffered)→pinned host→GPU）

该模式会先做一次不计时的顺序读取（热 cache），再对第二次 disk→GPU 的端到端链路计时（包含 H2D）。

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=safetensors_hot_disk_baseline \
  --safetensors_dir=/path/to/shards_dir \
  --device_id=0 \
  --io_threads=4 \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### H2D 基线（pinned host → GPU）

该模式测的是“pinned host → GPU”的纯 H2D 吞吐（不读磁盘；源数据来自 pinned buffer pool）。

说明：
- `--tp_world_size` / `--tp_devices` 用于选择同时参与 H2D 的 GPU 数量与 device id 列表；该模式不依赖 `--device_id`。
- 若你怀疑多 GPU H2D 的上限被 NUMA 影响，可用以下 flags 控制 pinned host pool 的 NUMA 放置：
  - `--pinned_numa_node=-1`：默认（不干预）
  - `--pinned_numa_node=N`：把 pinned host slab 绑定到 NUMA node N（best-effort）
  - `--pinned_numa_node=-2`：自动从 CUDA device 的 `/sys/bus/pci/devices/<bus_id>/numa_node` 推断（TP>1 需要 `--h2d_per_gpu_pinned_pool=true`）
  - `--pinned_numa_prefault=true`：在 `cudaHostRegister` 前触页，避免“pin 时隐式 fault”导致的不可控 NUMA 放置
  - `--h2d_per_gpu_pinned_pool=true`：`h2d_baseline` 为每张 GPU 分配独立 pinned host pool（便于做 NUMA-local 对照）

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=h2d_baseline \
  --tp_world_size=1 \
  --tp_devices=0 \
  --h2d_bench_bytes=$((8*1024*1024*1024)) \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

## Strategy D：预物化（materialize once, load many）

该策略面向“重复加载同一个模型（相同 TP 配置 / 相同计划文件）”的场景：第一次用 Strategy C 生成 output layout 后落盘成连续文件；后续加载可用顺序 I/O 读回，避免 Strategy C 为 `axis=1` 付出的 1.3–2.0× 读放大。

### D.1 生成（materialize）

会在 `--materialized_dir` 下生成：
- `tpN/rankX.bin`：原始二进制 bytes（按 output layout 连续存储）
- `tpN/rankX.meta.json`：包含 `output_bytes` 与 `data_path` 等信息

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=materialize_d \
  --strategy=c \
  --safetensors_dir=/mnt/host0/Qwen2.5-32B-Instruct \
  --load_plan_json_path=/path/to/loading-meta.json \
  --tp_world_size=4 \
  --tp_rank=0 \
  --device_id=0 \
  --use_pinned_host_buffer=true \
  --materialized_dir=/mnt/host0/Qwen2.5-32B-Instruct/tensorcast_materialized
```

### D.2 加载基线（materialized_disk_baseline）

按 meta json 读取对应的 `rankX.bin`，并测 disk→GPU 吞吐；可用 `--disk_io_mode=direct` 做冷读（O_DIRECT）：

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=materialized_disk_baseline \
  --materialized_meta_path=/mnt/host0/Qwen2.5-32B-Instruct/tensorcast_materialized/tp4/rank0.meta.json \
  --device_id=0 \
  --disk_io_mode=direct \
  --use_pinned_host_buffer=true
```

多 GPU 并发示例（测聚合带宽）：

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=h2d_baseline \
  --tp_world_size=4 \
  --tp_devices=0,1,2,3 \
  --h2d_bench_bytes=$((8*1024*1024*1024)) \
  --bbuf_size_kb=262144 \
  --buffer_chunks=8 \
  --use_pinned_host_buffer=true
```

### 磁盘基线（disk_*）

```bash
# 顺序读取大文件
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=disk_baseline \
  --disk_bench_path=/tmp/tensorcast_disk_bench.bin \
  --disk_bench_bytes=$((8*1024*1024*1024)) \
  --disk_io_mode=auto

# 碎片化访问模式
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=disk_fragmentation \
  --disk_bench_path=/tmp/tensorcast_disk_bench.bin \
  --disk_frag_segment_bytes=$((256*1024)) \
  --disk_frag_segments=32768 \
  --disk_frag_stride_bytes=$((4*1024*1024)) \
  --disk_io_mode=auto
```

### GPU Peer 基线（gpu_peer_baseline）

```bash
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=gpu_peer_baseline \
  --gpu_peer_bytes=$((1024*1024*1024))
```

### NCCL 微基准（nccl_*）

```bash
# 吞吐扫参（按 size 扫）
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=nccl_baseline \
  --tp_world_size=2 \
  --tp_devices=0,1 \
  --nccl_op=broadcast \
  --nccl_iters=100 \
  --nccl_warmup=10 \
  --nccl_min_bytes=1024 \
  --nccl_max_bytes=$((1024*1024*1024))

# 启动开销（每次调用的平均开销）
bazel run //core/store/materialization/benchmarks:safetensors_load_strategy_benchmark -- \
  --mode=nccl_launch_tax \
  --tp_world_size=2 \
  --tp_devices=0,1 \
  --nccl_op=broadcast \
  --nccl_overhead_iters=10000
```

## 稳定测量建议

- 对磁盘相关模式，建议控制 page cache（热缓存/冷缓存），并尽量保持 CPU 频率与系统负载稳定。
- 对比策略 A vs B 时，请固定 `--io_threads`、`--bbuf_size_kb`、`--buffer_chunks`、`--gpu_sched_*`，否则结果不可比。

## 影响结果的隐性环境/配置

即使参数（flags）完全一致，下列“隐性”配置/环境也可能显著改变吞吐/延迟结果。

### `loader` / `safetensors_disk_baseline`

- **OS page cache 状态**：热缓存 vs 冷缓存会主导 disk→host 时间；重复跑常测到的是 RAM + copy，而非真实存储。
- **文件系统与挂载参数**：ext4/xfs、`noatime`、DAX、加密、RAID、网络文件系统等。
- **CPU 调度/电源策略**：频率 governor、后台负载、cgroup 限制、NUMA 绑核/绑内存。
- **CUDA 设备可见性/顺序**：`CUDA_VISIBLE_DEVICES` 会改变 `--device_id=0` 对应的物理 GPU，也会改变 PCIe/NVLink 拓扑关系。
- **Pinned 内存行为**：主机 pinned 内存的限制/压力会影响 bounce buffer（中转缓冲）吞吐；策略 B 强制要求 `--use_pinned_host_buffer=true`。
- **GPU 时钟/温控**：persistence mode、application clocks、热降频、MIG 等都会改变 H2D/copy 速率。

### `disk_baseline` / `disk_fragmentation`

- 除了上一节所有对磁盘与 GPU copy 有影响的项之外，还包括：
- **Direct I/O 选择与回退**：`--disk_io_mode=auto|direct|buffered` 与对齐/文件系统支持强相关；`auto` 可能静默回退（输出里会打印 `direct_io_*` 字段）。
- **块设备参数**：内核 I/O scheduler、队列深度、readahead、writeback 等；NVMe/SATA/网络盘差异很大。
- **文件物理布局/碎片**：物理碎片程度、extent 布局会显著影响 `disk_fragmentation` 的结果。

### `gpu_peer_baseline`

- **拓扑与 peer access**：PCIe vs NVLink、NUMA，以及 peer access 是否支持/是否启用（输出 `peer_access=true|false`）。
- **设备映射**：该基准固定 src=GPU0、dst=GPU1；`CUDA_VISIBLE_DEVICES` 可能把它们映射到不同物理 GPU。
- **GPU 时钟/温控**：application clocks 与热降频会影响 memcpy 带宽。

### `nccl_baseline` / `nccl_launch_tax`

- **拓扑与传输**：NVLink vs PCIe；IB/RDMA 是否可用；网卡/接口选择。
- **NCCL 环境变量**：`NCCL_BLOCKING_WAIT`（也可通过 `--nccl_blocking_wait` 设置），以及常见调参项 `NCCL_ALGO`、`NCCL_PROTO`、`NCCL_P2P_LEVEL`、`NCCL_IB_DISABLE`、`NCCL_SOCKET_IFNAME`、`NCCL_TOPO_FILE`、`NCCL_DEBUG`。
- **CUDA 设备可见性/顺序**：`CUDA_VISIBLE_DEVICES` + `--tp_devices` 共同决定 rank↔device 映射。
- **CPU 调度**：NCCL progress 对 CPU 争用敏感；尽量保持后台负载稳定。
