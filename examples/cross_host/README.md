# Cross-Host Benchmark Helpers

这个目录集中放置“多机 put/get benchmark”脚本：

- `cross_host_matrix_runner.py`：统一矩阵 runner（负责重启两侧 daemon、循环 put/get、收集结果、清理 artifact）。
- `cross_host_put_once.py`：单轮 put helper。
- `cross_host_get_once.py`：单轮 get helper（含可见性等待与 comm bytes delta 采样）。
- `cross_host_deregister_once.py`：单轮 deregister helper（`wait=true`，用于验证释放收敛）。

## 1. 前置条件

1. 本机（CPU 节点）启动 Global Store（blocking）。
2. 两台 GPU worker 已通过 `brainctl` 启动，能拿到：
- `put_proc` / `get_proc`（process id）
- 两侧 daemon advertise IP
- 两侧 daemon gRPC 地址（`host:port`）
3. 两侧 worker 都有同一份代码目录（默认 `/data/workspace/tensorcast-280`）和 `.venv`。

## 2. 标准运行方式

示例（forward link，A put -> B get）：

```bash
source .venv/bin/activate

python examples/cross_host/cross_host_matrix_runner.py \
  --case-name compact_fwd_put_to_get_c20b16w16_g0 \
  --conn 20 --buffers 16 --maxw 16 --expected-gpu-channels 0 \
  --size-mib 1024 --warmup 1 --iterations 4 \
  --put-proc <PUT_PROCESS_ID> \
  --get-proc <GET_PROCESS_ID> \
  --put-adv-ip <PUT_WORKER_IP> \
  --get-adv-ip <GET_WORKER_IP> \
  --put-daemon-addr <PUT_WORKER_IP>:62001 \
  --get-daemon-addr <GET_WORKER_IP>:62011 \
  --gs-addr <GS_HOST_IP>:50051 \
  --daemon-config examples/config/store_daemon_config_cross_host_64g_compact.yaml \
  --out-dir /tmp/tc_cross_20260221/results
```

反向链路（reverse link）只需要交换 put/get 两侧参数（`--put-*` 与 `--get-*`）。

## 3. 输出说明

runner 会输出：

- `SUMMARY {...}`：当前 case 的聚合结果。
- `OUTPUT <path>.json`：完整结果文件，包含：
- `summary`：`put_sec_mean` / `e2e_gibps_mean` / `transfer_gibps_mean` / `visibility_wait_sec_p90` 等。
- `records`：逐轮明细（put/get/cleanup）。
- `params`：本次运行关键参数快照。

## 4. 常见问题

1. `no active lease found; proceeding with stateless retire`
- 表示该次 deregister 未命中 active lease，runner 仍会继续本地 retire。

2. daemon startup preflight 失败（内存不足）
- 优先使用 `examples/config/store_daemon_config_cross_host_64g_compact.yaml`，降低 pinned/stable 配置档位。

3. `comm_gpu` 切片不足启动失败
- 调大 `pinned_memory.classes[name=comm_gpu].pool_bytes`，或降低 `conn/buffers/maxw` 组合。
