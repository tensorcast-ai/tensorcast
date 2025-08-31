# RFC-0011 – OpenTelemetry 统一可观测性验收方案（Tracing / Metrics / Logs）

本文档给出一套可执行的验收步骤与命令，覆盖 RFC-0011 目标的端到端验证：
- Tracing：Python Global Store、C++ StoreDaemon、客户端链路；P2P 链路 Span 与 Link
- Metrics：GS Prometheus 暴露与 Collector 汇聚
- Logs：日志注入 trace_id/span_id（Python/C++）
- 回退与性能开关：采样率调控、禁用 SDK

文末附常用命令速查与 CI 任务清单。所有命令均在仓库根目录执行。

---

## 1. 前置条件

- 运行环境
  - Linux 开发机，具备 Bazel、uv（Python 包管理/运行），可用 Docker（用于 Collector）
  - 如无 GPU，Fake CUDA 路径已在 Bazel 侧支持；不影响 Python 端
- 端口占用
  - OTel Collector：OTLP gRPC 4317（或 HTTP 4318）、Prometheus exporter 9464
  - Global Store：gRPC 50051、metrics 8001
  - Daemon：gRPC 8073（示例）
- 基本环境变量（可按需覆盖）
  - `export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317`
  - `export OTEL_TRACES_SAMPLER=parentbased_traceidratio`
  - `export OTEL_TRACES_SAMPLER_ARG=1.0`

提示：仓库已内置最小化 Collector 配置 `tools/otel/collector-dev.yaml`。

---

## 2. 快速验收（一键 Smoke）

使用内置脚本一次性启动 Collector、Global Store、Daemon，并发起客户端调用，生成 spans。

```bash
# 进入仓库根目录
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  uv run tools/otel/e2e_smoke.py --collector --otlp http://127.0.0.1:4317

# 结束后查看输出目录（日志包含 collector/global_store/daemon）
ls build/e2e-smoke-logs
```

验收要点：
- Collector 日志（若开启）出现来自 `tensorcast-global-store`、`tensorcast-store-daemon`、`tensorcast-daemon-smoke` 的 spans
- `build/e2e-smoke-logs/daemon.log` 中日志行包含 `trace_id=<32hex> span_id=<16hex>`

如需手动操作或更细粒度验证，请按下文分阶段逐项验证。

---

## 3. 分阶段详细验收

### Phase 0：Python 端 Tracing PoC（控制台/Collector）

控制台导出冒烟：
```bash
TC_OTEL_CONSOLE_EXPORTER=1 OTEL_SERVICE_NAME=tensorcast-smoke \
  uv run tools/otel_smoke.py --use-aio
```
期望：控制台打印 `Smoke/ManualSpan`、`Smoke/GrpcSyncCall`、`Smoke/GrpcAioCall` 等 span。

Collector 导出冒烟：
```bash
docker run -d --rm --name otelcol-dev --network host \
  -v "$(pwd)/tools/otel/collector-dev.yaml":/etc/otelcol/config.yaml:ro \
  otel/opentelemetry-collector:latest
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  uv run tools/otel_smoke.py
```
期望：`docker logs -f otelcol-dev` 可看到 `resource.service.name=tensorcast-smoke` 的 spans。

收尾：`docker rm -f otelcol-dev`

---

### Phase 1：gRPC 贯通（客户端 → Global Store）

1) 启动 Collector（如未启动）：
```bash
docker run -d --rm --name otelcol-dev --network host \
  -v "$(pwd)/tools/otel/collector-dev.yaml":/etc/otelcol/config.yaml:ro \
  otel/opentelemetry-collector:latest
```

2) 启动 Global Store（导出到 Collector）：
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  uv run -m tensorcast.global_store --port 50051 --metrics-port 8001
```

3) 客户端触发 gRPC（另一终端执行）：
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
OTEL_SERVICE_NAME=tensorcast-gs-smoke \
  uv run tools/gs_smoke_client.py --host 127.0.0.1 --port 50051 --list-workers
```

验收要点：
- Collector 日志出现 client+server spans，`resource.service.name` 分别为 `tensorcast-gs-smoke` 和 `tensorcast-global-store`
- Span 属性含 `rpc.system=grpc`, `rpc.method=HealthCheck`/`ListActiveWorkers`

收尾：停止 Collector（如需要）`docker rm -f otelcol-dev`

---

### Phase 1（续）：gRPC 贯通（客户端 → C++ Daemon）

1) 编译 Daemon（Fake CUDA）：
```bash
bazel build --define use_fake_cuda=true //daemon:tensorcast_daemon
```

2) 启动 Collector（如未启动）：
```bash
docker run -d --rm --name otelcol-dev --network host \
  -v "$(pwd)/tools/otel/collector-dev.yaml":/etc/otelcol/config.yaml:ro \
  otel/opentelemetry-collector:latest
```

3) 启动 Global Store（同上步骤）并保持运行。

4) 启动 Daemon（导出到 Collector，日志 sink 注入 trace/span）：
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  ./bazel-bin/daemon/tensorcast_daemon \
  --listen_addr=0.0.0.0:8073 \
  --global_store_addr=127.0.0.1:50051 \
  --p2p_port=9090
```
可选：将日志写文件，并验证注入字段
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
TC_LOG_SINK_FILE=/tmp/daemon.log \
  ./bazel-bin/daemon/tensorcast_daemon --listen_addr=0.0.0.0:8073 --global_store_addr=127.0.0.1:50051
```

5) 触发一次 Daemon RPC（另一终端执行）：
```bash
uv run python - <<'PY'
import os, grpc
from tensorcast.observability.otel import setup_otel
from tensorcast.proto import store_daemon_pb2 as pb, store_daemon_pb2_grpc as rpc
os.environ.setdefault('OTEL_EXPORTER_OTLP_ENDPOINT','http://127.0.0.1:4317')
setup_otel('tensorcast-daemon-smoke','smoke-client')
ch = grpc.insecure_channel('127.0.0.1:8073')
stub = rpc.StoreDaemonStub(ch)
print(stub.GetWorkerStatus(pb.GetWorkerStatusRequest(), timeout=2.0))
ch.close()
PY
```

验收要点：
- Collector 出现 `StoreDaemon/GetWorkerStatus` server span（`resource.service.name=tensorcast-store-daemon`）
- `/tmp/daemon.log`（如设置）行内包含 `trace_id`/`span_id`，非 “-”

---

### Phase 2：P2P 链路（Span + Link）

运行 P2P 测试（Fake CUDA 环境可用）：
```bash
bazel test //core/store:store_engine_p2p_loader_test \
  --define use_fake_cuda=true \
  --test_env=OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317
```

验收要点：
- Collector 看到 `StoreEngine/P2PIngest` spans；后端如支持 Links，可查到与入口 gRPC server span 的 Link 关系

---

### Phase 3：Metrics 验证

1) 直接检查 GS metrics：
```bash
curl -s http://127.0.0.1:8001/metrics | head
```
2) Collector 暴露（默认 9464）：
```bash
curl -s http://127.0.0.1:9464/metrics | grep -E 'tc_|process_' | head
```

验收要点：
- 出现 `tc_` 系列指标（内存池、加载、P2P 吞吐等）与基础 `process_` 指标

---

### Phase 4：Logs 注入 Trace

Python：
```bash
uv run python - <<'PY'
import logging
from opentelemetry import trace
from tensorcast.observability.otel import setup_otel
setup_otel('tensorcast-log-smoke','dev')
tr=trace.get_tracer(__name__)
with tr.start_as_current_span('Smoke/LogSpan'):
    logging.getLogger().setLevel(logging.INFO)
    logging.info('hello-with-trace')
PY
```
期望：日志中包含 `trace_id`/`span_id` 字段。

C++：
- 参考上文 Phase 1 的 `/tmp/daemon.log` 验证

---

## 4. 负向与回退验收

1) 禁用 SDK：
```bash
OTEL_SDK_DISABLED=1 uv run -m tensorcast.global_store --port 50051 --metrics-port 8001
# 触发 RPC：不再导出 spans，功能正常；日志 trace/span 为 "-"
```

2) 客户端严格模式：
```bash
uv run python - <<'PY'
from tensorcast.torch_util import load_dict
try:
    load_dict(disk_path='x', device_id=0, storage_path='y', wait_for_completion=False)
except Exception as e:
    print('ERR:', e)
PY

# 预期：报错 "OpenTelemetry provider not configured..."

TC_OTEL_CLIENT_AUTO_INIT=1 \
uv run python - <<'PY'
from tensorcast.torch_util import load_dict
try:
    load_dict(disk_path='x', device_id=0, storage_path='y', wait_for_completion=False)
except Exception as e:
    print('ERR (expected from later path):', e)
PY
```
注：第二次运行将不再因 OTel 初始化报错；如参数无效，会在后续业务处报错。

---

## 5. 性能与稳定性开关

- 采样率快速调整：
  - 压测/生产降载：`OTEL_TRACES_SAMPLER=parentbased_traceidratio` + `OTEL_TRACES_SAMPLER_ARG=0.01`（1%）
  - 临时止损：`OTEL_TRACES_SAMPLER_ARG=0`
- 导出协议切换：`OTEL_EXPORTER_OTLP_PROTOCOL=grpc|http/protobuf`；HTTP 端点需使用 `/v1/traces` 路径
- gRPC 明文：`OTEL_EXPORTER_OTLP_INSECURE=true`（仅 gRPC 导出）

---

## 6. 故障排查要点

- 无 span：确认 Collector 在 4317/4318 监听且协议与 SDK 配置一致
- 只有 GS 有 span：确认客户端进程安装并启用 gRPC instrumentation（或设置 `TC_OTEL_CLIENT_AUTO_INIT=1`）
- 跨服务链路未串联：确认通道/拦截器复用、代理未剥离 metadata；C++/Python 传播器均为 W3C tracecontext
- 性能抖动：降低采样率；必要时置 0；C++ 端 API-only 模式开销更低

---

## 7. 常用命令速查（无 Makefile）

以下命令等价于之前的 make 目标，便于快速执行：

- 启动 Collector：
  ```bash
  docker run -d --rm --name otelcol-dev --network host \
    -v "$(pwd)/tools/otel/collector-dev.yaml":/etc/otelcol/config.yaml:ro \
    otel/opentelemetry-collector:latest
  ```
- 停止 Collector：`docker rm -f otelcol-dev`
- Collector 日志：`docker logs -f otelcol-dev`
- Python 控制台冒烟：
  ```bash
  TC_OTEL_CONSOLE_EXPORTER=1 OTEL_SERVICE_NAME=tensorcast-smoke \
    uv run tools/otel_smoke.py --use-aio
  ```
- Global Store 客户端冒烟（要求 GS 已启动）：
  ```bash
  OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 OTEL_SERVICE_NAME=tensorcast-gs-smoke \
    uv run tools/gs_smoke_client.py --host 127.0.0.1 --port 50051 --list-workers
  ```
- 构建 Daemon（Fake CUDA）：`bazel build --define use_fake_cuda=true //daemon:tensorcast_daemon`
- 快速查看 metrics：
  ```bash
  echo "[GS metrics]" && curl -s http://127.0.0.1:8001/metrics | head -n 20
  echo "[Collector metrics]" && curl -s http://127.0.0.1:9464/metrics | head -n 20
  ```

---

## 8. CI 任务清单（Blueprint）

可在 CI 中做轻量验收（Collector 可省略，用 ConsoleSpanExporter 或直接依赖 OTLP 端点）：

1) 安装依赖与构建
```bash
uv sync
bazel build --define use_fake_cuda=true //daemon:tensorcast_daemon
```

2) Python 冒烟（无 Collector，控制台导出）
```bash
TC_OTEL_CONSOLE_EXPORTER=1 OTEL_SERVICE_NAME=ci-otel-smoke \
  uv run tools/otel_smoke.py
```

3) 端到端试跑（如 CI 允许 Docker 与 host 网络）：
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  uv run tools/otel/e2e_smoke.py --collector --otlp http://127.0.0.1:4317
```

4) 基本质量
```bash
uv run ruff check .
uv run mypy ./tensorcast
```

---

## 9. 验收通过标准（汇总）

- Tracing：Collector 中可见 Global Store、Daemon、客户端 spans；P2P 流程存在 `StoreEngine/P2PIngest`（与入口 gRPC Span 通过 Link 关联）
- Propagation：Python ↔ C++ 调用链在后端可串联（或通过 Link 关联）
- Metrics：GS `/metrics` 可直接访问；Collector Prometheus exporter 可抓取 `tc_` 指标
- Logs：Python/C++ 日志行包含有效 `trace_id`/`span_id`；禁用 SDK 时退化为 “-” 不影响功能
- 回退：`OTEL_SDK_DISABLED=1` 能快速止损；客户端严格模式行为符合预期（提示主动初始化或开启 auto-init）

若需要，我可以将上述步骤进一步固化为 CI 工作流文件（如 GitHub Actions）并提交。
