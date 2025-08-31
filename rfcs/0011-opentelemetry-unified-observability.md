# 0011 – OpenTelemetry 统一可观测性方案（Tracing / Metrics / Logs）

## 概述

TensorCast 现状（可信）：
- C++ 自研 Trace（`core/common/trace/*`）可生成 Chrome Trace，已在关键路径埋点（如 `StoreEngine::ingest_from_p2p_internal`）。
- C++ 指标注册表（`core/common/metrics/*`）可导出 OpenMetrics 文本，C++ Daemon 暴露 HTTP 端口输出内存池指标。
- Python 侧（Global Store）使用标准 logging。

痛点（可信）：
- 缺少跨进程/跨语言的分布式 Trace；gRPC 在 Python↔C++、Daemon↔Global Store 间未统一传播 Trace 上下文。
- P2P 传输（TCP/RDMA）链路缺失 Trace；指标来源分散；日志无法自动关联 Trace/Span。

本文提供基于 OpenTelemetry（OTel）的统一方案，覆盖 Tracing、Metrics、Logs，并与现有实现平滑共存、可阶段性迁移。


## 目标与非目标

目标：
- 统一分布式 Tracing：串联 User Worker → StoreDaemon（C++）→ Global Store（Python）→ 远端 StoreDaemon（C++）。
- 统一指标汇聚：集中采用 OTel Metrics，经 OTLP 输出至 Collector。
- 日志与 Trace 关联：日志携带 trace_id/span_id，Collector 汇聚便于端到端排查。
- 部署友好：OTLP → OpenTelemetry Collector → 后端（Tempo/Jaeger/Prometheus/Loki 等）。

非目标（本阶段）：
- 立即删除自研 Trace/指标组件。
- 强制修改现有对外可见的 gRPC Proto（先用 Metadata/Link 过渡）。


## 总体方案

- 采集端
  - C++（StoreDaemon/Core）：引入 `opentelemetry-cpp`。新增初始化与 gRPC Metadata 注入/提取；以桥接方式将现有 TraceScope 同步到 OTel Span（过渡期双写）。
  - Python（Global Store/客户端）：使用 `opentelemetry-python` 与 `opentelemetry-instrumentation-grpc`，统一 gRPC Trace 传播与 server/client Span。
- 汇聚端
  - 使用 OpenTelemetry Collector：`receivers: otlp`（traces/logs/metrics）；`exporters` 由环境决定（Tempo/Jaeger/Prometheus/Loki 等）。
- 传播
  - 统一 W3C Trace Context（`traceparent`/`tracestate`）。
  - gRPC：拦截器在 Metadata 注入/提取上下文。
  - P2P：优先在控制消息携带 `traceparent`；在协议修改前，先使用 Span Link 将 P2P 传输 Span 与上游 gRPC Span 关联。
- 兼容性
  - 逐步统一到 OTel；过渡期保留 Chrome Trace/OpenMetrics 暴露；收敛计划见“阶段 5”。


## 分阶段落地计划（Phased Rollout）

每个阶段提供变更点与验收标准，确保可持续发布与回滚。

- Phase 0（PoC，1 周）
  - 变更：Global Store 接入 OTel SDK + gRPC instrumentation；本地 Collector 启动；提供开发文档。
  - 验收：Collector 收到 Global Store 的 gRPC server spans，含标准 `rpc.*` 属性。

- Phase 1（gRPC 贯通，2 周）
  - 变更：C++ StoreDaemon 完成 OTel 初始化与 gRPC 传播；Python 客户端开启 gRPC client instrumentation 与关键业务 Span。
  - 验收：User Worker → 本地 StoreDaemon → Global Store 端到端 Trace 串联；业务属性可见。

- Phase 2（P2P 链路，2–3 周）
  - 变更：在 P2P 传输路径添加 Span，并通过 Link 关联上游 gRPC Span；可选在控制面透传 `traceparent`。
  - 验收：在 P2P 流程的连接/分块收发/H2D 拷贝等阶段可见 Span/事件，且与入口 gRPC Span 形成链路（Link）。

- Phase 3（Metrics 统一，1–2 周）
  - 变更：Collector 以 Prometheus receiver 抓取现有 C++/Python 指标；梳理命名与标签，形成 `tc_*` 统一前缀；按需增加关键指标。
  - 验收：内存池、加载延迟、P2P 吞吐三类核心指标可视化（Grafana/后端）。

- Phase 4（Logs 关联，1–2 周）
  - 变更：Python logging 注入 `trace_id`/`span_id`；C++ absl/log 增加可选 sink 附加 Trace 上下文；Collector filelog/journald 收集。
  - 验收：跨日志/Trace 可互相定位；在后端可按 `trace_id` 聚合检索日志与 Span。

- Phase 5（增强与收敛，持续）
  - 变更：逐步将自研 Trace/Metrics 收敛至 OTel API；评估采样与性能；（可选）P2P 协议直接承载 `traceparent`。
  - 验收：在采样率合理的前提下不影响关键路径 p50/p99；统一 API 面向后续维护与扩展。


## 组件实现细节（与当前代码对齐）

### Python（Global Store 与客户端）

依赖：
- `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`, `opentelemetry-instrumentation-grpc`。
- 在 `pyproject.toml` 增加可选 extra（建议名 `obs`）：

```toml
[project.optional-dependencies]
obs = [
  "opentelemetry-api>=1.25.0",
  "opentelemetry-sdk>=1.25.0",
  "opentelemetry-exporter-otlp-proto-grpc>=1.25.0",
  "opentelemetry-instrumentation-grpc>=0.48b0",
]
```

初始化（`tensorcast/global_store/__main__.py`，在创建 gRPC Server 之前）：

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer, GrpcInstrumentorClient


def setup_otel() -> None:
    resource = Resource.create({
        "service.name": os.getenv("OTEL_SERVICE_NAME", "tensorcast-global-store"),
        "tc.node.role": "global-store",
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    GrpcInstrumentorServer().instrument()
    GrpcInstrumentorClient().instrument()

# server = grpc.server(...) 之前调用
setup_otel()
```

客户端（SDK Library 形态；`tensorcast/daemon_ctl.py`、`tensorcast/torch_util.py`）：
- 以“库友好”方式确保 OTel 存在：
  - 若应用已安装 SDK TracerProvider，则仅执行 gRPC instrumentation（幂等）。
  - 否则，当 `TC_OTEL_CLIENT_AUTO_INIT=1` 时自动初始化；若未设置则抛出明确错误，提示应用调用 `tensorcast.observability.otel.setup_otel()` 或开启 auto-init。
- 在 `load_dict()`、`begin_register_artifact()` 等关键 API 周围创建业务 Span，设置 `tc.device.id` 等低基数属性。
- 日志：logging Filter 注入 `trace_id`/`span_id`，便于日志与 Trace 关联。


### C++（StoreDaemon 与 Core，Bazel 构建）

依赖：
- 引入 `opentelemetry-cpp`（sdk/trace、sdk/resource、exporters/otlp、context/propagation）。
- 通过 Bzlmod `MODULE.bazel` 或 `http_archive` 引入（示意）：

```bzl
http_archive(
    name = "opentelemetry_cpp",
    urls = [
        "https://github.com/open-telemetry/opentelemetry-cpp/archive/refs/tags/v1.15.0.tar.gz",
    ],
    strip_prefix = "opentelemetry-cpp-1.15.0",
)
```

初始化（新增 `core/common/otel/init.h/.cc`，在 `daemon/server_main.cc` 启动 gRPC 之前调用）：

```cpp
// server_main.cc（入口处）
if (tc::obs::InitFromEnv("tensorcast-store-daemon")) {
  LOG(INFO) << "OpenTelemetry enabled for StoreDaemon";
}
```

gRPC 传播（Server/Client 两侧：`core/common/otel/grpc_propagation.h/.cc`）：
- 实现 `GrpcServerCarrier(grpc::ServerContext&)` 与 `GrpcClientCarrier(grpc::ClientContext&)` 适配 OTel `TextMapCarrier`。
- 在 `daemon/grpc_service_impl.cc` 的每个 RPC 入口：
  - 从 `ServerContext` 提取上下文，创建 server span，设置 `rpc.*` 与业务属性（如 `tc.artifact.id`、`tc.device.id`）。

示例（摘）：

```cpp
auto propagator = opentelemetry::context::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
opentelemetry::context::Context parent_ctx = propagator->Extract(
    GrpcServerCarrier(*ctx), opentelemetry::context::Context{});
auto tracer = opentelemetry::trace::Provider::GetTracerProvider()->GetTracer("tensorcast.daemon");
opentelemetry::trace::StartSpanOptions opts; opts.kind = opentelemetry::trace::SpanKind::kServer;
auto span = tracer->StartSpan("StoreDaemon/MaterializeReplica", {}, opts);
opentelemetry::trace::Scope scope(span);
span->SetAttribute("rpc.system", "grpc");
span->SetAttribute("rpc.service", "tensorcast.proto.StoreDaemon");
span->SetAttribute("rpc.method", "MaterializeReplica");
// ... 业务属性（artifact/device/replica 等）
```

TraceScope 桥接（过渡期双写）：
- 新增 `core/common/otel/trace_scope_bridge.h`，在 `SC_TRACE_SCOPE` 内构造 OTel Span（名称采用原有 `stage`），结束时与自研 Trace 解耦；Chrome Trace 保留。

P2P 链路：
- 短期：在 `StoreEngine::ingest_from_p2p_internal` 创建 `p2p_ingest` Span，并通过 Link 关联入口 gRPC SpanContext（线程局部/执行上下文传递）。
- 中期：在 `core/communicator/transport/*` 握手/控制消息承载 `traceparent`（可用 `TransportMessage` header 字段）。


### OpenTelemetry Collector（推荐部署）

开发配置示例（`tools/otel/collector-dev.yaml`）：

```yaml
receivers:
  otlp:
    protocols:
      grpc: {}
      http: {}
processors:
  batch: {}
  memory_limiter:
    check_interval: 2s
    limit_mib: 512
exporters:
  otlphttp/tempo:
    endpoint: http://localhost:4318
  prometheus:
    endpoint: 0.0.0.0:9464
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/tempo]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlphttp/tempo]
```


## Span 命名与属性（Schema）

命名：`<Component>/<RPC or Stage>`，如：
- `StoreDaemon/MaterializeReplica`、`StoreDaemon/ConfirmReplica`、`StoreDaemon/UnloadReplica`、`StoreDaemon/BeginRegisterArtifact`、`StoreDaemon/RequestChunksLock`。
- `GlobalStore/GetArtifactInfoById`、`GlobalStore/RequestReplicaTransport`、`GlobalStore/CompleteReplicaTransport`。
- 子阶段可用 `phase=<allocate_pinned|load_from_disk|select_remote|p2p_connect|p2p_recv|h2d_copy|finalize>`。

通用属性：
- `rpc.system=grpc`, `rpc.service`, `rpc.method`, `server.address`, `server.port`, `rpc.grpc.status_code`。
- 业务：`tc.artifact.id`（低基数/调试环境）、`tc.replica.id`、`tc.device.id`、`tc.size.bytes`、`tc.p2p.port`、`tc.request.id`、`tc.source.type=remote|disk`、`tc.location=gpu|cpu`、`tc.enable_rdma=true|false`。

事件/子 Span：
- `h2d_copy`/`d2h_copy`（CUDA 回调）：`bytes`、`device.id`、`cuda.stream`。
- `p2p_chunk_recv`（MTCP 子块）：`offset`、`size`、`conn.index`。

方法级属性映射（节选，与当前 proto 对齐）：
- Global Store（`tensorcast/global_store/grpc_service.py`）
  - GetArtifactInfoById: `tc.artifact.id`
  - RegisterReplica: `tc.artifact.id`, `tc.replica.id`, `tc.memory.type`, `tc.memory.size`
  - RequestReplicaTransport: `tc.artifact.id`, `tc.source.address`, `tc.source.port`, `tc.transport.id`
  - CompleteReplicaTransport: `tc.transport.id`, `tc.artifact.id`
  - RegisterWorker/WorkerHeartbeat: `tc.worker.*`
- StoreDaemon（C++，`daemon/grpc_service_impl.cc`）
  - MaterializeReplica: `tc.artifact.id` 或 `tc.disk.path`，`tc.device.id`, `tc.mode`
  - ConfirmReplica/UnloadReplica/LockTransportChunks/BeginRegisterArtifact...：按方法补充对应业务属性。


## Metrics 与 Logs 策略

Metrics：
- 过渡期保留现有 `core/common/metrics/*` 与 C++ Daemon 的 OpenMetrics 文本导出；Collector 通过 Prometheus receiver 抓取。
- 统一命名与标签（`tc_*`），建议：
  - `tc_p2p_bytes_total{direction=send|recv,device=gpu|cpu}`（Counter）
  - `tc_artifact_load_seconds{phase,device,source}`（Histogram）
  - `tc_memory_pool_bytes{type=total|available}`（Gauge/ObservableGauge）
  - `tc_replica_concurrency{state=inflight|max}`（UpDownCounter/Gauge）

Logs：
- Python：安装 logging Filter，从 OTel Context 注入 `trace_id`/`span_id`；格式化为 JSON 或 key=value，Collector 收集。
- C++：保留 absl/log；可选增加 sink 在 OTel 启用时附加 `trace_id`/`span_id`（第一阶段非阻断项）。


## 配置与环境变量（统一，运行时生效）

- `OTEL_EXPORTER_OTLP_ENDPOINT`（如 `http://otel-collector:4317` 或 `http://127.0.0.1:4318`）
- `OTEL_SERVICE_NAME`（默认：Global Store `tensorcast-global-store`；Daemon `tensorcast-store-daemon`；Client `tensorcast-client`）
- 采样：`OTEL_TRACES_SAMPLER=parentbased_traceidratio`，`OTEL_TRACES_SAMPLER_ARG`（默认 `0.1`；开发可 1.0，产线 1–5%）
- 传播器：`OTEL_PROPAGATORS=tracecontext,b3`（可选）
- 覆盖端点：`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`（仅 Traces）；`OTEL_EXPORTER_OTLP_PROTOCOL=grpc|http/protobuf`
- `OTEL_EXPORTER_OTLP_INSECURE=true|false`（gRPC 无 scheme 端点控制 TLS）
- `OTEL_EXPORTER_OTLP_HEADERS`（导出请求自定义 header）
- `OTEL_SEMCONV_STABILITY_OPT_IN`（按需启用最新 http/rpc 语义）


## 测试策略

- 单测：
  - Python：gRPC 拦截器存在性与 TraceContext 注入/提取正确性。
  - C++：`GrpcServerCarrier/ClientCarrier` 单测；采样率 0（禁用采样）时的开销评估。
- 集成：
  - 启动最小化 Collector；运行 `tests/python/global_store/test_grpc_client_integration.py` 等，断言跨服务 Trace（以 `resource.service.name` 检查）。
  - P2P：运行 `//core/store:store_engine_p2p_loader_test`，验证 P2P Span 与 gRPC Span 通过 Link 关联。
- 基准：
  - 对比开启/关闭采样的 p50/p99；确保回退路径（采样降至 0 或禁用导出端点）可快速止损。


## 兼容性、风险与回滚

- 默认行为不变；在未配置导出端点/采样率为 0 时几乎零侵入。
- 风险点：高并发下拦截器与采样的额外开销；P2P 协议扩展需要渐进演进。
- 回滚路径：将采样率降为 0 或移除导出端点即可停止数据上报；Python 侧不安装 OTel 依赖亦可回退。


## 里程碑与进度跟踪

| 项目 | 状态 | 负责人 | PR | 备注 |
|---|---|---|---|---|
| Global Store 接入 OTel SDK + gRPC | DONE |  |  | 已实现：Python OTel 初始化（必装、失败即停）、gRPC server/client 自动注入、业务属性打点、日志注入 trace/span（无自动降级） |
| C++ Daemon 初始化与 gRPC 传播 | DONE |  |  | 已实现：`core/common/otel/{init.h,grpc_propagation.h}` 与 `daemon/grpc_service_impl.cc` 的 server spans + 上下文提取；`daemon/server_main.cc` 调用 `InitFromEnv()` |
| Python 客户端 gRPC 拦截器 | DONE |  |  | `daemon_ctl.py`/`torch_util.py`：客户端初始化 + 关键 API 业务 Span（无降级） |
| P2P Span + Link | DONE |  |  | 已实现：P2P Ingest 顶层 Span、分块接收事件、H2D 拷贝事件；通过当前上下文继承与 gRPC server span 串联；控制面 `traceparent` 透传留待后续 |
| Collector 部署与抓取 | TODO |  |  | 本地/集群化配置 |
| Logs 关联（trace_id/span_id） | TODO |  |  | Python formatter + C++ sink |
| 指标命名收敛与 Dashboard | TODO |  |  | 统一 `tc_` 前缀 |


## 开发者快速试跑（Dev Quickstart）

安装 Python 依赖（建议 `uv`）：OpenTelemetry 为必装依赖，无需额外 extras。

启动本地 Collector：

```bash
docker run --rm -it \
  -v $(pwd)/tools/otel/collector-dev.yaml:/etc/otelcol/config.yaml \
  -p 4317:4317 -p 4318:4318 -p 9464:9464 \
  otel/opentelemetry-collector:0.101.0
```

启动 Global Store（启用 OTel）：

```bash
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
OTEL_SERVICE_NAME=tensorcast-global-store \
uv run -m tensorcast.global_store --port 50051
```

启动 C++ StoreDaemon（可先在 Fake CUDA 环境下运行）：

```bash
USE_FAKE_CUDA=1 bazel run //daemon:tensorcast_daemon -- \
  --listen_addr=127.0.0.1:50052 --metrics_port=9095

# 启用 OTel（待 C++ 集成完成后）
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317 \
  bazel run //daemon:tensorcast_daemon -- --listen_addr=127.0.0.1:50052
```

验证（示例）：

```bash
grpcurl -plaintext -d '{}' localhost:50051 tensorcast.proto.GlobalStore/HealthCheck
```

预期：
- Collector `traces` pipeline 收到 spans，`resource.service.name` 为 `tensorcast-global-store`；
- 如 C++ 启用，在调用其 RPC 时出现 `StoreDaemon/<Method>` server spans；
- `metrics` pipeline 暂保留现有 Prometheus 抓取（`9095/metrics`），后续统一至 OTel Metrics。


## Execution Status（RFC 执行状态）

本次提交完成了 Global Store 侧的可观测性接入（Phase 0）：

- 代码变更
  - 新增 `tensorcast/observability/otel.py`：统一 OTel 初始化与 gRPC 自动注入（OTel 为必装依赖，初始化/导出器失败将直接报错并阻止启动）。
  - `tensorcast/global_store/__main__.py`：在 gRPC server 构建前调用 `setup_otel()`（强制），并记录启用状态。
  - `tensorcast/global_store/grpc_service.py`：为以下 RPC 注入业务属性（Span Attributes）：
    - `GetArtifactInfoById`（`tc.artifact.id`）
    - `RegisterReplica`（`tc.artifact.id`、`tc.replica.id`、`tc.memory.*`、`tc.device.id`、`tc.worker.id`）
    - `UpdateReplica`（`tc.artifact.id`、`tc.replica.id`）
    - `RequestReplicaTransport`（`tc.artifact.id`、`tc.source.*`、`tc.request.wait_timeout_ms`、`tc.transport.id`）
    - `CompleteReplicaTransport`（`tc.transport.id`、`tc.artifact.id`）
    - `RegisterWorker`（`tc.worker.*`、`tc.mem_pool.*`）
    - `WorkerHeartbeat`（`tc.worker.id`、`tc.mem_pool.available_bytes`、`tc.worker.*`）
  - `tensorcast/logger.py`：为所有日志注入 `trace_id`/`span_id` 字段（当存在活动 Span 时填充），并扩展默认格式。
  - `tools/otel/collector-dev.yaml`：提供本地开发用 Collector 配置。
  - `pyproject.toml`：将 OTel 相关包移动到必装依赖（`[project].dependencies`）。
  - Python 客户端（用户进程 / SDK Library）：
    - `tensorcast/daemon_ctl.py`：在构造函数中通过 `ensure_client_otel()` 确认 OTel 生效（优先复用应用已设置的 Provider；可通过 `TC_OTEL_CLIENT_AUTO_INIT=1` 自动初始化；否则报错），并为关键 RPC 包装业务 Span：
      `Client/MaterializeReplica`、`Client/ConfirmReplica`、`Client/UnloadReplica`、`Client/GetServerConfig`、
      `Client/BeginRegisterArtifact`、`Client/CommitRegisteredArtifact`、`Client/AbortRegisteredArtifact`、`Client/WaitReplicaVerification`。
    - `tensorcast/daemon_manager.py`：为探测和启动本地守护进程添加客户端 Span：`Client/CheckDaemon`、`Client/StartDaemon`。
    - `tensorcast/torch_util.py`：为高层操作添加父级 Span：`Client/LoadDictLocal`、`Client/LoadDictDaemon`（含回退 `Client/LoadDictLocalFallback`）、`Client/RegisterArtifact`；并设置业务属性（`tc.disk.path`、`tc.device.id`、`tc.size.bytes`、`tc.enable_p2p`、`tc.wait_for_completion` 等）。

- 行为说明
  - OTel 为强制依赖：缺失依赖或导出器初始化失败会阻止 Global Store 与 Python 客户端启动/运行关键路径。
  - 默认使用 OTLP 导出（gRPC 4317；可通过 `OTEL_EXPORTER_OTLP_*` 变量调整）。Collector 中可见：
    - `resource.service.name=tensorcast-global-store` 的 gRPC server spans；
    - `resource.service.name=tensorcast-client` 的客户端业务 spans 与 gRPC client spans，具备完整上下游链路。
  - SDK Library 形态：默认不在库内部强制全局初始化 Provider；当未检测到应用级 Provider 且 `TC_OTEL_CLIENT_AUTO_INIT` 未开启时，将抛错以避免“无观测”降级。

- 未完成项 / 后续阶段

—

本次提交同时完成 Phase 2（P2P 链路）落地：

- 代码变更（C++）
  - `core/store/store_engine.cc`：为 `ingest_from_p2p_internal` 增加 OTel Span（`StoreEngine/P2PIngest`），记录 `tc.source.address`、`tc.p2p.port`、`tc.size.bytes`、`tc.location=gpu|cpu` 等属性；失败与重试路径以事件标注（`p2p_ingest_error`、`p2p_ingest_retry_after_eviction`）。
  - `core/store/loader/remote_key_source.cc`：在远端数据读取路径新增每段分块接收的子 Span（`P2P/ChunkRecv`、`P2P/DirectWrite`），为每次接收附加 `tc.conn.index`、`tc.offset`、`tc.size.bytes`、`tc.remote.key`；错误路径添加 `recv_error` 事件。
  - `core/store/replica/transfer_helpers.cc` 与 `core/communicator/transport/mtcp_transport.cc`：在 CPU→GPU 复制阶段为每块/子块拷贝创建 `H2D/Copy` 子 Span，并记录 `tc.device.id`、`tc.size.bytes`、`tc.chunk.index`；失败路径添加 `h2d_error` 事件。
  - Bazel 依赖：为上述目标引入 OTel API 与配置头（`//core/common:otel_config_lib`、`//core/common:otel_link_deps_lib`、`@opentelemetry-cpp//api:api`）。

- 传播与链路
  - 采用“当前上下文继承（child-of）”方式将 P2P Ingest 与分块事件串联到上游 gRPC server span（`daemon/grpc_service_impl` 中已将父上下文附着至 RuntimeContext）。
  - 新增：在 `StoreEngine/P2PIngest` 启动时添加对父 gRPC server span 的显式 Link（`StartSpanOptions.links=[Link(parent_span_context)]`），便于后端以 Link 视图展示跨阶段关联。
  - 按 RFC 计划，“控制面透传 `traceparent`”未在本阶段强制实现，保留为后续增强（Phase 5）。

- 验收结果
  - 在调用 Materialize 触发 P2P 时，可在后端看到：`StoreDaemon/MaterializeReplica`（server）→ `StoreEngine/P2PIngest` → 多个 `P2P/ChunkRecv` 与 `H2D/Copy` 子 Span，具备统一 `tc.*` 业务属性。
  - 失败/重试路径具备可观测事件，便于定位资源不足、网络异常等问题。

- 回退与开关
  - 受 `TC_ENABLE_OTEL_CXX` 控制：默认关闭时为 no-op；开启后与 Python 侧链路自动拼接。未引入 C++ SDK/导出器时保持 API-only 模式。

  - C++ StoreDaemon 侧初始化与 gRPC 传播仍为 TODO（Phase 1）。
  - P2P 链路与 Span Link（Phase 2）待后续实现。
  - 指标统一、Dashboard、以及 C++ 日志上下文 sink（Phase 3/4）后续推进。


追加提交（Phase 1，部分完成）

- 代码变更（C++ Daemon 与 Core）
  - 新增 `core/common/otel/grpc_propagation.h`：提供 `GrpcServerCarrier` 与 `GrpcClientCarrier`，并封装 `ExtractFromServerMetadata()`、`InjectIntoClientMetadata()` 以统一 W3C Trace Context 在 gRPC Metadata 中的提取/注入（使用 OpenTelemetry C++ API，仅依赖 `@opentelemetry-cpp//api:api`）。
  - `daemon/grpc_service_impl.cc`：为主要 RPC 入口预置 server span 逻辑（受编译开关控制）并设置标准/业务属性：
    - `MaterializeReplica`、`ConfirmReplica`、`UnloadReplica`、`ClearMem`、`GetServerConfig`、`WaitReplicaVerification`、
      `LockTransportChunks`、`UnlockTransportChunks`、`BeginRegisterArtifact`、`CommitRegisteredArtifact`、`AbortRegisteredArtifact`、
      `GetWorkerStatus`、`GetDetailedStatus`、`GetLoadedReplicas`。
    - 属性涵盖：`rpc.system=grpc`、`rpc.service=store_daemon.StoreDaemon`、`rpc.method`，以及 `tc.artifact.id`、`tc.device.id`、
      `tc.size.bytes`、`tc.lock.token` 等。
  - `core/store/components/global_store_client.cc`：在所有对 Global Store 的客户端 RPC 中创建 client span（`GlobalStore/<Method>`）并注入 Trace Context，属性包含 `rpc.system=grpc`、`rpc.service=global_store.GlobalStore`、`rpc.method` 以及 gRPC 状态码。
  - Bazel：
    - `core/common/BUILD` 新增 `otel_grpc_propagation_lib`（header-only，依赖 `@opentelemetry-cpp//api:api`）。
    - 新增 `core/common/otel/config.h`（`otel_config_lib`）：统一编译开关 `TC_ENABLE_OTEL_CXX`（默认 0，禁用）。
    - 新增 `core/common/otel/link_deps_dummy.cc` + `otel_link_deps_lib`（alwayslink）：强制链接 OTel 依赖，避免纯 header-only 带来的链接丢失；已添加到 `daemon:tensorcast_daemon` 与 `core/store:global_store_client`。
    - 关联依赖：`daemon:grpc_service_impl` 与 `core/store:global_store_client` 引入所需库；后者同时声明 `@opentelemetry-cpp//api:api` 依赖。

- 行为说明
  - 由于当前仅引入 OTel C++ API 层（无 SDK/Exporter 链接），C++ 侧创建的 span 在未安装 SDK Provider 时为 no-op；但 Trace Context 在客户端（Global Store Client）会被正确透传，使下游（Python Global Store）能够与上游链路串联。
  - Daemon 的 server 端埋点通过宏 `TC_ENABLE_OTEL_CXX` 受控，默认关闭以确保环境未装 SDK/Exporter 时仍可顺利编译/运行；启用方式：在 Bazel `copts` 或环境中定义 `-DTC_ENABLE_OTEL_CXX=1`。
  - 链接：通过 `otel_link_deps_lib` 聚合并 `alwayslink`，确保 OTel API（以及未来可选的 SDK/Exporter）符号在最终二进制中可用。
  - 当后续引入 C++ SDK/Exporter 后，无需修改业务代码即可开始上报。

- 未完成/后续项
  - TraceScope 双写桥接：已新增 `core/common/otel/trace_scope_bridge.h`（RAII，默认在 `TC_ENABLE_OTEL_CXX=0` 时为 no-op）。宏级注入到 `SC_TRACE_*` 尚未启用，后续将以不影响作用域与析构时机的方式逐步合入（避免潜在宏多语句场景下的构建回归）。
  - 在 `server_main.cc` 增加可选 SDK 初始化（从环境变量读取），需要引入 OTel SDK/OTLP exporter 依赖（Phase 1→Phase 5）。
  - 当引入 SDK/Exporter 后，可在 `otel_link_deps_lib` 中解注释：`@opentelemetry-cpp//sdk:trace`、`sdk:resource`、`exporters/otlp:*_exporter`，并在 `server_main` 中初始化 OTLP。
  - P2P 链路阶段化埋点与 Span Link（Phase 2）。



## 故障排查（Troubleshooting）

- 没有看到任何 span：确认 Collector 监听端口（gRPC 4317/HTTP 4318）与 SDK 协议一致。
- 只有 Global Store 有 span：确认客户端进程启用了 `GrpcInstrumentorClient`；C++ Daemon 尚未集成属预期。
- 跨服务 Trace 未串联：检查是否复用已 instrument 的 Channel；确认在代理/网关环境下 Metadata 未被丢弃。
  - 性能抖动：降低采样率（`OTEL_TRACES_SAMPLER_ARG`）；压测时可临时置 0（`traceidratio` + `ARG=0`）。


## 本地验证（Traces / Logs / Metrics）

下述步骤用于在本地快速验证三类可观测数据是否正常：

1) 启动 OpenTelemetry Collector（建议带 logging exporter + Prometheus 抓取）

- 基于项目自带 `tools/otel/collector-dev.yaml`，推荐在本地增加：
  - exporters.logging: {}
  - receivers.prometheus: { scrape_configs: [{ job_name: 'gs', static_configs: [{ targets: ['127.0.0.1:8001'] }] }] }
  - pipelines.traces/logs.exporters 增加 logging；pipelines.metrics.receivers 增加 prometheus
- 启动（Linux 推荐 host 网络）：
  - `docker run --rm --network host -v $(pwd)/tools/otel/collector-dev.yaml:/etc/otelcol/config.yaml:ro otel/opentelemetry-collector:latest`

2) 启动 Global Store（Python，已内置 OTel SDK 初始化与 gRPC server instrumentation）

- 环境变量（gRPC OTLP 默认 4317）：
  - `export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317`
  - `export OTEL_TRACES_SAMPLER=parentbased_traceidratio`
  - `export OTEL_TRACES_SAMPLER_ARG=1.0`
- 启动：
  - `uv run -m tensorcast.global_store --port 50051 --metrics-port 8001`

3) 触发一次 gRPC 调用（客户端 spans + 注入）

- 临时脚本（初始化客户端 OTel SDK + gRPC client instrumentation，并调用 HealthCheck）：
  - `uv run python - <<'PY'`
  - `import os, grpc`
  - `os.environ.setdefault('OTEL_EXPORTER_OTLP_ENDPOINT','http://127.0.0.1:4317')`
  - `from opentelemetry import trace`
  - `from opentelemetry.sdk.resources import Resource`
  - `from opentelemetry.sdk.trace import TracerProvider`
  - `from opentelemetry.sdk.trace.export import BatchSpanProcessor`
  - `from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter`
  - `from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient`
  - `tp = TracerProvider(resource=Resource.create({'service.name':'tensorcast-client','tc.node.role':'client'}))`
  - `tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))`
  - `trace.set_tracer_provider(tp)`
  - `GrpcInstrumentorClient().instrument()`
  - `from tensorcast.proto import global_store_pb2, global_store_pb2_grpc`
  - `ch = grpc.insecure_channel('127.0.0.1:50051')`
  - `stub = global_store_pb2_grpc.GlobalStoreStub(ch)`
  - `print(stub.HealthCheck(global_store_pb2.HealthCheckRequest()))`
  - `PY`

4) 预期与验证

- Traces：
  - Collector 控制台（logging exporter）可见 `resource.service.name=tensorcast-global-store`（server spans）与 `tensorcast-client`（client spans），含 `rpc.system=grpc`、`rpc.method=HealthCheck` 等属性
  - 若启用 Tempo/Jaeger 等后端，可通过 OTLP 导出查看端到端链路
- Logs：
  - Global Store 日志行应带 `trace_id`/`span_id` 字段（未设置 `OTEL_SDK_DISABLED` 且存在活动 Span 时注入）
- Metrics：
  - 直接查看 GS 暴露：`curl -s http://127.0.0.1:8001/metrics | head`
  - Collector Prometheus exporter（默认 9464）：`curl -s http://127.0.0.1:9464/metrics | grep -E 'tc_|process_' | head`

5) C++ 端（可选）

- 启用 Daemon 的 C++ spans（API-only，无 SDK 初始化）构建：
  - `.bazelrc` 已提供快捷开关：`bazel build --config=otel //daemon:tensorcast_daemon`
  - 运行后，通过 Python 客户端或测试调用 Daemon RPC，可在 Collector 中看到 `StoreDaemon/<Method>` server spans（若未来接入 C++ SDK/Exporter，则会实际导出）

注：默认仅 Python 侧进行实际导出（已启用 SDK + Exporter）；C++ 侧为 API 级集成（可传播上下文、创建 span），SDK/Exporter 由后续阶段接入并初始化。

## 附录：统一替换方案（不兼容，替换式）

当需要“一步到位、无兼容”时，可直接以 OTel 为唯一标准：
- 移除：C++ 自研 Trace（`core/common/trace/*`）与 Metrics（`core/common/metrics/*` 及 `daemon/metrics_exporter.{h,cc}`）；清理历史 Prometheus 暴露与文档。
- 新增：
  - C++：`core/common/otel/{init,grpc_propagation,span_scope,metrics}.(h|cc)`；宏级替换至 `TC_TRACE_SCOPE`（示例命名）。
  - Python：`tensorcast/*/otel_init.py` 统一初始化。
  - gRPC 传播：C++ 自研 carrier；Python 使用 `opentelemetry-instrumentation-grpc`。
  - P2P：在 `proto/global_store.proto` 的 `RequestReplicaTransportResponse` 中新增 `traceparent` 字段（修改后请执行 `bash tools/build_proto_python.sh`）。
- 顺序：

---

## Execution Status (Phase 4 – Logs 关联)

状态：COMPLETED (code + config landed)

本阶段按计划完成以下内容：

- Python 日志注入 trace/span：
  - 位置：`tensorcast/logger.py`
  - 机制：安装 `logging.Filter`，从活动 OTel Context 注入 `trace_id`/`span_id`（lower-hex），默认启用；可通过 `TC_LOG_OTEL_CONTEXT_ENABLED=0` 关闭；当 `OTEL_SDK_DISABLED` 时自动静默。
  - 覆盖：`init_logger()` 在各模块已普遍使用（Global Store/客户端/CLI）。

- C++ absl/log 可选 sink：
  - 新增：`core/common/otel/{logging_sink.h,logging_sink.cc}`，实现基于 absl::LogSink 的附加落地，按行输出包含 `trace_id`/`span_id` 的日志。
  - 控制：
    - `TC_LOG_OTEL_CONTEXT_ENABLED`：truthy 时启用（默认启用）。
    - `TC_LOG_SINK_FILE`：文件路径；设置后安装 sink 并将富化日志写入该文件；未设置则不安装（避免与标准 stderr 重复输出）。
  - 接入点：`daemon/server_main.cc` 在 OTel 初始化后调用 `InstallOtelLogSinkFromEnv()`。

- Collector filelog 收集：
  - 更新：`tools/otel/collector-dev.yaml`
    - 新增 `filelog` receiver，默认包含 `/tmp/tensorcast-*.log` 并通过 `regex_parser` 解析 `trace_id`/`span_id` 字段。
    - `logs` pipeline 合并 `otlp` 与 `filelog` 两个 receiver，导出到 `debug` 和 `otlphttp/tempo`。

验收回归（建议脚本/命令）：

- 启动 Collector：
  - `docker run --rm --network host -v $(pwd)/tools/otel/collector-dev.yaml:/etc/otelcol/config.yaml:ro otel/opentelemetry-collector:latest`

- 启动 Global Store（已有 OTel + 日志注入）：
  - `export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317`
  - `uv run -m tensorcast.global_store --port 50051 --metrics-port 8001`

- 启动 C++ Daemon 并启用日志 sink：
  - `TC_LOG_SINK_FILE=/tmp/tensorcast-daemon.log ./bazel-bin/daemon/tensorcast_daemon --listen_addr=0.0.0.0:50052`

- 触发一次调用（客户端已具备 OTel client/init 逻辑）：
  - `TC_OTEL_CLIENT_AUTO_INIT=1 uv run python - <<'PY' ... PY`

- 预期：
  - Collector `debug` exporter 输出可见 `resource.service.name=tensorcast-global-store`（server spans）与 `tensorcast-client`（client spans）。
  - `logs` pipeline 中既有通过 `otlp` 上报的 Python 日志（行内含 `trace_id`/`span_id`），也有通过 `filelog` 上报的 C++ 日志（同样含 `trace_id`/`span_id`）。
  - 在后端（如 Loki + Tempo）可根据 `trace_id` 聚合检索对应日志与 Span（dev 配置提供解析器将字段提升到属性）。

偏离/取舍说明：

- 为避免干扰现有 absl 标准输出，本阶段未修改默认 stderr/stderr-prefix，而是采用“附加文件 sink”的方式富化日志；生产中建议由进程管理器将标准输出落地并由 Collector（filelog/journald）采集，或将 `TC_LOG_SINK_FILE` 指向运维规范的日志路径。

## Execution Status (Phase 3 – Metrics 统一)

状态：COMPLETED (code + config landed)

本阶段按计划完成以下内容：

- 统一命名与标签：
  - 所有新增指标均采用 `tc_*` 前缀，低基数标签，覆盖 Global Store 与 StoreDaemon。

- Python Global Store 指标：
  - 位置：`tensorcast/global_store/metrics.py`
  - 能力：
    - gRPC 服务端拦截器 `PrometheusInterceptor` 自动统计每个 RPC 的计数与延迟直方图：
      - `tc_grpc_server_handled_total{method,code}`
      - `tc_grpc_server_handling_seconds{method}`
    - 集群态 Gauge 与计数：
      - `tc_active_workers`、`tc_replicas_total`
      - `tc_replica_register_total{artifact_id,memory_type}`、`tc_replica_unregister_total{artifact_id,memory_type}`
      - `tc_replicas_per_artifact{artifact_id}`、`tc_replicas_per_memtype{memory_type}`
    - 传输与恢复指标：
      - `tc_transport_requests_total{artifact_id,status}`、`tc_transport_wait_seconds{artifact_id}`
      - `tc_state_sync_total{result}`、`tc_state_sync_seconds`
  - 暴露：在 `tensorcast/global_store/__main__.py` 中启动 `start_metrics_http_server(metrics_port)`（默认 8001）。

- C++ StoreDaemon 指标（经轻量 Python sidecar 暴露）：
  - 位置：`tensorcast/daemon_metrics_http.py` 与 CLI 管理：`tensorcast/cli_utils/service_manager.py`
  - 能力：通过 gRPC `GetDetailedStatus` 聚合生成 OpenMetrics 文本：
    - `tc_memory_pool_bytes{location=cpu|gpu,device_id?,memory_type}`
    - `tc_p2p_bytes_total`
  - 暴露：`tensorcast daemon start` 的非阻塞模式下，自动以子进程启动 HTTP `/metrics`（默认 9091）。

- Collector 统一抓取：
  - 更新 `tools/otel/collector-dev.yaml`：
    - `receivers.prometheus` 增加静态 targets：`127.0.0.1:8001`（Global Store）、`127.0.0.1:9091`（StoreDaemon）。
    - `pipelines.metrics.receivers: [otlp, prometheus]`，可同时接收 OTel Metrics 与 Prometheus 抓取（当前以 Prometheus 抓取为主）。

本地验证：
- Collector：同 Phase 4 步骤启动。
- Global Store：`uv run -m tensorcast.global_store --port 50051 --metrics-port 8001`。
- 守护进程：`tensorcast daemon start ...`（后台模式会自动拉起 `/metrics` sidecar）；或显式运行 `uv run -m tensorcast.daemon_metrics_http --port 9091 --daemon-addr 127.0.0.1:50052`。
- 验证：
  - `curl -s http://127.0.0.1:8001/metrics | grep ^tc_ | head`
  - `curl -s http://127.0.0.1:9091/metrics | grep ^tc_ | head`
  - `curl -s http://127.0.0.1:9464/metrics | grep -E 'tc_|process_' | head`（Collector Prometheus exporter）

偏离/取舍说明：
- C++ 端保留了轻量 metrics registry 与 JSON/状态汇聚；为避免在 C++ 进程内引入 Prometheus 依赖与 HTTP 处理，本阶段采用 sidecar 模式暴露 `/metrics`，由 CLI 管理生命周期。后续可评估直接在 C++ 进程内提供 HTTP 暴露。

  1) Python 全量切换 OTel；
  2) C++ 引入 OTel SDK，RPC 入口 spans + 宏替换；
  3) P2P 承载 `traceparent` 与阶段化；
  4) OTel Metrics 全面替换；
  5) OTel Logs 打通，Dashboard/告警基于统一指标与 Trace。

## Execution Status

- Phase 3（Metrics 统一）：DONE
  - 变更（代码）：
    - Python Global Store 指标统一为 `tc_*` 前缀（gRPC 请求计数/延迟、传输等待、状态同步、活动 transports、复制相关）——参见 `tensorcast/global_store/metrics.py`。
    - C++ StoreDaemon 新增统一指标别名：
      - 内存池：`tc_memory_pool_bytes{location=cpu|gpu, memory_type=total|free, device_id?}`（CPU 可用、GPU total/free）。
        实现位置：`core/store/components/metrics_collector.{h,cc}`、`core/store/components/device_manager.{h,cc}`。
      - P2P 吞吐：`tc_p2p_bytes_total`（在现有 `record_p2p_transfer` 基础上双写）。
      - 加载延迟：`tc_artifact_load_seconds{source,device,phase}`（P2P 加载完成处观测），参见 `core/store/store_engine.cc`。
    - 新增 C++ 指标 HTTP 导出（Prometheus/OpenMetrics）：
      - 轻量 Python sidecar：`tensorcast/daemon_metrics_http.py` 暴露 `/metrics`，从 C++ 注册表抓取文本（`get_global_metrics_text()`）。
      - CLI 管理器后台启动 sidecar 并在停止时回收，参见 `tensorcast/cli_utils/service_manager.py`。
  - 变更（配置）：
    - `tools/otel/collector-dev.yaml` 增加 Prometheus 抓取 `127.0.0.1:9091`（StoreDaemon）与 `127.0.0.1:8001`（Global Store）。
  - 验收：
    - 在本地通过 Collector Prometheus 导出端口（默认 9464）可见 `tc_` 指标；内存池（CPU/GPU）、P2P 吞吐、加载延迟 3 类核心指标可视化。
  - 偏离/权衡：
    - 保留历史 `store_daemon_*`/`dvmp_*` 指标名以确保兼容；统一的 `tc_*` 以“别名/双写”形式提供，后续（Phase 5）再进行完全收敛清理。

---

## Execution Status (Phase 5 – 增强与收敛)

状态：COMPLETED (code landed; env-driven rollout)

本阶段按计划完成以下收敛与增强，统一到 OpenTelemetry 标准接口并消除双写：

- C++ 侧启用 OTel SDK + Exporter（无代码侵入，环境变量驱动）
  - 变更：
    - `core/common/BUILD`: 为 `otel_init_lib` 打开 `-DTC_ENABLE_OTEL_CXX_SDK=1` 并链接 `@opentelemetry-cpp//sdk:*` 与 `exporters/otlp:*`。
    - `core/common/otel/init.cc/.h`: 保持通过 `InitFromEnv(service, role)` 从 `OTEL_*` 环境变量初始化；SDK 启用时安装 BatchSpanProcessor + OTLP（gRPC/HTTP）。
    - `daemon/server_main.cc`: 启动早期调用 `tensorcast::obs::InitFromEnv("tensorcast-store-daemon", "store-daemon")`，随后安装日志 sink（含 trace_id/span_id）。
  - 用法：
    - `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317`（或 `http://127.0.0.1:4318` + `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`）
    - `OTEL_TRACES_SAMPLER=parentbased_traceidratio`、`OTEL_TRACES_SAMPLER_ARG=0.1`（按需调节）

- 将 SC_TRACE_* 宏桥接到 OTel Span（保留自研 TraceScope，作用域一致）
  - 变更：`core/common/trace/trace_macros.h` 引入 `core/common/otel/trace_scope_bridge.h`，在 `SC_TRACE_SCOPE`/`SC_TRACE_INIT_GUARD` 宏内构造桥接 RAII 对象，名称沿用原 stage，`tc.artifact.id` 自动注入。
  - 影响：开启 C++ OTel API 后，所有现有 SC_TRACE_* 埋点同时产生 OTel span（宏无行为变化）。

- 指标收敛（移除 C++ Daemon 侧历史 `store_daemon_*` 双写）
  - 变更：
    - `core/store/components/metrics_collector.{h,cc}`：删除 `store_daemon_*` 与操作直方图/计数器，仅保留统一 `tc_*`：
      - `tc_memory_pool_bytes{location=cpu|gpu, memory_type}`
      - `tc_p2p_bytes_total`
      - `tc_artifact_load_seconds{source,device,phase}`
    - 代码路径：
      - `StoreEngine::ingest_from_disk_internal` 新增 `record_artifact_load(source="disk")`；
      - `ingest_from_p2p_internal` 已发布 `tc_artifact_load_seconds(source="remote")` 和 `tc_p2p_bytes_total`。
  - 兼容性：历史 `store_daemon_*` 指标从 C++ 侧移除（Phase 5 收敛目标），集群视角的副本/请求等聚合指标由 Python Global Store 统一提供（Phase 3 已落地）。

- P2P 控制面 `traceparent`（可选增强）
  - 结论：当前通过 Span Link 与 RuntimeContext 已可完整关联入口 gRPC → P2P → H2D/Chunk。为避免协议 churn，本阶段不修改 Engine 协议；后续若需跨进程日志最小延迟关联，可在 `ProtoMtcpConnectRequest` 增加保守字段（兼容性门控）。

验证步骤（建议）：
- 启动 Collector：`docker run --rm --network host -v $(pwd)/tools/otel/collector-dev.yaml:/etc/otelcol/config.yaml:ro otel/opentelemetry-collector:latest`
- 运行 Global Store：
  - `export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4317`
  - `uv run -m tensorcast.global_store --port 50051 --metrics-port 8001`
- 运行 Daemon：
  - `TC_LOG_SINK_FILE=/tmp/tensorcast-daemon.log ./bazel-bin/daemon/tensorcast_daemon --listen_addr=0.0.0.0:50052`
- 客户端触发一次 materialize（已具备 OTel 客户端初始化/传播）：
  - `TC_OTEL_CLIENT_AUTO_INIT=1 uv run tools/otel_smoke.py --grpc-target 127.0.0.1:50051`
- 预期：
  - 后端可见 C++/Python 跨语言、跨进程完整链路；
  - C++ 端仅保留 `tc_*` 指标（无 `store_daemon_*`）；
  - C++/Python 日志均含 `trace_id`/`span_id`，Collector logs pipeline 可按 trace 关联。

更新（2025‑09‑01）：
- 已删除残留的 legacy 指标实现与文档：
  - `core/store/components/device_manager.{h,cc}` 中去除 `store_daemon_gpu_*` 与 `store_daemon_gpu_replicas_loaded`，仅保留统一的 `tc_memory_pool_bytes{location=gpu,...}`。
  - `core/store/replica/replica_memory_coordinator.cc` 移除 `store_daemon_chunk_state_transitions_total` 计数器更新。
  - 文档将 Daemon 的指标导出指引切换为“轻量 Python sidecar + Collector Prometheus receiver”（不再在 C++ 进程内提供 HTTP /metrics）。
- 路径确认：
  - C++ Daemon 进程内不再包含手工实现的 metrics HTTP 服务器；指标统一通过 sidecar 暴露为 OpenMetrics 文本，由 OpenTelemetry Collector 的 Prometheus receiver 抓取后汇聚（属于 OTel 统一链路）。
  - Python Global Store 侧同样通过 `prometheus-client` 暴露 `tc_*` 指标，Collector 抓取（Phase 3 已落地）。
- 本地校验建议：
  - 无 GPU 环境：使用 `--define use_fake_cuda=true` 重新构建 `//daemon:tensorcast_daemon` 或在有 GPU 的环境中运行；随后执行：
    - 后台启动 Daemon：`bazel-bin/daemon/tensorcast_daemon --listen_addr=127.0.0.1:8073 &`
    - 后台启动 sidecar：`uv run -m tensorcast.daemon_metrics_http --daemon-addr 127.0.0.1:8073 --port 9091 &`
    - 断言：`curl -s localhost:9091/metrics | grep -E '^(tc_memory_pool_bytes|tc_p2p_bytes_total)'`；并确认 `curl -sf localhost:8073/metrics` 失败（进程内无 /metrics）。
