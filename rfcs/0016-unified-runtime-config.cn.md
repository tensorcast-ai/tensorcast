# 0016-统一运行时配置系统（单一配置文件、Proto Schema、无环境变量、无热更新）

## 摘要

- 目标：以“单一标准配置文件”为中心，统一 TensorCast 的运行时配置（启动/服务层到数据面：gRPC、StoreEngine、通信层、内存、生命周期、HA、可观测性），完全移除基于环境变量与分散 flags 的配置路径；并提供跨 C++/Python 的同一套强类型 Schema 与加载器。
- 方式：以 Protobuf 作为唯一权威 Schema，YAML/JSON 作为运维友好载体；二者通过通用加载器（YAML→JSON→Proto）打通。StoreDaemon 与 Global Store 等进程均以 `--config=/path/to/tensorcast.yaml` 方式启动。配置变更通过“重启进程”生效（本期不支持热更新/SIGHUP 重载）。
- 收益：配置集中、可审计、可版本化；运维与开发人员对齐一套标准；严格拒收未知字段确保行为可预测；避免 flag/ENV 漫游导致的行为漂移。

## 背景与现状

当前配置来源分散：
- Store Daemon（C++）：大量 Abseil flags（`daemon/server_main.cc`），局部使用 ENV（例如周期性淘汰策略在 `grpc_service_impl.cc` 通过 `TC_DAEMON_*` 环境变量控制）。
- 通信层：已有统一 YAML/JSON→Proto 装载（`core/communicator/config_io.{h,cc}` + `tensorcast.communicator.v1.CommunicatorConfig`），但入口仍由 daemon flag `--comm_config_path` 传入。
- Python 侧：
  - `tensorcast/daemon_config.py` 提供 Pydantic 模型与 YAML 加载，CLI 将 YAML 译为 C++ flags。
  - Global Store 强依赖环境变量（`tensorcast/global_store/config/settings.py`）。
  - 其他模块零散读取 ENV（如 OTel、`torch_util.py` 的 `STORAGE_PATH` 回退等）。

问题：
- 配置分散（flags、ENV、多份 YAML），无法“一目了然”。
- 行为漂移风险（多处默认值/优先级/回退策略不一致）。
- 变更很难落地到所有进程（尤其是 Python 与 C++ 之间的差异）。
- 无统一的“热更新”边界（哪些字段可热更、哪些需要重启不清晰）。

## 设计目标（Goals）

- 单一入口：每个进程仅接受一个 `--config` 文件；文件内涵盖所有该进程所需配置。无其它来源（无 ENV、无多份 YAML、无 flags 叠加）。
- 强类型 Schema：以 Protobuf 定义完整配置；所有默认值与校验集中在加载器与构造路径。
- 统一格式：支持 YAML/JSON；YAML 面向运维（注释友好），JSON 面向工具链；二者都映射到同一个 Proto。
- 零环境变量：移除运行期 ENV 影响路径（OTel 也转入显式配置）。
- 一次性切换：完成落地后统一到单一来源（`--config`），不维护多来源优先级与回退策略。
- 可观测：配置加载日志、配置版本与校验摘要暴露到状态接口。

## 非目标（Non‑Goals）

- 不支持配置热更新/在线编辑（无 SIGHUP/`--reload`）。所有配置变更需重启对应进程生效。
- 不支持 TLS 证书轮转与在线重载（证书/密钥/CA 在启动时加载）。
- 不支持多来源合并与优先级（无 ENV、无 flags、无多份文件叠加）。

## 统一 Schema 与文件结构

以新包 `tensorcast.config.v1` 为统一 Schema 命名空间（通过 Buf 管理）。对不同进程定义独立接口类型：`DaemonConfig`、`GlobalStoreConfig`、`ClientConfig`（互不嵌套，不使用 oneof 封装）。统一引入 `SocketAddress {host, port}`，所有监听地址与对端连接均使用该结构，避免 `host+port` 与 `"host:port"` 两种表示法混用。

- DaemonConfig（Store Daemon 进程）
  - server：监听地址（`server.listen`）、可选 P2P 监听（`server.p2p_listen`）、线程、存储路径；`server.grpc` 为 gRPC 端参数
  - lifecycle：淘汰与进程扫描频率（`proc_check_interval`），会话/锁 sweep & TTL，GPU 显存比例等（合并重复监控间隔）
  - high_availability：Global Store 端点（`global_store_endpoints`）、心跳/同步/重试
  - communicator：`tensorcast.communicator.v1.CommunicatorConfig`（复用现有 Proto）
  - observability：统一 OTel（端点/协议/采样/服务名）与 Logging（枚举级别），`otel_cxx` 仅保留 C++ 专属开关（headers/insecure/console/sdk_disabled）
  - compatibility：ConfirmReplica 严格模式、超时映射等历史兼容开关
  - engine：StoreEngine 深层参数（*_bytes、streaming 并发、pinned_allocation_timeout 等）

- GlobalStoreConfig（Global Store 进程）
  - database：DuckDB 文件等
  - server：监听地址（`server.listen`）、线程、gRPC 端参数
  - worker_policy：默认心跳间隔、超时、清理周期
  - web_ui：UI 监听与 CORS（`cors_allowed_origins` 为列表）
  - observability：OTel 设置与日志

- ClientConfig（可选，客户端/工具使用）
  - daemon 启动/连接目标（`daemon.target` 为 `SocketAddress`）、默认存储根路径、默认行为（verification/wait/timeout）、可观测性

示例（DaemonConfig，最新结构）：

```yaml
server:
  listen:
    host: 0.0.0.0
    port: 50052
  # 可选：若不设置，默认与 listen.host 相同
  p2p_listen:
    host: 0.0.0.0
    port: 65090
  storage_path: /data/models
  num_threads: 16
  grpc:
    max_message_size_mb: 128
    max_concurrent_streams: 0             # 0 表示使用 gRPC 默认
    keepalive_time: 120s                  # 发送 KA PING 的周期（仅当连接空闲）
    keepalive_timeout: 20s
    max_connection_idle: 0s               # 0s=禁用
    max_connection_age: 0s                # 0s=禁用
    tcp_nodelay: true
    so_reuseport: false
    tls:
      enabled: false
      cert_file: /etc/tensorcast/tls/server.crt
      key_file: /etc/tensorcast/tls/server.key
      client_ca_file: /etc/tensorcast/tls/ca.crt
lifecycle:
  gpu_memory_limit_fraction: 0.75
  eviction_check_interval: 30s
  proc_check_interval: 5s                 # 合并了原 pid_watch_interval
  sessions_sweep_interval: 10s
  locks_sweep_interval: 10s
  verification_sweep_interval: 500ms
  sessions_ttl: 60s
  locks_ttl: 120s
high_availability:
  enabled: true
  heartbeat_interval: 5s
  periodic_sync_interval: 600s
  max_retries: -1
  registration_retry_delay: 1s
  global_store_endpoints:
    - host: 127.0.0.1
      port: 50051
communicator:
  enable_rdma: false
  stager:
    stage_cpu_for_rdma: true
    stage_chunk_mb_cpu: 4
    stage_chunk_mb_gpu: 16
    buffers_per_flow: 4
  rdma:
    outstanding_wr: 64
    ack_ttl_ms: 30000
    traffic_class: 186
    qp_timeout: 20
    qp_retry: 7
  pool:
    preregister_mr: true
    pool_size_bytes: 8589934592
    chunk_bytes: 67108864
  transport:
    tcp_conn_count: 8
    connect_timeout_sec: 10
    tcp_tos: 0
engine:
  mem_pool_size_bytes: 8589934592       # 8GB（也可写“8GB”，由 Loader 解析）
  chunk_bytes: 268435456                # 256MB（也可写“256MB”）
  dvmp_chunk_size_bytes: 268435456      # 256MB（也可写“256MB”）
  streaming_buffer_max_concurrent_sessions: 1
  pinned_allocation_timeout: 30s
  p2p_fallback_disk_dir: ""
observability:
  otel:
    enabled: true
    exporter_otlp_endpoint: http://127.0.0.1:4317
    exporter_protocol: grpc # or http/protobuf
    service_name: tensorcast-store-daemon
    sampler: parentbased_traceidratio
    sampler_arg: 1.0
  logging:
    level: INFO         # DEBUG/INFO/WARN/ERROR（强类型枚举）
    file: /var/log/tensorcast/daemon.log  # 可选
    otel_context_enabled: true        # 原 TC_LOG_OTEL_CONTEXT_ENABLED
    sink_file: ""                     # 原 TC_LOG_SINK_FILE（附加 OTel-enriched sink）
  tracing:
    chrome_trace_dir: ""             # 原 SC_TRACE_OUTPUT_DIR（留空表示关闭）
  otel_cxx:
    sdk_disabled: false               # 原 OTEL_SDK_DISABLED
    exporter_insecure: false          # 原 OTEL_EXPORTER_OTLP_INSECURE
    exporter_headers: {}              # 原 OTEL_EXPORTER_OTLP_HEADERS（k=v）
    console_exporter: false           # 原 TC_OTEL_CXX_CONSOLE（可选调试）
compatibility:
  confirm_requires_disk_path: false
  verification_timeout_status: ok  # enum: ok|deadline
  evict_on_dead_pid: false
checkpoint:
  streaming:
    num_buffers: 4
    io_chunk_bytes: 268435456       # 256MB
    pinned_pool_bytes: 2147483648   # 2GB
debug:
  cuda:
    enable_same_process_ipc_fallback: false
```

Global Store（示例 GlobalStoreConfig）：

```yaml
database:
  db_file: /var/lib/tensorcast/global_store.db
server:
  listen:
    host: 0.0.0.0
    port: 50051
  max_workers: 20
  grpc:
    max_message_size_mb: 128
    keepalive_time: 120s
worker_policy:
  heartbeat_timeout: 30s
  cleanup_interval: 60s
  default_heartbeat_interval: 5s
web_ui:
  enabled: true
  host: 0.0.0.0
  port: 9000
  cors_allowed_origins: ["*"]
observability:
  otel:
    enabled: true
    exporter_otlp_endpoint: http://127.0.0.1:4317
    service_name: tensorcast-global-store
  logging:
    level: INFO
meta:
  schema_version: v1
  description: "全局存储服务配置"
```

客户端（示例 ClientConfig）：

```yaml
daemon:
  target:
    host: 127.0.0.1
    port: 50052
  bin_path: /usr/local/bin/tensorcast_daemon
  python_interpreter: /usr/bin/python3
storage:
  default_root: /data/models
use_host_pid: false
connect_only: false
defaults:
  pinned_allocation_timeout: 30s
  enable_verification: true
  wait_for_completion: true
observability:
  logging:
    level: INFO
meta:
  schema_version: v1
  description: "客户端与 CLI 默认"
```

### 字段默认与校验
- 统一在加载器中完成：Proto3 无原生默认值与 presence，沿用 communicator 的 `NormalizeDefaults(...)` 模式（仅对数值/时间类进行默认，避免覆盖显式 false 的布尔值）。必要处对标量采用 `optional` 以保留 presence（区分“未设置”与“显式 false/0”）。
- 严格未知字段策略：对未知键一律报错（fail‑fast），不提供放宽开关；解析实现关闭 `ignore_unknown_fields`，并对 YAML→JSON 的中间树做键检查以定位错误。
- 字段去重（消除重复控制位）：
  - 移除 `server.enable_p2p_access`，以 `communicator.enable_rdma` 为准（避免重复开关）。
  - `observability.otel` 作为语言无关抽象（唯一来源：端点/协议/采样/服务名）；C++ 侧 `otel_cxx.*` 仅保留 headers/insecure/console/sdk_disabled，避免与 `observability.otel` 重复。

### 单位与时间（强校验）
- 字节类字段使用数值（`*_bytes`）或在加载器中支持 `KB/MB/GB` 后缀解析；时间采用 `google.protobuf.Duration` 或在加载器中统一解析 `ms/s/m` 后缀；linter 与运行时实现保持一致。

### Schema 变更原则与标准添加流程（必须遵循）
- 原则：任何运行时配置的新增/调整，必须以 Proto Schema 为唯一权威入口，不允许先写代码或由 ENV/flags 旁路引入新配置。
- 标准流程（一次提交内尽量完成闭环）：
  1) 修改 Proto：在 `tensorcast.config.v1` 下新增字段（遵循命名规范与分组；枚举补齐 `*_UNSPECIFIED=0`；如为重命名/迁移，保留 `reserved` 字段号/名称）。运行 `buf lint` 与 `tools/build_proto_python.sh` 生成并通过构建。
  2) 修改代码：更新 C++/Python Loader 的默认与校验逻辑（NormalizeDefaults、单位/时长解析、严格拒收未知键），贯通到引擎/服务使用点。移除与该字段相关的 ENV/flags 读路径。
  3) 更新文档：同步修改本 RFC、对应模块 README、示例 YAML（`examples/config/*.yaml`），并在 RFC 的 Execution Status 中记录变更与动机。必要时补充迁移映射（旧 → 新）。
  4) 校验与评审：本地通过 lint/构建/测试，提交时保证 CI 绿；评审关注“一个配置一个含义一个位置”的约束是否被满足。
  5) 弃用与兼容：如涉及旧字段，先标为弃用并在 Loader 打印一次性告警；删除时保留 `reserved`，确保向后兼容与演进有序。

## 统一加载器与启动方式

### C++（Daemon 为例）
- 新增：`core/common/config/daemon_config_io.{h,cc}`
  - `absl::StatusOr<tensorcast::config::v1::DaemonConfig> LoadDaemonConfigFromFile(const std::string& path);`
  - 自动识别 `.yaml/.yml` 或 `.json`；YAML 走 `yaml-cpp` → `nlohmann::json` → `google::protobuf::util::JsonStringToMessage` 路径（复用 communicator 方案）。
  - `NormalizeDefaults(DaemonConfig*)` 集中默认值；未知字段严格拒收；关闭 `ignore_unknown_fields`；在 Loader 中实现单位/时间后缀解析。
- `daemon/server_main.cc`
  - 仅接受 `--config`（唯一入口）；如未提供，拒绝启动并给出错误信息（附示例配置路径）。
  - 构造 `StoreEngineOptions`、`WorkerLifecycleManager::Options`、`StoreDaemonServiceImpl::CompatConfig` 等。
  - 若配置中包含 `communicator`，直接初始化 `CommunicationManager`；删除 `--comm_config_path` 与相关 flags。
- `daemon/grpc_service_impl.cc`
  - 移除 ENV 控制的周期性淘汰：改为读取 `lifecycle.*` 字段。
  - PID watcher、verification sweeper 等间隔从配置读取。
  - 新增 `checkpoint.streaming.*` 配置读取路径，替代 `STREAMING_*` 环境变量在保存/加载路径中的使用（对应 `core/checkpoint/checkpoint{,_streaming}.cc`）。
- 不提供 SIGHUP/热重载。配置变更需重启服务生效。

### Python
- 将 `tensorcast/daemon_config.py` 演进为“从 Proto Schema 生成/镜像”的 Pydantic（或直接使用生成的 Python Proto + 轻量封装）。
  - 短期内保持与新 Proto 结构 1:1；长远以 Proto 作为唯一权威 Schema。
- `tensorcast/daemon_manager.py` 与 CLI：
  - 启动 C++ 二进制时，仅传递 `--config=/path/to/file.yaml`。
  - 完全移除对逐项 flags 与 ENV 的拼接/读取。
- Global Store：提供与 Daemon 同风格的 `GlobalStoreConfig` 文件加载（替代 `from_env()`），CLI/入口统一接受 `--config`。
- 统一日志：`observability.logging.level/file` 替代 `LOG_LEVEL`。
- 统一客户端：新增 ClientConfig（见下），替代 `TENSORCAST_*` 客户端 ENV。
 - 客户端默认项（新）：`client.defaults.*`，例如 `pinned_allocation_timeout`、`wait_for_completion`、`enable_verification`，以统一 SDK 函数在未显式传参时的默认行为。

### YAML/JSON 互转工具
- 提供 `tools/config lint/format` 小工具：校验未知字段/单位后缀（MB/GB 等）、输出规范化 JSON 以便审计。

## 配置变更与重启策略

- 本期不支持热更新/在线编辑，亦不支持 TLS 轮转。
- 修改配置文件后，需重启对应进程（Daemon/Global Store/客户端工具）以使配置生效。
- 状态接口可暴露：配置 schema 版本、fingerprint（sha256）、加载时间与文件路径，便于审计；不包含“待重启字段”概念。

## 切换与弃用策略（一次性切换）

- 自本 RFC 落地起，统一以 `--config` 为唯一入口；不再接受 ENV/flags 作为运行时配置来源。
- 为帮助迁移，进程启动时可扫描已知 ENV/flags 并打印一次性弃用警告（这些值均被忽略，不参与合并/覆盖）。
- 示例/K8s/文档将同时切换到 `--config` 唯一路径。

## 统一来源与遗留删除策略（Flags/ENV 彻底下线）

目标：完全移除 ENV/flags 对运行时行为的影响，代码层面不再保留读取路径，避免“影子配置”。

阶段推进（建议以版本节拍执行）：
- Phase A（当前 − 迁移观察期）
  - 行为：仅 `--config` 生效；对已知 ENV/flags 做一次性“检测+告警”，记录到日志与状态接口；不参与合并/覆盖。
  - 文档：保留迁移映射表（旧 → 新），示例统一改为 `--config`。
- Phase B（R+1 − 构建与运行禁用）
  - 行为：默认禁止任何 ENV/flags 被读取；遗留桥接代码放入独立编译单元并默认不链接。若必须临时启用，需在开发环境以特定编译开关开启（生产构建禁止）。
  - CI：新增守卫规则：
    - C++：禁止 `getenv()`/`std::getenv`/`absl::GetFlag`（白名单：测试目录），Bazel `lint` 或 `clang-tidy` 规则卡口。
    - Python：禁止 `os.getenv()`（白名单：测试/工具），`ruff` 自定义检查或 pre-commit 钩子拦截。
  - 代码：将 ENV/flags 检测与告警逻辑 behind `#ifdef TENSORCAST_LEGACY_CFG_BRIDGE`（默认关闭）。
- Phase C（R+2 − 硬删除）
  - 行为：删除全部 ENV/flags 读取与检测代码、弃用映射常量、相关测试；仅保留 RFC 与 MIGRATION.md 中的迁移说明。
  - 文档：移除所有遗留提法；保留一份历史迁移指引。

观测与回滚：
- 在 Phase A 期间导出“遗留配置检测计数/来源键名 TopN”指标，辅助发现未迁移节点；如遇紧急问题，可在非生产构建临时启用桥接（Phase B 机制）。

## 影响面与风险

- 风险：一次性迁移可能影响现有部署（依赖 ENV/flags 的脚本/容器）。
  - 缓解：分阶段弃用、示例与 K8s 清单同步更新、详尽的迁移指南与 linter。
- 风险：Proto Schema 扩展需要跨 C++/Python 同步生成。
  - 缓解：Buf pipeline（`tools/build_proto_python.sh`）与 Bazel 目标齐备，CI 校验生成产物。

## 实施计划（分阶段）

P0 Schema & 工具链
- 新增 Proto：`proto/tensorcast/config/v1/daemon_config.proto`、`proto/tensorcast/config/v1/global_store_config.proto`、`proto/tensorcast/config/v1/common.proto`、`proto/tensorcast/config/v1/client_config.proto`（引用 `tensorcast.communicator.v1.CommunicatorConfig`）。
- 运行 `bash tools/build_proto_python.sh` 生成 Python stub；Bazel 加入 C++ 目标。

P1 Daemon 读配与生命周期去 ENV
- `core/common/config/daemon_config_io.{h,cc}`：YAML/JSON 装载与默认值归一。
- `daemon/server_main.cc` 接入 `--config`，按 Schema 构造引擎与服务；删除 `--comm_config_path`。
- `daemon/grpc_service_impl.cc`：移除 `TC_DAEMON_*` 环境变量；改为读 `lifecycle.*`。
 - `core/checkpoint/checkpoint{,_streaming}.cc`：移除 `STREAMING_*` 环境变量；改为读 `checkpoint.streaming.*`。
 - `core/common/cuda_real.cc`：将 `TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK` 迁移为 `debug.cuda.enable_same_process_ipc_fallback`（保留极短兼容）。

P2 CLI 与示例切换
- `tensorcast/daemon_manager.py` 与 `tensorcast.cli`：仅传 `--config`。
- 更新 `docker/k8s` 清单与 `examples/config/*.yaml`（提供统一示例）。
 - 为 Global Store 增加统一示例（含 metrics 与 web_ui）。

P3 Global Store 去 ENV（一次性切换）
- 新增 `GlobalStoreConfig` Proto 与加载器；入口统一支持 `--config`；移除 `from_env()`。
 - Python 端 `tensorcast/global_store/config/settings.py` 删除 `from_env()`，仅支持 `--config`；将以下 ENV 全量迁移（如检测到，仅打印一次性弃用提示并忽略）：
   - `GLOBAL_STORE_DB_PATH` → `database.db_file`
   - `GLOBAL_STORE_PORT` → `server.listen.port`
   - `GLOBAL_STORE_MAX_WORKERS` → `server.max_workers`
   - `GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS` → `worker_policy.heartbeat_timeout`
   - `GLOBAL_STORE_CLEANUP_INTERVAL_MS` → `worker_policy.cleanup_interval`
   - `GLOBAL_STORE_HEARTBEAT_INTERVAL_MS` → `worker_policy.default_heartbeat_interval`
   - `GLOBAL_STORE_OPTIMIZE_INTERVAL_MS` → `maintenance.optimize_interval_ms`（新增子块）
   - `GLOBAL_STORE_METRICS_PORT` → `observability.metrics.port`（如需要）
   - `GLOBAL_STORE_UI_*` → `web_ui.*`（port/host/static_dir/cors_allowed_origins/enabled/log_file）

P4 可观测性统一
- 将 OTel 主要参数（exporter/sampler/service_name 等）纳入 `observability.otel`，固定“配置文件唯一来源”；如检测到相关 ENV，仅打印一次性弃用警告并忽略。
- 将日志等级与文件输出纳入 `observability.logging`；删除 `LOG_LEVEL` 的生效路径（仅保留弃用提示）。
- Python 端 OTel 扩展 ENV 映射：
  - `TC_OTEL_CONSOLE_EXPORTER` → `observability.otel.console_exporter`
  - `TC_OTEL_CLIENT_AUTO_INIT` → `observability.otel.auto_init_client`
  - `TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS` → `observability.otel.allow_high_card_attrs`
 - C++ 端：`otel_cxx.*` 与日志 sink ENV 映射按上文收敛，默认继承 `observability.otel`。

（热更新与 SIGHUP 不在本期范围，取消该阶段；配置版本与校验摘要仍在状态接口中暴露，便于审计。）

P5 遗留配置硬删除（Flags/ENV 全面移除）
- 代码：删除 ENV/flags 读取与检测路径；去除 `TENSORCAST_LEGACY_CFG_BRIDGE` 编译开关与残留；移除相关测试用例与工具脚本。
- CI：启用强制检查，阻止新引入 `getenv/os.getenv/absl::GetFlag`；在评审模板中增加“配置项必须来自 Proto”的检查项。
- 文档：更新 README/模块 README，清理所有遗留字段与 ENV/flags 提法；保留 MIGRATION.md 历史说明。
- 验收：全仓搜索无遗留使用；构建日志无相关告警；部署样例与 K8s 清单仅存在 `--config`。

## 文件与代码变更清单（概览）

- 新增
  - `proto/tensorcast/config/v1/{daemon_config.proto, global_store_config.proto, client_config.proto, common.proto}`
  - `core/common/config/daemon_config_io.{h,cc}`
  - `tools/config/{lint,format}.py`（可选）
- 修改
  - `daemon/server_main.cc`：新增 `--config`、删除/禁用与 config 重叠的 flags；内联 communicator 初始化；HA 选项从配置读取。
  - `daemon/grpc_service_impl.cc`：ENV→配置；移除热更入口。
  - `core/checkpoint/checkpoint{,_streaming}.cc`：ENV→配置（`checkpoint.streaming.*`）。
  - `tensorcast/daemon_manager.py` 与 `tensorcast.cli`：仅传 `--config`。
  - `tensorcast/global_store/...`：新增文件加载入口，替代 `from_env()`。
  - 文档：`docs/deployment/store-daemon.md`、`docs/deployment/global-store-deployment.md`、`daemon/README.md`、仓库顶层 `README.md`。

## 迁移指南（运维/开发）

- 生成/调整配置文件：基于 `examples/config/store_daemon_config.yaml` 升级为统一 Schema；或使用 `tools/config format` 生成骨架。
- 启动：
  - Daemon：`bazel-bin/daemon/tensorcast_daemon --config=/etc/tensorcast/store-daemon.yaml`
  - Global Store：`uv run -m tensorcast.global_store --config=/etc/tensorcast/global-store.yaml`
- 移除脚本中对 ENV 的依赖；K8s 用 ConfigMap/Secret 挂载配置文件，容器只需传递 `--config`。

### 客户端/CLI 配置（ClientConfig）

用于 Python 客户端库、工具与 CLI 启动行为（不由 C++ 守护进程消费）：

```yaml
daemon:
  address: 127.0.0.1:8073     # 原 get_daemon_address 缺省
  bin_path: /usr/local/bin/tensorcast_daemon   # 原 TENSORCAST_DAEMON_BIN
  python_interpreter: /usr/bin/python3         # 原 TENSORCAST_PYTHON（仅 CLI 用）
client:
  use_host_pid: false         # 原 TENSORCAST_USE_HOST_PID
  storage:
    default_root: ./models    # 原 STORAGE_PATH 缺省
  connect_only: false         # 原 TENSORCAST_FORCE_CONNECT_ONLY
  defaults:
    pinned_allocation_timeout: 30s
    enable_verification: true
    wait_for_completion: true
observability:
  logging:
    level: INFO
  otel:
    auto_init_client: true    # 原 TC_OTEL_CLIENT_AUTO_INIT
    allow_high_card_attrs: false  # 原 TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS
    console_exporter: false       # 原 TC_OTEL_CONSOLE_EXPORTER
```

CLI/SDK 读取顺序：仅 `--config`（显式）。不存在多来源合并或优先级；若检测到兼容 ENV/旧 flags，仅打印一次性弃用提示并忽略。

### Daemon 现有 flags 的映射（完整清单）

- `--listen_addr` → `server.listen.host/port`
- `--storage_path` → `server.storage_path`
- `--p2p_port` → `server.p2p_listen.port`
- `--mem_pool_size` → `engine.mem_pool_size_bytes`
- `--chunk_size` → `engine.chunk_bytes`
- `--io_threads` → `server.num_threads`
- `--auto_register_disk_loads` → `compatibility.auto_register_disk_loads`（新增字段）
- `--global_store_addr` → `high_availability.global_store_endpoints`
- `--comm_config_path` → 删除；改为内嵌 `communicator.*`
- `--force_full_digest_on_load` → `compatibility.force_full_digest_on_load`（新增字段）
- `--heartbeat_interval_ms` → `high_availability.heartbeat_interval`
- `--chunk_sync_interval_ms` → `high_availability.periodic_sync_interval`
- `--confirm_requires_disk_path` → `compatibility.confirm_requires_disk_path`
- `--enable_p2p_access` → 已移除；以 `communicator.enable_rdma` 为准
- `--evict_on_dead_pid` → `compatibility.evict_on_dead_pid`
- `--verification_timeout_status` → `compatibility.verification_timeout_status`
- `--enable_periodic_eviction` → `lifecycle.enable_periodic_eviction`（新增字段）
- `--eviction_check_interval_ms` → `lifecycle.eviction_check_interval`
- `--gpu_memory_limit_fraction` → `lifecycle.gpu_memory_limit_fraction`

### C++/Python 现有 ENV 的映射（完整清单）

- 守护进程（C++）：
  - `TC_DAEMON_ENABLE_PERIODIC_EVICTION` → `lifecycle.enable_periodic_eviction`
  - `TC_DAEMON_GPU_MEMORY_LIMIT_FRACTION` → `lifecycle.gpu_memory_limit_fraction`
  - `TC_DAEMON_EVICTION_CHECK_INTERVAL_MS` → `lifecycle.eviction_check_interval`
  - `TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK` → `debug.cuda.enable_same_process_ipc_fallback`
  - `STREAMING_CHUNK_SIZE_MB` → `checkpoint.streaming.io_chunk_bytes`
  - `STREAMING_POOL_SIZE_GB` → `checkpoint.streaming.pinned_pool_bytes`
  - `STREAMING_NUM_BUFFERS` → `checkpoint.streaming.num_buffers`
  - OTel（C++ 侧）
    - `OTEL_SDK_DISABLED` → `observability.otel_cxx.sdk_disabled`
    - `OTEL_EXPORTER_OTLP_PROTOCOL` → `observability.otel.exporter_protocol`
    - `OTEL_EXPORTER_OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` → `observability.otel.exporter_otlp_endpoint`
    - `OTEL_EXPORTER_OTLP_INSECURE` → `observability.otel_cxx.exporter_insecure`
    - `OTEL_EXPORTER_OTLP_HEADERS` → `observability.otel_cxx.exporter_headers`
    - `TC_OTEL_CXX_CONSOLE` → `observability.otel_cxx.console_exporter`
  - 日志（C++ 侧）→ `observability.logging.*`
    - `TC_LOG_OTEL_CONTEXT_ENABLED` → `otel_context_enabled`
    - `TC_LOG_SINK_FILE` → `sink_file`
    - `TENSORCAST_LOG_LEVEL` → `level`
    - `TENSORCAST_VLOG_LEVEL` → `vlog_level`（新增）
  - Tracing（C++ 侧）
    - `SC_TRACE_OUTPUT_DIR` → `observability.tracing.chrome_trace_dir`
  - P2P 回退（C++ 侧）
    - `TENSORCAST_FALLBACK_MODEL_DIR` → `engine.p2p_fallback_disk_dir`

- 客户端/CLI（Python）：
  - `LOG_LEVEL` → `observability.logging.level`
  - `STORAGE_PATH` → `client.storage.default_root`
  - `TENSORCAST_PYTHON` → `daemon.python_interpreter`（仅 CLI 使用）
  - `TENSORCAST_DAEMON_BIN` → `daemon.bin_path`
  - `TENSORCAST_USE_HOST_PID` → `client.use_host_pid`
  - `TENSORCAST_FORCE_CONNECT_ONLY` → `client.connect_only`
  - `TENSORCAST_COMM_ENABLE_RDMA` → `communicator.enable_rdma`（仅开发环境遗留，建议删除）
  - OTel：
    - `OTEL_SERVICE_NAME` → `observability.otel.service_name`（若显式配置则忽略 ENV）
    - `OTEL_EXPORTER_OTLP_ENDPOINT` / `*_PROTOCOL` → `observability.otel.*`
    - `OTEL_TRACES_SAMPLER` / `_ARG` → `observability.otel.sampler/sampler_arg`
    - `TC_OTEL_CONSOLE_EXPORTER` → `observability.otel.console_exporter`
    - `TC_OTEL_CLIENT_AUTO_INIT` → `observability.otel.auto_init_client`
    - `TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS` → `observability.otel.allow_high_card_attrs`

- Global Store（Python）：
  - `GLOBAL_STORE_DB_PATH` → `database.db_file`
  - `GLOBAL_STORE_PORT` → `server.listen.port`
  - `GLOBAL_STORE_MAX_WORKERS` → `server.max_workers`
  - `GLOBAL_STORE_HEARTBEAT_TIMEOUT_MS` → `worker_policy.heartbeat_timeout`
  - `GLOBAL_STORE_CLEANUP_INTERVAL_MS` → `worker_policy.cleanup_interval`
  - `GLOBAL_STORE_HEARTBEAT_INTERVAL_MS` → `worker_policy.default_heartbeat_interval`
  - `GLOBAL_STORE_OPTIMIZE_INTERVAL_MS` → `maintenance.optimize_interval_ms`
  - `GLOBAL_STORE_METRICS_PORT` → `observability.metrics.port`（如使用）
  - `GLOBAL_STORE_UI_*` → `web_ui.*`（包含 `cors_allowed_origins`）
  - `GLOBAL_STORE_TRANSPORT_WAIT_RETRY_INTERVAL_MS` → `worker_policy.transport_wait_retry_interval_ms`

备注（非运行时）：`USE_FAKE_CUDA`、`BUILD_CORE/EXTENSION` 等构建环境变量不纳入运行时统一配置范围。

备注（测试专用 ENV，保留在测试框架中，不纳入运行时配置）：
- `TENSORCAST_RUN_CPP_DAEMON_IT`（控制集成测试开关）等。

## gRPC 服务端高级参数（server.grpc.*）

虽然当前 `daemon/server_main.cc` 尚未设置 `grpc::ServerBuilder` 的通道参数，本 RFC 预留 `server.grpc.*` 字段（均为启动期参数，运行时不可变）：
- 限流：`max_concurrent_streams`、每连接高级参数（idle/age）
- 报文：`max_message_size_mb`
- Keepalive：`keepalive_time`、`keepalive_timeout`
- TCP：`tcp_nodelay`、`so_reuseport`
- TLS：`tls.enabled` + 证书/密钥/CA 与可选 mTLS 校验

实现策略：优先在 Daemon 与 Global Store 的 gRPC 引导处读取并应用通道参数；TLS 配置为启动期只读，变更需重启。

### Proto 概览与命名规范（避免重复维护代码）

- 权威 Schema：以 Proto 为唯一来源，详细定义请直接参考仓库文件：
  - `proto/tensorcast/config/v1/common.proto`（SocketAddress, GrpcServer, Observability, ConfigMeta）
  - `proto/tensorcast/config/v1/daemon_config.proto`（DaemonConfig）
  - `proto/tensorcast/config/v1/global_store_config.proto`（GlobalStoreConfig）
  - `proto/tensorcast/config/v1/client_config.proto`（ClientConfig）
  - 包名：`tensorcast.config.v1`

- 结构概览（高层，不在文档中重复列出字段代码）：
  - DaemonConfig
    - server：listen、可选 p2p_listen、storage_path、num_threads、grpc
    - lifecycle：gpu_memory_limit_fraction、eviction/proc/sessions/locks/verification 间隔与 TTL
    - high_availability：global_store_endpoints、heartbeat/periodic_sync/max_retries/registration_retry_delay
    - communicator：复用 `tensorcast.communicator.v1.CommunicatorConfig`
    - engine：mem_pool_size_bytes、chunk_bytes、dvmp_chunk_size_bytes、streaming_buffer_max_concurrent_sessions、pinned_allocation_timeout、p2p_fallback_disk_dir
    - observability：Logging（枚举级别）、OTel（endpoint/protocol/sampler/service_name）、OTelCxx（headers/insecure/console/sdk_disabled）、Tracing
    - compatibility：confirm_requires_disk_path、verification_timeout_status、evict_on_dead_pid、auto_register_disk_loads、force_full_digest_on_load
    - checkpoint.streaming：num_buffers、io_chunk_bytes、pinned_pool_bytes
    - debug.cuda：enable_same_process_ipc_fallback
    - meta：schema_version、description
  - GlobalStoreConfig
    - database：db_file
    - server：listen、max_workers、grpc
    - worker_policy：heartbeat_timeout、cleanup_interval、default_heartbeat_interval
    - web_ui：host、port、cors_allowed_origins
    - observability、meta
  - ClientConfig
    - daemon：target（SocketAddress）、bin_path、python_interpreter
    - storage：default_root
    - use_host_pid、connect_only
    - defaults：pinned_allocation_timeout、enable_verification、wait_for_completion
    - observability、meta

- 命名与建模规范
  - 时长字段统一使用 `google.protobuf.Duration`，命名以 `*_interval`/`*_timeout`/`*_ttl`/`*_dur` 结尾。
  - 字节字段统一以 `*_bytes` 命名；Loader 支持 `KB/MB/GB` 后缀解析。
  - 枚举零值使用 `*_UNSPECIFIED = 0`；避免魔术字符串。
  - 迁移时对废弃字段保留 `reserved` 号与名称，避免冲突。

## StoreEngine 深层参数（engine.*）

将 `StoreEngineOptions` 中未暴露到 flags 的选项显式配置化：
- `dvmp_chunk_size`（默认 256MB）：影响 DVMP 颗粒度
- `streaming_buffer_max_concurrent_sessions`（默认 1）：共享 streaming pinned pool 的并发租约数
- `pinned_allocation_timeout`（默认 30s）：与请求级别 `MaterializeReplicaRequest.pinned_allocation_timeout_ms` 的优先级关系为：请求显式值 > engine 默认


## 与 0013 的关系（Communicator 已统一）

- 本 RFC 直接复用 0013 中 `CommunicatorConfig` 与 `config_io` 的加载/默认化策略，将其作为 Daemon 顶层配置的一个子块；不再需要 `--comm_config_path`。

## 运行时可调项盘点（代码位置 → 归并字段）

- Daemon（C++）
  - Flags（daemon/server_main.cc）：
    - `listen_addr` → `server.listen.host/port`
    - `storage_path` → `server.storage_path`
    - `p2p_port` → `server.p2p_listen.port`
    - `mem_pool_size` → `engine.mem_pool_size_bytes`
    - `chunk_size` → `engine.chunk_bytes`
    - `io_threads` → `server.num_threads`
    - `auto_register_disk_loads` → `compatibility.auto_register_disk_loads`
    - `global_store_addr` → `high_availability.global_store_endpoints`
    - `force_full_digest_on_load` → `compatibility.force_full_digest_on_load`
    - 其它（heartbeat/chunk_sync/confirm_requires_disk_path/evict_on_dead_pid/verification_timeout_status/enable_periodic_eviction/eviction_check_interval_ms/gpu_memory_limit_fraction）→ `lifecycle/compatibility/HA` 对应字段（将 *_ms 映射为 Duration）
  - ENV（daemon/grpc_service_impl.cc）：TC_DAEMON_*（周期淘汰）→ lifecycle.*；TC_OTEL_ALLOW_HIGH_CARDINALITY_ATTRS → observability.otel.allow_high_card_attrs。
  - 线程与清扫：sessions/locks/verif/pid（grpc_service_impl.{h,cc}）→ lifecycle.sweep_* 与 ttl_*。
- StoreEngine（C++）
  - Options（core/store/store_engine_options.h）：pinned_memory_timeout、dvmp_chunk_size、streaming_buffer_max_concurrent_sessions → server./engine.*；force_full_digest_on_load → compatibility.force_full_digest_on_load。
  - P2P 回退（core/store/loader/p2p_loader.cc）：TENSORCAST_FALLBACK_MODEL_DIR → engine.p2p_fallback_disk_dir。
- Checkpoint（C++）
  - STREAMING_*（core/checkpoint/checkpoint{,_streaming}.cc）→ checkpoint.streaming.*。
- CUDA/调试（C++）
  - TENSORCAST_ENABLE_IPC_SAME_PROCESS_FALLBACK（core/common/cuda_real.cc）→ debug.cuda.enable_same_process_ipc_fallback。
- 日志/OTel（C++）
  - TENSORCAST_LOG_LEVEL/TENSORCAST_VLOG_LEVEL（core/common/logging_init.cc）→ observability.logging.level/vlog_level。
  - OTEL_* & TC_*（core/common/otel/init.cc, logging_sink.cc）→ observability.otel_cxx.* 与 observability.logging.*。
  - SC_TRACE_OUTPUT_DIR（core/common/trace/trace_manager.h）→ observability.tracing.chrome_trace_dir。
- Global Store（Python）
  - GlobalStoreConfig.from_env（tensorcast/global_store/config/settings.py）：GLOBAL_STORE_* → database/server/worker_policy/web_ui/maintenance/observability。
  - CLI（tensorcast/global_store/__main__.py）：--port/--workers/--db-file/--metrics-port/--webui-log-file → 对应配置字段。
- Python 客户端/SDK
  - LOG_LEVEL（tensorcast/logger.py）→ observability.logging.level。
  - STORAGE_PATH（tensorcast/torch_util.py）→ client.storage.default_root。
  - TENSORCAST_DAEMON_BIN/TENSORCAST_PYTHON（daemon_manager.py, cli_utils）→ daemon.bin_path/python_interpreter。
  - TENSORCAST_USE_HOST_PID（daemon_ctl.py）→ client.use_host_pid。
  - TENSORCAST_FORCE_CONNECT_ONLY（config.py）→ client.connect_only。
  - OTel（tensorcast/observability/otel.py）：OTEL_*、TC_OTEL_* → observability.otel.*。

## 开放问题（Open Questions）

- OTel 覆盖优先级：固定为“配置文件唯一来源”，不再保留 ENV 覆盖路线；若发现相关 ENV，仅打印一次性弃用警告并忽略。
- 客户端配置（ClientConfig）：是否要引入统一文件标准，还是继续靠项目/应用方自定义？建议提供但不强制。
- Daemon TLS：本期不支持证书轮转与在线重载；后续如需轮转，将另立 RFC 讨论热插拔机制与失效回退策略。
- 速记单位：统一支持 `KB/MB/GB` 与字节整型两种写法；linter 应输出展开值与原始值，避免歧义。

---

## 附：Mermaid 视图

统一配置加载（以 Daemon 为例）

```mermaid
flowchart LR
  A[config.yaml/json] --> B[yaml-cpp]
  B --> C[nlohmann::json]
  C --> D[JsonStringToMessage]
  D --> E[DaemonConfig (proto)]
  E --> F[NormalizeDefaults]
  F --> G[StoreEngine/Comm/HA]
```

（热更新已明确不在本期范围，移除此流程图）

## 实施工单清单（代码触点与任务）

本节将“代码触点”细化为可执行工单，分模块列出文件与变更点，便于拆分与并行推进。每项建议附带验收要点（Acceptance）。

### Proto 与构建工具

- 新增 Proto Schema（统一配置）
  - 文件：`proto/tensorcast/config/v1/{common.proto, daemon_config.proto, global_store_config.proto, client_config.proto}`
  - 内容：本文档定义的各分组；`daemon_config.proto` 引用 `tensorcast.communicator.v1.CommunicatorConfig`
  - 验收：Bazel 编译通过；Python stub 由 `tools/build_proto_python.sh` 正确生成
- Bazel 目标
  - 为上述 proto 新增 C++ 目标并接入 `//daemon`
  - 验收：`bazel build //daemon:tensorcast_daemon` 通过（Fake CUDA/真 CUDA 均可）

### C++（Daemon + Core）

- 统一配置加载器
  - 文件：`core/common/config/daemon_config_io.{h,cc}`
  - 动作：YAML/JSON→Proto 加载、NormalizeDefaults、未知键严格拒收；支持 `MB/GB` 单位与 `ms/s/m` 时长解析
  - 验收：无配置/最小配置/完整配置三套样例均能成功加载并给出一致默认

- Daemon 入口
  - 文件：`daemon/server_main.cc`
  - 动作：新增 `--config`（唯一入口）；从配置构造 `StoreEngineOptions`、`CommunicationManager`、`WorkerLifecycleManager::Options`、gRPC ServerBuilder（含 TLS/keepalive/报文限制）；日志与 OTel 初始化；不安装 SIGHUP/热重载逻辑
  - 验收：仅 `--config` 即可启动；任意配置变更需重启进程生效

- 服务实现读取配置
  - 文件：`daemon/grpc_service_impl.{h,cc}`
  - 动作：引入线程安全 `DaemonRuntimeConfig` 视图；用配置替代 ENV（周期淘汰 `TC_DAEMON_*`、高基数属性门控）；sweep/TTL/间隔全部可配；状态 RPC 暴露配置版本/摘要
  - 验收：ENV 不再影响行为；配置项生效可观测（日志/状态 RPC）

- HA 生命周期
  - 文件：`daemon/worker_lifecycle_manager.{h,cc}`
  - 动作：心跳/同步间隔改读配置；监控参数从配置注入
  - 验收：修改心跳/同步频率后，重启生效

- 引擎与加载路径
  - 文件：`core/store/store_engine_options.h`, `core/store/store_engine.{h,cc}`
  - 动作：映射 pinned_memory_timeout、dvmp_chunk_size、streaming_buffer_max_concurrent_sessions、p2p_fallback_disk_dir
  - 验收：`GetServerConfig`/状态 RPC 能反映这些设置；行为与阈值符合预期

- Checkpoint Streaming 配置化
  - 文件：`core/checkpoint/checkpoint{,_streaming}.cc`
  - 动作：移除 `STREAMING_*` ENV；读取 `checkpoint.streaming.*`
  - 验收：大体积读写在新配置下可控（num_buffers/chunk_bytes/pool_bytes）

- CUDA IPC 测试回退
  - 文件：`core/common/cuda_real.cc`
  - 动作：用 `debug.cuda.enable_same_process_ipc_fallback` 替代 ENV；默认关闭
  - 验收：开启时单进程导入/打开 IPC 句柄用原指针回退；关闭时报错与真实 CUDA 一致

- OTel/日志（C++）
  - 文件：`core/common/otel/init.{h,cc}`, `core/common/otel/logging_sink.{h,cc}`, `core/common/logging_init.cc`
  - 动作：新增 `init_from_config`，接受 `observability.otel_cxx`；日志等级/文件/vlog/sink 由配置驱动
  - 验收：不依赖 ENV 也能完整启用 OTel 与日志；控制台/文件输出均可配置

- P2P 回退目录
  - 文件：`core/store/loader/p2p_loader.cc`
  - 动作：用 `engine.p2p_fallback_disk_dir` 替代 `TENSORCAST_FALLBACK_MODEL_DIR`
  - 验收：配置目录有效时正常回退；无效时记录告警并走纯远端

### Python（Global Store + Client/CLI）

- Global Store 统一配置
  - 文件：`tensorcast/global_store/config/settings.py`, `tensorcast/global_store/__main__.py`
  - 动作：新增文件加载入口与 `--config`；移除 `from_env()`；WebUI/Metrics 由配置控制
  - 验收：仅 `--config` 可启动；ENV 设置若存在仅打印一次性弃用提示并忽略

- Client/CLI 配置（ClientConfig）
  - 文件：新增轻量 Loader（Pydantic 或 proto+json_format）
  - 消费点：
    - `tensorcast/daemon_manager.py`：仅传 `--config`；支持 `client.connect_only` / `daemon.bin_path/python_interpreter`
    - `tensorcast/daemon_ctl.py`：支持 `client.use_host_pid`
    - `tensorcast/torch_util.py`：默认存储路径 `client.storage.default_root`；支持 `client.defaults.*`
    - `tensorcast/logger.py`, `tensorcast/observability/otel.py`：增加 from-config 路径
    - `tensorcast/daemon_config.py`：对齐新 Schema，增加 engine/server.grpc/observability/checkpoint/debug 等字段
  - 验收：客户端 API 在未显式传参时，继承配置默认项；CLI 使用统一配置启动 C++ 守护进程

### 兼容与弃用

- 启动期 ENV 扫描并打印一次性弃用提示（映射表见上）
- `--config` 在时，忽略重叠 flags 并告警；后续支持 `--no-legacy-flags`
- 验收：常见 ENV/flags 被侦测且提示清晰，不干扰启动

### 文档 / 示例 / K8s

- 文档：更新 `docs/deployment/store-daemon.md`, `docs/deployment/global-store-deployment.md`, `daemon/README.md`, `README.md`
- 示例：`examples/config/{store_daemon_config.yaml, global_store_config.yaml, client_config.yaml}`
- K8s：`docker/k8s/store_daemon_set.yaml` 统一 `--config`，去除 ENV
- 验收：示例可直接运行通过，文档链路一致

### 配置 Linter/Formatter（建议）

- 文件：`tools/config/lint.py`, `tools/config/format.py`
- 动作：校验未知键/单位/类型；输出规范化 JSON/展开值；CI 针对示例与片段运行
- 验收：错误能被捕获并给出明确定位

### 测试更新

- C++：去除 ENV 依赖（`STREAMING_*`, `TC_DAEMON_*`）；新增 Loader 默认与错误用例测试
- Python：Global Store/Client 转文件配置；增加 from-config 覆盖
- 验收：现有测试绿色；新增测试覆盖核心路径

### 推进节奏（里程碑）

- P0：Proto、C++/Python Loader、`--config`、嵌入 communicator；删除 `--comm_config_path` 与相关 flags；未知字段严格拒收；单位/时间解析落地；示例与 K8s 改造。
- P1：Daemon 去 ENV（淘汰/OTel gating）、sweeper 配置化、Engine 映射、Checkpoint/P2P 回退；状态接口暴露配置 fingerprint。
- P2：Global Store 文件配置、Client/CLI 接入；C++/Python 观测性按配置生效；工具链 `tools/config lint/format`。

## Execution Status

- 2025-09-07 — P0/Step 1 完成：初始 Proto 起草与落地
  - 新增 Schema（通过 Buf 管理）：
    - `proto/tensorcast/config/v1/common.proto` — 提供 `GrpcServer` 与跨组件复用的 `Observability`（Logging/OTel/OTelCxx/Tracing）。
    - `proto/tensorcast/config/v1/daemon_config.proto` — `DaemonConfig` 顶层配置，包含 `server/lifecycle/high_availability/communicator/engine/observability/compatibility/checkpoint/debug`。
    - `proto/tensorcast/config/v1/global_store_config.proto` — `GlobalStoreConfig` 顶层配置，包含 `database/server/worker_policy/web_ui/observability`。
    - `proto/tensorcast/config/v1/client_config.proto` — `ClientConfig` 顶层配置（客户端/CLI 默认项与可观测性）。
  - Bazel 目标：`//proto/tensorcast/config/v1:config_proto_lib`、`config_cc` 与 `config_proto_lint` 已添加；`buf lint` 通过。
  - 生成 Python Stub：已运行 `bash tools/build_proto_python.sh`（bazel 模式），产物位于 `proto/gen/python/`。
  - 与 RFC 差异（必要、提升长期可维护性）：
    - 去重：移除 `server.enable_p2p_access`，以 `communicator.enable_rdma` 为准（已在 proto 中删除并保留 reserved 字段号，防止误复用）。
    - 命名清晰化：`DaemonConfig` 中的 `Server` 重命名为 `DaemonServer`，`GlobalStoreConfig` 中 `GsServer` 重命名为 `GlobalServer`（字段名仍为 `server`，YAML 结构不变，类型名仅用于代码层面避免歧义）。
    - 观测性强类型：`Observability.Logging.level` 改为枚举 `LogLevel`（避免字符串拼写错误），OTLP 导出协议改为枚举 `OTelProtocol`（`grpc`/`http_protobuf`）。同时去重：`OTelCxx` 不再重复定义 endpoint/protocol，仅保留 C++ 专属开关（headers/insecure/console/sdk_disabled），统一以 `Observability.OTel` 为单一来源。
    - Web UI CORS：`web_ui.cors_origins` 改为 `web_ui.cors_allowed_origins`（repeated string，便于显式列举多个来源）。
    - 统一地址模型：新增 `SocketAddress {host, port}`，用于所有监听与对端引用，避免在不同地方混用 `host+port` 与 `"host:port"` 字符串。具体：
      - `DaemonServer.listen`、`DaemonServer.p2p_listen`（保留旧字段号 reserved）
      - `GlobalServer.listen`（保留旧字段号 reserved）
      - `HighAvailability.global_store_endpoints`（替代 `global_store_address` 字符串）
      - `Client.DaemonLauncher.target`（替代 `address` 字符串）
    - 顶层元信息：新增可选 `meta`（`ConfigMeta { schema_version, description }`）便于审计与变更记录；不会影响运行参数。
    - Lint 规范：`VerificationTimeoutStatus` 与 `OTelProtocol` 等枚举均补齐 `*_UNSPECIFIED=0`，并遵循前缀规范通过 Buf STANDARD。
    - 命名去歧义（避免相同名不同义）：`checkpoint.streaming.chunk_bytes` → `io_chunk_bytes`，`checkpoint.streaming.pool_bytes` → `pinned_pool_bytes`；同时将 `Lifecycle.pid_watch_interval` 合并进 `proc_check_interval`（字段号 reserved），避免两个进程扫描间隔含义重叠。
  - 尚未改动：ClientConfig 规划留待 P2；C++/Python Loader 与 `--config` 接入在后续步骤实施。

后续计划（对齐“实施计划”）：
- P0/Step 2：实现 C++ Loader（YAML/JSON→Proto，严格拒收未知字段，默认/单位归一）、`daemon/server_main.cc` 接入 `--config` 并移除 `--comm_config_path`。
- P1：Daemon 去 ENV、sweeper 配置化、Engine/Checkpoint/P2P 映射，状态接口展示配置摘要。
- P2：Global Store/Client 文件配置与 CLI 接入、观测性按配置生效、`tools/config lint/format`。
