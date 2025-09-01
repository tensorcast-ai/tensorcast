## 0011 – OpenTelemetry Unified Observability (Concise Guide)

This document is a high-level guide to TensorCast's OpenTelemetry (OTel) adoption across Tracing, Metrics, and Logs. It is intended for contributors extending observability in this repository. Focus: understand current support, core conventions, and how to add instrumentation—without deep implementation details.

### What’s already supported (snapshot)

- **Tracing (end-to-end)**
  - **Cross-language gRPC propagation**: Python Global Store ↔ C++ StoreDaemon/Core using W3C Trace Context; both client (SDK) and server are instrumented.
  - **C++ bridge to in-house trace**: `SC_TRACE_*` macros are bridged to OTel spans (same scope semantics).
  - **P2P path coverage**: Engine/transport key stages emit spans/events; parent-child or Links connect them to the gRPC entry span.
- **Metrics (unified prefix)**
  - Unified `tc_*` metrics cover memory pools, P2P throughput, load latency, and gRPC request telemetry.
  - Export via OTLP or let the Collector scrape via Prometheus receiver; C++ in-process registries/HTTP `/metrics` are removed.
- **Logs (trace correlation)**
  - Python logs auto-inject `trace_id`/`span_id` (when an active span exists).
  - C++ optionally installs an absl::LogSink that writes `trace_id`/`span_id` to a file for the Collector to ingest.

All features are environment-driven. With no exporter configured or with sampling at 0, overhead is near-zero.

### Design principles (when adding/modifying instrumentation)

- **Consistent naming**: Spans use `<Component>/<Operation>`; metrics use `tc_*`; log fields use `trace_id`, `span_id`.
- **Low-cardinality attributes**: Business attributes use `tc.*` and avoid high-cardinality values (e.g., use `tc.artifact.id` only when appropriate for debug/test).
- **Correct semantics**: Prefer child-of for causal chains; use Span Links across async or staged boundaries.
- **Cost awareness**: Avoid overly fine-grained spans on hot paths; control overhead via sampling.
- **API-first**: Use OTel APIs and existing wrappers in Python/C++; avoid bespoke/dead-end implementations.

### Extension guide (where/how to add)

#### Python (Global Store / Client)

- **Entry & initialization**
  - Unified init: `tensorcast/observability/otel.py` (shared by server and client contexts).
  - Global Store: set Provider + gRPC server instrumentation before server startup (`tensorcast/global_store/__main__.py`).
  - Client SDK: `tensorcast/daemon_ctl.py`, `tensorcast/torch_util.py` ensure Provider + gRPC client instrumentation via `ensure_client_otel()` (supports `TC_OTEL_CLIENT_AUTO_INIT=1`).
- **Where to add spans**
  - gRPC server handlers: `tensorcast/global_store/grpc_service.py` (set standard `rpc.*` and necessary `tc.*` attributes per RPC).
  - High-level APIs: `tensorcast/torch_util.py` (wrap user-facing operations with parent spans and add business attributes).
- **Logs & metrics**
  - Logs: `tensorcast/logger.py` injects `trace_id`/`span_id` into log records.
  - Metrics: extend `tensorcast/global_store/metrics.py` as needed, preserve `tc_*` naming and low-cardinality labels.

Practical tips: set `service.name` explicitly; attach low-cardinality `tc.*` attributes (e.g., `tc.device.id`, `tc.size.bytes`); avoid mass creation of short-lived spans in tight loops.

#### C++ (StoreDaemon / Core)

- **Entry & common wrappers**
  - SDK init: `core/common/otel/init.h/.cc` provides `InitFromEnv(service, role)`; call early in `daemon/server_main.cc`.
  - gRPC propagation: `core/common/otel/grpc_propagation.h` extracts/injects Trace Context on `grpc::{Server,Client}Context`.
  - Trace bridge: `core/common/otel/trace_scope_bridge.h` bridges `SC_TRACE_*` to OTel spans (no business code changes needed).
- **Where to add spans/events**
  - Daemon RPCs: `daemon/grpc_service_impl.cc` (server spans + `rpc.*` + key `tc.*`).
  - Engine/transport: `core/store/store_engine.cc`, `core/communicator/transport/*` (stage-level P2P spans/events; use Links to connect to the entry gRPC span when needed).
- **Logs & metrics**
  - Logs: `core/common/otel/logging_sink.*` can write enriched logs (with `trace_id`/`span_id`) to a file.
  - Metrics: use unified `tc_*` metrics under `core/store/components/*` (memory pool, P2P throughput, load latency).

Practical tips: aggregate hot-path work into stage spans; rely on existing init/propagation/bridge helpers to avoid duplication.

### Naming & attributes (minimum set)

- **Span names**: `StoreDaemon/MaterializeReplica`, `GlobalStore/RequestReplicaTransport`, `StoreEngine/P2PIngest`.
- **Common attributes**:
  - Standard: `rpc.system=grpc`, `rpc.service`, `rpc.method`, `rpc.grpc.status_code`.
  - Business (low-cardinality): `tc.artifact.id`, `tc.replica.id`, `tc.device.id`, `tc.size.bytes`, `tc.source.type=remote|disk`, `tc.location=gpu|cpu`.

### Run & verify (minimal)

- **Key environment variables**
  - `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g., `http://127.0.0.1:4317` or `http://127.0.0.1:4318`)
  - `OTEL_EXPORTER_OTLP_PROTOCOL=grpc|http/protobuf`
  - `OTEL_SERVICE_NAME` (e.g., `tensorcast-global-store`, `tensorcast-store-daemon`, `tensorcast-client`)
  - `OTEL_TRACES_SAMPLER=parentbased_traceidratio` and `OTEL_TRACES_SAMPLER_ARG`
  - Client optional: `TC_OTEL_CLIENT_AUTO_INIT=1`
- **Local verification (recommended)**
  - Start Collector via `tools/otel/collector-dev.yaml`.
  - Start Global Store and Daemon (Fake CUDA is supported).
  - Use `tools/otel_smoke.py` or any gRPC call to verify cross-service traces/logs/metrics visibility.

### Code map (quick index)

- **Python**
  - `tensorcast/observability/otel.py` (init & gRPC instrumentation)
  - `tensorcast/global_store/__main__.py`, `tensorcast/global_store/grpc_service.py` (server entry & RPCs)
  - `tensorcast/daemon_ctl.py`, `tensorcast/torch_util.py` (client & higher-level APIs)
  - `tensorcast/logger.py`, `tensorcast/global_store/metrics.py`
- **C++**
  - `core/common/otel/{init.h,grpc_propagation.h,trace_scope_bridge.h,logging_sink.*}`
  - `daemon/server_main.cc`, `daemon/grpc_service_impl.cc`
  - `core/store/store_engine.cc`, `core/communicator/transport/*`
  - Metrics: `core/store/components/*`
- **Config & tools**: `tools/otel/collector-dev.yaml`

### Contributor checklist (self-review when adding observability)

- Span naming follows `<Component>/<Operation>`; metrics use `tc_*`.
- Only low-cardinality `tc.*` attributes; avoid fine granularity on hot paths.
- New gRPC paths correctly propagate context (Python and C++ sides).
- P2P/async stages are linked back to the entry gRPC span when appropriate.
- Logs can correlate to traces (or C++ log sink is installed).
- Default behavior can fall back to low overhead (sampling/endpoint disabled).

### Rollback & compatibility

- Lower sampling to 0 or remove exporters to “turn off” signal export; C++ runs in API-only (no-op) mode without SDK init.
- The in-house trace remains bridged; macro scope semantics are preserved.



