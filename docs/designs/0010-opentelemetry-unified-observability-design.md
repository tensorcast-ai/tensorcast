---
slug: 0010-opentelemetry-unified-observability-design
title: OpenTelemetry Unified Observability (Design)
links:
  plan: ../plans/0010-opentelemetry-unified-observability.md
---

# Summary

Introduce a unified, cross-language OpenTelemetry (OTel) observability layer across TensorCast components (Python Global Store, C++ Store Daemon, and C++ Core). The design standardizes tracing, metrics, and logs with consistent naming, low overhead, and seamless context propagation across gRPC boundaries. It preserves existing in-house trace macro semantics via an OTel bridge and enables trace–log correlation with minimal code changes.

# Goals / Non‑Goals

Goals
- End-to-end traces across Global Store ↔ Store Daemon ↔ Core, including P2P data paths.
- Single naming system: spans as `<Component>/<Operation>`, metrics prefixed `tc_*`, logs enriched with `trace_id` and `span_id`.
- gRPC context propagation using W3C Trace Context across Python and C++.
- Low overhead by default: no-op when sampling is 0 or no exporter is configured.
- Clean API surfaces and helper utilities to minimize bespoke instrumentation.

Non‑Goals
- Building a custom observability stack or replacing OTel SDKs/Collector.
- Defining every fine-grained span; micro-instrumentation of hot loops is intentionally out of scope.
- Changing business logic, data models, or persistence schema.
- Mandating a specific backend; any OTLP-compatible backend or the OTel Collector is acceptable.

# Architecture & Interfaces

Overview
- Signals: Tracing (primary), Metrics (unified prefix), Logs (trace correlation).
- Transport: W3C Trace Context propagated over gRPC (client and server sides in both Python and C++).
- Bridge: Existing `SC_TRACE_*` C++ macros map to OTel spans with equivalent scope semantics.

Components & Entry Points
- Python
  - Initialization: `tensorcast/observability/otel.py` sets Tracer/Meter/Logger providers and gRPC instrumentation.
  - Global Store: initialize OTel at process start and instrument RPC handlers in `tensorcast/global_store/grpc_service.py`.
  - Clients & APIs: client init in `tensorcast/daemon_ctl.py` and high-level APIs in `tensorcast/api` create parent spans and attach low-cardinality `tc.*` attributes. Client SDK defaults to OTel disabled; enable by setting `observability.otel.enabled: true` in `ClientConfig`.
  - Logs/Metrics: `tensorcast/logger.py` injects `trace_id`/`span_id`; `tensorcast/global_store/metrics.py` defines `tc_*` metrics.
- C++
  - Initialization: `core/common/otel/init.h/.cc` provides `init_from_config(obs, role)`; call from daemon/server entry.
  - Propagation: `core/common/otel/grpc_propagation.h` extracts/injects context on `grpc::{Server,Client}Context`.
  - Trace bridge: `core/common/otel/trace_scope_bridge.h` maps `SC_TRACE_*` macro scopes to OTel spans/events.
  - Logs: `core/common/otel/logging_sink.*` optionally records logs annotated with `trace_id`/`span_id` for Collector ingestion.
  - Engine/Transport: instrument stage-level spans/events in `core/store/store_engine.cc` and `core/communicator/transport/*`.

Span Model & Conventions
- Span names: `<Component>/<Operation>` (examples: `GlobalStore/RequestReplicaTransport`, `StoreDaemon/MaterializeReplica`, `StoreEngine/P2PIngest`).
- Relationships: use child-of for causal chains; use Span Links for async or staged boundaries (e.g., P2P pipeline stages).
- Attributes
  - Standard: `rpc.system=grpc`, `rpc.service`, `rpc.method`, `rpc.grpc.status_code`.
  - Business (low-cardinality): `tc.artifact.id`, `tc.replica.id`, `tc.device.id`, `tc.size.bytes`, `tc.source.type=remote|disk`, `tc.location=gpu|cpu`.
- Cost controls: avoid span creation in tight loops; prefer stage or batch spans. Tune rate via sampling.

Metrics
- Unified prefix: all metrics start with `tc_*`.
- Scope: memory pool usage, P2P throughput, load latencies, and gRPC request telemetry.
- Export: OTLP push or Collector scrape via Prometheus receiver; deprecated in-process HTTP `/metrics` endpoints are removed.

Logs
- Python logs automatically include `trace_id` and `span_id` when an active span exists.
- C++ can install an `absl::LogSink` that writes enriched records for Collector ingestion.

Configuration
- Observability is driven by configuration (e.g., `observability.otel.*`):
  - `service.name` and service role (daemon, global_store, client)
  - exporter endpoint(s), protocol, and timeouts
  - sampling ratio and span limits
  - enable/disable signals (traces, metrics, logs)
- With no exporter or with sampling=0, runtime overhead remains near zero.

Verification (Dev Loop)
- Start an OTel Collector via `tools/otel/collector-dev.yaml`.
- Start Global Store and Daemon (Fake CUDA is supported for CPU-only machines).
- Trigger any client operation (or `tools/otel_smoke.py`) to see cross-service traces, metrics, and correlated logs.

# Schema Changes (if any)

None. This design does not introduce or modify persistent data schemas.

# Trade‑offs & Risks

- Overhead vs. fidelity: coarse stage-level spans are preferred on hot paths; sampling mitigates cost but reduces visibility.
- Attribute cardinality: enforce low-cardinality `tc.*` attributes to avoid backend cost explosions and high-cardinality pitfalls.
- Cross-language consistency: rely on shared propagation helpers and naming conventions to avoid drift.
- Bridge semantics: the `SC_TRACE_*` mapping must preserve scope lifetimes; regressions could break existing debug workflows.
- PII and data hygiene: avoid embedding sensitive data in attributes or logs.
- Operational dependencies: Collector/backend outages must degrade gracefully (non-blocking, bounded buffering).

# Compatibility & Acceptance Criteria

Compatibility
- Client SDK default: OTel is disabled unless explicitly enabled via `ClientConfig` (`observability.otel.enabled: true`).
- Default behavior is no-op if exporters are disabled or sampling=0.
- Existing `SC_TRACE_*` macro semantics are preserved via the bridge.
- Cross-language propagation uses W3C Trace Context; legacy code without OTel remains functional.

Acceptance Criteria
- Traces: A single request produces a coherent trace across Global Store, Daemon, and Core, including P2P stages with appropriate child relationships or links.
- Metrics: `tc_*` metrics for memory pools, P2P throughput, load latency, and gRPC telemetry are emitted and visible via OTLP or Prometheus.
- Logs: Log records in Python and (optionally) C++ include `trace_id`/`span_id` for correlation.
- Config: Toggling sampling and exporters affects cost and signal export without code changes.

# References

- Code surfaces
  - Python: `tensorcast/observability/otel.py`, `tensorcast/global_store/__main__.py`, `tensorcast/global_store/grpc_service.py`, `tensorcast/daemon_ctl.py`, `tensorcast/api`, `tensorcast/logger.py`, `tensorcast/global_store/metrics.py`
  - C++: `core/common/otel/{init.h,grpc_propagation.h,trace_scope_bridge.h,logging_sink.*}`,
    `daemon/server_main.cc`, `daemon/grpc_service_impl.cc`,
    `core/store/store_engine.cc`, `core/communicator/transport/*`, `core/store/components/*`
- Tools & configs: `tools/otel/collector-dev.yaml`

