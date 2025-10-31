---
slug: 0021-global-store-dashboard
title: Global Store Dashboard (Design)
status: accepted
areas: ["global_store", "infra"]
related_code:
  - tensorcast/global_store/grpc_service.py
  - tensorcast/global_store/metrics.py
  - proto/tensorcast/global_store/v1/global_store.proto
  - tensorcast/global_store/README.md
  - tensorcast/dashboard/api.py
  - tensorcast/dashboard/gs_client.py
  - tensorcast/dashboard/schemas.py
  - tensorcast/dashboard/webui/
---

# Summary

Provide a lightweight, read‑only dashboard for the Global Store that surfaces cluster state (workers, replicas, transports), content‑addressed artifact details (views/leaves), and core service health. Metrics integration is optional: when the dashboard receives external visualization config (e.g., `GRAFANA_HOST`, `GRAFANA_DASHBOARD_UID`), the UI embeds specified panels; otherwise the metrics section remains hidden. No changes are made to Global Store’s data model or control flow.

The initial delivery favors an independent service consisting of a Dashboard Backend and a static Dashboard Frontend. The Backend consumes Global Store’s public gRPC surface and exposes a simple REST JSON API to the Frontend. This keeps coupling low and allows incremental evolution.

The dashboard UI is optional. The Global Store service and the wider system operate normally without launching the dashboard.

# Goals / Non‑Goals

Goals
- Cluster overview: active workers, total replicas, replicas by memory type, in‑flight transports.
- Browsable lists: workers (heartbeats, pools, acceptance), replicas (filters + pagination via ListReplicasV2), recent transports.
- Artifact insight: GetArtifactInfoById (replicas, view metadata, leaf coverage), QueryChunkLocations (chunk placement/state).
- Health and metrics: optionally embed external metric panels; hide the section when not configured.
- Operational safety: read‑only; no control‑plane mutations.

Non‑Goals (initial)
- No embedded write operations (e.g., unregister, eviction, key mapping updates).
- No per‑daemon local logs/metrics streaming. (Future: optional node agents.)
- No schema changes in DuckDB; no extra tables.
- No mandatory metrics backend dependency; metrics section appears only when visualization config is provided.

# Architecture & Interfaces

## High‑Level Architecture

```mermaid
flowchart LR
  subgraph Dashboard Frontend
    FE[Static Web UI\n(HTML/JS/CSS)]
    GFX[Optional Embedded Panels]
  end

  subgraph Dashboard Backend
    API[HTTP JSON API (REST)]
    GSClient[gRPC client\n(GlobalStoreService)]
  end

  FE <-->|fetch JSON| API
  FE --->|if configured| GFX
  API --> GSClient

  subgraph Global Store
    GSgRPC[GlobalStore gRPC]
  end

  GSClient -.-> GSgRPC

  subgraph External Visualization (optional)
    GRAF[Grafana]
  end
  GFX -.-> GRAF
```

Rationale
- Decouple: ship dashboard without modifying Global Store runtime; deploy independently.
- Stability: reuse public gRPC; no private DB access. Metrics are optional via embedded external panels when configured.
- Evolvability: later add optional node‑level agents if per‑daemon logs/metrics are needed.

## JSON API (Dashboard Backend; REST only in MVP)

Read‑only endpoints that map 1:1 (or many:1) to Global Store gRPC calls and metrics aggregation. Exact schema is intentionally thin to minimize impedance with existing protos.

- `GET /api/health`
  - Maps to `HealthCheck`.
  - Returns `{ status: "OK"|... }`.

- `GET /api/workers?include_unavailable=0|1`
  - Maps to `ListActiveWorkers(include_unavailable)`.
  - Returns condensed worker rows with `worker_id, node_id, node_address, grpc_port, p2p_port, mem_pool_total/available, accepting_new_requests, last_heartbeat_ts, state_version, status`.

- `GET /api/replicas?artifact_id=&node_id=&node_address=&memory_type=&device_id=&page_token=&page_size=`
  - Maps to `ListReplicasV2` with filter fields and pagination.
  - Returns `{ replicas: [...], page_info: { next_page_token, total_size? } }`.

- `GET /api/artifacts/{artifact_id}?include=replicas,view,leaves&space=canonical|view&view_id=&leaf_indices=1,2,...`
  - Maps to `GetArtifactInfoById`.
  - Query params:
    - `include=replicas,view,leaves` selects sections to include.
    - `space=canonical|view` selects index space.
    - `view_id` optional when `space=view`.
    - `leaf_indices=1,2,...` optional selection for leaves.
  - Returns selected `replicas`, `view_meta`, `leaves`, and `partial_coverage`.

- `GET /api/chunks?artifact_id=&chunk_indices=1,2,...`
  - Maps to `QueryChunkLocations`.
  - Returns flattened chunk placement/state records.

Auxiliary endpoints
- `GET /api/config` — returns sanitized UI configuration (Grafana host/UID/panels). No secrets.
- `GET /healthz` — liveness (process up)
- `GET /readyz` — readiness (GS reachable and healthy)
- `GET /metrics` — Prometheus metrics for the dashboard backend itself

Notes
- Pagination mirrors proto: `page_token` is an opaque string compatible with `ListReplicasV2`; clients must not assume integer semantics.
- `memory_type` filter uses proto enum values (`RAM|GPU|DISK`).
- Time values are returned as RFC3339 (UTC) strings.
- The MVP does not parse or proxy Prometheus metrics; when visualization config is provided, the UI embeds external panels on the Metrics page.

## Implementation Stack & Packaging

- Backend: Python + FastAPI + Pydantic models for strict request/response schemas. Uses the existing Global Store gRPC stubs.
- Frontend: React + TypeScript + Vite + Tailwind (shadcn/ui) under `tensorcast/dashboard/webui`.
  - Router `basename` 与 Vite `base` 通过 `BASE_PATH`/`VITE_BASE_PATH` 对齐，支持子路径部署。
  - 不直接读取 Prometheus；Metrics 页面仅在 `GRAFANA_*` 配置存在时显示嵌入面板。
  - 前端运行时以 `import.meta.env.BASE_URL` 作为 Router `basename`，构建时由 `VITE_BASE_PATH` 写入 Vite `base`。
- Packaging: 后端随 Python wheel 分发；前端构建产物位于 `tensorcast/dashboard/static`（Vite build 的 outDir 指向该目录）。
- Entry point: ASGI 应用 `tensorcast.dashboard.api:app`（使用 Uvicorn 等 ASGI 服务器启动）。

Directory sketch (non-normative):
- `tensorcast/dashboard/api.py` — FastAPI app and routers
- `tensorcast/dashboard/schemas.py` — Pydantic models
- `tensorcast/dashboard/gs_client.py` — thin gRPC client wrapping GS RPCs
- `tensorcast/dashboard/webui/` — web UI source (React/Vite/Tailwind/shadcn)
- `tensorcast/dashboard/static/` — compiled frontend (Vite build 输出)

## API Contracts & Error Semantics

Conventions
- Content type: `application/json; charset=utf-8`.
- Timestamps: RFC3339 (UTC), e.g., `2025-02-01T12:34:56Z`.
- Pagination: `page_size` default 100, maximum 500. `page_token` is an opaque string; clients must not assume structure.
- Enumerations: use proto enum strings (e.g., `RAM|GPU|DISK`).

Error mapping (gRPC → HTTP)
- `OK` → 200
- `INVALID_ARGUMENT` → 400
- `NOT_FOUND` → 404
- `UNAUTHENTICATED` → 401
- `PERMISSION_DENIED` → 403
- `UNAVAILABLE` → 503
- `DEADLINE_EXCEEDED` → 504
- `ALREADY_EXISTS` → 409 (not used in read-only MVP)
- `INTERNAL|UNKNOWN` → 500

Error body (uniform)
```json
{
  "error": {
    "code": "NOT_FOUND",
    "http_status": 404,
    "message": "artifact not found",
    "details": {},
    "trace_id": "2f7c1c2b9e1d7a63"
  }
}
```

Endpoints — request/response examples（与 `tensorcast/dashboard/schemas.py` 对齐）

- GET `/api/health`
  - Response 200
  ```json
  { "status": "OK" }
  ```

- GET `/api/workers?include_unavailable=1`
  - Response 200
  ```json
  {
    "workers": [
      {
        "worker_id": "w-123",
        "node_id": "n-1",
        "node_address": "10.0.0.1",
        "grpc_port": 7070,
        "p2p_port": 9090,
        "mem_pool_total": 17179869184,
        "mem_pool_available": 8589934592,
        "accepting_new_requests": true,
        "last_heartbeat_ts": "2025-01-24T12:00:03Z",
        "state_version": 42,
        "status": "ACTIVE"
      }
    ]
  }
  ```

- GET `/api/replicas?artifact_id=af-001&memory_type=GPU&page_size=100`
  - Response 200
  ```json
  {
    "replicas": [
      {
        "artifact_id": "af-001",
        "node_id": "n-1",
        "node_address": "10.0.0.1",
        "device_id": 0,
        "memory_type": "GPU",
        "bytes": 1048576,
        "created_ts": "2025-01-24T12:01:00Z"
      }
    ],
    "page_info": {
      "next_page_token": "opaque-token"
    }
  }
  ```

- GET `/api/artifacts/af-001?include=replicas,view,leaves&space=canonical&leaf_indices=0,1,2`
  - Response 200
  ```json
  {
    "artifact_id": "af-001",
    "replicas": [
      {
        "node_id": "n-1",
        "node_address": "10.0.0.1",
        "device_id": 0,
        "memory_type": "GPU",
        "bytes": 1048576,
        "created_ts": "2025-01-24T12:01:00Z"
      }
    ],
    "view_meta": {
      "view_spec_json": "{}",
      "view_size": 1048576,
      "view_data_hash": "abcd...",
      "verified_at": "2025-01-24T12:01:00Z"
    },
    "leaves": [
      { "index": 0, "digest_b64": "..." },
      { "index": 1, "digest_b64": "..." }
    ],
    "partial_coverage": [
      { "space_kind": "CANONICAL", "space_id": "default", "missing": [ { "offset": 0, "length": 4096 } ] }
    ]
  }
  ```

- GET `/api/chunks?artifact_id=af-001&chunk_indices=0,1,2`
  - Response 200
  ```json
  {
    "chunks": [
      {
        "index": 0,
        "node_id": "n-1",
        "node_address": "10.0.0.1",
        "p2p_port": 9090,
        "state": "HOT",
        "node_load_ratio": 0.42,
        "device_uuid": "GPU-...",
        "replica": 0
      }
    ]
  }
  ```

## UI Surfaces (MVP)

- Overview
  - Cards: Active Workers, Replicas Total, Replicas by MemType, Active Transports.
  - Metrics charts: shown only when external visualization is configured; otherwise omitted.
- Workers
  - Table with filters and soft refresh (e.g., 5s‑15s interval).
- Replicas
  - Filterable table; supports pagination; quick memory type/device filters.
- Artifact Detail
  - Replica list, View meta, Leaves sample/coverage gaps, Chunk placement heatmap.
- Metrics
  - Optional: when `GRAFANA_*` config is provided, embed panels via iframe or similar; when absent, hide the page.

## Metrics Integration Options

MVP behavior
- Optional external embedding: when the dashboard process receives the following configuration, the UI embeds predefined panels on the Metrics page:
  - `GRAFANA_HOST`: base URL of the visualization server
  - `GRAFANA_DASHBOARD_UID`: target dashboard UID
  - `GRAFANA_PANEL_IDS`: comma‑separated panel IDs to embed
  - `GRAFANA_AUTH` (optional): read‑only token or header if the server requires authentication
- When not configured, the Metrics page is not shown.

Future (non‑MVP)
- If needed, add a lightweight local summary path that parses a small set of Global Store `/metrics` counters/histograms and exposes a JSON for the UI. Off by default.

## Real‑time Update Strategy

- MVP: short polling; no WebSocket/SSE.
- Future: optional SSE/WS for hot tables (e.g., transport queue).

## Security & Deployment

- Default posture: internal‑only, read‑only; deploy behind a gateway/reverse proxy in production.
- No built‑in authentication in the dashboard. Rely on internal network controls and/or reverse proxy policies if access restriction is needed.
- TLS: serve HTTPS directly (cert/key) or terminate at the reverse proxy.
- CORS: configurable allowlist via `CORS_ALLOWED_ORIGINS` (comma‑separated). Default is disabled (no cross‑origin access).
- CSP: 后端当前未默认设置 CSP 标头；建议在反向代理侧设置严格的 `Content-Security-Policy`（当启用 Grafana 嵌入时仅允许相应 `frame-src`）。后续可在后端增设可配置的 CSP 中间件。
- Do not expose Grafana tokens to the browser. Prefer anonymous viewer mode, or terminate Grafana auth at the reverse proxy and allow only panel paths from the dashboard origin.
- Path prefix: support hosting under a subpath (e.g., `/tensorcast-dashboard`) via `BASE_PATH`.
- Health probes:
  - `GET /healthz` — process liveness (always 200 when process is running).
  - `GET /readyz` — readiness (200 when GS becomes reachable and initial gRPC probe succeeds).
- Deployment: standalone process/container; configure GS gRPC endpoint via env; visualization `GRAFANA_*` config is optional.
- Optional component: launching the dashboard UI is not required for running the Global Store or the wider system.

## Configuration & Startup

Environment variables
- Global Store
  - `TENSORCAST_GS_ADDR` (optional, default `127.0.0.1:50051`): `host:port` of the GS gRPC endpoint.
  - `TENSORCAST_GS_SECURE` (optional, default `false`): enable TLS for GS.
  - `TENSORCAST_GS_CA_CERT` (optional): CA certificate path for GS mTLS.
Environment variables
- Global Store
  - `TENSORCAST_GS_ADDR` (optional, default `127.0.0.1:50051`): `host:port` of the GS gRPC endpoint.
  - `TENSORCAST_GS_SECURE` (optional, default `false`): enable TLS for GS.
  - `TENSORCAST_GS_CA_CERT` (optional): CA certificate path for GS mTLS.
- Dashboard server
  - `PORT` (default `8080`): HTTP port.
  - `HOST` (default `0.0.0.0`): bind address.
  - `BASE_PATH` (optional): path prefix when behind a reverse proxy.
  - `CORS_ALLOWED_ORIGINS` (optional): comma‑separated origins.
  - `DASHBOARD_TLS_CERT_FILE` / `DASHBOARD_TLS_KEY_FILE` (optional): enable HTTPS serving.
- Metrics embedding (optional)
  - `GRAFANA_HOST`, `GRAFANA_DASHBOARD_UID`, `GRAFANA_PANEL_IDS`, `GRAFANA_AUTH` (read‑only token if absolutely required; not forwarded to the browser).
- Observability
  - `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME=tensorcast-dashboard`.
  - `LOG_LEVEL` (default `INFO`).

Startup (examples)
```bash
# Run with Uvicorn (ASGI), using uv
uv run uvicorn tensorcast.dashboard.api:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"

# With BASE_PATH support (served behind a reverse proxy)
BASE_PATH=/tensorcast-dashboard \
  uv run uvicorn tensorcast.dashboard.api:app --host 0.0.0.0 --port 8080
```

## Observability (Dashboard)

- Tracing: initialize OpenTelemetry at process start; propagate trace context to downstream GS gRPC calls. Annotate spans with RPC method, status, and latency.
- Metrics: expose `/metrics` for dashboard backend self‑metrics (request count/latency, gRPC call count/latency, error rates). This endpoint surfaces only dashboard internals, not GS metrics.
- Logging: structured logs with `trace_id`/`span_id` correlation. Use existing logging utilities.

## Testing & Data Replay

- Contract tests for REST endpoints (golden request/response) using a GS gRPC stub.
- Frontend E2E (Playwright/Cypress) for Overview, Workers, Replicas, Artifact Detail.
- Sample data and a lightweight recorder/replayer to enable local development without a live cluster.

# Schema Changes

None. The dashboard is read‑only and uses existing gRPC/metrics. No DuckDB tables are added or modified.

# Trade‑offs & Risks

- Pros
  - Minimal coupling; rapid iteration; no DB coupling.
  - Reuses stable public APIs and Prometheus.
- Cons / Risks
  - Extra hop adds slight latency; mitigated by coarse refresh and pagination.
- Prometheus exposition parsing on the Dashboard Backend would be simplistic if enabled in the future—ensure bounded cardinality (already enforced in metrics design).
  - Without node agents, daemon‑local logs/metrics aren’t visible yet.

# Compatibility & Acceptance Criteria

Compatibility
- No changes to Global Store server behavior or schema.
- Uses existing proto errors/status mapping; tolerant to future additive RPC fields.

Acceptance Criteria (MVP)
- Overview shows accurate counts from gRPC within one refresh interval.
- Workers and Replicas lists are responsive at typical cluster sizes; pagination works.
- Artifact detail renders replicas, view meta (when present), and chunk locations.
- Health endpoint reflects GS HealthCheck; UI surfaces errors clearly.
- Metrics page is visible only when `GRAFANA_*` config is provided; otherwise hidden.
- No writes/mutations to GS; dashboard operates with the read‑only principle.

# References

- Architecture overview: docs/architecture/architecture-overview.md
- Global Store internals & data model: tensorcast/global_store/README.md
- Global Store gRPC proto: proto/tensorcast/global_store/v1/global_store.proto
- Common proto (Pagination/Memory types): proto/tensorcast/common/v1/common.proto
- Prometheus metrics in GS: tensorcast/global_store/metrics.py
- Observability design: docs/designs/0010-opentelemetry-unified-observability-design.md
