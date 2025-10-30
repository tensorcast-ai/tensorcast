TensorCast Dashboard Backend
============================

Overview
--------

The Dashboard Backend is a lightweight, read‑only HTTP JSON service that surfaces Global Store cluster state and artifact details for a web UI. It consumes the existing Global Store gRPC APIs and exposes a thin REST layer to the frontend. Metrics embedding is optional: when configured, the UI can embed external Grafana panels; otherwise the metrics page remains hidden.

Key properties:

- Read‑only; no control‑plane mutations
- Reuses public Global Store gRPC; no direct DB access
- Optional static frontend can be served by the backend
- Prometheus metrics exposed for the backend itself


Architecture
------------

- ASGI app: `tensorcast.dashboard.api:app` (FastAPI)
- gRPC client: thin async wrapper over Global Store stubs
- Optional static assets: served from `tensorcast/dashboard/static/` if present


Quickstart
----------

1) Set the Global Store endpoint (required):

```bash
export TENSORCAST_GS_ADDR="<host>:<port>"
```

2) Start the backend (ASGI via Uvicorn, using uv):

```bash
uv run uvicorn tensorcast.dashboard.api:app \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"
```

With a path prefix (behind a reverse proxy):

```bash
BASE_PATH=/tensorcast-dashboard \
  uv run uvicorn tensorcast.dashboard.api:app --host 0.0.0.0 --port 8080
```


Configuration
-------------

Environment variables:

- Global Store
  - `TENSORCAST_GS_ADDR` (required): `host:port` of the GS gRPC endpoint
  - `TENSORCAST_GS_SECURE` (optional, default `false`): enable TLS for GS
  - `TENSORCAST_GS_CA_CERT` (optional): CA certificate path for GS mTLS
  - `TENSORCAST_GS_TIMEOUT_SEC` (optional, default `5.0`): per‑RPC timeout (seconds)

- Server
  - `HOST` (default `0.0.0.0`): bind address
  - `PORT` (default `8080`): HTTP port
  - `BASE_PATH` (optional): path prefix when behind a reverse proxy
  - `CORS_ALLOWED_ORIGINS` (optional): comma‑separated origins (e.g., `https://a.com,https://b.com`)

- Metrics embedding (optional)
  - `GRAFANA_HOST`: base URL of the visualization server
  - `GRAFANA_DASHBOARD_UID`: target dashboard UID
  - `GRAFANA_PANEL_IDS`: comma‑separated panel IDs to embed
  - `GRAFANA_AUTH` (optional): read‑only token or header if the server requires authentication (never forwarded to the browser)

- Observability (backend self‑telemetry)
  - `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`
  - `OTEL_SERVICE_NAME=tensorcast-dashboard`
  - `LOG_LEVEL` (default `INFO`)


API Surface (REST)
------------------

Content type: `application/json; charset=utf-8`. Timestamps are RFC3339 (UTC, `...Z`).

- `GET /api/health`
  - Maps to GS `HealthCheck`. Returns `{ "status": "OK"|"ERROR" }`.

- `GET /api/workers?include_unavailable=0|1`
  - Maps to `ListActiveWorkers(include_unavailable)`.
  - Returns worker rows with `worker_id, node_id, node_address, grpc_port, p2p_port, mem_pool_total/available, accepting_new_requests, last_heartbeat_ts, state_version, status`.

- `GET /api/replicas?artifact_id=&node_id=&node_address=&memory_type=&device_id=&page_token=&page_size=`
  - Maps to `ListReplicasV2` with filters & pagination.
  - Pagination: `page_size` default 100, maximum 500; `page_token` is opaque.
  - `memory_type` must be one of `RAM|GPU|DISK`.

- `GET /api/artifacts/{artifact_id}?include=replicas,view,leaves&space=canonical|view&view_id=&leaf_indices=1,2,...`
  - Maps to `GetArtifactInfoById`.
  - `include` selects `replicas`, `view`, `leaves` (default: replicas only).
  - `space=canonical|view` (when `view`, `view_id` is required).
  - `leaf_indices` is a comma‑separated, non‑negative integer list.

- `GET /api/chunks?artifact_id=&chunk_indices=1,2,...`
  - Maps to `QueryChunkLocations`. Returns flattened chunk placement/state records.

- `GET /api/config`
  - Returns sanitized UI configuration (e.g., Grafana host, dashboard UID, panel IDs). No secrets are exposed.

Health & Metrics Endpoints:

- `GET /healthz` — liveness (always 200 when process is up)
- `GET /readyz` — readiness (probes GS and returns 200 only when GS is healthy)
- `GET /metrics` — Prometheus metrics for the dashboard backend itself


Error Semantics
---------------

gRPC → HTTP mapping:

- `INVALID_ARGUMENT` → 400
- `NOT_FOUND` → 404
- `UNAUTHENTICATED` → 401
- `PERMISSION_DENIED` → 403
- `ALREADY_EXISTS` → 409 (not used in read‑only API)
- `UNAVAILABLE` → 503
- `DEADLINE_EXCEEDED` → 504
- Unknown/others → 500

Uniform error body:

```json
{
  "error": {
    "code": "NOT_FOUND",
    "http_status": 404,
    "message": "artifact not found",
    "details": {},
    "trace_id": "<trace-id>"
  }
}
```


Static Assets
-------------

If a compiled frontend exists under `tensorcast/dashboard/static/`, it is served at `/static` (HTML and assets). API endpoints remain under `/api/*`.


Security & Deployment
---------------------

- Default posture: internal‑only, read‑only; deploy behind a gateway/reverse proxy in production
- No built‑in authentication; rely on network controls and/or reverse proxy policies
- TLS to Global Store is supported when `TENSORCAST_GS_SECURE=true` and CA is provided
- CORS: allowlist via `CORS_ALLOWED_ORIGINS` (disabled by default)
- CSP: the backend does not set CSP headers by default. Configure a strict `Content-Security-Policy` at your reverse proxy (when Grafana embedding is enabled, allow only the required `frame-src`). Backend‑side CSP middleware may be added in the future.
- Do not expose Grafana tokens to browsers; prefer anonymous viewer or terminate Grafana auth at a reverse proxy


Development & Testing
---------------------

- Run the app (using uv):

```bash
uv run uvicorn tensorcast.dashboard.api:app --host 0.0.0.0 --port 8080
```

- Lint & type‑check (see repository guidelines):

```bash
uv run ruff check .
uv run ruff format .
uv run mypy ./tensorcast
```

- Tests (examples):

```bash
uv run pytest tests/python/dashboard/test_api_params.py -q
```


Compatibility
-------------

- Uses existing Global Store gRPC API; tolerant to additive RPC fields
- No schema changes; read‑only behavior


Mock Global Store (for local development)
----------------------------------------

When no real Global Store is available, run the built‑in mock gRPC server to power the dashboard end‑to‑end.

1) Start the mock server:

```bash
uv run tensorcast/dashboard/mock_gs_server.py --port 50051
```

2) Start the dashboard backend pointing to the mock:

```bash
TENSORCAST_GS_ADDR=127.0.0.1:50051 \
TENSORCAST_GS_SECURE=0 \
uv run uvicorn tensorcast.dashboard.api:app --reload --host 0.0.0.0 --port 8080
```

The mock implements minimal, static responses for:

- HealthCheck
- ListActiveWorkers
- ListReplicasV2
- GetArtifactInfoById
- QueryChunkLocations

