#  Copyright (c) 2025, TensorCast Team.

"""FastAPI application factory for the Global Store dashboard backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Annotated

import grpc
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from tensorcast.dashboard.config import DashboardSettings, get_settings
from tensorcast.dashboard.gs_client import (
    ClientConfig,
    GlobalStoreClient,
    GlobalStoreStatusError,
)
from tensorcast.dashboard.metrics import REQUEST_COUNTER, REQUEST_LATENCY
from tensorcast.dashboard.schemas import (
    ArtifactDetailResponse,
    ChunkLocationsResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReplicasResponse,
    WorkersResponse,
)
from tensorcast.logger import init_logger
from tensorcast.observability.otel import setup_otel
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.global_store.v1 import global_store_pb2

logger = init_logger(__name__)


def _current_trace_id() -> str | None:
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return None
    return f"{context.trace_id:032x}"


def _error_response(
    *,
    code: str,
    http_status: int,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    trace_id = _current_trace_id()
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            http_status=http_status,
            message=message,
            details=details or {},
            trace_id=trace_id,
        )
    )
    return JSONResponse(status_code=http_status, content=body.model_dump())


def _memory_type_from_param(
    value: str | None,
) -> common_pb2.MemoryType | None:
    if value is None:
        return None
    lookup = {
        "RAM": common_pb2.MemoryType.MEMORY_TYPE_RAM,
        "GPU": common_pb2.MemoryType.MEMORY_TYPE_GPU,
        "DISK": common_pb2.MemoryType.MEMORY_TYPE_DISK,
    }
    key = value.strip().upper()
    if key not in lookup:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported memory_type '{value}'. Expected one of RAM,GPU,DISK.",
        )
    return lookup[key]


def _parse_leaf_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token, 10)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid leaf index '{token}'",
            ) from exc
        if value < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Leaf indices must be non-negative",
            )
        indices.append(value)
    return indices or None


def _parse_chunk_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token, 10)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chunk index '{token}'",
            ) from exc
        if value < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chunk indices must be non-negative",
            )
        indices.append(value)
    return indices or None


def _include_flags(include_param: str | None) -> tuple[bool, bool, bool]:
    if include_param is None or not include_param.strip():
        return True, False, False
    parts = {part.strip().lower() for part in include_param.split(",") if part.strip()}
    allowed = {"replicas", "view", "leaves"}
    invalid = parts.difference(allowed)
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported include values: {', '.join(sorted(invalid))}",
        )
    return (
        "replicas" in parts,
        "view" in parts,
        "leaves" in parts,
    )


def _space_flags(space: str | None, view_id: str | None) -> tuple[bool, str | None]:
    if space is None:
        return True, None
    normalized = space.strip().lower()
    if normalized == "canonical":
        return True, None
    if normalized == "view":
        if not view_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="view_id is required when space=view",
            )
        return False, view_id
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="space must be either 'canonical' or 'view'",
    )


def _grpc_to_http_status(code: grpc.StatusCode) -> int:
    mapping = {
        grpc.StatusCode.INVALID_ARGUMENT: status.HTTP_400_BAD_REQUEST,
        grpc.StatusCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
        grpc.StatusCode.UNAUTHENTICATED: status.HTTP_401_UNAUTHORIZED,
        grpc.StatusCode.PERMISSION_DENIED: status.HTTP_403_FORBIDDEN,
        grpc.StatusCode.ALREADY_EXISTS: status.HTTP_409_CONFLICT,
        grpc.StatusCode.UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED: status.HTTP_504_GATEWAY_TIMEOUT,
    }
    return mapping.get(code, status.HTTP_500_INTERNAL_SERVER_ERROR)


def _create_client(settings: DashboardSettings) -> GlobalStoreClient:
    config = ClientConfig(
        target=settings.gs_endpoint,
        secure=settings.gs_secure,
        ca_cert=settings.gs_ca_cert,
        timeout_sec=settings.request_timeout_sec,
    )
    return GlobalStoreClient(config)


def create_app(settings: DashboardSettings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    try:
        setup_otel(service_default="tensorcast-dashboard", role="dashboard")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialise OpenTelemetry: %s", exc)

    client = _create_client(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await client.connect()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialise Global Store client: %s", exc)
        yield
        await client.close()

    app = FastAPI(
        title="TensorCast Dashboard",
        root_path=resolved_settings.root_path,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.gs_client = client

    if resolved_settings.has_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_allowed_origins),
            allow_methods=["GET"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = perf_counter()
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except HTTPException as exc:
            status_code = exc.status_code
            raise
        except Exception:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise
        finally:
            duration = perf_counter() - start
            route = request.scope.get(
                "route"
            )  # FastAPI attaches the Route object here.
            path = getattr(route, "path", request.url.path)
            method = request.method
            REQUEST_COUNTER.labels(
                endpoint=path, method=method, status=str(status_code)
            ).inc()
            REQUEST_LATENCY.labels(endpoint=path, method=method).observe(duration)

    class _CSPMiddleware(BaseHTTPMiddleware):
        def __init__(self, app: ASGIApp, grafana_host: str | None):
            super().__init__(app)
            self._grafana_host = grafana_host

        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            if self._grafana_host:
                csp = f"default-src 'self'; frame-src {self._grafana_host}; "
            else:
                csp = "default-src 'self'"
            # Do not expose any auth tokens to browser; we only set CSP.
            response.headers.setdefault("Content-Security-Policy", csp)
            return response

    # Attach CSP middleware based on Grafana config
    app.add_middleware(
        _CSPMiddleware,
        grafana_host=(
            resolved_settings.grafana.host if resolved_settings.grafana else None
        ),
    )

    # Optionally serve static frontend assets if present
    static_dir = Path(__file__).with_name("static")
    if static_dir.exists() and static_dir.is_dir():
        # Mount under /static to avoid shadowing API routes
        app.mount(
            "/static", StaticFiles(directory=str(static_dir), html=True), name="static"
        )

    def get_client_dependency(request: Request) -> GlobalStoreClient:
        return request.app.state.gs_client

    @app.exception_handler(GlobalStoreStatusError)
    async def handle_status_error(
        _: Request, exc: GlobalStoreStatusError
    ) -> JSONResponse:
        return _error_response(
            code=exc.status_name,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=str(exc),
        )

    @app.exception_handler(grpc.aio.AioRpcError)
    async def handle_grpc_error(_: Request, exc: grpc.aio.AioRpcError) -> JSONResponse:
        code = exc.code()
        details = exc.details() or "Global Store RPC failed"
        http_code = (
            _grpc_to_http_status(code)
            if code is not None
            else status.HTTP_502_BAD_GATEWAY
        )
        code_name = code.name if code is not None else "UNKNOWN"
        return _error_response(
            code=code_name,
            http_status=http_code,
            message=details,
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
    ) -> dict[str, str]:
        try:
            response = await client.health_check()
        except grpc.aio.AioRpcError as exc:
            raise HTTPException(
                status_code=_grpc_to_http_status(exc.code()),
                detail=exc.details() or "Global Store unavailable",
            ) from exc
        if response.status != global_store_pb2.Status.STATUS_OK:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Global Store reported unhealthy",
            )
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/health", response_model=HealthResponse)
    async def api_health(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
    ) -> HealthResponse:
        response = await client.health_check()
        return HealthResponse.from_proto(response)

    @app.get("/api/workers", response_model=WorkersResponse)
    async def api_workers(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
        include_unavailable: bool = Query(
            default=False, description="Include workers marked unavailable"
        ),
    ) -> WorkersResponse:
        response = await client.list_active_workers(
            include_unavailable=include_unavailable
        )
        return WorkersResponse.from_proto(response)

    @app.get("/api/replicas", response_model=ReplicasResponse)
    async def api_replicas(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
        artifact_id: str | None = Query(default=None),
        node_id: str | None = Query(default=None),
        node_address: str | None = Query(default=None),
        memory_type: str | None = Query(default=None),
        device_id: int | None = Query(default=None),
        page_token: str | None = Query(default=None),
        page_size: int | None = Query(default=100, ge=1, le=500),
    ) -> ReplicasResponse:
        memory_type_value = (
            _memory_type_from_param(memory_type) if memory_type else None
        )
        response = await client.list_replicas(
            artifact_id=artifact_id,
            node_id=node_id,
            node_address=node_address,
            memory_type=memory_type_value,
            device_id=device_id,
            page_token=page_token,
            page_size=page_size,
        )
        return ReplicasResponse.from_proto(response)

    @app.get("/api/artifacts/{artifact_id}", response_model=ArtifactDetailResponse)
    async def api_artifact_detail(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
        artifact_id: str,
        include: str | None = Query(
            default=None, description="Comma-separated sections (replicas,view,leaves)"
        ),
        space: str | None = Query(default=None, description="canonical or view"),
        view_id: str | None = Query(default=None),
        leaf_indices: str | None = Query(default=None),
    ) -> ArtifactDetailResponse:
        include_replicas, include_view, include_leaves = _include_flags(include)
        canonical, resolved_view_id = _space_flags(space, view_id)
        indices = _parse_leaf_indices(leaf_indices)
        response = await client.get_artifact_info(
            artifact_id,
            include_replicas=include_replicas,
            include_view=include_view,
            include_leaves=include_leaves,
            space_canonical=canonical,
            view_id=resolved_view_id,
            leaf_indices=indices,
        )
        return ArtifactDetailResponse.from_proto(artifact_id, response)

    @app.get("/api/chunks", response_model=ChunkLocationsResponse)
    async def api_chunk_locations(
        client: Annotated[GlobalStoreClient, Depends(get_client_dependency)],
        artifact_id: str = Query(...),
        chunk_indices: str | None = Query(default=None),
    ) -> ChunkLocationsResponse:
        indices = _parse_chunk_indices(chunk_indices)
        response = await client.query_chunk_locations(artifact_id, indices)
        return ChunkLocationsResponse.from_proto(response)

    @app.get("/api/config", include_in_schema=False)
    async def api_config(request: Request) -> JSONResponse:
        settings: DashboardSettings = request.app.state.settings
        grafana = settings.grafana
        body: dict[str, object] = {
            "grafana": None,
        }
        if grafana is not None:
            body["grafana"] = {
                "host": grafana.host,
                "dashboard_uid": grafana.dashboard_uid,
                "panel_ids": list(grafana.panel_ids),
            }
        return JSONResponse(body)

    return app


# Expose module-level ASGI app for `uv run uvicorn tensorcast.dashboard.api:app`
app = create_app()
