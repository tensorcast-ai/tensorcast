#  Copyright (c) 2025, StepCast Team.

"""FastAPI application for Global Store Web UI."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from scstore.global_store.webui_backend.api import api_router
from scstore.global_store.webui_backend.grpc_client import (
    GlobalStoreClient,
    GlobalStoreClientConfig,
    close_global_store_client,
    get_global_store_client,
)
from scstore.global_store.webui_backend.websocket import WebSocketManager

if TYPE_CHECKING:
    from scstore.global_store.config.settings import GlobalStoreConfig

logger = logging.getLogger(__name__)


class WebUIApp:
    """Web UI application for Global Store."""

    def __init__(
        self,
        grpc_config: GlobalStoreClientConfig,
        config: "GlobalStoreConfig",
    ):
        """Initialize Web UI application."""
        self.grpc_config = grpc_config
        self.config = config
        self.grpc_client: GlobalStoreClient | None = None
        self.ws_manager = WebSocketManager()
        self.app = self._create_app()
        self._setup_routes()
        self._setup_middleware()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting Global Store Web UI")
            # ------------------------------------------------------------------
            # Initialize gRPC client.  The Web UI should still start even if the
            # Global Store backend is temporarily unavailable (e.g. starting up
            # a few seconds later or running on a different machine that is not
            # reachable yet).  Therefore we wrap the connection in a try/except
            # block.  On failure we merely log the error and continue with
            # ``self.grpc_client`` set to ``None``.  The WebSocketManager is
            # already prepared to handle a ``None`` client – it will simply
            # skip polling until a valid client is injected later using
            # ``set_grpc_client``.
            # ------------------------------------------------------------------

            try:
                self.grpc_client = await get_global_store_client(self.grpc_config)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Failed to connect to Global Store gRPC server at %s:%d – "
                    "continuing without backend. The Web UI will operate in a "
                    "degraded state until the server becomes reachable. Error: %s",
                    self.grpc_config.host,
                    self.grpc_config.port,
                    exc,
                )
                self.grpc_client = None

            # Start WebSocket background task (polling only works once the
            # gRPC client is available, but we can start the broadcaster right
            # away so that clients can connect).
            self.ws_manager.set_grpc_client(self.grpc_client)
            asyncio.create_task(self.ws_manager.broadcast_updates())
            yield
            # Shutdown
            logger.info("Shutting down Global Store Web UI")
            # Close gRPC client if we managed to connect during startup.
            if self.grpc_client is not None:
                await close_global_store_client()

        return FastAPI(
            title="Global Store Web UI",
            description="Real-time monitoring dashboard for Global Store",
            version="1.0.0",
            lifespan=lifespan,
        )

    def _setup_routes(self):
        """Set up application routes."""
        # Store app instance for dependency injection
        self.app.extra = {"webui_app": self}
        # Include API routes
        self.app.include_router(api_router, prefix="/api")

        # WebSocket endpoint
        @self.app.websocket("/ws/stream")
        async def websocket_endpoint(websocket: WebSocket):
            await self.ws_manager.connect(websocket)
            try:
                while True:
                    data = await websocket.receive_json()
                    # Handle subscription requests
                    if data.get("action") == "subscribe":
                        topics = data.get("topics", [])
                        await self.ws_manager.subscribe(websocket, topics)
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket)

        # Static files - serve frontend build
        static_dir = self._get_static_dir()
        if static_dir and static_dir.exists():
            self.app.mount(
                "/assets", StaticFiles(directory=static_dir / "assets"), name="assets"
            )

            @self.app.get("/", response_class=HTMLResponse)
            @self.app.get("/{path:path}", response_class=HTMLResponse)
            async def serve_spa(path: str = ""):
                """Serve the SPA for all routes."""
                index_file = static_dir / "index.html"
                if index_file.exists():
                    return index_file.read_text()

                # Return 404 with helpful message for missing static resources
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=404,
                    detail="Frontend not built. Please run 'bash tools/build_ui.sh' to build the UI.",
                )

        # Prometheus metrics endpoint (exposes UI process metrics)
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            @self.app.get("/metrics")
            async def metrics_endpoint():
                """Expose Prometheus metrics for the Web UI itself."""
                data = generate_latest()
                from fastapi.responses import Response

                return Response(data, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            # prometheus_client is already a dependency of scstore, but guard just in case
            logger.warning(
                "prometheus_client not available – /metrics endpoint disabled"
            )

    def _setup_middleware(self):
        """Set up middleware."""
        # CORS middleware - configure based on environment
        allowed_origins = ["*"]  # Default for development

        allowed_origins = (
            self.config.ui_cors_origins.split(",")
            if self.config.ui_cors_origins
            else ["*"]
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    def _get_static_dir(self) -> Path | None:
        """Get static directory path."""
        if self.config.ui_static_dir:
            return Path(self.config.ui_static_dir)

        # Default to package directory
        package_dir = Path(__file__).parent / "build"
        if package_dir.exists():
            return package_dir

        return None

    async def get_grpc_client(self) -> GlobalStoreClient:
        """Get gRPC client."""
        if self.grpc_client is None:
            raise RuntimeError("gRPC client not initialized")
        return self.grpc_client

    def notify_update(self, topic: str, payload: dict[str, Any]):
        """Notify WebSocket clients of updates."""
        asyncio.create_task(self.ws_manager.send_update(topic, payload))


# The run_fastapi_app function has been removed since the Web UI
# now runs as a separate process with its own entry point
