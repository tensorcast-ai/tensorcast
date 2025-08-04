#  Copyright (c) 2025, StepCast Team.

"""Health check service for StoreDaemon."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from scstore.logger import init_logger

logger = init_logger(__name__)


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP health check handler."""

    def __init__(self, servicer, *args, **kwargs):
        self.servicer = servicer
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET request."""
        try:
            if self.path == "/health":
                self._handle_health_check()
            elif self.path == "/ready":
                self._handle_readiness_check()
            elif self.path == "/status":
                self._handle_status_check()
            else:
                self._send_response(404, {"error": "Not found"})
        except Exception as e:
            logger.exception(f"Health check error: {e}")
            self._send_response(500, {"error": "Internal server error"})

    def _handle_health_check(self):
        """Basic health check."""
        health_status = {
            "status": "healthy" if not self.servicer.shutting_down else "unhealthy",
            "timestamp": time.time(),
            "uptime_seconds": int(time.time() - self.servicer.start_time),
        }

        status_code = 200 if not self.servicer.shutting_down else 503
        self._send_response(status_code, health_status)

    def _handle_readiness_check(self):
        """Readiness check."""
        is_ready = (
            not self.servicer.shutting_down
            and self.servicer.checkpoint_store is not None
        )

        readiness_status = {
            "ready": is_ready,
            "timestamp": time.time(),
            "details": {
                "shutting_down": self.servicer.shutting_down,
                "checkpoint_store_initialized": self.servicer.checkpoint_store
                is not None,
                "worker_registered": bool(self.servicer.worker_id)
                if self.servicer.global_store_enabled
                else True,
            },
        }

        status_code = 200 if is_ready else 503
        self._send_response(status_code, readiness_status)

    def _handle_status_check(self):
        """Detailed status check."""
        try:
            # Get memory info
            mem_total = self.servicer.checkpoint_store.get_mem_pool_size()
            mem_available = self.servicer.checkpoint_store.get_available_memory()
            mem_used = mem_total - mem_available

            status_info = {
                "worker": {
                    "id": self.servicer.worker_id or "unregistered",
                    "registered": bool(self.servicer.worker_id),
                    "healthy": not self.servicer.shutting_down,
                    "shutting_down": self.servicer.shutting_down,
                    "uptime_seconds": int(time.time() - self.servicer.start_time),
                },
                "memory": {
                    "pool_total_bytes": mem_total,
                    "pool_available_bytes": mem_available,
                    "pool_used_bytes": mem_used,
                    "pool_usage_percent": round((mem_used / mem_total) * 100, 2)
                    if mem_total > 0
                    else 0,
                },
                "operations": {"active_count": self.servicer.active_operations.get()},
                "network": {
                    "node_id": self.servicer.node_id,
                    "node_address": self.servicer.node_address,
                    "grpc_port": self.servicer.grpc_port,
                    "p2p_port": self.servicer.node_port,
                },
                "config": {
                    "enable_p2p_engine": self.servicer.enable_p2p_engine,
                    "global_store_enabled": self.servicer.global_store_enabled,
                    "enable_p2p_access": self.servicer.enable_p2p_access,
                },
                "timestamp": time.time(),
            }

            self._send_response(200, status_info)

        except Exception as e:
            logger.exception(f"Error getting detailed status: {e}")
            self._send_response(500, {"error": str(e)})

    def _send_response(self, status_code: int, data: dict):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response_data = json.dumps(data, indent=2)
        self.wfile.write(response_data.encode("utf-8"))

    def log_message(self, format, *args):
        """Disable default log output."""
        pass


class HealthCheckServer:
    """Health check HTTP server."""

    def __init__(self, servicer, port: int = 8080):
        self.servicer = servicer
        self.port = port
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self):
        """Start health check server."""
        try:
            # Create handler class, bind servicer
            def handler_class(*args, **kwargs):
                return HealthCheckHandler(self.servicer, *args, **kwargs)

            self.httpd = HTTPServer(("0.0.0.0", self.port), handler_class)
            self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.thread.start()

            logger.info(f"Health check server started on port {self.port}")
            logger.info("Health check endpoints:")
            logger.info("  - GET /health  - Basic health status")
            logger.info("  - GET /ready   - Readiness probe")
            logger.info("  - GET /status  - Detailed status")

        except Exception as e:
            logger.error(f"Failed to start health check server: {e}")

    def stop(self):
        """Stop health check server."""
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            logger.info("Health check server stopped")
