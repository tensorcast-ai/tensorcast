#  Copyright (c) 2025, TensorCast Team.

"""
Minimal HTTP server exposing StoreDaemon metrics in OpenMetrics text format.

This runs as a small sidecar next to the C++ StoreDaemon. It queries the
daemon via gRPC `GetDetailedStatus` and converts the response to `tc_*`
Prometheus metrics so that an OpenTelemetry Collector (Prometheus receiver)
can scrape them.

Usage:
  uv run -m tensorcast.daemon_metrics_http --port 9091 --daemon-addr 127.0.0.1:8073
"""

from __future__ import annotations

import argparse
import logging
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import grpc

from tensorcast.proto import store_daemon_pb2, store_daemon_pb2_grpc

LOG = logging.getLogger("tensorcast.daemon_metrics_http")


def _escape_label_value(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _compose_metrics_from_status(
    status: store_daemon_pb2.GetDetailedStatusResponse,
) -> bytes:
    lines: list[str] = []

    # Types
    lines.append("# TYPE tc_memory_pool_bytes gauge")
    lines.append("# TYPE tc_p2p_bytes_total counter")

    mp = status.memory_pool_info
    # CPU memory pool
    if mp.total_size_bytes:
        lines.append(
            f'tc_memory_pool_bytes{{location="cpu",memory_type="total"}} {mp.total_size_bytes}'
        )
    lines.append(
        f'tc_memory_pool_bytes{{location="cpu",memory_type="available"}} {mp.available_bytes}'
    )

    # GPU memory per device
    for gpu in status.gpu_devices:
        did = _escape_label_value(str(gpu.device_id))
        lines.append(
            f'tc_memory_pool_bytes{{location="gpu",device_id="{did}",memory_type="total"}} {gpu.total_memory_bytes}'
        )
        lines.append(
            f'tc_memory_pool_bytes{{location="gpu",device_id="{did}",memory_type="free"}} {gpu.free_memory_bytes}'
        )

    # P2P throughput
    ci = status.communication_info
    lines.append(f"tc_p2p_bytes_total {ci.total_bytes_transferred}")

    lines.append("# EOF")
    return ("\n".join(lines)).encode("utf-8")


class _MetricsHandler(BaseHTTPRequestHandler):
    daemon_addr: str = "127.0.0.1:8073"

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        try:
            channel = grpc.insecure_channel(self.daemon_addr)
            stub = store_daemon_pb2_grpc.StoreDaemonStub(channel)
            req = store_daemon_pb2.GetDetailedStatusRequest()
            status = stub.GetDetailedStatus(req, timeout=1.5)
            body = _compose_metrics_from_status(status)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Failed to query daemon status: %s", exc)
            body = (
                b"# TYPE tc_daemon_metrics_errors_total counter\n"
                b'tc_daemon_metrics_errors_total{reason="rpc_failure"} 1\n'
                b"# EOF\n"
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        with suppress(Exception):  # pragma: no cover - best-effort
            self.wfile.write(body)

    # Silence default logging
    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - shadow built-in
        LOG.debug("%s - %s", self.address_string(), format % args)


def serve(
    port: int, host: str = "0.0.0.0", daemon_addr: str = "127.0.0.1:8073"
) -> None:
    LOG.info("Starting daemon metrics HTTP server on %s:%d", host, port)
    httpd: Optional[HTTPServer] = None
    try:
        _MetricsHandler.daemon_addr = daemon_addr
        httpd = HTTPServer((host, port), _MetricsHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual exit
        pass
    finally:
        if httpd is not None:
            with suppress(Exception):
                httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TensorCast Daemon metrics HTTP server"
    )
    parser.add_argument("--port", type=int, default=9091, help="HTTP port for /metrics")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    parser.add_argument(
        "--daemon-addr",
        type=str,
        required=True,
        help="StoreDaemon gRPC address, e.g. 127.0.0.1:8073",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    serve(args.port, args.host, args.daemon_addr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
