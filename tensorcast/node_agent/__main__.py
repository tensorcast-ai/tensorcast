#  Copyright (c) 2026, TensorCast Team.

"""Node Agent entrypoint (gRPC service)."""

from __future__ import annotations

import argparse
import importlib
import threading
import time
from concurrent import futures

import grpc

from tensorcast import __version__ as _tc_version
from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.engine_adapter import EngineAdapter
from tensorcast.global_store.composite_stub import GlobalStoreCompositeStub
from tensorcast.logger import init_logger
from tensorcast.node_agent.config import NodeAgentConfig
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.node_agent.server import add_servicer_to_server
from tensorcast.observability.otel import (
    setup_otel_from_observability as _setup_otel_from_observability,
)
from tensorcast.proto.global_store.v1 import global_store_pb2

logger = init_logger(__name__)

_NODE_AGENT_HOST_KIND = "node_agent_grpc"
_WILDCARD_LISTEN_HOSTS = {"", "0.0.0.0", "::", "[::]"}


def _load_transform_plugins(adapter: EngineAdapter, paths: list[str]) -> None:
    for path in paths:
        module = importlib.import_module(path)
        register = getattr(module, "register", None) or getattr(
            module, "register_transforms", None
        )
        if not callable(register):
            raise RuntimeError(
                f"Transform plugin '{path}' must expose register(engine_adapter)"
            )
        register(adapter)


def _resolve_worker_id_from_daemon(client: DaemonCtl, *, daemon_id: str) -> str | None:
    response = client.get_worker_status()
    resolved_daemon_id = str(getattr(response, "daemon_id", "") or "").strip()
    if resolved_daemon_id and resolved_daemon_id != str(daemon_id).strip():
        raise RuntimeError(
            "Connected daemon identity does not match node agent config: "
            f"expected={daemon_id} actual={resolved_daemon_id}"
        )
    worker_id = str(getattr(response, "worker_id", "") or "").strip()
    return worker_id or None


def _register_instance(
    stub: GlobalStoreCompositeStub,
    *,
    config: NodeAgentConfig,
    daemon_id: str,
    worker_id: str | None,
    execution_endpoint: str,
) -> None:
    capability_flags = 0
    capability_flags |= (
        1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED
    )
    if config.signals_endpoint:
        capability_flags |= (
            1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED
        )
    req = global_store_pb2.RegisterInstanceRequest(
        instance_id=config.instance_id,
        daemon_id=daemon_id,
        engine=config.engine,
        signals_endpoint=config.signals_endpoint or "",
        labels=dict(config.labels),
        capability_flags=capability_flags,
        execution_endpoint=execution_endpoint,
        execution_host_kind=_NODE_AGENT_HOST_KIND,
    )
    if worker_id:
        req.worker_id = worker_id
    resp = stub.RegisterInstance(req)
    if resp.status != global_store_pb2.Status.STATUS_OK:
        raise RuntimeError("RegisterInstance failed")


def _resolve_execution_endpoint(config: NodeAgentConfig, bound_port: int) -> str:
    if config.execution_endpoint:
        return config.execution_endpoint
    if config.listen_host in _WILDCARD_LISTEN_HOSTS:
        raise RuntimeError(
            "Node Agent instance registration requires identity.execution_endpoint "
            "when server.listen.host is wildcard"
        )
    return f"{config.listen_host}:{bound_port}"


def _heartbeat_loop(
    *,
    stub: GlobalStoreCompositeStub,
    config: NodeAgentConfig,
    daemon_client: DaemonCtl,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        worker_id = config.worker_id or _resolve_worker_id_from_daemon(
            daemon_client, daemon_id=config.daemon_id
        )
        capability_flags = 0
        capability_flags |= (
            1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_NODE_AGENT_ENABLED
        )
        if config.signals_endpoint:
            capability_flags |= (
                1 << global_store_pb2.INSTANCE_CAPABILITY_FLAG_EXECUTION_SIGNALS_ENABLED
            )
        req = global_store_pb2.InstanceHeartbeatRequest(
            instance_id=config.instance_id,
            capability_flags=capability_flags,
        )
        if worker_id:
            req.worker_id = worker_id
        stub.InstanceHeartbeat(req)
        stop_event.wait(config.heartbeat_interval_ms / 1000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorCast Node Agent")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Node Agent config (YAML/JSON)",
    )
    args = parser.parse_args()

    config = NodeAgentConfig.from_file(args.config)
    pb_cfg = NodeAgentConfig.load_proto_from_file(args.config)

    try:
        _setup_otel_from_observability(pb_cfg.observability, role="node-agent")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize OpenTelemetry: %s", exc)

    secret = (
        config.capability_secret.encode("utf-8") if config.capability_secret else None
    )
    adapter = EngineAdapter(
        instance_id=config.instance_id,
        engine=config.engine,
        capability_secret=secret,
        default_target_ttl_ms=config.target_ttl_ms,
    )
    if config.transform_plugins:
        _load_transform_plugins(adapter, config.transform_plugins)

    executor = NodeAgentExecutor(
        daemon_id=config.daemon_id,
        daemon_address=config.daemon_address,
        instance_id=config.instance_id,
        version=_tc_version,
        engine_adapter=adapter,
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    add_servicer_to_server(server, executor)
    bind_addr = f"{config.listen_host}:{config.listen_port}"
    bound_port = server.add_insecure_port(bind_addr)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind Node Agent on {bind_addr}")
    server.start()
    logger.info(
        "Node Agent started on %s:%d (instance_id=%s)",
        config.listen_host,
        bound_port,
        config.instance_id,
    )

    stop_event = threading.Event()
    hb_thread: threading.Thread | None = None
    stub: GlobalStoreCompositeStub | None = None
    daemon_client: DaemonCtl | None = None
    if config.register_instance and config.global_store_endpoints:
        target = config.global_store_endpoints[0]
        channel = grpc.insecure_channel(target)
        stub = GlobalStoreCompositeStub(channel)
        daemon_client = DaemonCtl(config.daemon_address)
        worker_id = config.worker_id or _resolve_worker_id_from_daemon(
            daemon_client, daemon_id=config.daemon_id
        )
        execution_endpoint = _resolve_execution_endpoint(config, bound_port)
        _register_instance(
            stub,
            config=config,
            daemon_id=config.daemon_id,
            worker_id=worker_id,
            execution_endpoint=execution_endpoint,
        )
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stub": stub,
                "config": config,
                "daemon_client": daemon_client,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        hb_thread.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutting down Node Agent")
    finally:
        stop_event.set()
        if hb_thread is not None:
            hb_thread.join(timeout=2.0)
        if stub is not None and config.register_instance:
            try:
                stub.UnregisterInstance(
                    global_store_pb2.UnregisterInstanceRequest(
                        instance_id=config.instance_id, is_graceful_shutdown=True
                    )
                )
            except Exception:
                logger.exception("Failed to unregister instance")
        if daemon_client is not None:
            daemon_client.close()
        server.stop(grace=2)


if __name__ == "__main__":
    main()
