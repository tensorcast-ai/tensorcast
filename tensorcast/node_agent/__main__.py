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
from tensorcast.engine_adapter import EngineAdapter
from tensorcast.logger import init_logger
from tensorcast.node_agent.config import NodeAgentConfig
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.node_agent.server import add_servicer_to_server
from tensorcast.observability.otel import (
    setup_otel_from_observability as _setup_otel_from_observability,
)
from tensorcast.proto.global_store.v1 import global_store_pb2, global_store_pb2_grpc

logger = init_logger(__name__)


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


def _resolve_worker_id(
    stub: global_store_pb2_grpc.GlobalStoreServiceStub, daemon_id: str
) -> str | None:
    resp = stub.ListActiveWorkers(global_store_pb2.ListActiveWorkersRequest())
    for worker in resp.workers:
        if worker.daemon_id == daemon_id:
            return worker.worker_id
    return None


def _register_instance(
    stub: global_store_pb2_grpc.GlobalStoreServiceStub,
    *,
    config: NodeAgentConfig,
    daemon_id: str,
    worker_id: str | None,
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
    )
    if worker_id:
        req.worker_id = worker_id
    resp = stub.RegisterInstance(req)
    if resp.status != global_store_pb2.Status.STATUS_OK:
        raise RuntimeError("RegisterInstance failed")


def _heartbeat_loop(
    *,
    stub: global_store_pb2_grpc.GlobalStoreServiceStub,
    config: NodeAgentConfig,
    daemon_id: str,
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        worker_id = config.worker_id or _resolve_worker_id(stub, daemon_id)
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
    stub: global_store_pb2_grpc.GlobalStoreServiceStub | None = None
    if config.register_instance and config.global_store_endpoints:
        target = config.global_store_endpoints[0]
        channel = grpc.insecure_channel(target)
        stub = global_store_pb2_grpc.GlobalStoreServiceStub(channel)
        worker_id = config.worker_id or _resolve_worker_id(stub, config.daemon_id)
        _register_instance(
            stub, config=config, daemon_id=config.daemon_id, worker_id=worker_id
        )
        hb_thread = threading.Thread(
            target=_heartbeat_loop,
            kwargs={
                "stub": stub,
                "config": config,
                "daemon_id": config.daemon_id,
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
        server.stop(grace=2)


if __name__ == "__main__":
    main()
