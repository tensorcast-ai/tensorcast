#  Copyright (c) 2026, TensorCast Team.

"""Node Agent configuration settings (file-based only)."""

from __future__ import annotations

import json
from typing import Any, Optional

import yaml
from google.protobuf import json_format as _pb_json
from pydantic import BaseModel, ConfigDict

from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import node_agent_config_pb2 as nac_pb2


def _dur_ms(dur) -> int:
    return int(dur.seconds * 1000 + dur.nanos // 1_000_000)


def _socket_address(addr) -> str | None:
    host = addr.host if addr is not None else ""
    port = int(addr.port) if addr is not None else 0
    if not host or port <= 0:
        return None
    return f"{host}:{port}"


class NodeAgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    listen_host: str = "127.0.0.1"
    listen_port: int = 0
    daemon_address: str
    daemon_id: str
    instance_id: str
    engine: str
    worker_id: Optional[str] = None
    signals_endpoint: Optional[str] = None
    labels: dict[str, str] = {}
    global_store_endpoints: list[str] = []
    heartbeat_interval_ms: int = 5000
    register_retry_delay_ms: int = 5000
    register_instance: bool = False
    target_ttl_ms: int = 30_000
    capability_secret: Optional[str] = None
    transform_plugins: list[str] = []

    @classmethod
    def from_file(cls, path: str) -> "NodeAgentConfig":
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        pb = nac_pb2.NodeAgentConfig()
        if isinstance(data, dict):
            normalize_enum_aliases_inplace(data, nac_pb2.NodeAgentConfig.DESCRIPTOR)
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)

        listen_host = "127.0.0.1"
        listen_port = 0
        if pb.server.HasField("listen"):
            listen_host = pb.server.listen.host or listen_host
            listen_port = int(pb.server.listen.port or 0)

        daemon_address = _socket_address(pb.daemon_target)
        if not daemon_address:
            raise ValueError("daemon_target must include host and port")

        daemon_id = pb.identity.daemon_id
        instance_id = pb.identity.instance_id
        engine = pb.identity.engine
        if not daemon_id or not instance_id or not engine:
            raise ValueError(
                "identity.daemon_id, identity.instance_id, and identity.engine are required"
            )

        endpoints = [
            addr
            for addr in (_socket_address(ep) for ep in pb.global_store.endpoints)
            if addr
        ]

        heartbeat_ms = (
            _dur_ms(pb.global_store.heartbeat_interval)
            if pb.global_store.HasField("heartbeat_interval")
            else 5000
        )
        retry_ms = (
            _dur_ms(pb.global_store.register_retry_delay)
            if pb.global_store.HasField("register_retry_delay")
            else 5000
        )
        register_instance = (
            bool(pb.global_store.register_instance)
            if pb.global_store.HasField("register_instance")
            else bool(endpoints)
        )

        target_ttl_ms = (
            _dur_ms(pb.engine_adapter.target_ttl)
            if pb.engine_adapter.HasField("target_ttl")
            else 30_000
        )

        return cls(
            listen_host=listen_host,
            listen_port=listen_port,
            daemon_address=daemon_address,
            daemon_id=daemon_id,
            instance_id=instance_id,
            engine=engine,
            worker_id=pb.identity.worker_id or None,
            signals_endpoint=pb.identity.signals_endpoint or None,
            labels=dict(pb.identity.labels) if pb.identity.labels else {},
            global_store_endpoints=endpoints,
            heartbeat_interval_ms=max(0, int(heartbeat_ms)),
            register_retry_delay_ms=max(0, int(retry_ms)),
            register_instance=register_instance,
            target_ttl_ms=max(1, int(target_ttl_ms)),
            capability_secret=pb.engine_adapter.capability_secret or None,
            transform_plugins=list(pb.engine_adapter.transform_plugins),
        )

    @staticmethod
    def load_proto_from_file(path: str) -> nac_pb2.NodeAgentConfig:
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data: Any = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        pb = nac_pb2.NodeAgentConfig()
        if isinstance(data, dict):
            normalize_enum_aliases_inplace(data, nac_pb2.NodeAgentConfig.DESCRIPTOR)
        _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
        return pb


__all__ = ["NodeAgentConfig"]
