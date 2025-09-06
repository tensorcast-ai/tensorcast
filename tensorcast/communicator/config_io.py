#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from google.protobuf import json_format

from tensorcast.proto.communicator.v1 import communicator_config_pb2 as pb


def _normalize_defaults(cfg: pb.CommunicatorConfig) -> None:
    # Stager
    if cfg.stager.stage_chunk_mb_cpu <= 0:
        cfg.stager.stage_chunk_mb_cpu = 4
    if cfg.stager.stage_chunk_mb_gpu <= 0:
        cfg.stager.stage_chunk_mb_gpu = 16
    if cfg.stager.buffers_per_flow <= 0:
        cfg.stager.buffers_per_flow = 4

    # RDMA
    if cfg.rdma.outstanding_wr <= 0:
        cfg.rdma.outstanding_wr = 64
    if cfg.rdma.ack_ttl_ms <= 0:
        cfg.rdma.ack_ttl_ms = 30000
    if cfg.rdma.traffic_class == 0:
        cfg.rdma.traffic_class = 186
    if cfg.rdma.qp_timeout <= 0:
        cfg.rdma.qp_timeout = 20
    if cfg.rdma.qp_retry <= 0:
        cfg.rdma.qp_retry = 7

    # Pool
    if cfg.pool.pool_size_bytes == 0:
        cfg.pool.pool_size_bytes = 8 * 1024 * 1024 * 1024
    if cfg.pool.chunk_bytes == 0:
        cfg.pool.chunk_bytes = 64 * 1024 * 1024

    # Transport
    if cfg.transport.tcp_conn_count <= 0:
        cfg.transport.tcp_conn_count = 8
    if cfg.transport.tcp_tos < 0:
        cfg.transport.tcp_tos = 0
    if cfg.transport.connect_timeout_sec <= 0:
        cfg.transport.connect_timeout_sec = 10


def from_yaml(path: str | Path) -> pb.CommunicatorConfig:
    """Load communicator config from YAML/JSON into a protobuf message.

    This is a light convenience wrapper for Python users. The Store Daemon
    and C++ core use the C++ loader directly.
    """
    p = Path(path)
    content = p.read_text()
    data: Any
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(content) or {}
    else:
        data = json.loads(content)

    # Normalize boolean defaults that depend on presence
    st = data.setdefault("stager", {})
    st.setdefault("stage_cpu_for_rdma", True)
    pl = data.setdefault("pool", {})
    pl.setdefault("preregister_mr", True)

    msg = pb.CommunicatorConfig()
    json_format.Parse(json.dumps(data), msg)
    _normalize_defaults(msg)
    return msg


def to_yaml(cfg: pb.CommunicatorConfig) -> str:
    """Serialize a CommunicatorConfig message to YAML text."""
    data = json_format.MessageToDict(cfg, preserving_proto_field_name=True)
    return yaml.safe_dump(data, sort_keys=False)
