#  Copyright (c) 2025, TensorCast Team.

import os
from pathlib import Path

import yaml
from google.protobuf import json_format as _pb_json

from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import (
    client_config_pb2 as cc_pb2,
    global_store_config_pb2 as gsc_pb2,
)
from tensorcast.global_store.config.settings import GlobalStoreConfig


def test_global_store_config_yaml_example_parses():
    cfg_path = Path("examples/config/global_store_config.yaml")
    assert cfg_path.exists(), f"missing example config: {cfg_path}"
    # Should not raise
    pb = GlobalStoreConfig.load_proto_from_file(str(cfg_path))
    assert isinstance(pb, gsc_pb2.GlobalStoreConfig)
    # And Pydantic conversion
    pyd = GlobalStoreConfig.from_file(str(cfg_path))
    assert pyd.port > 0


def test_enum_alias_normalization_observability():
    data = {
        "observability": {
            "otel": {
                "enabled": True,
                "exporter_otlp_endpoint": "http://127.0.0.1:4317",
                "exporter_protocol": "grpc",
                "service_name": "svc",
            },
            "logging": {"level": "warning"},
        }
    }
    # Normalize and parse GlobalStoreConfig
    normalize_enum_aliases_inplace(data, gsc_pb2.GlobalStoreConfig.DESCRIPTOR)
    pb = gsc_pb2.GlobalStoreConfig()
    _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
    assert (
        pb.observability.otel.exporter_protocol
        == pb.observability.OTelProtocol.O_TEL_PROTOCOL_GRPC
    )
    assert pb.observability.logging.level == pb.observability.LOG_LEVEL_WARN


def test_client_config_enum_aliases():
    data = {
        "observability": {
            "otel": {"enabled": True, "exporter_protocol": "http/protobuf"},
            "logging": {"level": "info"},
        }
    }
    normalize_enum_aliases_inplace(data, cc_pb2.ClientConfig.DESCRIPTOR)
    pb = cc_pb2.ClientConfig()
    _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
    assert (
        pb.observability.otel.exporter_protocol
        == pb.observability.OTelProtocol.O_TEL_PROTOCOL_HTTP_PROTOBUF
    )
    assert pb.observability.logging.level == pb.observability.LOG_LEVEL_INFO

