#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
from typing import Any

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

from google.protobuf import json_format as _pb_json

from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import client_config_pb2 as cc_pb2


def load_client_config(path: str) -> cc_pb2.ClientConfig:
    """Load ClientConfig proto from YAML or JSON (strict, unknown keys rejected)."""
    if path.endswith(".yaml") or path.endswith(".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs")
        with open(path, "r", encoding="utf-8") as f:
            data: Any = yaml.safe_load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    pb = cc_pb2.ClientConfig()
    if isinstance(data, dict):
        normalize_enum_aliases_inplace(data, cc_pb2.ClientConfig.DESCRIPTOR)
    _pb_json.ParseDict(data, pb, ignore_unknown_fields=False)
    return pb
