#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None

from google.protobuf import json_format as _pb_json

from tensorcast.cli_utils.paths import home_dir
from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import client_config_pb2 as cc_pb2


def discover_client_config() -> Path | None:
    """Locate a default ClientConfig file.

    Order:
    1) $TENSORCAST_CLIENT_CONFIG
    2) ~/.tensorcast/config/client.(yaml|yml|json) or client_config.(yaml|yml|json)
    """

    env = os.environ.get("TENSORCAST_CLIENT_CONFIG")
    if env:
        candidate = Path(env).expanduser()
        if candidate.exists():
            return candidate

    cfg_dir = home_dir() / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "client.yaml",
        "client.yml",
        "client.json",
        "client_config.yaml",
        "client_config.yml",
        "client_config.json",
    ):
        candidate = cfg_dir / name
        if candidate.exists():
            return candidate
    return None


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
