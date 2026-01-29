#  Copyright (c) 2025-2026, TensorCast Team.

"""
Proto-based loader for the Store Daemon unified runtime configuration.

Loads YAML/JSON into tensorcast.config.v1.DaemonConfig with strict
validation and cross-language normalization for enum aliases, durations,
and byte-size fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml
from google.protobuf.json_format import MessageToDict, MessageToJson, ParseDict

from tensorcast.common.config.normalize import normalize_enum_aliases_inplace
from tensorcast.proto.config.v1 import daemon_config_pb2 as cfg_pb

_DURATION_SUFFIXES = {
    "ms": 1 / 1000.0,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}

_BYTE_SUFFIXES = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
}


def _to_seconds_string(value: Any) -> str:
    """Normalize various duration inputs to protobuf JSON seconds string.

    Accepts:
    - numeric (int/float): treated as seconds
    - strings with ms/s/m/h suffixes (e.g., "500ms", "2s", "3m", "1h")
    - canonical protobuf strings like "1.5s" (returned as-is)
    """
    if value is None:
        return "0s"
    if isinstance(value, (int, float)):
        return f"{float(value)}s" if isinstance(value, float) else f"{int(value)}s"
    if not isinstance(value, str):
        raise ValueError(f"Unsupported duration value type: {type(value)}")

    s = value.strip().lower()
    # Already canonical (ends with 's' and contains only digits/dot)
    if re.fullmatch(r"\d+(?:\.\d+)?s", s):
        return s

    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*(ms|s|m|h)\s*", s)
    if not m:
        # Try bare number => seconds
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return f"{s}s"
        raise ValueError(f"Invalid duration literal: {value}")

    num = float(m.group(1))
    unit = m.group(2)
    seconds = num * _DURATION_SUFFIXES[unit]
    # Use minimal representation: strip trailing .0
    return f"{seconds if seconds % 1 else int(seconds)}s"


def _to_num_bytes(value: Any) -> int:
    """Normalize humanized byte size to integer bytes.

    Accepts integers or strings like "256MB", "8gb", "1024".
    """
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"Unsupported bytes value type: {type(value)}")

    s = value.strip().lower()
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*([a-z]*)\s*", s)
    if not m:
        raise ValueError(f"Invalid bytes literal: {value}")
    num = float(m.group(1))
    unit = m.group(2) or "b"
    if unit not in _BYTE_SUFFIXES:
        raise ValueError(f"Unknown byte unit: {unit}")
    return int(num * _BYTE_SUFFIXES[unit])


def _validate_unknown_keys(tree: Any, desc, path: str = "") -> list[str]:
    """Recursively validate there are no unknown keys against a descriptor.

    Returns list of dotted paths for unknown keys.
    """
    from google.protobuf.descriptor import FieldDescriptor

    errs: list[str] = []
    if tree is None:
        return errs
    if isinstance(tree, dict):
        field_map = {f.name: f for f in desc.fields}
        for k, v in tree.items():
            f = field_map.get(k)
            if f is None:
                errs.append(f"{path}.{k}" if path else k)
                continue
            if f.type == FieldDescriptor.TYPE_MESSAGE:
                if f.label == FieldDescriptor.LABEL_REPEATED:
                    if isinstance(v, list):
                        for i, item in enumerate(v):
                            errs.extend(
                                _validate_unknown_keys(
                                    item,
                                    f.message_type,
                                    f"{path}.{k}[{i}]" if path else f"{k}[{i}]",
                                )
                            )
                else:
                    errs.extend(
                        _validate_unknown_keys(
                            v, f.message_type, f"{path}.{k}" if path else k
                        )
                    )
    elif isinstance(tree, list):
        for i, item in enumerate(tree):
            errs.extend(_validate_unknown_keys(item, desc, f"{path}[{i}]"))
    return errs


def _normalize_units_inplace(data: Any, desc) -> None:
    """Normalize durations and *_bytes fields according to the descriptor."""
    from google.protobuf.descriptor import FieldDescriptor

    if data is None:
        return
    if isinstance(data, dict):
        field_map = {f.name: f for f in desc.fields}
        for k, v in list(data.items()):
            f = field_map.get(k)
            if f is None:
                continue
            if f.type == FieldDescriptor.TYPE_MESSAGE:
                # Duration special-case
                if f.message_type.full_name == "google.protobuf.Duration":
                    # Convert to canonical JSON string (e.g., "120s"); support repeated if needed
                    if f.label == FieldDescriptor.LABEL_REPEATED and isinstance(
                        v, list
                    ):
                        data[k] = [_to_seconds_string(item) for item in v]
                    else:
                        data[k] = _to_seconds_string(v)
                else:
                    if f.label == FieldDescriptor.LABEL_REPEATED:
                        if isinstance(v, list):
                            for item in v:
                                _normalize_units_inplace(item, f.message_type)
                    else:
                        _normalize_units_inplace(v, f.message_type)
            else:
                # Byte-size fields by naming convention *_bytes
                if f.type in (
                    FieldDescriptor.TYPE_UINT32,
                    FieldDescriptor.TYPE_UINT64,
                ) and k.endswith("_bytes"):
                    data[k] = _to_num_bytes(v)
    elif isinstance(data, list):
        for item in data:
            _normalize_units_inplace(item, desc)


def _normalize_defaults_inplace(msg: cfg_pb.DaemonConfig) -> None:
    engine = msg.engine

    if engine.HasField("memory_tiers"):
        mt = engine.memory_tiers
        if mt.preemptible_low_watermark_ratio <= 0:
            mt.preemptible_low_watermark_ratio = 0.4

    bm = engine.byte_mapping
    if not bm.HasField("enable_strided_execution"):
        bm.enable_strided_execution = True
    if not bm.HasField("enable_direct_write_at"):
        bm.enable_direct_write_at = True
    if bm.program_cache_entries == 0:
        bm.program_cache_entries = 256
    if bm.strided_run_min_ranges == 0:
        bm.strided_run_min_ranges = 128
    if bm.strided_min_row_len_bytes == 0:
        bm.strided_min_row_len_bytes = 4096
    if bm.strided_max_amplification == 0:
        bm.strided_max_amplification = 8
    if bm.strided_block_target_bytes == 0:
        bm.strided_block_target_bytes = 16 * 1024 * 1024
    if bm.strided_block_max_bytes == 0:
        bm.strided_block_max_bytes = 64 * 1024 * 1024


def _parse_override_value(value: str) -> Any:
    if value == "":
        return ""
    parsed = yaml.safe_load(value)
    if parsed is None:
        raise ValueError(
            "Override value cannot be null; use an explicit empty string or []"
        )
    return parsed


def _parse_override_path(raw_key: str) -> list[str]:
    key = raw_key.strip()
    if not key:
        raise ValueError("Override key cannot be empty")
    if key.startswith(".") or key.endswith(".") or ".." in key:
        raise ValueError(f"Invalid override key '{raw_key}'")
    parts = key.split(".")
    if any(not part for part in parts):
        raise ValueError(f"Invalid override key '{raw_key}'")
    return parts


def _validate_override_path(path: list[str], desc) -> None:
    from google.protobuf.descriptor import FieldDescriptor

    current = desc
    for idx, part in enumerate(path):
        field = current.fields_by_name.get(part)
        if field is None:
            raise ValueError(f"Unknown config key '{'.'.join(path[: idx + 1])}'")
        if idx < len(path) - 1:
            if field.type != FieldDescriptor.TYPE_MESSAGE:
                raise ValueError(
                    f"Config key '{'.'.join(path[: idx + 1])}' is not a message; "
                    f"cannot set subfield '{path[idx + 1]}'"
                )
            if (
                field.label == FieldDescriptor.LABEL_REPEATED
                and not field.message_type.GetOptions().map_entry
            ):
                raise ValueError(
                    f"Config key '{'.'.join(path[: idx + 1])}' is repeated; "
                    "set the full list instead of subfields"
                )
            if field.message_type.GetOptions().map_entry:
                raise ValueError(
                    f"Config key '{'.'.join(path[: idx + 1])}' is a map; "
                    "set the full mapping instead of subfields"
                )
            if (
                field.label == FieldDescriptor.LABEL_REPEATED
                and field.message_type.GetOptions().map_entry
            ):
                raise ValueError(
                    f"Config key '{'.'.join(path[: idx + 1])}' is a map; "
                    "set the full mapping instead of subfields"
                )
            current = field.message_type


def _apply_override(tree: dict[str, Any], path: list[str], value: Any) -> None:
    node: dict[str, Any] = tree
    for part in path[:-1]:
        existing = node.get(part)
        if existing is None:
            node[part] = {}
            node = node[part]
        elif isinstance(existing, dict):
            node = existing
        else:
            raise ValueError(
                f"Override path '{'.'.join(path)}' conflicts with earlier value at '{part}'"
            )
    leaf = path[-1]
    if leaf in node:
        raise ValueError(f"Duplicate override for '{'.'.join(path)}'")
    node[leaf] = value


def _build_override_tree(overrides: Iterable[str], desc) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for raw in overrides:
        if raw is None:
            continue
        if "=" not in raw:
            raise ValueError(f"Override '{raw}' must be in key=value form")
        key, value = raw.split("=", 1)
        path = _parse_override_path(key)
        _validate_override_path(path, desc)
        parsed_value = _parse_override_value(value)
        _apply_override(tree, path, parsed_value)
    return tree


def apply_daemon_config_overrides(
    msg: cfg_pb.DaemonConfig, overrides: Iterable[str] | None
) -> None:
    if not overrides:
        return
    overlay = _build_override_tree(overrides, cfg_pb.DaemonConfig.DESCRIPTOR)
    if not overlay:
        return
    normalize_enum_aliases_inplace(overlay, cfg_pb.DaemonConfig.DESCRIPTOR)
    _normalize_units_inplace(overlay, cfg_pb.DaemonConfig.DESCRIPTOR)
    ParseDict(overlay, msg, ignore_unknown_fields=False)
    _normalize_defaults_inplace(msg)


def load_daemon_config(path: str | Path) -> cfg_pb.DaemonConfig:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw: Any = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Validate keys early for precise error messages
    errs = _validate_unknown_keys(raw, cfg_pb.DaemonConfig.DESCRIPTOR)
    if errs:
        raise ValueError(f"Unknown config keys: {', '.join(errs)}")

    # Normalize enums, durations, and byte sizes
    normalize_enum_aliases_inplace(raw, cfg_pb.DaemonConfig.DESCRIPTOR)
    _normalize_units_inplace(raw, cfg_pb.DaemonConfig.DESCRIPTOR)

    msg = cfg_pb.DaemonConfig()
    ParseDict(raw, msg, ignore_unknown_fields=False)
    _normalize_defaults_inplace(msg)
    return msg


def dump_daemon_config(msg: cfg_pb.DaemonConfig, path: str | Path) -> None:
    """Write DaemonConfig to YAML or JSON file based on extension.

    - .yaml/.yml: dumps snake_case keys using MessageToDict(preserving_proto_field_name=True)
    - otherwise: writes canonical protobuf JSON via MessageToJson
    """
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = out_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = MessageToDict(
            msg,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        with open(out_path, "w", encoding="utf-8") as fp:
            yaml.safe_dump(data, fp, sort_keys=False)
        return

    # Default: JSON
    json_str = MessageToJson(msg, always_print_fields_with_no_presence=True)
    out_path.write_text(json_str, encoding="utf-8")
