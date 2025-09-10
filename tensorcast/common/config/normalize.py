#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import re
from typing import Any, Iterable

from google.protobuf.descriptor import Descriptor, EnumDescriptor, FieldDescriptor


def _longest_common_prefix(parts: Iterable[str]) -> str:
    parts = list(parts)
    if not parts:
        return ""
    s1 = min(parts)
    s2 = max(parts)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


def _normalize_enum_string(enum: EnumDescriptor, value: str) -> str | int:
    """Map user-friendly strings to canonical protobuf JSON enum names.

    Examples:
    - grpc -> O_TEL_PROTOCOL_GRPC
    - http/protobuf -> O_TEL_PROTOCOL_HTTP_PROTOBUF
    - info -> LOG_LEVEL_INFO
    - WARNING -> LOG_LEVEL_WARN (synonym)

    If mapping fails, returns the original string unchanged. Numeric strings
    (e.g., "1") are converted to int to let the parser accept them.
    """
    if not isinstance(value, str):
        return value

    raw = value
    s = value.strip()
    if s == "":
        return raw

    # Allow numeric form
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            return raw

    names = [v.name for v in enum.values]
    name_set = set(names)
    # Normalize: uppercase and collapse non-alnum to underscore
    norm = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()

    # Direct match
    if norm in name_set:
        return norm

    # Common prefix heuristic (e.g., LOG_LEVEL_, O_TEL_PROTOCOL_)
    prefix = _longest_common_prefix(names)
    if prefix:
        candidate = f"{prefix}{norm}"
        if candidate in name_set:
            return candidate

        # Handle common synonyms implicitly (WARNING -> WARN)
        synonyms = {"WARNING": "WARN"}
        alt = synonyms.get(norm)
        if alt:
            alt_candidate = f"{prefix}{alt}"
            if alt_candidate in name_set:
                return alt_candidate

    # Suffix match as a last resort (handles prefixes we didn't infer cleanly)
    for n in names:
        if n.endswith("_" + norm) or n == norm:
            return n

    return raw


def normalize_enum_aliases_inplace(data: Any, message: Descriptor) -> None:
    """Recursively normalize enum fields within a dict according to a message descriptor.

    Mutates `data` (dict/list) in place, rewriting string enum aliases into the
    canonical protobuf JSON names or integer values accepted by ParseDict.
    """
    if data is None:
        return

    if isinstance(data, dict):
        # Build a mapping of field name -> FieldDescriptor for quick lookup
        field_map: dict[str, FieldDescriptor] = {f.name: f for f in message.fields}
        for key, val in list(data.items()):
            field = field_map.get(key)
            if field is None:
                # Unknown key at this level; leave as-is (strict parsing will catch)
                continue

            if field.type == FieldDescriptor.TYPE_MESSAGE:
                if field.label == FieldDescriptor.LABEL_REPEATED:
                    if isinstance(val, list):
                        for item in val:
                            normalize_enum_aliases_inplace(item, field.message_type)
                else:
                    normalize_enum_aliases_inplace(val, field.message_type)

            elif field.type == FieldDescriptor.TYPE_ENUM:
                if field.label == FieldDescriptor.LABEL_REPEATED:
                    if isinstance(val, list):
                        for i, item in enumerate(val):
                            if isinstance(item, str):
                                data[key][i] = _normalize_enum_string(
                                    field.enum_type, item
                                )
                else:
                    if isinstance(val, str):
                        data[key] = _normalize_enum_string(field.enum_type, val)

    elif isinstance(data, list):
        # Repeated of messages at root (rare)
        for item in data:
            normalize_enum_aliases_inplace(item, message)
