#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral semantic validation helpers for serving recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast


def evaluate_semantic_validation_spec(spec: Any, actual_payload: Any) -> Any:
    if spec.kind == "none":
        return None
    actual = _jsonable(actual_payload)
    if spec.kind == "framework_semantic_probes":
        return actual
    if spec.kind == "explicit":
        expected = _jsonable(spec.payload)
        if actual != expected:
            raise RuntimeError(
                "TensorCast semantic validation failed for explicit probe "
                f"spec: expected={expected!r}, actual={actual!r}"
            )
        return actual
    raise RuntimeError(
        f"Unsupported TensorCast semantic validation spec kind: {spec.kind!r}"
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonable(item)
            for key, item in dataclasses.asdict(cast(Any, value)).items()
        }
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


__all__ = ["evaluate_semantic_validation_spec"]
