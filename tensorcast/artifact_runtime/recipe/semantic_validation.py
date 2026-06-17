#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral semantic validation helpers for runtime recipes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, cast


def evaluate_semantic_validation_spec(spec: Any, actual_payload: Any) -> Any:
    if spec.kind == "none":
        return None
    actual = _jsonable(actual_payload)
    if spec.kind == "framework_semantic_probes":
        return _compare_semantic_payload(
            label="framework probe",
            expected=_jsonable(spec.payload),
            actual=actual,
        )
    if spec.kind == "explicit":
        return _compare_semantic_payload(
            label="explicit probe",
            expected=_jsonable(spec.payload),
            actual=actual,
        )
    raise RuntimeError(
        f"Unsupported TensorCast semantic validation spec kind: {spec.kind!r}"
    )


def _compare_semantic_payload(
    *,
    label: str,
    expected: Any,
    actual: Any,
) -> Any:
    if actual != expected:
        raise RuntimeError(
            f"TensorCast semantic validation failed for {label} "
            f"spec: expected={expected!r}, actual={actual!r}"
        )
    return actual


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
