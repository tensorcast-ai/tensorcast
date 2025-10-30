#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest
from fastapi import HTTPException

from tensorcast.dashboard.api import (
    _include_flags,
    _memory_type_from_param,
    _parse_chunk_indices,
    _parse_leaf_indices,
    _space_flags,
)
from tensorcast.proto.common.v1 import common_pb2


def test_memory_type_from_param_accepts_case_insensitive_values() -> None:
    assert _memory_type_from_param("ram") == common_pb2.MemoryType.MEMORY_TYPE_RAM
    assert _memory_type_from_param("GPU") == common_pb2.MemoryType.MEMORY_TYPE_GPU


def test_memory_type_from_param_rejects_unknown_value() -> None:
    with pytest.raises(HTTPException) as exc:
        _memory_type_from_param("ssd")
    assert exc.value.status_code == 400


def test_include_flags_defaults_to_replicas_only() -> None:
    include_replicas, include_view, include_leaves = _include_flags(None)
    assert include_replicas is True
    assert include_view is False
    assert include_leaves is False


def test_include_flags_rejects_invalid_entries() -> None:
    with pytest.raises(HTTPException) as exc:
        _include_flags("replicas,foo")
    assert exc.value.status_code == 400


def test_space_flags_requires_view_id_for_view_space() -> None:
    with pytest.raises(HTTPException) as exc:
        _space_flags("view", None)
    assert exc.value.status_code == 400


def test_parse_leaf_indices_rejects_negative() -> None:
    with pytest.raises(HTTPException) as exc:
        _parse_leaf_indices("1,-1")
    assert exc.value.status_code == 400


def test_parse_chunk_indices_ignores_whitespace() -> None:
    result = _parse_chunk_indices(" 1 , 2 , 3 ")
    assert result == [1, 2, 3]
