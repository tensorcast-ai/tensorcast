#  Copyright (c) 2026, TensorCast Team.

"""Framework-neutral recipe build identity and cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _jsonable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class RecipeBuildIdentity:
    model_hash: str
    model_id: str
    model_revision: str | None
    dtype: str
    runtime_version: str
    framework_name: str
    framework_version: str
    adapter_version: str
    serving_abi_version: str
    trace_cache_schema_version: int
    tp_rank: int
    tp_world_size: int
    topology_ref: Any | None = None
    member_ref: Any | None = None
    placement: Any | None = None

    def base_payload(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "model": self.model_id,
            "revision": self.model_revision,
            "dtype": self.dtype,
            "version": self.runtime_version,
            "trace_cache_schema_version": self.trace_cache_schema_version,
            "tp_rank": self.tp_rank,
            "tp_world_size": self.tp_world_size,
            "topology_ref": _jsonable(self.topology_ref),
            "member_ref": _jsonable(self.member_ref),
            "placement": _jsonable(self.placement),
        }

    def recipe_payload(self, *, metadata_fingerprint: str) -> dict[str, Any]:
        payload = self.base_payload()
        payload.update(
            {
                "metadata_fingerprint": metadata_fingerprint,
                "framework_name": self.framework_name,
                "framework_version": self.framework_version,
                "adapter_version": self.adapter_version,
                "serving_abi_version": self.serving_abi_version,
            }
        )
        return payload

    def trace_payload(self, *, metadata_fingerprint: str) -> dict[str, Any]:
        payload = self.base_payload()
        payload["metadata_fingerprint"] = metadata_fingerprint
        return payload


def stable_recipe_build_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def compute_trace_cache_key(
    identity: RecipeBuildIdentity,
    *,
    metadata_fingerprint: str,
) -> str:
    return stable_recipe_build_hash(
        identity.trace_payload(metadata_fingerprint=metadata_fingerprint)
    )


def compute_recipe_cache_key(
    identity: RecipeBuildIdentity,
    *,
    metadata_fingerprint: str,
) -> str:
    return stable_recipe_build_hash(
        identity.recipe_payload(metadata_fingerprint=metadata_fingerprint)
    )


def trace_cache_path(*, cache_dir: str, cache_key: str, tp_rank: int) -> str:
    return os.path.join(cache_dir, f"tensorcast_trace_{cache_key}_tp{tp_rank}.json")


def recipe_cache_path(*, cache_dir: str, cache_key: str, tp_rank: int) -> str:
    return os.path.join(cache_dir, f"tensorcast_recipe_{cache_key}_tp{tp_rank}.json")


class RecipeBuildSession:
    """Small core-owned shell for stable recipe build cache identity."""

    def __init__(self, identity: RecipeBuildIdentity) -> None:
        self.identity = identity

    def trace_cache_key(self, *, metadata_fingerprint: str) -> str:
        return compute_trace_cache_key(
            self.identity,
            metadata_fingerprint=metadata_fingerprint,
        )

    def recipe_cache_key(self, *, metadata_fingerprint: str) -> str:
        return compute_recipe_cache_key(
            self.identity,
            metadata_fingerprint=metadata_fingerprint,
        )

    def trace_cache_path(
        self,
        *,
        metadata_fingerprint: str,
        cache_dir: str,
    ) -> str:
        return trace_cache_path(
            cache_dir=cache_dir,
            cache_key=self.trace_cache_key(metadata_fingerprint=metadata_fingerprint),
            tp_rank=self.identity.tp_rank,
        )

    def recipe_cache_path(
        self,
        *,
        metadata_fingerprint: str,
        cache_dir: str,
    ) -> str:
        return recipe_cache_path(
            cache_dir=cache_dir,
            cache_key=self.recipe_cache_key(metadata_fingerprint=metadata_fingerprint),
            tp_rank=self.identity.tp_rank,
        )


__all__ = [
    "RecipeBuildIdentity",
    "RecipeBuildSession",
    "compute_recipe_cache_key",
    "compute_trace_cache_key",
    "recipe_cache_path",
    "stable_recipe_build_hash",
    "trace_cache_path",
]
