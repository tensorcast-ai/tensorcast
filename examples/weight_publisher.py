#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, field_validator

import tensorcast
from tensorcast.api._config import OverflowPolicy, StorePolicy, StorePolicyProfile
from tensorcast.api.store.types import PersistenceStatusResult, TensorDict

logger = logging.getLogger(__name__)

_CGID_SUFFIX_ALLOWED = re.compile(r"[-._~A-Za-z0-9]+")


class WeightPublisherConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    keep_last: int = 2
    history_path: str | None = None
    # Durable policies trigger managed shared-disk persistence; daemon storage_path is required.
    policy: StorePolicy | str | None = StorePolicyProfile.DURABLE.value
    overflow_policy: OverflowPolicy | str = OverflowPolicy.SPILL
    wait_persistence: bool = True
    allow_degraded_persistence: bool = False
    persistence_timeout_s: float = 600.0
    persistence_poll_interval_s: float = 2.0
    use_cgid: bool = True
    cgid_prefix: str = "weights"
    key_template: str = "model:{model_name}:v{version}"
    trigger_reload: bool = True
    reload_url: str | None = None
    reload_timeout_s: float = 10.0
    model_overrides: Mapping[str, object] | None = None
    gc_wait: bool = True
    gc_drain_timeout_s: float | None = 30.0

    @field_validator("model_name")
    @classmethod
    def _validate_model_name(cls, value: object) -> str:
        name = str(value).strip()
        if not name:
            raise ValueError("model_name must be non-empty")
        return name

    @field_validator("keep_last")
    @classmethod
    def _validate_keep_last(cls, value: object) -> int:
        keep = int(str(value).strip())
        if keep < 0:
            raise ValueError("keep_last must be >= 0")
        return keep

    @field_validator("policy", mode="before")
    @classmethod
    def _normalize_policy(cls, value: object) -> StorePolicy | str | None:
        if value is None or value == "":
            return None
        if isinstance(value, StorePolicy):
            return value
        if isinstance(value, StorePolicyProfile):
            return value.value
        return str(value)

    @field_validator("overflow_policy", mode="before")
    @classmethod
    def _normalize_overflow(cls, value: object) -> OverflowPolicy | str:
        if value is None:
            return OverflowPolicy.SPILL
        if isinstance(value, OverflowPolicy):
            return value
        return OverflowPolicy.parse(value)

    @field_validator("key_template")
    @classmethod
    def _validate_key_template(cls, value: object) -> str:
        template = str(value).strip()
        if "{version}" not in template:
            raise ValueError("key_template must include '{version}'")
        if "{model_name}" not in template:
            raise ValueError("key_template must include '{model_name}'")
        return template


class WeightPublisher:
    def __init__(self, config: WeightPublisherConfig) -> None:
        self._config = config
        self._history_path = Path(config.history_path) if config.history_path else None

    def publish(self, tensors: TensorDict, *, version: int) -> str:
        if version < 0:
            raise ValueError("version must be >= 0")
        artifact_key = self._build_key(version)
        requested_id = self._new_artifact_id(version)
        policy = self._build_policy()
        registered = tensorcast.put(
            tensors,
            artifact_id=requested_id,
            key=artifact_key,
            policy=policy,
        )
        artifact_id = registered.artifact_id

        if self._config.wait_persistence:
            self._wait_for_persistence(registered.persistence_task_id, artifact_id)

        if self._config.trigger_reload:
            self._trigger_reload(version)

        self._gc_old_artifacts(version, artifact_id)
        return artifact_id

    def _build_key(self, version: int) -> str:
        key = self._config.key_template.format(
            model_name=self._config.model_name, version=version
        ).strip()
        if not key:
            raise ValueError("resolved artifact key is empty")
        return key

    def _build_policy(self) -> StorePolicy | str | None:
        policy = self._config.policy
        if policy is None:
            return None
        parsed = StorePolicy.parse(policy)
        if parsed is None:
            return None
        return parsed.model_copy(
            update={
                "overflow_policy": OverflowPolicy.parse(self._config.overflow_policy)
            }
        )

    def _new_artifact_id(self, version: int) -> str | None:
        if not self._config.use_cgid:
            return None
        suffix = self._build_cgid_suffix(version)
        return f"cgid:{suffix}"

    def _build_cgid_suffix(self, version: int) -> str:
        key_part = _sanitize_cgid_component(self._config.model_name)
        prefix_part = _sanitize_cgid_component(self._config.cgid_prefix)
        version_part = _sanitize_cgid_component(str(version))
        stamp = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex
        pieces = [p for p in (prefix_part, key_part, version_part, stamp, nonce) if p]
        suffix = "-".join(pieces)
        if len(suffix) < 8:
            suffix = f"{suffix}-{nonce}"
        return suffix[:200]

    def _wait_for_persistence(self, task_id: str | None, artifact_id: str) -> None:
        if not task_id:
            return
        deadline = time.monotonic() + self._config.persistence_timeout_s
        while True:
            status = tensorcast.query_persistence_status(task_id=task_id)
            if _is_persistence_terminal(
                status, self._config.allow_degraded_persistence
            ):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"persistence timeout: task_id={task_id} artifact_id={artifact_id} state={status.state}"
                )
            time.sleep(self._config.persistence_poll_interval_s)

    def _trigger_reload(self, version: int) -> None:
        url = self._config.reload_url
        if not url:
            raise ValueError("reload_url must be set when trigger_reload is True")
        payload = {
            "weight_version": int(version),
            "model_overrides": self._config.model_overrides,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._config.reload_timeout_s
            ) as resp:
                resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"reload failed: http={exc.code} body={exc.read().decode('utf-8', 'ignore')}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"reload failed: {exc}") from exc

    def _gc_old_artifacts(self, version: int, latest_artifact_id: str) -> None:
        keep = self._config.keep_last
        if keep <= 0:
            return
        history = self._load_history()
        history.append((version, latest_artifact_id))
        by_version = {ver: artifact_id for ver, artifact_id in history if artifact_id}
        ordered = sorted(by_version.items(), key=lambda item: item[0], reverse=True)
        to_keep = ordered[:keep]
        to_drop = ordered[keep:]
        for _, artifact_id in to_drop:
            try:
                tensorcast.deregister_artifact(
                    artifact_id,
                    wait=self._config.gc_wait,
                    drain_timeout_s=self._config.gc_drain_timeout_s,
                )
            except Exception:
                logger.exception("deregister failed: %s", artifact_id)
        self._store_history(to_keep)

    def _load_history(self) -> list[tuple[int, str]]:
        path = self._history_path
        if path is None:
            return []
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("history load failed: %s", path)
            return []
        if not isinstance(raw, list):
            return []
        out: list[tuple[int, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            version = item.get("version")
            artifact_id = item.get("artifact_id")
            if (
                isinstance(version, int)
                and isinstance(artifact_id, str)
                and artifact_id.strip()
            ):
                out.append((version, artifact_id.strip()))
        return out

    def _store_history(self, artifact_ids: Sequence[tuple[int, str]]) -> None:
        path = self._history_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"version": version, "artifact_id": artifact_id}
            for version, artifact_id in artifact_ids
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sanitize_cgid_component(value: str) -> str:
    candidate = re.sub(r"[^-._~A-Za-z0-9]+", "_", value.strip())
    candidate = candidate.strip("_")
    if not candidate:
        return ""
    if not _CGID_SUFFIX_ALLOWED.fullmatch(candidate):
        return ""
    return candidate


def _is_persistence_terminal(
    status: PersistenceStatusResult, allow_degraded: bool
) -> bool:
    if status.state == "success":
        return True
    if status.state == "failed":
        raise RuntimeError(
            f"persistence failed: task_id={status.task_id} error={status.last_error}"
        )
    if status.state == "degraded":
        if allow_degraded:
            return True
        raise RuntimeError(
            f"persistence degraded: task_id={status.task_id} reason={status.degraded_reason}"
        )
    return False


__all__ = ["WeightPublisher", "WeightPublisherConfig"]
