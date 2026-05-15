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
from tensorcast.api._config import (
    OverflowPolicy,
    PlanType,
    RegisterArtifactOptions,
    StorePolicy,
    StorePolicyProfile,
)
from tensorcast.api.operation import OperationStatus
from tensorcast.api.store import artifact as resolve_artifact
from tensorcast.api.store.types import PersistenceStatusResult, TensorDict

logger = logging.getLogger(__name__)

_CGID_SUFFIX_ALLOWED = re.compile(r"[-._~A-Za-z0-9]+")


class WeightPublisherConfig(BaseModel):
    """Configuration for publishing weights to Tensorcast and triggering reloads.

    Two supported publish modes:
    - publish(): tc.put(CUDA TensorDict) under a versioned key (durable policy).
    - publish_from_disk(): tc.from_disk(HF safetensors folder) bridge for systems
      that already export weights to a shared filesystem (e.g. Steptron).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    keep_last: int = 2
    pre_publish_trim_margin: int = 1
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
    # vLLM's online update protocol uses {weight_version}.
    key_template: str = "model:{model_name}:v{weight_version}"
    trigger_reload: bool = True
    reload_url: str | None = None

    # Optional: Stepcast multi-endpoint reload (discover endpoints via router).
    # If set, this takes precedence over reload_url.
    stepcast_router: str | None = None  # e.g. "stepcast-router:9200"
    stepcast_served_model_name: str | None = None  # defaults to model_name
    stepcast_reset_prefix_cache: bool = True
    stepcast_ack: bool = True
    stepcast_ack_timeout_s: float = 900.0
    stepcast_ack_poll_interval_s: float = 2.0

    # When publishing from a HF folder via tensorcast.from_disk(...).
    from_disk_verify_checksums: bool = True

    # Verify that the key resolves to the expected artifact_id after publish.
    # This catches accidental key reuse (protocol violation) which would
    # otherwise silently keep the old mapping.
    verify_key_mapping: bool = True
    key_mapping_timeout_s: float = 30.0
    key_mapping_poll_interval_s: float = 0.5

    # vLLM drain timeout while reloading (passed as ?drain_timeout_s=...).
    vllm_drain_timeout_s: float = 300.0

    reload_timeout_s: float = 10.0
    model_overrides: Mapping[str, object] | None = None
    gc_wait: bool = True
    gc_drain_timeout_s: float | None = 30.0
    gc_require_drained: bool = True
    # For DRAM_STABLE publish, default to direct CPU streaming into replica
    # memory. This avoids daemon-side full-size GPU staging.
    stage_on_gpu: bool = False

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

    @field_validator("pre_publish_trim_margin")
    @classmethod
    def _validate_pre_publish_trim_margin(cls, value: object) -> int:
        margin = int(str(value).strip())
        if margin < 0:
            raise ValueError("pre_publish_trim_margin must be >= 0")
        return margin

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
        if "{weight_version}" not in template:
            raise ValueError("key_template must include '{weight_version}'")
        if "{model_name}" not in template:
            raise ValueError("key_template must include '{model_name}'")
        return template


class WeightPublisher:
    def __init__(self, config: WeightPublisherConfig) -> None:
        self._config = config
        self._history_path = Path(config.history_path) if config.history_path else None
        self._last_publish_breakdown_s: dict[str, float] = {}

    def last_publish_breakdown_s(self) -> dict[str, float]:
        return dict(self._last_publish_breakdown_s)

    def publish(self, tensors: TensorDict, *, version: int) -> str:
        if version < 0:
            raise ValueError("version must be >= 0")
        total_start = time.monotonic()
        pre_trim_start = time.monotonic()
        self._maybe_trim_before_publish(version=version)
        pre_publish_trim_s = time.monotonic() - pre_trim_start
        artifact_key = self._build_key(version)
        requested_id = self._new_artifact_id(version)
        policy = self._build_policy()
        register_options = RegisterArtifactOptions(
            plan=PlanType.DRAM_STABLE,
            stage_on_gpu=bool(self._config.stage_on_gpu),
            release_gpu_on_commit=True,
        )
        put_start = time.monotonic()
        registered = tensorcast.put(
            tensors,
            artifact_id=requested_id,
            key=artifact_key,
            policy=policy,
            options=register_options,
        )
        put_s = time.monotonic() - put_start
        artifact_id = registered.artifact_id

        wait_persistence_s = 0.0
        if self._config.wait_persistence:
            persistence_start = time.monotonic()
            self._wait_for_persistence(registered, artifact_id)
            wait_persistence_s = time.monotonic() - persistence_start

        verify_key_mapping_s = 0.0
        if self._config.verify_key_mapping:
            verify_start = time.monotonic()
            self._wait_for_key_mapping(
                artifact_key=artifact_key,
                expected_artifact_id=str(artifact_id),
            )
            verify_key_mapping_s = time.monotonic() - verify_start

        trigger_reload_s = 0.0
        if self._config.trigger_reload:
            reload_start = time.monotonic()
            self._trigger_reload(version)
            trigger_reload_s = time.monotonic() - reload_start

        gc_start = time.monotonic()
        self._gc_old_artifacts(version, artifact_id)
        gc_s = time.monotonic() - gc_start
        total_s = time.monotonic() - total_start
        self._last_publish_breakdown_s = {
            "pre_publish_trim_s": float(pre_publish_trim_s),
            "put_s": float(put_s),
            "wait_persistence_s": float(wait_persistence_s),
            "verify_key_mapping_s": float(verify_key_mapping_s),
            "trigger_reload_s": float(trigger_reload_s),
            "gc_s": float(gc_s),
            "total_s": float(total_s),
        }
        logger.info(
            "publish_breakdown version=%s key=%s artifact_id=%s pre_trim_s=%.3f put_s=%.3f verify_key_s=%.3f gc_s=%.3f total_s=%.3f",
            version,
            artifact_key,
            artifact_id,
            pre_publish_trim_s,
            put_s,
            verify_key_mapping_s,
            gc_s,
            total_s,
        )
        return artifact_id

    def publish_from_disk(self, hf_dir: str | Path, *, version: int) -> str:
        """Bridge publisher for HF safetensors export folders.

        This resolves the HF folder into a Tensorcast artifact and publishes a
        *versioned key mapping* via the daemon. This does not require CUDA
        tensors and is intended as an incremental rollout path.
        """
        if version < 0:
            raise ValueError("version must be >= 0")
        total_start = time.monotonic()
        pre_trim_start = time.monotonic()
        self._maybe_trim_before_publish(version=version)
        pre_publish_trim_s = time.monotonic() - pre_trim_start
        artifact_key = self._build_key(version)
        from_disk_start = time.monotonic()
        artifact = tensorcast.from_disk(
            str(hf_dir),
            key=artifact_key,
            verify_checksums=bool(self._config.from_disk_verify_checksums),
        )
        from_disk_s = time.monotonic() - from_disk_start
        artifact_id = getattr(artifact, "artifact_id", "") or ""

        verify_key_mapping_s = 0.0
        if self._config.verify_key_mapping:
            verify_start = time.monotonic()
            self._wait_for_key_mapping(
                artifact_key=artifact_key,
                expected_artifact_id=str(artifact_id),
            )
            verify_key_mapping_s = time.monotonic() - verify_start

        trigger_reload_s = 0.0
        if self._config.trigger_reload:
            reload_start = time.monotonic()
            self._trigger_reload(version)
            trigger_reload_s = time.monotonic() - reload_start

        gc_s = 0.0
        if artifact_id:
            gc_start = time.monotonic()
            self._gc_old_artifacts(version, artifact_id)
            gc_s = time.monotonic() - gc_start
        total_s = time.monotonic() - total_start
        self._last_publish_breakdown_s = {
            "pre_publish_trim_s": float(pre_publish_trim_s),
            "from_disk_s": float(from_disk_s),
            "verify_key_mapping_s": float(verify_key_mapping_s),
            "trigger_reload_s": float(trigger_reload_s),
            "gc_s": float(gc_s),
            "total_s": float(total_s),
        }
        return artifact_id

    def _build_key(self, version: int) -> str:
        key = self._config.key_template.format(
            model_name=self._config.model_name,
            weight_version=version,
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

    def _wait_for_persistence(self, registered: object, artifact_id: str) -> None:
        task_id = getattr(registered, "persistence_task_id", None)
        if not task_id:
            return
        if hasattr(registered, "persistence_operation"):
            operation = registered.persistence_operation()
        else:
            operation = tensorcast.persistence_operation(task_id=task_id)
        deadline = time.monotonic() + self._config.persistence_timeout_s
        while True:
            status = operation.status()
            if _is_persistence_operation_terminal(
                status, task_id=task_id, allow_degraded=self._config.allow_degraded_persistence
            ):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"persistence timeout: task_id={task_id} artifact_id={artifact_id} state={status.state}"
                )
            time.sleep(self._config.persistence_poll_interval_s)

    def _trigger_reload(self, version: int) -> None:
        if self._config.stepcast_router:
            self._trigger_stepcast_reload(version)
            return

        url = self._config.reload_url
        if not url:
            raise ValueError(
                "Either reload_url or stepcast_router must be set when trigger_reload is True"
            )
        _http_post_json(
            url,
            {
                "weight_version": int(version),
                "model_overrides": self._config.model_overrides,
            },
            timeout_s=self._config.reload_timeout_s,
        )

    def _trigger_stepcast_reload(self, version: int) -> None:
        router = self._config.stepcast_router
        assert router
        served = self._config.stepcast_served_model_name or self._config.model_name
        endpoints = _stepcast_endpoints(router=router, served_model_name=served)
        logger.info("Stepcast endpoints(%s): %s", len(endpoints), endpoints)

        payload = {
            "weight_version": int(version),
            "model_overrides": self._config.model_overrides,
        }
        for ep in endpoints:
            _http_post_json(
                f"http://{ep}/set_model_weight?drain_timeout_s={float(self._config.vllm_drain_timeout_s)}",
                payload,
                timeout_s=self._config.reload_timeout_s,
            )
            if self._config.stepcast_reset_prefix_cache:
                _http_post_bytes(
                    f"http://{ep}/reset_prefix_cache",
                    timeout_s=self._config.reload_timeout_s,
                )

        if not self._config.stepcast_ack:
            return

        deadline = time.monotonic() + float(self._config.stepcast_ack_timeout_s)
        for ep in endpoints:
            _wait_weight_version(
                endpoint=ep,
                version=int(version),
                deadline=deadline,
                poll_interval_s=float(self._config.stepcast_ack_poll_interval_s),
            )

    def _maybe_trim_before_publish(self, *, version: int) -> None:
        keep = self._effective_pre_publish_keep_last()
        history = self._load_history()
        if not history:
            return
        kept = self._apply_retention(
            history=history,
            keep=keep,
            reason=f"pre_publish(version={version})",
        )
        logger.info(
            "pre_publish trim applied: version=%s keep=%s kept_versions=%s",
            version,
            keep,
            [ver for ver, _ in kept],
        )

    def _effective_pre_publish_keep_last(self) -> int:
        keep = int(self._config.keep_last)
        margin = int(self._config.pre_publish_trim_margin)
        return max(0, keep - margin)

    def _gc_old_artifacts(self, version: int, latest_artifact_id: str) -> None:
        keep = self._config.keep_last
        if keep <= 0:
            return
        history = self._load_history()
        history.append((version, latest_artifact_id))
        self._apply_retention(
            history=history,
            keep=int(keep),
            reason=f"post_publish(version={version})",
        )

    def _apply_retention(
        self,
        *,
        history: Sequence[tuple[int, str]],
        keep: int,
        reason: str,
    ) -> list[tuple[int, str]]:
        clamped_keep = max(0, int(keep))
        by_version = {ver: artifact_id for ver, artifact_id in history if artifact_id}
        ordered = sorted(by_version.items(), key=lambda item: item[0], reverse=True)
        to_keep = ordered[:clamped_keep] if clamped_keep > 0 else []
        to_drop = ordered[clamped_keep:]
        for _, artifact_id in to_drop:
            # Retention policy intentionally keeps key mappings append-only.
            # keep_last only controls replica/shared-disk residency.
            try:
                outcome = tensorcast.deregister_artifact(
                    artifact_id,
                    wait=self._config.gc_wait,
                    drain_timeout_s=self._config.gc_drain_timeout_s,
                )
                logger.info(
                    "deregister reason=%s artifact_id=%s drained=%s removed=%s released_region_ids=%s message=%s",
                    reason,
                    artifact_id,
                    bool(outcome.drained),
                    bool(outcome.removed),
                    list(outcome.released_region_ids),
                    outcome.message,
                )
                if (
                    self._config.gc_wait
                    and self._config.gc_require_drained
                    and not bool(outcome.drained)
                ):
                    raise RuntimeError(
                        "deregister did not drain in gc_wait mode: "
                        f"artifact_id={artifact_id}, message={outcome.message}"
                    )
            except Exception:
                logger.exception("deregister failed: %s", artifact_id)
                raise
        self._store_history(to_keep)
        return list(to_keep)

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

    def _wait_for_key_mapping(
        self, *, artifact_key: str, expected_artifact_id: str
    ) -> None:
        """Ensure key mapping is visible and points to the expected artifact.

        Tensorcast's daemon publish is best-effort and will keep an existing key
        mapping. This check makes protocol violations loud.
        """
        deadline = time.monotonic() + float(self._config.key_mapping_timeout_s)
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            try:
                resolved = resolve_artifact(key=str(artifact_key)).artifact_id
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(float(self._config.key_mapping_poll_interval_s))
                continue
            if resolved == expected_artifact_id:
                return
            raise RuntimeError(
                "Tensorcast key mapping mismatch: "
                f"key={artifact_key} expected_artifact_id={expected_artifact_id} "
                f"resolved_artifact_id={resolved}"
            )
        raise TimeoutError(
            f"Timeout waiting for key mapping: key={artifact_key} expected_artifact_id={expected_artifact_id} last_err={last_err}"
        )


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


def _is_persistence_operation_terminal(
    status: OperationStatus, *, task_id: str, allow_degraded: bool
) -> bool:
    if status.state == "success":
        return True
    if status.state in {"failed", "cancelled"}:
        message = (
            status.error.message
            if status.error is not None and status.error.message
            else status.message or "operation failed"
        )
        raise RuntimeError(f"persistence failed: task_id={task_id} error={message}")
    if status.state == "degraded":
        if allow_degraded:
            return True
        message = (
            status.error.message
            if status.error is not None and status.error.message
            else status.message or "operation degraded"
        )
        raise RuntimeError(f"persistence degraded: task_id={task_id} reason={message}")
    return False


def _http_post_bytes(url: str, *, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"http post failed: url={url} http={exc.code} body={exc.read().decode('utf-8', 'ignore')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"http post failed: url={url} err={exc}") from exc


def _http_post_json(url: str, payload: dict[str, object], *, timeout_s: float) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"http post failed: url={url} http={exc.code} body={exc.read().decode('utf-8', 'ignore')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"http post failed: url={url} err={exc}") from exc


def _http_get_json(url: str, *, timeout_s: float = 10.0) -> dict[str, object]:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"http get failed: url={url} http={exc.code} body={exc.read().decode('utf-8', 'ignore')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"http get failed: url={url} err={exc}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _stepcast_endpoints(*, router: str, served_model_name: str) -> list[str]:
    data = _http_get_json(f"http://{router}/v1/model/{served_model_name}")
    raw_eps = data.get("endpoints", [])
    endpoints: list[str] = []
    if isinstance(raw_eps, list):
        for ep in raw_eps:
            if isinstance(ep, (list, tuple)) and ep:
                endpoints.append(str(ep[0]))
            elif isinstance(ep, str) and ep.strip():
                endpoints.append(ep.strip())
    endpoints = sorted(set(endpoints))
    if not endpoints:
        raise RuntimeError(
            f"no endpoints from stepcast router={router} model={served_model_name} payload={data}"
        )
    return endpoints


def _wait_weight_version(
    *,
    endpoint: str,
    version: int,
    deadline: float,
    poll_interval_s: float,
) -> None:
    url = f"http://{endpoint}/weight_version"
    while time.monotonic() < deadline:
        info = _http_get_json(url, timeout_s=10.0)
        if isinstance(info, dict) and info.get("weight_version") == int(version):
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"timeout waiting endpoint={endpoint} to reach weight_version={version}"
    )


__all__ = ["WeightPublisher", "WeightPublisherConfig"]
