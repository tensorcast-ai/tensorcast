#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorcast.artifact_runtime.local_ready_prewarm import (
    MATERIALIZING_READY_WRITE_ENV,
    RETAINED_MANIFEST_WRITE_ENV,
    SOURCE_PATH_FILTER_ENV,
    TARGET_PLAN_MANIFEST_B64_ENV,
    TARGET_PLAN_MANIFEST_CACHE_DIR_ENV,
    TARGET_PLAN_MANIFEST_ENV,
    TARGET_PLAN_MANIFEST_JSON_ENV,
    TARGET_PLAN_MANIFEST_SHA256_ENV,
    TARGET_PLAN_MANIFEST_WRITE_ENV,
    target_plan_manifest_cache_path,
    target_plan_manifest_cache_source_index_path,
)

RETAINED_MANIFEST_ENV = "TENSORCAST_RETAINED_BINDING_MANIFEST"
PREWARM_LOG_ENV = "TENSORCAST_LOCAL_READY_PREWARM_LOG"
MATERIALIZING_READY_TIMEOUT_S_ENV = (
    "TENSORCAST_LOCAL_READY_MATERIALIZING_READY_TIMEOUT_S"
)
LOCAL_READY_RUN_DIR_ENV = "TENSORCAST_LOCAL_READY_RUN_DIR"
TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV = (
    "TENSORCAST_LOCAL_READY_TARGET_PLAN_CACHE_READY_TIMEOUT_S"
)


@dataclass(frozen=True, slots=True)
class LocalReadyLaunchConfig:
    env: dict[str, str]
    materializing_ready_marker: Path
    prewarm_log: Path
    timeout_s: float


def _split_env_paths(raw: str | None) -> list[str]:
    return [entry for entry in str(raw or "").split(os.pathsep) if entry]


def _normalize_sha256_digest(value: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid target-plan manifest sha256: {value!r}")
    return digest


def _default_marker_path(retained_manifest_write: str) -> Path:
    path = Path(retained_manifest_write).expanduser()
    return path.with_name(path.name + ".ready.json")


def _default_log_path(marker_path: Path) -> Path:
    return marker_path.with_name(marker_path.name + ".prewarm.log")


def _timeout_from_env(env: Mapping[str, str]) -> float:
    raw = env.get(MATERIALIZING_READY_TIMEOUT_S_ENV)
    return float(raw) if raw else 300.0


def _target_plan_cache_ready_timeout_from_env(env: Mapping[str, str]) -> float | None:
    raw = env.get(TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV)
    if raw is None or not str(raw).strip():
        return None
    timeout_s = float(raw)
    if timeout_s < 0:
        raise ValueError(
            f"{TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV} must be non-negative"
        )
    return timeout_s


def _default_retained_manifest_write(
    env: Mapping[str, str],
    *,
    manifest_paths: Sequence[str],
    inline_manifest: str | None,
) -> str | None:
    run_dir = env.get(LOCAL_READY_RUN_DIR_ENV)
    if run_dir:
        return str(Path(run_dir).expanduser() / "retained-binding-manifest.json")

    cache_dir = env.get(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV)
    if cache_dir and (
        env.get(TARGET_PLAN_MANIFEST_SHA256_ENV) or env.get(SOURCE_PATH_FILTER_ENV)
    ):
        identity_payload = {
            "schema_version": 1,
            "cache_dir": str(Path(cache_dir).expanduser()),
            "manifest_sha256": env.get(TARGET_PLAN_MANIFEST_SHA256_ENV) or "",
            "source_path": env.get(SOURCE_PATH_FILTER_ENV) or "",
        }
        digest = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return str(
            Path(cache_dir).expanduser()
            / "retained_manifests"
            / digest[:2]
            / f"{digest}.json"
        )

    if manifest_paths:
        path = Path(manifest_paths[0]).expanduser()
        return str(path.with_name(path.name + ".retained.json"))

    if inline_manifest:
        return None
    return None


def prepare_launch_environment(
    env: Mapping[str, str] | None = None,
    *,
    retained_manifest_write: str | os.PathLike[str] | None = None,
    run_dir: str | os.PathLike[str] | None = None,
    materializing_ready_write: str | os.PathLike[str] | None = None,
    prewarm_log: str | os.PathLike[str] | None = None,
    timeout_s: float | None = None,
) -> LocalReadyLaunchConfig:
    launch_env = dict(os.environ if env is None else env)
    if run_dir is not None:
        launch_env[LOCAL_READY_RUN_DIR_ENV] = str(run_dir)
    if retained_manifest_write is not None:
        launch_env[RETAINED_MANIFEST_WRITE_ENV] = str(retained_manifest_write)
    manifest_paths = _split_env_paths(
        launch_env.get(TARGET_PLAN_MANIFEST_ENV)
    ) or _split_env_paths(launch_env.get(TARGET_PLAN_MANIFEST_WRITE_ENV))
    inline_manifest = launch_env.get(TARGET_PLAN_MANIFEST_JSON_ENV) or launch_env.get(
        TARGET_PLAN_MANIFEST_B64_ENV
    )
    cache_identity = launch_env.get(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV) and (
        launch_env.get(TARGET_PLAN_MANIFEST_SHA256_ENV)
        or launch_env.get(SOURCE_PATH_FILTER_ENV)
    )
    if not manifest_paths and not inline_manifest and not cache_identity:
        raise ValueError(
            "set a TensorCast local-ready target-plan manifest before "
            f"launching prewarm: {TARGET_PLAN_MANIFEST_ENV}, "
            f"{TARGET_PLAN_MANIFEST_JSON_ENV}, or "
            f"{TARGET_PLAN_MANIFEST_B64_ENV}, or "
            f"{TARGET_PLAN_MANIFEST_CACHE_DIR_ENV} with "
            f"{TARGET_PLAN_MANIFEST_SHA256_ENV} or {SOURCE_PATH_FILTER_ENV}. Use "
            f"{TARGET_PLAN_MANIFEST_SHA256_ENV} with path manifests to bind "
            "the env-carried manifest identity."
        )

    retained_manifest_write = launch_env.get(
        RETAINED_MANIFEST_WRITE_ENV
    ) or _default_retained_manifest_write(
        launch_env,
        manifest_paths=manifest_paths,
        inline_manifest=inline_manifest,
    )
    if not retained_manifest_write:
        raise ValueError(
            f"set {RETAINED_MANIFEST_WRITE_ENV} or {LOCAL_READY_RUN_DIR_ENV} "
            "before launching local-ready prewarm"
        )
    launch_env[RETAINED_MANIFEST_WRITE_ENV] = retained_manifest_write
    if not launch_env.get(RETAINED_MANIFEST_ENV):
        launch_env[RETAINED_MANIFEST_ENV] = retained_manifest_write

    marker_raw = (
        str(materializing_ready_write)
        if materializing_ready_write is not None
        else launch_env.get(MATERIALIZING_READY_WRITE_ENV)
    )
    marker_path = (
        Path(marker_raw).expanduser()
        if marker_raw
        else _default_marker_path(retained_manifest_write)
    )
    launch_env[MATERIALIZING_READY_WRITE_ENV] = str(marker_path)

    log_raw = (
        str(prewarm_log) if prewarm_log is not None else launch_env.get(PREWARM_LOG_ENV)
    )
    log_path = Path(log_raw).expanduser() if log_raw else _default_log_path(marker_path)
    launch_env[PREWARM_LOG_ENV] = str(log_path)

    resolved_timeout_s = (
        float(timeout_s) if timeout_s is not None else _timeout_from_env(launch_env)
    )
    if resolved_timeout_s <= 0:
        raise ValueError(f"{MATERIALIZING_READY_TIMEOUT_S_ENV} must be positive")
    return LocalReadyLaunchConfig(
        env=launch_env,
        materializing_ready_marker=marker_path,
        prewarm_log=log_path,
        timeout_s=resolved_timeout_s,
    )


class _TargetPlanCacheNotReady(RuntimeError):
    pass


def _target_plan_cache_ready_once(
    env: MutableMapping[str, str],
) -> dict[str, Any] | None:
    cache_dir = env.get(TARGET_PLAN_MANIFEST_CACHE_DIR_ENV)
    if not cache_dir:
        return None

    explicit_sha256s = [
        _normalize_sha256_digest(entry)
        for entry in _split_env_paths(env.get(TARGET_PLAN_MANIFEST_SHA256_ENV))
    ]
    if explicit_sha256s:
        manifest_paths: list[str] = []
        missing_paths: list[str] = []
        for digest in explicit_sha256s:
            path = target_plan_manifest_cache_path(cache_dir, digest)
            manifest_paths.append(str(path))
            if not path.exists():
                missing_paths.append(str(path))
        if missing_paths:
            raise _TargetPlanCacheNotReady(
                f"target-plan manifest cache file is not ready: paths={missing_paths}"
            )
        return {
            "schema_version": 1,
            "event": "target_plan_manifest_cache_ready",
            "cache_dir": str(Path(cache_dir).expanduser()),
            "manifest_sha256s": explicit_sha256s,
            "manifest_paths": manifest_paths,
        }

    source_path = env.get(SOURCE_PATH_FILTER_ENV)
    if not source_path:
        return None

    index_path = target_plan_manifest_cache_source_index_path(cache_dir, source_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target-plan manifest cache index is not an object")
    if payload.get("ready") is not True:
        raise _TargetPlanCacheNotReady(
            "target-plan manifest cache index is not ready: "
            f"path={index_path} record_count={payload.get('record_count')} "
            f"expected_records={payload.get('expected_records')}"
        )

    digest = _normalize_sha256_digest(str(payload.get("manifest_sha256") or ""))
    manifest_path = target_plan_manifest_cache_path(cache_dir, digest)
    if not manifest_path.exists():
        raise _TargetPlanCacheNotReady(
            "target-plan manifest cache index is ready but manifest file is "
            f"missing: path={manifest_path}"
        )
    env[TARGET_PLAN_MANIFEST_SHA256_ENV] = digest

    resolved = dict(payload)
    resolved["event"] = "target_plan_manifest_cache_ready"
    resolved["manifest_path"] = str(manifest_path)
    return resolved


def wait_for_target_plan_cache_ready(
    env: MutableMapping[str, str],
    *,
    timeout_s: float = 0.0,
) -> dict[str, Any] | None:
    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")
    deadline = time.monotonic() + timeout_s
    sleep_s = 0.02
    last_error: BaseException | None = None
    while True:
        try:
            return _target_plan_cache_ready_once(env)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            _TargetPlanCacheNotReady,
        ) as exc:
            last_error = exc

        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for TensorCast target-plan cache "
                "readiness" + (f": {last_error}" if last_error is not None else "")
            ) from last_error
        time.sleep(min(sleep_s, max(0.0, deadline - time.monotonic())))
        sleep_s = min(0.2, sleep_s * 1.5)


def _maybe_wait_for_target_plan_cache_ready(
    env: MutableMapping[str, str],
    *,
    timeout_s: float | None,
) -> dict[str, Any] | None:
    resolved_timeout_s = (
        float(timeout_s)
        if timeout_s is not None
        else _target_plan_cache_ready_timeout_from_env(env)
    )
    if resolved_timeout_s is None:
        return None
    return wait_for_target_plan_cache_ready(env, timeout_s=resolved_timeout_s)


def _tail_text(path: Path, *, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def start_prewarm_process(
    config: LocalReadyLaunchConfig,
    *,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> Any:
    config.prewarm_log.parent.mkdir(parents=True, exist_ok=True)
    log_file = config.prewarm_log.open("a", encoding="utf-8")
    try:
        return popen_factory(
            [
                sys.executable,
                "-m",
                "tensorcast.artifact_runtime.local_ready_prewarm",
            ],
            env=config.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()


def wait_for_materializing_ready_marker(
    marker_path: str | os.PathLike[str],
    *,
    prewarm_proc: Any | None = None,
    prewarm_log: str | os.PathLike[str] | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    marker = Path(marker_path).expanduser()
    log_path = Path(prewarm_log).expanduser() if prewarm_log else None
    deadline = time.monotonic() + timeout_s
    sleep_s = 0.02
    last_error: Exception | None = None
    while True:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("marker payload is not a JSON object")
            if payload.get("ready") is True:
                return payload
            if payload.get("ready") is False:
                detail = payload.get("error_message") or payload.get("event")
                raise RuntimeError(
                    "TensorCast local-ready prewarm reported failure"
                    + (f": {detail}" if detail else "")
                )
        except FileNotFoundError as exc:
            last_error = exc
        except json.JSONDecodeError as exc:
            last_error = exc

        if prewarm_proc is not None:
            rc = prewarm_proc.poll()
            if rc is not None:
                tail = _tail_text(log_path) if log_path is not None else ""
                raise RuntimeError(
                    "TensorCast local-ready prewarmer exited before "
                    f"materializing ready marker: rc={rc}"
                    + (f"\n{tail}" if tail else "")
                ) from last_error

        if time.monotonic() >= deadline:
            tail = _tail_text(log_path) if log_path is not None else ""
            raise TimeoutError(
                "timed out waiting for TensorCast local-ready materializing "
                f"marker: path={marker}" + (f"\n{tail}" if tail else "")
            ) from last_error
        time.sleep(min(sleep_s, max(0.0, deadline - time.monotonic())))
        sleep_s = min(0.2, sleep_s * 1.5)


def _terminate_process(proc: Any) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except Exception:
        proc.kill()


def launch_after_materializing_ready(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    retained_manifest_write: str | os.PathLike[str] | None = None,
    run_dir: str | os.PathLike[str] | None = None,
    target_plan_cache_ready_timeout_s: float | None = None,
    materializing_ready_write: str | os.PathLike[str] | None = None,
    prewarm_log: str | os.PathLike[str] | None = None,
    timeout_s: float | None = None,
    exec_fn: Callable[[str, Sequence[str], MutableMapping[str, str]], Any] = os.execvpe,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    if not command:
        raise ValueError("command is required")
    config = prepare_launch_environment(
        env,
        retained_manifest_write=retained_manifest_write,
        run_dir=run_dir,
        materializing_ready_write=materializing_ready_write,
        prewarm_log=prewarm_log,
        timeout_s=timeout_s,
    )
    _maybe_wait_for_target_plan_cache_ready(
        config.env,
        timeout_s=target_plan_cache_ready_timeout_s,
    )
    proc = start_prewarm_process(config, popen_factory=popen_factory)
    try:
        marker = wait_for_materializing_ready_marker(
            config.materializing_ready_marker,
            prewarm_proc=proc,
            prewarm_log=config.prewarm_log,
            timeout_s=config.timeout_s,
        )
    except Exception:
        _terminate_process(proc)
        raise
    try:
        exec_fn(command[0], list(command), config.env)
    except Exception:
        _terminate_process(proc)
        raise
    return marker


def _send_signal(proc: Any, signum: int) -> None:
    if proc is None or proc.poll() is not None:
        return
    send_signal = getattr(proc, "send_signal", None)
    if callable(send_signal):
        send_signal(signum)
    elif signum == signal.SIGTERM:
        proc.terminate()


class _ForwardSignals:
    def __init__(self, *procs: Any) -> None:
        self._procs = procs
        self._old_handlers: dict[int, Any] = {}

    def __enter__(self) -> "_ForwardSignals":
        for signum in (
            signal.SIGINT,
            signal.SIGTERM,
            getattr(signal, "SIGHUP", None),
        ):
            if signum is None:
                continue
            try:
                self._old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
            except (OSError, ValueError):
                continue
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for signum, old_handler in self._old_handlers.items():
            try:
                signal.signal(signum, old_handler)
            except (OSError, ValueError):
                continue

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        for proc in self._procs:
            try:
                _send_signal(proc, signum)
            except Exception:
                continue


def _wait_supervised_command(
    command_proc: Any,
    *,
    prewarm_proc: Any,
    poll_interval_s: float = 0.1,
) -> int:
    prewarm_completed = False
    while True:
        rc = command_proc.poll()
        prewarm_rc = None
        if prewarm_proc is not None and not prewarm_completed:
            with suppress(Exception):
                prewarm_rc = prewarm_proc.poll()
            if prewarm_rc is not None:
                prewarm_completed = True
                with suppress(Exception):
                    prewarm_proc.wait(timeout=0)
                if int(prewarm_rc) != 0 and rc is None:
                    _terminate_process(command_proc)
                    return int(prewarm_rc)
        if rc is not None:
            try:
                return int(command_proc.wait())
            finally:
                if (
                    prewarm_proc is not None
                    and not prewarm_completed
                    and prewarm_proc.poll() is None
                ):
                    _terminate_process(prewarm_proc)
                elif prewarm_proc is not None:
                    with suppress(Exception):
                        prewarm_proc.wait(timeout=0)
        time.sleep(poll_interval_s)


def run_after_materializing_ready(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    retained_manifest_write: str | os.PathLike[str] | None = None,
    run_dir: str | os.PathLike[str] | None = None,
    target_plan_cache_ready_timeout_s: float | None = None,
    materializing_ready_write: str | os.PathLike[str] | None = None,
    prewarm_log: str | os.PathLike[str] | None = None,
    timeout_s: float | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    command_popen_factory: Callable[..., Any] = subprocess.Popen,
) -> int:
    if not command:
        raise ValueError("command is required")
    config = prepare_launch_environment(
        env,
        retained_manifest_write=retained_manifest_write,
        run_dir=run_dir,
        materializing_ready_write=materializing_ready_write,
        prewarm_log=prewarm_log,
        timeout_s=timeout_s,
    )
    _maybe_wait_for_target_plan_cache_ready(
        config.env,
        timeout_s=target_plan_cache_ready_timeout_s,
    )
    prewarm_proc = start_prewarm_process(config, popen_factory=popen_factory)
    try:
        wait_for_materializing_ready_marker(
            config.materializing_ready_marker,
            prewarm_proc=prewarm_proc,
            prewarm_log=config.prewarm_log,
            timeout_s=config.timeout_s,
        )
        command_proc = command_popen_factory(list(command), env=config.env)
    except Exception:
        _terminate_process(prewarm_proc)
        raise

    try:
        with _ForwardSignals(command_proc, prewarm_proc):
            return _wait_supervised_command(
                command_proc,
                prewarm_proc=prewarm_proc,
            )
    except BaseException:
        _terminate_process(command_proc)
        _terminate_process(prewarm_proc)
        raise


def _normalize_command(raw_command: Sequence[str]) -> list[str]:
    command = list(raw_command)
    if command and command[0] == "--":
        command = command[1:]
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Start TensorCast local-ready target-plan prewarm, wait for "
            "materializing retained records, then exec a command."
        )
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help=(
            "Parent-side timeout for the materializing ready marker. Defaults "
            f"to {MATERIALIZING_READY_TIMEOUT_S_ENV} or 300s."
        ),
    )
    parser.add_argument(
        "--retained-manifest-write",
        default=None,
        help=(
            "Retained binding manifest output path. Defaults to "
            f"{RETAINED_MANIFEST_WRITE_ENV}, "
            f"{LOCAL_READY_RUN_DIR_ENV}/retained-binding-manifest.json, "
            "a target-plan cache-derived path, or <manifest>.retained.json."
        ),
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Run directory used to derive retained manifest, ready marker, "
            f"and prewarm log paths. Also sets {LOCAL_READY_RUN_DIR_ENV}."
        ),
    )
    parser.add_argument(
        "--target-plan-cache-ready-timeout-s",
        type=float,
        default=None,
        help=(
            "Wait this long for a target-plan cache manifest or source-path "
            f"index before starting prewarm. Defaults to "
            f"{TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV}; unset means no wait."
        ),
    )
    parser.add_argument(
        "--materializing-ready-write",
        default=None,
        help=(
            "Ready marker path. Defaults to "
            f"{MATERIALIZING_READY_WRITE_ENV} or "
            f"{RETAINED_MANIFEST_WRITE_ENV}.ready.json."
        ),
    )
    parser.add_argument(
        "--prewarm-log",
        default=None,
        help=(
            "Prewarmer log path. Defaults to "
            f"{PREWARM_LOG_ENV} or <ready-marker>.prewarm.log."
        ),
    )
    parser.add_argument(
        "--supervise",
        action="store_true",
        help=(
            "Run the command as a child process and reap the prewarmer instead "
            "of execing the command in-place."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after prewarm materializing readiness.",
    )
    args = parser.parse_args(argv)
    command = _normalize_command(args.command)
    if not command:
        raise SystemExit("command is required")
    if args.supervise:
        return run_after_materializing_ready(
            command,
            retained_manifest_write=args.retained_manifest_write,
            run_dir=args.run_dir,
            target_plan_cache_ready_timeout_s=(args.target_plan_cache_ready_timeout_s),
            materializing_ready_write=args.materializing_ready_write,
            prewarm_log=args.prewarm_log,
            timeout_s=args.timeout_s,
        )
    launch_after_materializing_ready(
        command,
        retained_manifest_write=args.retained_manifest_write,
        run_dir=args.run_dir,
        target_plan_cache_ready_timeout_s=(args.target_plan_cache_ready_timeout_s),
        materializing_ready_write=args.materializing_ready_write,
        prewarm_log=args.prewarm_log,
        timeout_s=args.timeout_s,
    )
    return 0


__all__ = [
    "LOCAL_READY_RUN_DIR_ENV",
    "LocalReadyLaunchConfig",
    "MATERIALIZING_READY_TIMEOUT_S_ENV",
    "PREWARM_LOG_ENV",
    "RETAINED_MANIFEST_ENV",
    "TARGET_PLAN_CACHE_READY_TIMEOUT_S_ENV",
    "launch_after_materializing_ready",
    "prepare_launch_environment",
    "run_after_materializing_ready",
    "start_prewarm_process",
    "wait_for_target_plan_cache_ready",
    "wait_for_materializing_ready_marker",
]

if __name__ == "__main__":
    raise SystemExit(main())
