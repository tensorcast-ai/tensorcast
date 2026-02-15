#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""End-to-end harness for tensorcast.tools.weight_publisher.

Scenarios:
1) single-host: run publisher and receiver concurrently on one node.
2) publisher: distributed publisher role.
3) receiver: distributed receiver role.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors.torch import save_file

import tensorcast as tc
from tensorcast import FallbackOptions
from tensorcast.api.store import artifact as resolve_artifact
from tensorcast.api.store.runtime import get_context as get_store_context
from tensorcast.tools.weight_publisher import WeightPublisher, WeightPublisherConfig


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "not_found" in msg
        or "not found" in msg
        or "key not found" in msg
        or "statuscode.not_found" in msg
        or "no available replicas" in msg
    )


@dataclass(frozen=True)
class PublishEvent:
    version: int
    key: str
    artifact_id: str
    export_dir: str
    published_at_s: float


@dataclass(frozen=True)
class ReceiveEvent:
    version: int
    key: str
    artifact_id: str
    received_at_s: float
    materialize_latency_s: float


def _ensure_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _ensure_non_negative(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _build_key(*, model_name: str, key_template: str, version: int) -> str:
    key = key_template.format(
        model_name=model_name,
        version=version,
        weight_version=version,
    ).strip()
    if not key:
        raise ValueError("resolved key is empty")
    return key


def _materialization_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("TENSORCAST_CUDA_BACKEND") == "fake":
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _prepare_export_dir(
    *,
    run_root: Path,
    version: int,
) -> Path:
    export_dir = run_root / "exports" / f"v{version:05d}"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    version_marker = torch.tensor([version], dtype=torch.int64)
    weight_probe = torch.arange(16, dtype=torch.float32).reshape(4, 4) + float(version)
    weight_probe[0, 0] = float(version)
    checksum_seed = int(version * 9973 + 17)
    rolling_checksum = torch.tensor(
        [checksum_seed, checksum_seed + 1, checksum_seed + 2],
        dtype=torch.int64,
    )

    save_file(
        {
            "version_marker": version_marker,
            "weight_probe": weight_probe,
            "rolling_checksum": rolling_checksum,
        },
        str(export_dir / "model.safetensors"),
    )
    return export_dir


def _validate_payload(*, version: int, tensors: dict[str, torch.Tensor]) -> None:
    expected = {"version_marker", "weight_probe", "rolling_checksum"}
    missing = expected - set(tensors)
    if missing:
        raise AssertionError(f"missing tensors in payload: {sorted(missing)}")

    marker = int(tensors["version_marker"].reshape(-1)[0].cpu().item())
    if marker != version:
        raise AssertionError(
            f"version_marker mismatch: expected={version}, actual={marker}"
        )

    probe_origin = float(tensors["weight_probe"][0, 0].cpu().item())
    if probe_origin != float(version):
        raise AssertionError(
            f"weight_probe[0,0] mismatch: expected={float(version)}, actual={probe_origin}"
        )

    checksum_seed = int(version * 9973 + 17)
    checksum_sum = int(tensors["rolling_checksum"].sum().cpu().item())
    expected_sum = checksum_seed + (checksum_seed + 1) + (checksum_seed + 2)
    if checksum_sum != expected_sum:
        raise AssertionError(
            f"rolling_checksum sum mismatch: expected={expected_sum}, actual={checksum_sum}"
        )


class WeightUpdatePublisher:
    def __init__(
        self,
        *,
        model_name: str,
        key_template: str,
        keep_last: int,
        history_path: Path,
        run_root: Path,
        check_poll_interval_s: float,
        check_timeout_s: float,
    ) -> None:
        self._model_name = model_name
        self._key_template = key_template
        self._run_root = run_root
        self._check_poll_interval_s = check_poll_interval_s
        self._check_timeout_s = check_timeout_s
        self._config = WeightPublisherConfig(
            model_name=model_name,
            keep_last=keep_last,
            history_path=str(history_path),
            key_template=key_template,
            trigger_reload=False,
            verify_key_mapping=True,
            from_disk_verify_checksums=True,
            wait_persistence=False,
        )
        self._publisher = WeightPublisher(self._config)

    def publish_versions(
        self,
        *,
        start_version: int,
        num_versions: int,
        publish_interval_s: float,
    ) -> list[PublishEvent]:
        _ensure_positive("num_versions", num_versions)
        _ensure_positive("start_version", start_version)

        events: list[PublishEvent] = []
        for offset in range(num_versions):
            version = start_version + offset
            self.publish_one_version(version=version, events=events)
            if offset + 1 < num_versions:
                time.sleep(max(0.0, publish_interval_s))
        return events

    def publish_one_version(
        self,
        *,
        version: int,
        events: list[PublishEvent],
    ) -> PublishEvent:
        export_dir = _prepare_export_dir(run_root=self._run_root, version=version)
        key = _build_key(
            model_name=self._model_name,
            key_template=self._key_template,
            version=version,
        )
        artifact_id = self._publisher.publish_from_disk(export_dir, version=version)
        event = PublishEvent(
            version=version,
            key=key,
            artifact_id=artifact_id,
            export_dir=str(export_dir),
            published_at_s=time.time(),
        )
        events.append(event)
        self._verify_retention_window(events=events)
        print(
            "[publisher] published",
            f"version={version}",
            f"key={key}",
            f"artifact_id={artifact_id}",
            flush=True,
        )
        return event

    def _verify_retention_window(self, *, events: list[PublishEvent]) -> None:
        keep_last = int(self._config.keep_last)
        if keep_last <= 0 or not events:
            return
        kept_versions = {event.version for event in events[-keep_last:]}
        for event in events:
            self._wait_key_mapping_state(
                key=event.key,
                expected_artifact_id=event.artifact_id,
            )
            expected_materializable = event.version in kept_versions
            self._wait_materialization_state(
                key=event.key,
                version=event.version,
                expected_materializable=expected_materializable,
            )

    def _probe_artifact_exists(self, artifact_id: str) -> bool:
        try:
            return resolve_artifact(artifact_id=artifact_id).exists()
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return False
            raise

    def _resolve_key_mapping(self, key: str) -> str | None:
        try:
            mapping = get_store_context().ensure_client().resolve_key_mapping(key)
            resolved = str(mapping.artifact_id or "").strip()
            if not resolved:
                return None
            return resolved
        except Exception as exc:  # noqa: BLE001
            if _is_not_found_error(exc):
                return None
            raise

    def _wait_key_mapping_state(self, *, key: str, expected_artifact_id: str) -> None:
        deadline = time.monotonic() + self._check_timeout_s
        last_resolved: str | None = None
        while time.monotonic() < deadline:
            resolved = self._resolve_key_mapping(key)
            last_resolved = resolved
            if resolved == expected_artifact_id:
                return
            time.sleep(self._check_poll_interval_s)
        raise AssertionError(
            f"key mapping mismatch: key={key}, "
            f"expected_artifact_id={expected_artifact_id}, last_resolved={last_resolved}"
        )

    def _probe_materializable(self, *, key: str, version: int) -> bool:
        try:
            artifact = resolve_artifact(key=key).with_fallback(
                self._fallback_for_checks()
            )
            tensors = artifact.tensor_dict(device="cpu")
            _validate_payload(version=version, tensors=tensors)
            return True
        except Exception:
            return False

    def _fallback_for_checks(self) -> FallbackOptions:
        return FallbackOptions(
            prefer="auto",
            allow_p2p=True,
            allow_disk=True,
            verify_checksums=False,
        )

    def _wait_materialization_state(
        self,
        *,
        key: str,
        version: int,
        expected_materializable: bool,
    ) -> None:
        deadline = time.monotonic() + self._check_timeout_s
        last_state: bool | None = None
        while time.monotonic() < deadline:
            materializable = self._probe_materializable(key=key, version=version)
            last_state = materializable
            if materializable == expected_materializable:
                return
            time.sleep(self._check_poll_interval_s)
        raise AssertionError(
            f"materialization retention mismatch: key={key}, version={version}, "
            f"expected_materializable={expected_materializable}, last_materializable={last_state}"
        )


class WeightUpdateReceiver:
    def __init__(
        self,
        *,
        model_name: str,
        key_template: str,
        poll_interval_s: float,
        per_version_timeout_s: float,
        fallback_prefer: str,
        materialize_device: str,
    ) -> None:
        self._model_name = model_name
        self._key_template = key_template
        self._poll_interval_s = poll_interval_s
        self._per_version_timeout_s = per_version_timeout_s
        self._fallback = FallbackOptions(
            prefer=fallback_prefer,  # pyright: ignore[reportArgumentType]
            allow_p2p=True,
            allow_disk=True,
            verify_checksums=False,
        )
        self._materialize_device = _materialization_device(materialize_device)

    def receive_versions(
        self,
        *,
        start_version: int,
        num_versions: int,
        on_event: Callable[[ReceiveEvent], None] | None = None,
    ) -> list[ReceiveEvent]:
        _ensure_positive("num_versions", num_versions)
        _ensure_positive("start_version", start_version)

        events: list[ReceiveEvent] = []
        previous_artifact_id: str | None = None
        for offset in range(num_versions):
            version = start_version + offset
            key = _build_key(
                model_name=self._model_name,
                key_template=self._key_template,
                version=version,
            )
            event = self._wait_one(version=version, key=key)
            if previous_artifact_id == event.artifact_id:
                raise AssertionError(
                    f"artifact_id reused across versions: version={version}, artifact_id={event.artifact_id}"
                )
            previous_artifact_id = event.artifact_id
            events.append(event)
            if on_event is not None:
                on_event(event)
            print(
                "[receiver] received",
                f"version={version}",
                f"key={key}",
                f"artifact_id={event.artifact_id}",
                f"latency_s={event.materialize_latency_s:.3f}",
                flush=True,
            )
        return events

    def _wait_one(self, *, version: int, key: str) -> ReceiveEvent:
        deadline = time.monotonic() + self._per_version_timeout_s
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                artifact = resolve_artifact(key=key).with_fallback(self._fallback)
                if not artifact.exists():
                    time.sleep(self._poll_interval_s)
                    continue
                start = time.monotonic()
                tensors = artifact.tensor_dict(device=self._materialize_device)
                latency_s = time.monotonic() - start
                _validate_payload(version=version, tensors=tensors)
                return ReceiveEvent(
                    version=version,
                    key=key,
                    artifact_id=artifact.artifact_id,
                    received_at_s=time.time(),
                    materialize_latency_s=latency_s,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(self._poll_interval_s)

        raise TimeoutError(
            f"receiver timeout for version={version}, key={key}, last_error={last_error}"
        )


def _init_tensorcast(args: argparse.Namespace) -> None:
    init_mode = str(args.init_mode)
    if init_mode == "connect":
        if args.connect_address:
            tc.init(mode="connect", address=str(args.connect_address))
        else:
            tc.init(mode="connect")
        return
    tc.init(
        mode=init_mode,  # pyright: ignore[reportArgumentType]
        daemon_config_path=str(args.daemon_config_path)
        if args.daemon_config_path
        else None,
        global_store_mode=args.global_store_mode,  # pyright: ignore[reportArgumentType]
        global_store_address=str(args.global_store_address)
        if args.global_store_address
        else None,
        global_store_config_path=str(args.global_store_config_path)
        if args.global_store_config_path
        else None,
        cluster_id=str(args.cluster_id) if args.cluster_id else None,
        allow_gs_fallback=bool(args.allow_gs_fallback),
    )


def _resolve_run_root(args: argparse.Namespace) -> Path:
    base = Path(str(args.weights_root)).expanduser().resolve()
    run_id = str(args.run_id).strip() if args.run_id else ""
    if not run_id:
        run_id = f"run-{int(time.time())}-{os.getpid()}"
    run_root = base / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _resolve_model_name(args: argparse.Namespace, *, run_root: Path | None) -> str:
    if args.model_name:
        return str(args.model_name).strip()
    if run_root is None:
        raise ValueError("model_name is required when run root is unavailable")
    suffix = run_root.name.replace("_", "-")
    return f"weight-publisher-e2e-{suffix}"


def _to_jsonable(
    items: list[PublishEvent] | list[ReceiveEvent],
) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def _write_summary(path: Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_publisher_runner(
    args: argparse.Namespace,
    *,
    run_root: Path,
    model_name: str,
) -> WeightUpdatePublisher:
    history_path = (
        Path(str(args.history_path)).expanduser().resolve()
        if args.history_path
        else run_root / "publisher_history.json"
    )
    return WeightUpdatePublisher(
        model_name=model_name,
        key_template=str(args.key_template),
        keep_last=_ensure_non_negative("keep_last", int(args.keep_last)),
        history_path=history_path,
        run_root=run_root,
        check_poll_interval_s=float(args.poll_interval_s),
        check_timeout_s=float(args.retention_timeout_s),
    )


def _build_receiver_runner(
    args: argparse.Namespace,
    *,
    model_name: str,
) -> WeightUpdateReceiver:
    return WeightUpdateReceiver(
        model_name=model_name,
        key_template=str(args.key_template),
        poll_interval_s=float(args.poll_interval_s),
        per_version_timeout_s=float(args.receiver_timeout_s),
        fallback_prefer=str(args.fallback_prefer),
        materialize_device=str(args.materialize_device),
    )


def _run_publisher(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    try:
        run_root = _resolve_run_root(args)
        model_name = _resolve_model_name(args, run_root=run_root)
        publisher = _build_publisher_runner(
            args, run_root=run_root, model_name=model_name
        )
        events = publisher.publish_versions(
            start_version=int(args.start_version),
            num_versions=int(args.num_versions),
            publish_interval_s=float(args.publish_interval_s),
        )
        summary = {
            "mode": "publisher",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "keep_last": int(args.keep_last),
            "run_root": str(run_root),
            "published": _to_jsonable(events),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else run_root / "publisher_summary.json"
        )
        _write_summary(output, summary)
        return 0
    finally:
        tc.shutdown()


def _run_receiver(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    try:
        model_name = _resolve_model_name(args, run_root=None)
        receiver = _build_receiver_runner(args, model_name=model_name)
        events = receiver.receive_versions(
            start_version=int(args.start_version),
            num_versions=int(args.num_versions),
        )
        summary = {
            "mode": "receiver",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "materialize_device": _materialization_device(str(args.materialize_device)),
            "received": _to_jsonable(events),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else None
        )
        _write_summary(output, summary)
        return 0
    finally:
        tc.shutdown()


def _run_single_host(args: argparse.Namespace) -> int:
    _init_tensorcast(args)
    try:
        run_root = _resolve_run_root(args)
        model_name = _resolve_model_name(args, run_root=run_root)
        publisher = _build_publisher_runner(
            args, run_root=run_root, model_name=model_name
        )
        receiver = _build_receiver_runner(args, model_name=model_name)

        receiver_holder: dict[str, Any] = {"events": None, "error": None}
        ack_condition = threading.Condition()
        acked_version = int(args.start_version) - 1

        def _on_receive_event(event: ReceiveEvent) -> None:
            nonlocal acked_version
            with ack_condition:
                acked_version = max(acked_version, event.version)
                ack_condition.notify_all()

        def _wait_for_receiver_ack(version: int, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            with ack_condition:
                while acked_version < version:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"receiver did not ack version={version} within {timeout_s}s"
                        )
                    ack_condition.wait(timeout=min(1.0, remaining))

        def _receiver_worker() -> None:
            try:
                receiver_holder["events"] = receiver.receive_versions(
                    start_version=int(args.start_version),
                    num_versions=int(args.num_versions),
                    on_event=_on_receive_event,
                )
            except Exception as exc:  # noqa: BLE001
                receiver_holder["error"] = exc

        worker = threading.Thread(
            target=_receiver_worker,
            name="weight-publisher-e2e-receiver",
            daemon=True,
        )
        worker.start()
        time.sleep(max(0.0, float(args.receiver_start_delay_s)))

        published: list[PublishEvent] = []
        start_version = int(args.start_version)
        num_versions = int(args.num_versions)
        for offset in range(num_versions):
            version = start_version + offset
            publisher.publish_one_version(version=version, events=published)
            _wait_for_receiver_ack(
                version=version,
                timeout_s=float(args.receiver_timeout_s),
            )
            if offset + 1 < num_versions:
                time.sleep(max(0.0, float(args.publish_interval_s)))

        join_timeout = max(
            5.0,
            float(args.receiver_timeout_s) * float(args.num_versions) + 10.0,
        )
        worker.join(timeout=join_timeout)
        if worker.is_alive():
            raise TimeoutError("receiver worker did not finish in expected time")
        if receiver_holder["error"] is not None:
            raise RuntimeError(
                f"receiver failed: {receiver_holder['error']}"
            ) from receiver_holder["error"]
        received = receiver_holder["events"]
        if received is None:
            raise RuntimeError("receiver produced no result")

        expected_versions = list(
            range(
                int(args.start_version),
                int(args.start_version) + int(args.num_versions),
            )
        )
        actual_versions = [event.version for event in received]
        if expected_versions != actual_versions:
            raise AssertionError(
                f"receiver version sequence mismatch: expected={expected_versions}, actual={actual_versions}"
            )

        summary = {
            "mode": "single-host",
            "model_name": model_name,
            "start_version": int(args.start_version),
            "num_versions": int(args.num_versions),
            "keep_last": int(args.keep_last),
            "run_root": str(run_root),
            "published": _to_jsonable(published),
            "received": _to_jsonable(received),
        }
        output = (
            Path(str(args.output_json)).expanduser().resolve()
            if args.output_json
            else run_root / "single_host_summary.json"
        )
        _write_summary(output, summary)
        return 0
    finally:
        tc.shutdown()


def _add_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_init_mode: str,
    default_global_store_mode: str,
) -> None:
    parser.add_argument(
        "--init-mode",
        choices=["connect", "create", "auto"],
        default=default_init_mode,
        help="TensorCast init mode.",
    )
    parser.add_argument(
        "--connect-address",
        default=None,
        help="Daemon address for connect mode, e.g. 127.0.0.1:8073.",
    )
    parser.add_argument(
        "--daemon-config-path",
        default=None,
        help="Daemon config path for create/auto mode.",
    )
    parser.add_argument(
        "--global-store-mode",
        choices=["none", "connect", "start"],
        default=default_global_store_mode,
        help="Global Store orchestration mode for create/auto mode.",
    )
    parser.add_argument(
        "--global-store-address",
        default=None,
        help="Global Store address for global_store_mode=connect.",
    )
    parser.add_argument(
        "--global-store-config-path",
        default=None,
        help="Global Store config path for global_store_mode=start.",
    )
    parser.add_argument(
        "--cluster-id",
        default=None,
        help="Optional cluster token to enforce for Global Store.",
    )
    parser.add_argument(
        "--allow-gs-fallback",
        action="store_true",
        help="Allow TensorCast to fall back when Global Store start/connect fails.",
    )


def _add_common_stream_args(
    parser: argparse.ArgumentParser,
    *,
    require_model_name: bool,
) -> None:
    parser.add_argument(
        "--model-name",
        required=require_model_name,
        help="Logical model name used by key_template.",
    )
    parser.add_argument(
        "--key-template",
        default="model:{model_name}:v{weight_version}",
        help="Versioned key template.",
    )
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--num-versions", type=int, default=3)
    parser.add_argument("--poll-interval-s", type=float, default=0.5)
    parser.add_argument("--receiver-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--fallback-prefer",
        choices=["auto", "local", "p2p", "disk"],
        default="auto",
    )
    parser.add_argument(
        "--materialize-device",
        default="auto",
        help="Receiver materialization device: auto/cpu/cuda:0...",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output summary path.",
    )


def _add_publisher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-interval-s", type=float, default=2.0)
    parser.add_argument("--keep-last", type=int, default=2)
    parser.add_argument("--retention-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--weights-root",
        default="/tmp/tensorcast_weight_publisher_e2e",
        help="Root directory for generated version exports.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier used under weights-root.",
    )
    parser.add_argument(
        "--history-path",
        default=None,
        help="Optional explicit publisher history path.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WeightPublisher E2E harness for single-host and distributed tests.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser(
        "single-host",
        help="Run publisher and receiver concurrently on one node.",
    )
    _add_runtime_args(
        single,
        default_init_mode="auto",
        default_global_store_mode="start",
    )
    _add_common_stream_args(single, require_model_name=False)
    _add_publisher_args(single)
    single.add_argument(
        "--receiver-start-delay-s",
        type=float,
        default=1.0,
        help="Delay after receiver thread starts before publishing begins.",
    )

    pub = subparsers.add_parser(
        "publisher",
        help="Run publisher role only (for distributed test).",
    )
    _add_runtime_args(
        pub,
        default_init_mode="connect",
        default_global_store_mode="none",
    )
    _add_common_stream_args(pub, require_model_name=False)
    _add_publisher_args(pub)

    recv = subparsers.add_parser(
        "receiver",
        help="Run receiver role only (for distributed test).",
    )
    _add_runtime_args(
        recv,
        default_init_mode="connect",
        default_global_store_mode="none",
    )
    _add_common_stream_args(recv, require_model_name=True)

    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "single-host":
        return _run_single_host(args)
    if args.mode == "publisher":
        return _run_publisher(args)
    if args.mode == "receiver":
        return _run_receiver(args)
    parser.error(f"unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
