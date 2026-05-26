#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT.as_posix()
FANOUT_RUNNER = (
    REPO_ROOT / "examples" / "cross_host" / "cross_host_fanout_runner.py"
).as_posix()


@dataclass(frozen=True)
class ChaosEventSpec:
    offset_sec: float
    target_role: str
    action: str
    duration_sec: float
    expected_impact: str
    target_index: int | None
    command: str | None


@dataclass(frozen=True)
class CaseSpec:
    name: str
    fanout_args: dict[str, Any]
    chaos_events: tuple[ChaosEventSpec, ...]
    expected_outcome: str
    expected_error_pattern: str | None


def run(
    cmd: list[str], *, timeout_sec: float | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )


def run_remote(
    process_id: str, inner_cmd: str, *, timeout_sec: float
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "orchestratorctl",
        "exec",
        f"process/{process_id}",
        "-n",
        "tensorcast",
        "--",
        "bash",
        "-lc",
        inner_cmd,
    ]
    return run(cmd, timeout_sec=timeout_sec)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def stable_case_token(case_name: str) -> str:
    return hashlib.sha1(case_name.encode("utf-8")).hexdigest()[:10]


def worker_home(case_name: str, role: str, getter_index: int | None = None) -> str:
    case_root = f"/tmp/tc_cross_20260221/{stable_case_token(case_name)}"
    if role == "seed":
        return f"{case_root}/seed_h"
    if role == "getter" and getter_index is not None:
        return f"{case_root}/g{getter_index + 1}_h"
    raise ValueError(
        f"unsupported role for worker_home: role={role} getter_index={getter_index}"
    )


def daemon_session(case_name: str, role: str, getter_index: int | None = None) -> str:
    if role == "seed":
        return f"tc-fanout-{case_name}-seed"
    if role == "getter" and getter_index is not None:
        return f"tc-fanout-{case_name}-get{getter_index + 1}"
    raise ValueError(
        f"unsupported role for daemon_session: role={role} getter_index={getter_index}"
    )


def normalize_bool_flag(flag: str, value: bool) -> str:
    return f"--{flag}" if value else f"--no-{flag}"


def cli_args_from_dict(args_dict: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for raw_key, raw_value in args_dict.items():
        key = str(raw_key).strip()
        if not key:
            continue
        flag = key.replace("_", "-")
        if isinstance(raw_value, bool):
            argv.append(normalize_bool_flag(flag, raw_value))
            continue
        if raw_value is None:
            continue
        argv.extend((f"--{flag}", str(raw_value)))
    return argv


def parse_chaos_events(raw_events: list[dict[str, Any]]) -> tuple[ChaosEventSpec, ...]:
    return tuple(
        ChaosEventSpec(
            offset_sec=float(item.get("offset_sec", 0.0)),
            target_role=str(item.get("target_role", "seed")),
            action=str(item.get("action", "sleep")),
            duration_sec=float(item.get("duration_sec", 0.0)),
            expected_impact=str(item.get("expected_impact", "")),
            target_index=(
                int(item["target_index"])
                if item.get("target_index") is not None
                else None
            ),
            command=(
                str(item["command"]).strip()
                if item.get("command") is not None
                else None
            ),
        )
        for item in raw_events
    )


def parse_case_specs(schema: dict[str, Any]) -> tuple[CaseSpec, ...]:
    defaults = schema.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("schema.defaults must be an object")
    default_fanout = defaults.get("fanout_args", {})
    if not isinstance(default_fanout, dict):
        raise ValueError("schema.defaults.fanout_args must be an object")
    cases_raw = schema.get("cases", [])
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("schema.cases must be a non-empty array")
    cases: list[CaseSpec] = []
    for raw in cases_raw:
        if not isinstance(raw, dict):
            raise ValueError("each case entry must be an object")
        name = str(raw.get("name", "")).strip()
        if not name:
            raise ValueError("case.name is required")
        fanout_args = dict(default_fanout)
        raw_fanout = raw.get("fanout_args", {})
        if not isinstance(raw_fanout, dict):
            raise ValueError(f"case[{name}].fanout_args must be an object")
        fanout_args.update(raw_fanout)
        expected_outcome = str(raw.get("expected_outcome", "success")).strip().lower()
        if expected_outcome not in {"success", "failure"}:
            raise ValueError(f"case[{name}].expected_outcome must be success|failure")
        expected_error_pattern = raw.get("expected_error_pattern")
        if expected_error_pattern is not None:
            expected_error_pattern = str(expected_error_pattern)
        raw_events = raw.get("chaos_events", [])
        if not isinstance(raw_events, list):
            raise ValueError(f"case[{name}].chaos_events must be an array")
        cases.append(
            CaseSpec(
                name=name,
                fanout_args=fanout_args,
                chaos_events=parse_chaos_events(raw_events),
                expected_outcome=expected_outcome,
                expected_error_pattern=expected_error_pattern,
            )
        )
    return tuple(cases)


def load_schema(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("case schema root must be an object")
    return payload


def extract_summary_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("SUMMARY "):
            continue
        try:
            parsed = json.loads(line[len("SUMMARY ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_output_path(stdout: str) -> Path | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("OUTPUT "):
            raw_path = line[len("OUTPUT ") :].strip()
            if raw_path:
                return Path(raw_path)
    return None


def role_processes(
    case_name: str, fanout_args: dict[str, Any], event: ChaosEventSpec
) -> list[tuple[str, str, int | None]]:
    seed_proc = str(fanout_args.get("seed_proc", "")).strip()
    getter_procs = split_csv(str(fanout_args.get("get_procs", "")))
    role = event.target_role
    if role == "seed":
        return [("seed", seed_proc, None)]
    if role == "getter":
        if event.target_index is None:
            raise ValueError("getter target requires target_index")
        if event.target_index < 0 or event.target_index >= len(getter_procs):
            raise ValueError(f"getter target_index out of range: {event.target_index}")
        return [("getter", getter_procs[event.target_index], event.target_index)]
    if role == "all_getters":
        return [("getter", pid, idx) for idx, pid in enumerate(getter_procs)]
    if role == "custom":
        if not event.command:
            raise ValueError("custom target requires command")
        return [("custom", "", None)]
    raise ValueError(f"unsupported target_role={role}")


def execute_chaos_event(
    *,
    case_name: str,
    fanout_args: dict[str, Any],
    event: ChaosEventSpec,
    timeout_sec: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    targets = role_processes(case_name, fanout_args, event)
    if event.action == "sleep":
        if event.duration_sec > 0:
            time.sleep(event.duration_sec)
        records.append(
            {
                "target_role": event.target_role,
                "action": event.action,
                "ok": True,
                "duration_sec": event.duration_sec,
                "stdout_tail": "",
                "stderr_tail": "",
            }
        )
        return records

    for role, process_id, getter_index in targets:
        if role == "custom":
            result = run(["bash", "-lc", str(event.command)], timeout_sec=timeout_sec)
            records.append(
                {
                    "target_role": role,
                    "action": event.action,
                    "ok": result.returncode == 0,
                    "returncode": int(result.returncode),
                    "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
                    "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
                }
            )
            continue

        if event.action == "daemon_stop":
            session = daemon_session(case_name, role, getter_index)
            home = worker_home(case_name, role, getter_index)
            inner_cmd = (
                "set -euo pipefail; "
                f"export TENSORCAST_HOME={shlex.quote(home)}; "
                "source .venv/bin/activate; "
                "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon stop "
                f"--session {shlex.quote(session)}"
            )
        elif event.action == "daemon_kill":
            inner_cmd = (
                "set -euo pipefail; "
                "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do "
                'kill -KILL "$pid" >/dev/null 2>&1 || true; '
                "done"
            )
        elif event.action == "command":
            if not event.command:
                raise ValueError("event.action=command requires event.command")
            inner_cmd = str(event.command)
        else:
            raise ValueError(f"unsupported chaos action={event.action}")

        result = run_remote(process_id, inner_cmd, timeout_sec=timeout_sec)
        records.append(
            {
                "target_role": role,
                "target_process_id": process_id,
                "target_index": getter_index,
                "action": event.action,
                "ok": result.returncode == 0,
                "returncode": int(result.returncode),
                "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
                "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
            }
        )
        if event.duration_sec > 0:
            time.sleep(event.duration_sec)
    return records


def evaluate_case_outcome(
    *,
    expected_outcome: str,
    expected_error_pattern: str | None,
    returncode: int,
    merged_output: str,
) -> tuple[str, bool]:
    expected_failure = expected_outcome == "failure"
    if not expected_failure:
        if returncode == 0:
            return "pass", True
        return "unexpected_failure", False

    if returncode == 0:
        return "unexpected_success", False

    if not expected_error_pattern:
        return "expected_failure_pass", True
    matched = (
        re.search(expected_error_pattern, merged_output, flags=re.IGNORECASE)
        is not None
    )
    return ("expected_failure_pass", True) if matched else ("unexpected_failure", False)


def cleanup_case_processes(
    *,
    case_name: str,
    fanout_args: dict[str, Any],
    timeout_sec: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seed_proc = str(fanout_args.get("seed_proc", "")).strip()
    getter_procs = split_csv(str(fanout_args.get("get_procs", "")))
    targets: list[tuple[str, str, int | None]] = []
    if seed_proc:
        targets.append(("seed", seed_proc, None))
    for idx, proc_id in enumerate(getter_procs):
        targets.append(("getter", proc_id, idx))

    for role, process_id, getter_index in targets:
        if not process_id:
            continue
        session = daemon_session(case_name, role, getter_index)
        home = worker_home(case_name, role, getter_index)
        inner_cmd = (
            "set -euo pipefail; "
            f"cd {shlex.quote(BENCH_ROOT)}; "
            f"export TENSORCAST_HOME={shlex.quote(home)}; "
            "source .venv/bin/activate; "
            "LD_LIBRARY_PATH=/data/cuda/compat tensorcast-cli daemon stop "
            f"--session {shlex.quote(session)} >/dev/null 2>&1 || true; "
            "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do "
            'kill -TERM "$pid" >/dev/null 2>&1 || true; '
            "done; "
            "sleep 1; "
            "for pid in $(pgrep -f '[t]ensorcast_daemon --config=' || true); do "
            'kill -KILL "$pid" >/dev/null 2>&1 || true; '
            "done"
        )
        try:
            result = run_remote(process_id, inner_cmd, timeout_sec=timeout_sec)
            records.append(
                {
                    "action": "case_failure_cleanup",
                    "target_role": role,
                    "target_index": getter_index,
                    "target_process_id": process_id,
                    "ok": result.returncode == 0,
                    "returncode": int(result.returncode),
                    "stdout_tail": "\n".join(result.stdout.splitlines()[-20:]),
                    "stderr_tail": "\n".join(result.stderr.splitlines()[-20:]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "action": "case_failure_cleanup",
                    "target_role": role,
                    "target_index": getter_index,
                    "target_process_id": process_id,
                    "ok": False,
                    "returncode": -1,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "error": str(exc),
                }
            )
    return records


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(
    *,
    run_id: str,
    case: CaseSpec,
    case_dir: Path,
    run_events_path: Path,
    chaos_seed: int,
    remote_timeout_sec: float,
    cleanup_on_failure: bool,
    cleanup_timeout_sec: float,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    case_name = case.name
    fanout_args = dict(case.fanout_args)
    fanout_args["case_name"] = case_name
    fanout_args["out_dir"] = case_dir.as_posix()

    chaos_rng = random.Random(f"{run_id}:{case_name}:{chaos_seed}")
    indexed_events = list(enumerate(case.chaos_events))
    indexed_events.sort(
        key=lambda item: (
            float(item[1].offset_sec),
            chaos_rng.random(),
            item[0],
        )
    )

    chaos_records: list[dict[str, Any]] = []
    stop_scheduler = threading.Event()
    scheduler_error: list[str] = []
    case_start_epoch = time.time()
    case_start_mono = time.monotonic()

    def scheduler() -> None:
        for order, event in indexed_events:
            if stop_scheduler.is_set():
                break
            target_at = case_start_mono + max(0.0, float(event.offset_sec))
            while True:
                now = time.monotonic()
                if now >= target_at:
                    break
                if stop_scheduler.is_set():
                    return
                time.sleep(min(0.05, target_at - now))
            correlation_id = f"{run_id}:{case_name}:event{order:03d}"
            start_epoch = time.time()
            base_record = {
                "run_id": run_id,
                "case_name": case_name,
                "type": "chaos_event",
                "correlation_id": correlation_id,
                "offset_sec": float(event.offset_sec),
                "target_role": event.target_role,
                "action": event.action,
                "expected_impact": event.expected_impact,
                "target_index": event.target_index,
            }
            try:
                rows = execute_chaos_event(
                    case_name=case_name,
                    fanout_args=fanout_args,
                    event=event,
                    timeout_sec=remote_timeout_sec,
                )
                for row in rows:
                    merged = dict(base_record)
                    merged["ts_epoch"] = float(start_epoch)
                    merged["event_phase"] = "done"
                    merged.update(row)
                    chaos_records.append(merged)
            except Exception as exc:  # noqa: BLE001
                scheduler_error.append(str(exc))
                merged = dict(base_record)
                merged["ts_epoch"] = float(start_epoch)
                merged["event_phase"] = "error"
                merged["ok"] = False
                merged["error"] = str(exc)
                chaos_records.append(merged)

    scheduler_thread = threading.Thread(target=scheduler, daemon=True)
    scheduler_thread.start()

    cmd = [sys.executable, FANOUT_RUNNER, *cli_args_from_dict(fanout_args)]
    runner_exec_error = ""
    try:
        proc = run(cmd, timeout_sec=max(1.0, remote_timeout_sec))
    except subprocess.TimeoutExpired as exc:
        runner_exec_error = (
            f"fanout runner timeout (timeout_sec={max(1.0, remote_timeout_sec):.1f})"
        )
        timeout_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        timeout_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        if timeout_stderr:
            timeout_stderr = f"{timeout_stderr}\n{runner_exec_error}"
        else:
            timeout_stderr = runner_exec_error
        proc = subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=timeout_stdout,
            stderr=timeout_stderr,
        )
    except Exception as exc:  # noqa: BLE001
        runner_exec_error = f"fanout runner execution error: {exc}"
        proc = subprocess.CompletedProcess(
            args=cmd,
            returncode=125,
            stdout="",
            stderr=runner_exec_error,
        )
    stop_scheduler.set()
    scheduler_thread.join(timeout=10.0)

    case_end_epoch = time.time()
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    merged_output = f"{stdout}\n{stderr}"
    status_label, passed = evaluate_case_outcome(
        expected_outcome=case.expected_outcome,
        expected_error_pattern=case.expected_error_pattern,
        returncode=int(proc.returncode),
        merged_output=merged_output,
    )
    case_failed = (
        int(proc.returncode) != 0 or bool(scheduler_error) or bool(runner_exec_error)
    )
    cleanup_records: list[dict[str, Any]] = []
    if cleanup_on_failure and case_failed:
        cleanup_records = cleanup_case_processes(
            case_name=case_name,
            fanout_args=fanout_args,
            timeout_sec=cleanup_timeout_sec,
        )

    fanout_output_path = extract_output_path(stdout)
    if fanout_output_path is None:
        candidate = case_dir / f"{case_name}.json"
        fanout_output_path = candidate if candidate.exists() else None

    fanout_summary: dict[str, Any] = {}
    fanout_events: list[dict[str, Any]] = []
    if fanout_output_path is not None and fanout_output_path.exists():
        try:
            loaded = json.loads(fanout_output_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                fanout_summary = (
                    loaded.get("summary", {})
                    if isinstance(loaded.get("summary"), dict)
                    else {}
                )
                fanout_events_raw = loaded.get("events", [])
                if isinstance(fanout_events_raw, list):
                    for entry in fanout_events_raw:
                        if isinstance(entry, dict):
                            event = dict(entry)
                            event["run_id"] = run_id
                            event["case_name"] = case_name
                            event["type"] = "traffic_event"
                            fanout_events.append(event)
        except json.JSONDecodeError:
            pass

    if not fanout_summary:
        fallback_summary = extract_summary_line(stdout)
        if fallback_summary is not None:
            fanout_summary = fallback_summary

    cleanup_events: list[dict[str, Any]] = []
    for row in cleanup_records:
        event = dict(row)
        event["run_id"] = run_id
        event["case_name"] = case_name
        event["type"] = "case_cleanup"
        event["ts_epoch"] = float(time.time())
        cleanup_events.append(event)

    all_events = [*chaos_records, *fanout_events, *cleanup_events]
    case_events_path = case_dir / "events.jsonl"
    if not run_events_path.exists():
        run_events_path.touch()
    if not case_events_path.exists():
        case_events_path.touch()
    append_jsonl(run_events_path, all_events)
    append_jsonl(case_events_path, all_events)

    classification = fanout_summary.get("failure_classification_counts", {})
    if not isinstance(classification, dict):
        classification = {}
    classification_payload = {
        "run_id": run_id,
        "case_name": case_name,
        "status": status_label,
        "expected_outcome": case.expected_outcome,
        "classification_counts": {
            "infra": int(classification.get("infra", 0)),
            "product": int(classification.get("product", 0)),
            "unknown": int(classification.get("unknown", 0)),
        },
    }

    metrics_payload: dict[str, Any] = {
        "run_id": run_id,
        "case_name": case_name,
        "all_get_complete": bool(fanout_summary.get("all_get_complete", False)),
        "source_cardinality_timeline": fanout_summary.get(
            "source_cardinality_timeline", []
        ),
        "recover_time_sec": float(fanout_summary.get("recover_time_sec", 0.0)),
        "put_success_rate": float(fanout_summary.get("put_success_rate", 0.0)),
        "get_success_rate": float(fanout_summary.get("get_success_rate", 0.0)),
        "comm_bytes_delta": int(fanout_summary.get("comm_bytes_delta", 0)),
        "comm_errors_delta": int(fanout_summary.get("comm_errors_delta", 0)),
        "retry_reason_buckets": fanout_summary.get("retry_reason_buckets", {}),
        "budget_exit_reason_buckets": fanout_summary.get(
            "budget_exit_reason_buckets", {}
        ),
    }

    result_payload: dict[str, Any] = {
        "run_id": run_id,
        "case_name": case_name,
        "expected_outcome": case.expected_outcome,
        "expected_error_pattern": case.expected_error_pattern,
        "status": status_label,
        "passed": bool(passed),
        "expected_failure_pass": status_label == "expected_failure_pass",
        "unexpected_failure": status_label == "unexpected_failure",
        "unexpected_success": status_label == "unexpected_success",
        "returncode": int(proc.returncode),
        "start_epoch": float(case_start_epoch),
        "end_epoch": float(case_end_epoch),
        "duration_sec": float(case_end_epoch - case_start_epoch),
        "command": " ".join(shlex.quote(part) for part in cmd),
        "scheduler_errors": scheduler_error,
        "runner_exec_error": runner_exec_error,
        "case_failed": bool(case_failed),
        "failure_cleanup_applied": bool(cleanup_on_failure and case_failed),
        "cleanup_error_count": int(
            sum(1 for row in cleanup_records if not bool(row.get("ok", False)))
        ),
        "cleanup_records": cleanup_records,
        "stdout_tail": "\n".join(stdout.splitlines()[-60:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-60:]),
        "fanout_output_path": fanout_output_path.as_posix()
        if fanout_output_path
        else "",
        "events_count": int(len(all_events)),
    }

    write_json(case_dir / "result.json", result_payload)
    write_json(case_dir / "metrics.json", metrics_payload)
    write_json(case_dir / "classification.json", classification_payload)

    return {
        "case_name": case_name,
        "status": status_label,
        "passed": bool(passed),
        "expected_outcome": case.expected_outcome,
        "expected_failure_pass": status_label == "expected_failure_pass",
        "unexpected_failure": status_label == "unexpected_failure",
        "unexpected_success": status_label == "unexpected_success",
        "summary": fanout_summary,
        "metrics": metrics_payload,
        "result": result_payload,
    }


def aggregate_summary(
    run_id: str, case_results: list[dict[str, Any]]
) -> dict[str, Any]:
    positive = [
        item for item in case_results if item.get("expected_outcome") == "success"
    ]
    negative = [
        item for item in case_results if item.get("expected_outcome") == "failure"
    ]

    all_get_complete = (
        all(
            bool(item.get("metrics", {}).get("all_get_complete", False))
            for item in positive
        )
        if positive
        else True
    )

    source_cardinality_timeline: list[dict[str, Any]] = []
    retry_reason_buckets: dict[str, int] = {}
    budget_exit_reason_buckets: dict[str, int] = {}
    comm_bytes_delta = 0
    comm_errors_delta = 0
    recover_time_sec_max = 0.0
    classification_counts = {"infra": 0, "product": 0, "unknown": 0}

    for item in case_results:
        case_name = str(item.get("case_name", ""))
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        comm_bytes_delta += int(metrics.get("comm_bytes_delta", 0))
        comm_errors_delta += int(metrics.get("comm_errors_delta", 0))
        recover_time_sec_max = max(
            recover_time_sec_max, float(metrics.get("recover_time_sec", 0.0))
        )

        timeline = metrics.get("source_cardinality_timeline", [])
        if isinstance(timeline, list):
            for row in timeline:
                if isinstance(row, dict):
                    merged = dict(row)
                    merged["case_name"] = case_name
                    source_cardinality_timeline.append(merged)

        for key in ("retry_reason_buckets", "budget_exit_reason_buckets"):
            raw = metrics.get(key, {})
            if not isinstance(raw, dict):
                continue
            sink = (
                retry_reason_buckets
                if key == "retry_reason_buckets"
                else budget_exit_reason_buckets
            )
            for reason, count in raw.items():
                sink[str(reason)] = sink.get(str(reason), 0) + int(count or 0)

        summary = item.get("summary", {})
        if isinstance(summary, dict):
            raw_cls = summary.get("failure_classification_counts", {})
            if isinstance(raw_cls, dict):
                for key in ("infra", "product", "unknown"):
                    classification_counts[key] += int(raw_cls.get(key, 0))

    expected_failure_pass = (
        all(bool(item.get("expected_failure_pass")) for item in negative)
        if negative
        else True
    )
    unexpected_failures = [
        str(item.get("case_name"))
        for item in case_results
        if bool(item.get("unexpected_failure"))
    ]
    unexpected_successes = [
        str(item.get("case_name"))
        for item in case_results
        if bool(item.get("unexpected_success"))
    ]

    return {
        "run_id": run_id,
        "case_count": int(len(case_results)),
        "pass_count": int(sum(1 for item in case_results if bool(item.get("passed")))),
        "fail_count": int(
            sum(1 for item in case_results if not bool(item.get("passed")))
        ),
        "all_get_complete": bool(all_get_complete),
        "source_cardinality_timeline": source_cardinality_timeline,
        "recover_time_sec": float(recover_time_sec_max),
        "comm_bytes_delta": int(comm_bytes_delta),
        "comm_errors_delta": int(comm_errors_delta),
        "retry_reason_buckets": retry_reason_buckets,
        "budget_exit_reason_buckets": budget_exit_reason_buckets,
        "expected_failure_pass": bool(expected_failure_pass),
        "unexpected_failures": unexpected_failures,
        "unexpected_successes": unexpected_successes,
        "failure_classification_counts": classification_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replayable multi-host chaos runner around cross_host_fanout_runner with "
            "event timeline and expected-failure gating."
        )
    )
    parser.add_argument(
        "--case-schema", required=True, help="JSON schema path for chaos cases"
    )
    parser.add_argument("--out-dir", required=True, help="Run output root")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--chaos-seed", type=int, default=7)
    parser.add_argument("--remote-timeout-sec", type=float, default=900.0)
    parser.add_argument(
        "--case-failure-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Best-effort daemon cleanup for seed/getters when a case fails.",
    )
    parser.add_argument(
        "--cleanup-timeout-sec",
        type=float,
        default=180.0,
        help="Timeout for each remote cleanup action.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema_path = Path(str(args.case_schema)).resolve()
    schema = load_schema(schema_path)
    cases = parse_case_specs(schema)

    run_id = str(args.run_id).strip() or f"{int(time.time())}"
    run_dir = Path(str(args.out_dir)).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_events_path = run_dir / "events.jsonl"
    if run_events_path.exists():
        run_events_path.unlink()
    run_events_path.touch()

    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_dir = run_dir / "cases" / case.name
        print(f"[chaos] case={case.name} expected={case.expected_outcome}", flush=True)
        case_result = run_case(
            run_id=run_id,
            case=case,
            case_dir=case_dir,
            run_events_path=run_events_path,
            chaos_seed=int(args.chaos_seed),
            remote_timeout_sec=float(args.remote_timeout_sec),
            cleanup_on_failure=bool(args.case_failure_cleanup),
            cleanup_timeout_sec=float(args.cleanup_timeout_sec),
        )
        case_results.append(case_result)
        print(
            f"[chaos] case={case.name} status={case_result['status']} "
            f"passed={case_result['passed']}",
            flush=True,
        )

    summary = aggregate_summary(run_id, case_results)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "cases.json",
        {
            "run_id": run_id,
            "cases": [
                {
                    "name": item["case_name"],
                    "status": item["status"],
                    "passed": item["passed"],
                    "expected_outcome": item["expected_outcome"],
                }
                for item in case_results
            ],
        },
    )

    print(f"SUMMARY {json.dumps(summary, ensure_ascii=False)}", flush=True)
    print(f"OUTPUT {run_dir}", flush=True)

    if summary["fail_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
