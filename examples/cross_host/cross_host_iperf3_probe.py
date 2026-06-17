#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import math
import os
import pwd
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REMOTE_USER_RE = re.compile(r"^[a-z_][a-z0-9_.-]{0,63}$")


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return float(ordered[idx])


def _summarize_series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "p50": float(_percentile(values, 0.5)),
        "p90": float(_percentile(values, 0.9)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
    }


def _normalize_non_root_user(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError("resolved empty workspace user for remote execution")
    if not REMOTE_USER_RE.fullmatch(value):
        raise ValueError(f"invalid workspace user for remote execution: {value!r}")
    if value == "root":
        raise ValueError(
            "workspace user resolved to root; refusing remote execution as root"
        )
    return value


def _resolve_workspace_user() -> str:
    env_user = str(os.environ.get("USER", "")).strip()
    if env_user:
        try:
            return _normalize_non_root_user(env_user)
        except ValueError:
            pass
    try:
        uid_user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError as exc:
        raise RuntimeError(
            f"failed to resolve workspace user from uid={os.getuid()}"
        ) from exc
    return _normalize_non_root_user(uid_user)


def _wrap_remote_inner_cmd_for_user(
    *,
    inner_cmd: str,
    run_as_user: str,
) -> str:
    quoted_inner = shlex.quote(str(inner_cmd))
    quoted_user = shlex.quote(str(run_as_user))
    return (
        "set -euo pipefail; "
        f"run_as_user={quoted_user}; "
        'if ! getent passwd "${run_as_user}" >/dev/null 2>&1; then '
        'echo "remote run-as user not found: ${run_as_user}" >&2; '
        "exit 97; "
        "fi; "
        'if [[ "$(id -un)" == "${run_as_user}" ]]; then '
        f"bash -lc {quoted_inner}; "
        "else "
        f'su - "${{run_as_user}}" -s /bin/bash -c {quoted_inner}; '
        "fi"
    )


def _tail_text(raw: str | None, *, max_chars: int = 4000) -> str:
    if raw is None:
        return ""
    text = str(raw)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _run(cmd: str, *, timeout_sec: float | None = None) -> str:
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_tail = _tail_text(exc.stdout)
        stderr_tail = _tail_text(exc.stderr)
        raise RuntimeError(
            "command timeout "
            f"(timeout_sec={timeout_sec}): {cmd}\n"
            f"[stdout tail]\n{stdout_tail}\n"
            f"[stderr tail]\n{stderr_tail}"
        ) from exc
    if proc.returncode != 0:
        stdout_tail = _tail_text(proc.stdout)
        stderr_tail = _tail_text(proc.stderr)
        raise RuntimeError(
            "command failed "
            f"(rc={proc.returncode}, timeout_sec={timeout_sec}): {cmd}\n"
            f"[stdout tail]\n{stdout_tail}\n"
            f"[stderr tail]\n{stderr_tail}"
        )
    return proc.stdout


def _run_remote(
    *,
    process_id: str,
    inner_cmd: str,
    run_as_user: str,
    timeout_sec: float,
) -> str:
    wrapped_cmd = _wrap_remote_inner_cmd_for_user(
        inner_cmd=inner_cmd,
        run_as_user=run_as_user,
    )
    cmd = (
        f"orchestratorctl exec process/{process_id} -n tensorcast -- bash -lc "
        f"{shlex.quote(wrapped_cmd)}"
    )
    return _run(cmd, timeout_sec=timeout_sec)


def _start_remote_server(
    *,
    process_id: str,
    port: int,
    run_as_user: str,
) -> subprocess.Popen[str]:
    inner_cmd = f"set -euo pipefail; iperf3 -s -1 -p {port} --json"
    wrapped_cmd = _wrap_remote_inner_cmd_for_user(
        inner_cmd=inner_cmd,
        run_as_user=run_as_user,
    )
    cmd = (
        f"orchestratorctl exec process/{process_id} -n tensorcast -- bash -lc "
        f"{shlex.quote(wrapped_cmd)}"
    )
    return subprocess.Popen(  # noqa: S603
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _parse_iperf_json_from_output(raw_output: str) -> dict[str, Any]:
    text = str(raw_output or "")
    # iperf3 --json may still include wrapper lines; parse the last full json object.
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no json object found in iperf output")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("iperf output json root is not object")
    return payload


def _extract_iperf_gibps(payload: dict[str, Any]) -> float:
    end = payload.get("end", {})
    if not isinstance(end, dict):
        end = {}
    for key in ("sum_received", "sum_sent", "sum"):
        node = end.get(key, {})
        if not isinstance(node, dict):
            continue
        bits_per_second = _safe_float(node.get("bits_per_second"))
        if bits_per_second is None or bits_per_second <= 0.0:
            continue
        return float(bits_per_second / 8.0 / float(1024**3))
    raise ValueError("failed to parse bits_per_second from iperf json end summary")


@dataclass(slots=True)
class ProbeDirectionResult:
    direction: str
    server_process: str
    client_process: str
    server_host: str
    port: int
    gibps: float | None
    error: str | None
    client_bits_per_second: float | None
    server_bits_per_second: float | None


@dataclass(slots=True)
class PairProbeResult:
    getter_name: str
    getter_process: str
    getter_ip: str
    seed_to_getter: ProbeDirectionResult
    getter_to_seed: ProbeDirectionResult


def _probe_direction(
    *,
    direction: str,
    server_proc: str,
    client_proc: str,
    server_host: str,
    port: int,
    run_as_user: str,
    duration_sec: int,
    parallel: int,
    remote_timeout_sec: float,
    startup_wait_sec: float,
) -> ProbeDirectionResult:
    server = _start_remote_server(
        process_id=server_proc,
        port=port,
        run_as_user=run_as_user,
    )
    time.sleep(max(0.1, startup_wait_sec))

    client_json: dict[str, Any] | None = None
    server_json: dict[str, Any] | None = None
    error: str | None = None
    gibps: float | None = None
    client_bits_per_second: float | None = None
    server_bits_per_second: float | None = None

    try:
        client_output = _run_remote(
            process_id=client_proc,
            run_as_user=run_as_user,
            inner_cmd=(
                "set -euo pipefail; "
                f"iperf3 -c {shlex.quote(server_host)} -p {int(port)} "
                f"-t {int(duration_sec)} -P {int(parallel)} --json"
            ),
            timeout_sec=remote_timeout_sec,
        )
        client_json = _parse_iperf_json_from_output(client_output)
    except Exception as exc:  # noqa: BLE001
        error = f"client_error: {exc}"

    try:
        server_stdout, server_stderr = server.communicate(timeout=remote_timeout_sec)
    except subprocess.TimeoutExpired:
        server.kill()
        server_stdout, server_stderr = server.communicate()
        timeout_msg = "server_error: timeout waiting iperf server output"
        error = timeout_msg if error is None else f"{error}; {timeout_msg}"
    else:
        if server.returncode != 0:
            server_msg = (
                "server_error: "
                f"rc={server.returncode} "
                f"stderr_tail={_tail_text(server_stderr)!r}"
            )
            error = server_msg if error is None else f"{error}; {server_msg}"
        else:
            try:
                server_json = _parse_iperf_json_from_output(server_stdout)
            except Exception as exc:  # noqa: BLE001
                server_msg = f"server_json_error: {exc}"
                error = server_msg if error is None else f"{error}; {server_msg}"

    if client_json is not None:
        client_end = client_json.get("end", {})
        if isinstance(client_end, dict):
            for key in ("sum_received", "sum_sent", "sum"):
                node = client_end.get(key, {})
                if not isinstance(node, dict):
                    continue
                value = _safe_float(node.get("bits_per_second"))
                if value is not None and value > 0.0:
                    client_bits_per_second = float(value)
                    break
    if server_json is not None:
        server_end = server_json.get("end", {})
        if isinstance(server_end, dict):
            for key in ("sum_received", "sum_sent", "sum"):
                node = server_end.get(key, {})
                if not isinstance(node, dict):
                    continue
                value = _safe_float(node.get("bits_per_second"))
                if value is not None and value > 0.0:
                    server_bits_per_second = float(value)
                    break

    if error is None and client_json is not None:
        try:
            gibps = _extract_iperf_gibps(client_json)
        except Exception as exc:  # noqa: BLE001
            error = f"parse_error: {exc}"

    return ProbeDirectionResult(
        direction=direction,
        server_process=server_proc,
        client_process=client_proc,
        server_host=server_host,
        port=int(port),
        gibps=gibps,
        error=error,
        client_bits_per_second=client_bits_per_second,
        server_bits_per_second=server_bits_per_second,
    )


def _render_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _build_markdown(
    *,
    summary: dict[str, Any],
    pairs: list[PairProbeResult],
) -> str:
    lines: list[str] = []
    lines.append(
        f"# Iperf3 Probe Summary ({datetime.now(tz=timezone.utc).isoformat()})"
    )
    lines.append("")
    lines.append(
        "- single_link_ref_gibps="
        f"{_render_float(_safe_float(summary.get('single_link_ref_gibps')))}"
    )
    lines.append(
        "- bidirectional_floor_gibps="
        f"{_render_float(_safe_float(summary.get('bidirectional_floor_gibps')))}"
    )
    lines.append("")
    lines.append(
        "| getter | seed->getter GiB/s | getter->seed GiB/s | pair_floor GiB/s | errors |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for pair in pairs:
        fwd = pair.seed_to_getter.gibps
        rev = pair.getter_to_seed.gibps
        floor = min([x for x in (fwd, rev) if x is not None], default=None)
        errors: list[str] = []
        if pair.seed_to_getter.error:
            errors.append(f"seed->getter: {pair.seed_to_getter.error}")
        if pair.getter_to_seed.error:
            errors.append(f"getter->seed: {pair.getter_to_seed.error}")
        lines.append(
            "| "
            + " | ".join(
                [
                    pair.getter_name,
                    _render_float(fwd),
                    _render_float(rev),
                    _render_float(floor),
                    "; ".join(errors) if errors else "",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _compute_summary(pairs: list[PairProbeResult]) -> dict[str, Any]:
    seed_to_getter_vals: list[float] = []
    getter_to_seed_vals: list[float] = []
    pair_floor_vals: list[float] = []
    for pair in pairs:
        if pair.seed_to_getter.gibps is not None:
            seed_to_getter_vals.append(pair.seed_to_getter.gibps)
        if pair.getter_to_seed.gibps is not None:
            getter_to_seed_vals.append(pair.getter_to_seed.gibps)
        values = [
            value
            for value in (pair.seed_to_getter.gibps, pair.getter_to_seed.gibps)
            if value is not None
        ]
        if values:
            pair_floor_vals.append(min(values))

    pair_floor_p50 = _safe_float(_summarize_series(pair_floor_vals).get("p50"))
    bidirectional_floor = float(min(pair_floor_vals)) if pair_floor_vals else None
    return {
        "seed_to_getter_gibps": _summarize_series(seed_to_getter_vals),
        "getter_to_seed_gibps": _summarize_series(getter_to_seed_vals),
        "pair_floor_gibps": _summarize_series(pair_floor_vals),
        # single-link reference used by early gate:
        # median of per-pair bidirectional floor.
        "single_link_ref_gibps": pair_floor_p50,
        # strict floor across all sampled pairs.
        "bidirectional_floor_gibps": bidirectional_floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe cross-host single-link ceiling via bidirectional iperf3."
    )
    parser.add_argument(
        "--seed-proc", required=True, help="Seed orchestratorctl process id."
    )
    parser.add_argument(
        "--seed-ip",
        required=True,
        help="Seed advertise IP for getter->seed iperf direction.",
    )
    parser.add_argument(
        "--get-procs",
        required=True,
        help="Comma-separated getter process ids.",
    )
    parser.add_argument(
        "--get-ips",
        required=True,
        help="Comma-separated getter advertise IPs.",
    )
    parser.add_argument(
        "--sample-getters",
        type=int,
        default=2,
        help="Probe first N getters (default: 2).",
    )
    parser.add_argument(
        "--duration-sec",
        type=int,
        default=20,
        help="iperf3 test duration per direction.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=20,
        help="iperf3 -P parallel streams.",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=63900,
        help="Base server port, incremented by pair index and direction.",
    )
    parser.add_argument(
        "--remote-timeout-sec",
        type=float,
        default=240.0,
        help="Timeout for each remote server/client execution.",
    )
    parser.add_argument(
        "--startup-wait-sec",
        type=float,
        default=1.0,
        help="Wait time after server starts before launching client.",
    )
    parser.add_argument(
        "--run-as-user",
        default="",
        help="Remote non-root user. Default resolves from current local user.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output json path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown path.",
    )
    args = parser.parse_args()

    get_procs = _parse_csv(args.get_procs)
    get_ips = _parse_csv(args.get_ips)
    if len(get_procs) != len(get_ips):
        raise SystemExit("--get-procs count must equal --get-ips count")
    if not get_procs:
        raise SystemExit("no getters provided")
    if args.sample_getters <= 0:
        raise SystemExit("--sample-getters must be > 0")
    if args.duration_sec <= 0:
        raise SystemExit("--duration-sec must be > 0")
    if args.parallel <= 0:
        raise SystemExit("--parallel must be > 0")
    if args.port_base <= 0:
        raise SystemExit("--port-base must be > 0")

    run_as_user = (
        _normalize_non_root_user(args.run_as_user)
        if str(args.run_as_user).strip()
        else _resolve_workspace_user()
    )
    sample_getters = min(args.sample_getters, len(get_procs))

    pairs: list[PairProbeResult] = []
    for idx in range(sample_getters):
        getter_proc = get_procs[idx]
        getter_ip = get_ips[idx]
        getter_name = f"get{idx + 1}"

        port_seed_to_getter = int(args.port_base + idx * 2)
        port_getter_to_seed = int(args.port_base + idx * 2 + 1)

        print(
            f"[iperf-probe] pair={getter_name} seed->getter port={port_seed_to_getter}"
        )
        seed_to_getter = _probe_direction(
            direction="seed_to_getter",
            server_proc=getter_proc,
            client_proc=args.seed_proc,
            server_host=getter_ip,
            port=port_seed_to_getter,
            run_as_user=run_as_user,
            duration_sec=args.duration_sec,
            parallel=args.parallel,
            remote_timeout_sec=args.remote_timeout_sec,
            startup_wait_sec=args.startup_wait_sec,
        )

        print(
            f"[iperf-probe] pair={getter_name} getter->seed port={port_getter_to_seed}"
        )
        getter_to_seed = _probe_direction(
            direction="getter_to_seed",
            server_proc=args.seed_proc,
            client_proc=getter_proc,
            server_host=str(args.seed_ip),
            port=port_getter_to_seed,
            run_as_user=run_as_user,
            duration_sec=args.duration_sec,
            parallel=args.parallel,
            remote_timeout_sec=args.remote_timeout_sec,
            startup_wait_sec=args.startup_wait_sec,
        )

        pairs.append(
            PairProbeResult(
                getter_name=getter_name,
                getter_process=getter_proc,
                getter_ip=getter_ip,
                seed_to_getter=seed_to_getter,
                getter_to_seed=getter_to_seed,
            )
        )

    summary = _compute_summary(pairs)
    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_as_user": run_as_user,
        "params": {
            "seed_proc": args.seed_proc,
            "seed_ip": str(args.seed_ip),
            "get_procs": get_procs,
            "get_ips": get_ips,
            "sample_getters": sample_getters,
            "duration_sec": int(args.duration_sec),
            "parallel": int(args.parallel),
            "port_base": int(args.port_base),
            "remote_timeout_sec": float(args.remote_timeout_sec),
            "startup_wait_sec": float(args.startup_wait_sec),
        },
        "pairs": [asdict(pair) for pair in pairs],
        "summary": summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[iperf-probe] json={args.out_json}")

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            _build_markdown(summary=summary, pairs=pairs),
            encoding="utf-8",
        )
        print(f"[iperf-probe] md={args.out_md}")


if __name__ == "__main__":
    main()
