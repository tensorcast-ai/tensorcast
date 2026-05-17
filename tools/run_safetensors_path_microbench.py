#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

"""Run a small TensorCast safetensors path-selection microbenchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCH_BIN = (
    REPO_ROOT
    / "bazel-bin/core/store/materialization/benchmarks/"
    "safetensors_load_strategy_benchmark"
)


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def _read_proc_status(pid: int) -> dict[str, int]:
    out: dict[str, int] = {}
    status_path = Path("/proc") / str(pid) / "status"
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                key, rest = line.split(":", 1)
                fields = rest.strip().split()
                if fields:
                    out[key] = int(fields[0])
    except FileNotFoundError:
        pass
    return out


def _sample_gpu_memory() -> dict[str, int]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    out: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        idx, used = [part.strip() for part in line.split(",", 1)]
        out[idx] = int(used)
    return out


def _parse_result_lines(log_text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        marker = None
        if " result strategy=" in line:
            marker = "result"
            payload = line.split(" result ", 1)[1]
        else:
            match = re.search(
                r"(safetensors_[a-z_]+_baseline|h2d_baseline|"
                r"nccl_[a-z_]+|disk_[a-z_]+) ",
                line,
            )
            if match is None:
                continue
            marker = match.group(1)
            payload = line[match.start() :]
        fields: dict[str, Any] = {"kind": marker, "raw": payload}
        for key, value in re.findall(r"([A-Za-z0-9_./()=-]+)=([^ ]+)", payload):
            fields[key] = value.rstrip(",")
        results.append(fields)
    return results


def _run_case(
    *,
    name: str,
    cmd: list[str],
    output_dir: Path,
    sample_interval_sec: float,
    env: dict[str, str],
) -> dict[str, Any]:
    log_path = output_dir / f"{name}.log"
    samples_path = output_dir / f"{name}.samples.jsonl"
    started_at = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    peak_proc: dict[str, int] = {}
    peak_gpu: dict[str, int] = {}
    with samples_path.open("w", encoding="utf-8") as samples_file:
        while proc.poll() is None:
            proc_status = _read_proc_status(proc.pid)
            gpu = _sample_gpu_memory()
            for key, value in proc_status.items():
                peak_proc[key] = max(peak_proc.get(key, 0), value)
            for key, value in gpu.items():
                peak_gpu[key] = max(peak_gpu.get(key, 0), value)
            samples_file.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "proc": proc_status,
                        "gpu_memory_mib": gpu,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            samples_file.flush()
            time.sleep(sample_interval_sec)

    stdout, _ = proc.communicate()
    ended_at = time.time()
    log_path.write_text(stdout or "", encoding="utf-8")
    return {
        "name": name,
        "cmd": cmd,
        "returncode": proc.returncode,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": ended_at - started_at,
        "log_path": str(log_path),
        "samples_path": str(samples_path),
        "peak_proc_kib": peak_proc,
        "peak_gpu_mib": peak_gpu,
        "results": _parse_result_lines(stdout or ""),
    }


def _run_concurrent_case(
    *,
    name: str,
    rank_cmds: list[tuple[int, list[str]]],
    output_dir: Path,
    sample_interval_sec: float,
    env: dict[str, str],
) -> dict[str, Any]:
    samples_path = output_dir / f"{name}.samples.jsonl"
    started_at = time.time()
    procs: list[dict[str, Any]] = []
    for rank, cmd in rank_cmds:
        log_path = output_dir / f"{name}_rank{rank}.log"
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        procs.append(
            {
                "rank": rank,
                "cmd": cmd,
                "log_path": log_path,
                "log_file": log_file,
                "proc": proc,
            }
        )

    peak_proc_by_rank: dict[str, dict[str, int]] = {}
    peak_gpu: dict[str, int] = {}
    with samples_path.open("w", encoding="utf-8") as samples_file:
        while any(item["proc"].poll() is None for item in procs):
            proc_samples: dict[str, dict[str, int]] = {}
            for item in procs:
                rank_key = str(item["rank"])
                proc_status = _read_proc_status(item["proc"].pid)
                proc_samples[rank_key] = proc_status
                rank_peak = peak_proc_by_rank.setdefault(rank_key, {})
                for key, value in proc_status.items():
                    rank_peak[key] = max(rank_peak.get(key, 0), value)
            gpu = _sample_gpu_memory()
            for key, value in gpu.items():
                peak_gpu[key] = max(peak_gpu.get(key, 0), value)
            samples_file.write(
                json.dumps(
                    {
                        "time": time.time(),
                        "procs": proc_samples,
                        "gpu_memory_mib": gpu,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            samples_file.flush()
            time.sleep(sample_interval_sec)

    ended_at = time.time()
    rank_results: list[dict[str, Any]] = []
    combined_results: list[dict[str, Any]] = []
    returncodes: list[int] = []
    for item in procs:
        proc = item["proc"]
        returncodes.append(proc.wait())
        item["log_file"].close()
        log_text = item["log_path"].read_text(encoding="utf-8")
        parsed_results = _parse_result_lines(log_text)
        for parsed in parsed_results:
            combined_results.append({"rank": item["rank"], **parsed})
        rank_results.append(
            {
                "rank": item["rank"],
                "cmd": item["cmd"],
                "returncode": proc.returncode,
                "log_path": str(item["log_path"]),
                "peak_proc_kib": peak_proc_by_rank.get(str(item["rank"]), {}),
                "results": parsed_results,
            }
        )

    failed = [rc for rc in returncodes if rc != 0]
    return {
        "name": name,
        "kind": "concurrent",
        "returncode": failed[0] if failed else 0,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_sec": ended_at - started_at,
        "samples_path": str(samples_path),
        "peak_proc_by_rank_kib": peak_proc_by_rank,
        "peak_gpu_mib": peak_gpu,
        "rank_results": rank_results,
        "results": combined_results,
    }


def _convert_trace_plan(args: argparse.Namespace, load_plan: Path) -> None:
    if load_plan.exists() and not args.force_convert:
        return
    if args.trace_plan_dir is None:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/trace_plan_to_load_plan.py"),
        "--model-dir",
        str(args.model_dir),
        "--trace-plan-dir",
        str(args.trace_plan_dir),
        "--output",
        str(load_plan),
    ]
    if args.tp_world_size:
        cmd.extend(["--tp-world-size", str(args.tp_world_size)])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def _base_bench_cmd(args: argparse.Namespace) -> list[str]:
    return [
        str(args.benchmark_bin),
        "--safetensors_dir",
        str(args.model_dir),
        "--use_pinned_host_buffer=true",
        f"--io_threads={args.io_threads}",
        f"--bbuf_size_kb={args.bbuf_size_kb}",
        f"--buffer_chunks={args.buffer_chunks}",
        f"--pinned_numa_node={args.pinned_numa_node}",
        f"--pinned_numa_prefault={str(args.pinned_numa_prefault).lower()}",
    ]


def _loader_strategies(args: argparse.Namespace) -> list[str]:
    loader_strategies: list[str] = []
    if "loader_b" in args.case:
        loader_strategies.append("b")
    if "loader_c" in args.case:
        loader_strategies.append("c")
    if args.include_strategy_a:
        loader_strategies.insert(0, "a")
    return loader_strategies


def _build_loader_cmd(
    args: argparse.Namespace,
    load_plan: Path,
    *,
    rank: int,
    strategy: str,
    devices: list[int],
    world_size: int,
) -> list[str]:
    return [
        *_base_bench_cmd(args),
        "--mode=loader",
        f"--strategy={strategy}",
        f"--tp_world_size={world_size}",
        f"--tp_rank={rank}",
        f"--device_id={devices[rank % len(devices)]}",
        f"--load_plan_json_path={load_plan}",
        f"--strategy_c_staging_bytes={args.strategy_c_staging_bytes}",
    ]


def _build_cases(args: argparse.Namespace, load_plan: Path) -> list[tuple[str, list[str]]]:
    devices = _parse_csv_ints(args.tp_devices)
    world_size = args.tp_world_size or len(devices)
    cases: list[tuple[str, list[str]]] = []
    base = _base_bench_cmd(args)

    if "odirect_host" in args.case:
        cases.append(
            (
                "odirect_host_full",
                [*base, "--mode=safetensors_o_direct_host_baseline"],
            )
        )
    if "buffered_host" in args.case:
        cases.append(("buffered_host_full", [*base, "--mode=safetensors_host_baseline"]))
    if "h2d_1gpu" in args.case:
        cases.append(
            (
                "h2d_1gpu",
                [
                    *base,
                    "--mode=h2d_baseline",
                    "--tp_world_size=1",
                    f"--tp_devices={devices[0]}",
                    f"--h2d_bench_bytes={args.h2d_bench_bytes}",
                ],
            )
        )
    if "h2d_tp" in args.case:
        cases.append(
            (
                f"h2d_tp{world_size}",
                [
                    *base,
                    "--mode=h2d_baseline",
                    f"--tp_world_size={world_size}",
                    f"--tp_devices={args.tp_devices}",
                    "--h2d_per_gpu_pinned_pool=true",
                    f"--h2d_bench_bytes={args.h2d_bench_bytes}",
                ],
            )
        )

    loader_strategies = _loader_strategies(args)
    for rank in args.rank:
        for strategy in loader_strategies:
            cases.append(
                (
                    f"loader_rank{rank}_strategy_{strategy}",
                    _build_loader_cmd(
                        args,
                        load_plan,
                        rank=rank,
                        strategy=strategy,
                        devices=devices,
                        world_size=world_size,
                    ),
                )
            )
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--trace-plan-dir", type=Path)
    parser.add_argument("--load-plan", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-bin", type=Path, default=DEFAULT_BENCH_BIN)
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument("--tp-world-size", type=int, default=0)
    parser.add_argument("--tp-devices", default="0,1,2,3")
    parser.add_argument("--rank", type=int, action="append")
    parser.add_argument(
        "--concurrent-ranks",
        action="store_true",
        help=(
            "Run loader strategy cases for the selected ranks concurrently. "
            "This models TP startup disk contention and records per-rank logs."
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "odirect_host",
            "buffered_host",
            "h2d_1gpu",
            "h2d_tp",
            "loader_b",
            "loader_c",
        ),
        default=[],
    )
    parser.add_argument("--include-strategy-a", action="store_true")
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--bbuf-size-kb", type=int, default=262144)
    parser.add_argument("--buffer-chunks", type=int, default=8)
    parser.add_argument("--pinned-numa-node", type=int, default=-2)
    parser.add_argument("--pinned-numa-prefault", action="store_true", default=True)
    parser.add_argument("--sample-interval-sec", type=float, default=0.5)
    parser.add_argument("--h2d-bench-bytes", type=int, default=8 * 1024 * 1024 * 1024)
    parser.add_argument("--strategy-c-staging-bytes", type=int, default=1024 * 1024 * 1024)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    devices = _parse_csv_ints(args.tp_devices)
    world_size = args.tp_world_size or len(devices)
    if args.rank is None:
        args.rank = list(range(world_size)) if args.concurrent_ranks else [0]
    bad_ranks = [rank for rank in args.rank if rank < 0 or rank >= world_size]
    if bad_ranks:
        raise SystemExit(f"--rank must be in [0,{world_size}); got {bad_ranks}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.case:
        args.case = (
            ["loader_c"]
            if args.concurrent_ranks
            else ["odirect_host", "h2d_1gpu", "h2d_tp", "loader_b", "loader_c"]
        )
    load_plan = args.load_plan or (args.output_dir / "load_plan.json")
    _convert_trace_plan(args, load_plan)
    if any(case.startswith("loader_") for case in args.case) and not load_plan.exists():
        raise SystemExit("--load-plan or --trace-plan-dir is required for loader cases")
    if args.concurrent_ranks:
        unsupported_cases = [
            case for case in args.case if case not in ("loader_b", "loader_c")
        ]
        if unsupported_cases:
            raise SystemExit(
                "--concurrent-ranks supports loader_b/loader_c only; got "
                f"{unsupported_cases}"
            )

    env = os.environ.copy()
    summary: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "load_plan": str(load_plan),
        "benchmark_bin": str(args.benchmark_bin),
        "concurrent_ranks": args.concurrent_ranks,
        "ranks": args.rank,
        "cases": [],
    }
    if args.concurrent_ranks:
        for strategy in _loader_strategies(args):
            name = f"loader_tp{world_size}_strategy_{strategy}_concurrent"
            print(f"running {name}", flush=True)
            rank_cmds = [
                (
                    rank,
                    _build_loader_cmd(
                        args,
                        load_plan,
                        rank=rank,
                        strategy=strategy,
                        devices=devices,
                        world_size=world_size,
                    ),
                )
                for rank in args.rank
            ]
            result = _run_concurrent_case(
                name=name,
                rank_cmds=rank_cmds,
                output_dir=args.output_dir,
                sample_interval_sec=args.sample_interval_sec,
                env=env,
            )
            summary["cases"].append(result)
            (args.output_dir / "summary.partial.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if result["returncode"] != 0:
                break
    else:
        for name, cmd in _build_cases(args, load_plan):
            print(f"running {name}", flush=True)
            result = _run_case(
                name=name,
                cmd=cmd,
                output_dir=args.output_dir,
                sample_interval_sec=args.sample_interval_sec,
                env=env,
            )
            summary["cases"].append(result)
            (args.output_dir / "summary.partial.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if result["returncode"] != 0:
                break

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [case for case in summary["cases"] if case["returncode"] != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
