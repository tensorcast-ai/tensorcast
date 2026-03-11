#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Minimal TP=8 TensorCast repro script for from_disk + subset + view.

This script is intentionally independent from vLLM. It reconstructs per-rank
subset/view arguments from TensorCast debug view args JSON files and runs:

    from_disk -> subset -> view -> tensor_dict

Expected input files are the dumps produced by vLLM's
`tensorcast_dump_view_args_path`, e.g.:
    /tmp/tensorcast_view_debug/tensorcast_view_args_tp0_pid123.json
    ...
    /tmp/tensorcast_view_debug/tensorcast_view_args_tp7_pid123.json

Examples:

1) Full TP=8 repro (default paths):
   python tools/tensorcast_tp8_from_disk_subset_view_repro.py

2) Use one specific run's files:
   python tools/tensorcast_tp8_from_disk_subset_view_repro.py \
     --args-pattern 'tensorcast_view_args_tp*_pid39754*.json'

3) Quick smoke test with fewer tensors:
   python tools/tensorcast_tp8_from_disk_subset_view_repro.py --name-limit 32

4) Materialize both CPU and GPU per rank:
   python tools/tensorcast_tp8_from_disk_subset_view_repro.py --dual-materialize
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TP_FILE_RE = re.compile(r".*_tp(\d+)_pid\d+\.json$")


@dataclass(frozen=True)
class WorkerTask:
    tp_rank: int
    args_file: str
    verify_checksums: bool
    show_progress: bool
    name_limit: int | None
    export_policy: str
    device_mode: str
    force_device: str | None
    set_cuda_visible_devices: bool
    worker_logs: bool
    dual_materialize: bool


def _dedupe_keep_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _parse_disk_path(artifact_ref: str) -> str:
    if not artifact_ref.startswith("disk:"):
        raise ValueError(
            f"Only disk artifacts are supported for this repro, got: {artifact_ref}"
        )
    return artifact_ref[len("disk:") :]


def _load_rank_args(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "tensorcast_slices" not in data:
        raise ValueError(f"Invalid args file (missing tensorcast_slices): {path}")
    return data


def _build_names_and_slices(
    data: dict[str, Any],
    name_limit: int | None,
) -> tuple[list[str], dict[str, list[tuple[int, slice]]]]:
    sliced_names = list(data.get("tensorcast_slices", {}).keys())
    full_read_names = list(data.get("full_read_names_sample", []))

    # Use the largest available set from debug args:
    # - all sliced tensors
    # - sampled full-read tensors
    names = _dedupe_keep_order(sliced_names + full_read_names)
    if name_limit is not None:
        names = names[: max(0, int(name_limit))]

    slices_raw = data.get("tensorcast_slices", {})
    slices: dict[str, list[tuple[int, slice]]] = {}
    for name in names:
        spec = slices_raw.get(name)
        if spec is None:
            continue
        dim = int(spec["dim"])
        start = int(spec["start"])
        end = int(spec["end"])
        slices[name] = [(dim, slice(start, end))]

    return names, slices


def _pick_device(
    data: dict[str, Any],
    tp_rank: int,
    mode: str,
    force_device: str | None,
) -> str:
    if force_device:
        return str(force_device)
    if mode == "cpu":
        return "cpu"
    if mode == "cuda_rank":
        return f"cuda:{tp_rank}"
    if mode == "from_args":
        return str(data.get("target_device", "cuda:0"))
    raise ValueError(f"Unsupported device_mode: {mode}")


def _pick_dual_gpu_device(
    data: dict[str, Any],
    task: WorkerTask,
    primary_device: str,
) -> str:
    if primary_device.startswith("cuda:"):
        return primary_device

    if task.force_device is not None:
        forced = str(task.force_device).strip()
        if forced.startswith("cuda:"):
            return forced

    target_from_args = str(data.get("target_device", "")).strip()
    if target_from_args.startswith("cuda:"):
        return target_from_args

    if task.set_cuda_visible_devices:
        # Each worker sees only one visible GPU in this mode.
        return "cuda:0"

    return f"cuda:{task.tp_rank}"


def _run_worker(task: WorkerTask) -> dict[str, Any]:
    start = time.time()
    rank_prefix = f"[tp{task.tp_rank}]"

    def _log(msg: str) -> None:
        if task.worker_logs:
            print(f"{rank_prefix} {msg}", flush=True)

    _log("worker_start")
    if task.set_cuda_visible_devices:
        # Mirror vLLM style process-local GPU view: each rank sees one GPU as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task.tp_rank)
        _log(f"set CUDA_VISIBLE_DEVICES={task.tp_rank}")

    # Import TensorCast only in worker process, after CUDA env is finalized.
    import tensorcast as tc

    args_data = _load_rank_args(task.args_file)
    tp_rank_in_file = int(args_data.get("tp_rank", task.tp_rank))

    names, slices = _build_names_and_slices(args_data, task.name_limit)
    disk_path = _parse_disk_path(str(args_data["artifact_ref"]))
    device = _pick_device(
        args_data,
        task.tp_rank,
        task.device_mode,
        task.force_device,
    )
    _log(
        f"resolved args_file={task.args_file} device={device} "
        f"subset={len(names)} slices={len(slices)}"
    )

    result: dict[str, Any] = {
        "tp_rank": int(task.tp_rank),
        "tp_rank_in_file": tp_rank_in_file,
        "args_file": task.args_file,
        "artifact_path": disk_path,
        "device": device,
        "dual_materialize": bool(task.dual_materialize),
        "subset_count": len(names),
        "slice_count": len(slices),
        "ok": False,
    }

    try:
        t0 = time.time()
        _log("from_disk_begin")
        artifact = tc.from_disk(
            disk_path,
            verify_checksums=task.verify_checksums,
            show_progress=task.show_progress,
        )
        artifact.describe()
        t1 = time.time()
        _log(f"from_disk_done sec={round(t1 - t0, 6)}")

        artifact_tp = artifact.subset(names).view(slices=slices)
        options = tc.GetArtifactOptions(
            export_policy=task.export_policy,
        )
        materialization: dict[str, dict[str, Any]] = {}

        if task.dual_materialize:
            gpu_device = _pick_dual_gpu_device(args_data, task, device)
            targets = [("cpu", "cpu"), ("gpu", gpu_device)]
            result["device"] = f"cpu+{gpu_device}"
        else:
            targets = [("primary", device)]

        _log(
            "materialize_begin "
            + ",".join(f"{name}:{target_device}" for name, target_device in targets)
        )

        for name, target_device in targets:
            m0 = time.time()
            tensor_dict = artifact_tp.tensor_dict(device=target_device, options=options)
            m1 = time.time()

            total_bytes = 0
            for tensor in tensor_dict.values():
                total_bytes += int(tensor.numel() * tensor.element_size())

            materialization[name] = {
                "device": target_device,
                "materialized_count": len(tensor_dict),
                "materialized_bytes": total_bytes,
                "materialize_sec": round(m1 - m0, 6),
            }
            _log(
                f"materialize_done target={name} device={target_device} "
                f"sec={round(m1 - m0, 6)} tensors={len(tensor_dict)}"
            )

        t2 = time.time()

        primary_key = "gpu" if task.dual_materialize else "primary"
        primary_stats = materialization[primary_key]

        result.update(
            {
                "ok": True,
                "materialized_count": int(primary_stats["materialized_count"]),
                "materialized_bytes": int(primary_stats["materialized_bytes"]),
                "from_disk_sec": round(t1 - t0, 6),
                "materialize_sec": round(t2 - t1, 6),
                "materialization": materialization,
                "total_sec": round(time.time() - start, 6),
            }
        )
        _log(f"worker_done ok=True total_sec={result['total_sec']}")
        return result
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "total_sec": round(time.time() - start, 6),
            }
        )
        _log(f"worker_done ok=False error_type={type(exc).__name__} error={exc}")
        return result


def _tp_rank_from_filename(path: str) -> int | None:
    m = TP_FILE_RE.match(Path(path).name)
    if not m:
        return None
    return int(m.group(1))


def _discover_args_files(
    args_dir: str,
    args_pattern: str,
    tp_world_size: int,
) -> dict[int, str]:
    pattern = str(Path(args_dir) / args_pattern)
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No args files matched: {pattern}")

    latest_by_tp: dict[int, tuple[float, str]] = {}
    for path in candidates:
        tp_rank = _tp_rank_from_filename(path)
        if tp_rank is None:
            continue
        mtime = Path(path).stat().st_mtime
        prev = latest_by_tp.get(tp_rank)
        if prev is None or mtime > prev[0]:
            latest_by_tp[tp_rank] = (mtime, path)

    selected: dict[int, str] = {
        tp_rank: item[1] for tp_rank, item in latest_by_tp.items()
    }

    missing = [rank for rank in range(tp_world_size) if rank not in selected]
    if missing:
        raise RuntimeError(
            f"Missing args files for ranks {missing}; matched pattern: {pattern}"
        )

    return {rank: selected[rank] for rank in range(tp_world_size)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TP=8 TensorCast minimal repro: from_disk + subset + view",
    )
    parser.add_argument(
        "--args-dir",
        default="/tmp/tensorcast_view_debug",
        help="Directory containing tensorcast_view_args_tp*_pid*.json",
    )
    parser.add_argument(
        "--args-pattern",
        default="tensorcast_view_args_tp*_pid*.json",
        help="Glob pattern inside --args-dir",
    )
    parser.add_argument(
        "--tp-world-size",
        type=int,
        default=8,
        help="Expected TP world size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel worker count",
    )
    parser.add_argument(
        "--name-limit",
        type=int,
        default=None,
        help="Optional cap on per-rank subset tensor count for quick smoke tests",
    )
    parser.add_argument(
        "--verify-checksums",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass verify_checksums to tc.from_disk",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass show_progress to tc.from_disk (per-worker TensorCast progress bars)",
    )
    parser.add_argument(
        "--overall-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show one aggregate progress bar in parent process (completed ranks / total)",
    )
    parser.add_argument(
        "--export-policy",
        default="force",
        help="GetArtifactOptions.export_policy",
    )
    parser.add_argument(
        "--device-mode",
        choices=["from_args", "cuda_rank", "cpu"],
        default="from_args",
        help="Materialization device selection mode",
    )
    parser.add_argument(
        "--force-device",
        default=None,
        help="Override device string for every rank, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--dual-materialize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Materialize both CPU and GPU in each worker (CPU first, then GPU)",
    )
    parser.add_argument(
        "--set-cuda-visible-devices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Set CUDA_VISIBLE_DEVICES=<tp_rank> in each worker",
    )
    parser.add_argument(
        "--worker-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-worker stage logs (start/from_disk/materialize/done)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write full JSON summary",
    )
    return parser.parse_args()


def _render_overall_progress(done: int, total: int) -> str:
    total = max(1, int(total))
    done = max(0, min(int(done), total))
    cols = shutil.get_terminal_size((120, 20)).columns
    bar_width = max(16, min(48, cols - 52))
    ratio = done / total
    fill = int(round(ratio * bar_width))
    bar = "#" * fill + "-" * (bar_width - fill)
    return f"[overall] [{bar}] {done}/{total} ({ratio * 100:6.2f}%)"


def main() -> int:
    args = _parse_args()

    files_by_rank = _discover_args_files(
        args_dir=args.args_dir,
        args_pattern=args.args_pattern,
        tp_world_size=args.tp_world_size,
    )

    print("[repro] selected args files:")
    for rank, path in files_by_rank.items():
        print(f"  tp{rank}: {path}")

    tasks = [
        WorkerTask(
            tp_rank=rank,
            args_file=path,
            verify_checksums=bool(args.verify_checksums),
            show_progress=bool(args.show_progress),
            name_limit=args.name_limit,
            export_policy=str(args.export_policy),
            device_mode=str(args.device_mode),
            force_device=(
                str(args.force_device) if args.force_device is not None else None
            ),
            set_cuda_visible_devices=bool(args.set_cuda_visible_devices),
            worker_logs=bool(args.worker_logs),
            dual_materialize=bool(args.dual_materialize),
        )
        for rank, path in files_by_rank.items()
    ]

    # Use spawn to avoid CUDA state sharing across workers.
    ctx = mp.get_context("spawn")

    started = time.time()
    results: list[dict[str, Any]] = []
    done = 0
    total = len(tasks)
    if args.overall_progress:
        print(_render_overall_progress(done, total), flush=True)
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)), mp_context=ctx
    ) as ex:
        future_to_rank = {ex.submit(_run_worker, task): task.tp_rank for task in tasks}
        for future in as_completed(future_to_rank):
            rank = future_to_rank[future]
            result = future.result()
            results.append(result)
            done += 1

            status = "OK" if result.get("ok") else "FAIL"
            print(
                f"[tp{rank}] {status} "
                f"subset={result.get('subset_count')} slices={result.get('slice_count')} "
                f"device={result.get('device')} total_sec={result.get('total_sec')}"
            )
            if not result.get("ok"):
                print(f"[tp{rank}] error_type={result.get('error_type')}")
                print(f"[tp{rank}] error={result.get('error')}")
            if args.overall_progress:
                print(_render_overall_progress(done, total), flush=True)

    results.sort(key=lambda x: int(x.get("tp_rank", 0)))
    failed = [r for r in results if not r.get("ok")]
    ok = [r for r in results if r.get("ok")]

    summary = {
        "tp_world_size": args.tp_world_size,
        "workers": args.workers,
        "elapsed_sec": round(time.time() - started, 6),
        "ok_count": len(ok),
        "fail_count": len(failed),
        "results": results,
    }

    print("[repro] summary:")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[repro] wrote json summary: {out_path}")

    # Print full traceback blocks for failures at the end.
    for item in failed:
        rank = item.get("tp_rank")
        print(f"[tp{rank}] traceback begin")
        print(item.get("traceback", ""))
        print(f"[tp{rank}] traceback end")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
