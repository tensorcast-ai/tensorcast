#!/usr/bin/env python3
# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class FanoutSnapshot:
    path: str
    mtime: float
    case_name: str
    workers: int
    wave1_getters: int | None
    wave2_getters: int | None
    size_mib: int | None
    l0_pass: bool
    p2p_ratio: float | None
    get_success_rate: float | None
    cluster_gibps_mean: float | None
    wave1_transfer_gibps_mean: float | None
    wave2_transfer_gibps_mean: float | None
    wave2_over_wave1_transfer_ratio: float | None


@dataclass(slots=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def _parse_round_workers(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        parsed = _safe_int(token)
        if parsed is None:
            continue
        if parsed < 2:
            continue
        if parsed in out:
            continue
        out.append(parsed)
    return out


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"json root is not object: {path}")
    return payload


def _parse_case(path: Path) -> FanoutSnapshot | None:
    payload = _read_json(path)
    if str(payload.get("mode", "")).strip().lower() != "fanout":
        return None

    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}

    getters = _safe_int(summary.get("getters"))
    if getters is None:
        get_procs = params.get("get_procs")
        if isinstance(get_procs, list):
            getters = len(get_procs)
    if getters is None:
        return None
    workers = getters + 1
    wave1_getters = _safe_int(summary.get("wave1_getters"))
    wave2_getters = _safe_int(summary.get("wave2_getters"))

    case_name = str(summary.get("case_name") or params.get("case_name") or path.stem)
    case_name = case_name.strip() or path.stem

    size_mib = _safe_int(summary.get("size_mib"))
    if size_mib is None:
        size_mib = _safe_int(params.get("size_mib"))

    p2p_ratio = _safe_float(summary.get("p2p_ratio"))
    all_get_complete = summary.get("all_get_complete")
    get_success_rate = _safe_float(summary.get("get_success_rate"))
    comm_errors_delta = _safe_int(summary.get("comm_errors_delta"))
    comm_bytes_mismatch_count = _safe_int(summary.get("comm_bytes_mismatch_count"))
    l0_pass = bool(
        p2p_ratio is not None
        and p2p_ratio >= 0.999999
        and all_get_complete is True
        and comm_errors_delta == 0
        and comm_bytes_mismatch_count == 0
    )

    stat = path.stat()
    return FanoutSnapshot(
        path=str(path),
        mtime=stat.st_mtime,
        case_name=case_name,
        workers=workers,
        wave1_getters=wave1_getters,
        wave2_getters=wave2_getters,
        size_mib=size_mib,
        l0_pass=l0_pass,
        p2p_ratio=p2p_ratio,
        get_success_rate=get_success_rate,
        cluster_gibps_mean=_safe_float(summary.get("cluster_gibps_mean")),
        wave1_transfer_gibps_mean=_safe_float(summary.get("wave1_transfer_gibps_mean")),
        wave2_transfer_gibps_mean=_safe_float(summary.get("wave2_transfer_gibps_mean")),
        wave2_over_wave1_transfer_ratio=_safe_float(
            summary.get("wave2_over_wave1_transfer_ratio")
        ),
    )


def _collect_cases(
    *,
    fanout_dir: Path,
    run_id: str,
    target_size_mib: int,
) -> dict[int, FanoutSnapshot]:
    by_worker: dict[int, FanoutSnapshot] = {}
    run_tag = f"suite_{run_id}_" if run_id else ""

    for path in sorted(fanout_dir.glob("*.json")):
        try:
            case = _parse_case(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip fanout file {path}: {exc}")
            continue
        if case is None:
            continue
        if run_tag and run_tag not in case.case_name:
            continue
        if case.size_mib != target_size_mib:
            continue

        prev = by_worker.get(case.workers)
        if prev is None or case.mtime >= prev.mtime:
            by_worker[case.workers] = case
    return by_worker


def _build_markdown(
    *,
    passed: bool,
    checks: list[GateCheck],
    selected_cases: list[FanoutSnapshot],
    round_workers: list[int],
    target_size_mib: int,
    min_cluster_scale_ratio: float,
    baseline_link_gibps: float,
    min_baseline_ratio: float,
    baseline_source: str,
    iperf_single_link_ref_gibps: float | None,
) -> str:
    lines: list[str] = []
    lines.append(
        f"# Scaleout Early Gate ({datetime.now(tz=timezone.utc).isoformat()})"
    )
    lines.append("")
    lines.append(f"- verdict={'PASS' if passed else 'FAIL'}")
    lines.append(f"- round_workers={','.join(str(w) for w in round_workers)}")
    lines.append(f"- target_size_mib={target_size_mib}")
    lines.append(f"- min_cluster_scale_ratio={min_cluster_scale_ratio:.3f}")
    lines.append(
        "- baseline_link_floor="
        f"{baseline_link_gibps:.3f}*{min_baseline_ratio:.3f}"
        f"={baseline_link_gibps * min_baseline_ratio:.3f} GiB/s"
    )
    lines.append(f"- baseline_source={baseline_source}")
    lines.append(
        "- iperf_single_link_ref_gibps="
        f"{_render_float(iperf_single_link_ref_gibps)}"
    )
    lines.append("")
    lines.append("## Wave Analysis")
    lines.append("")
    lines.append(
        "| workers | wave1_getters | wave2_getters | expected_wave2/wave1(worker-ratio) | observed_wave2/wave1(transfer) | wave1_util_vs_baseline | wave2_util_vs_baseline |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for case in sorted(selected_cases, key=lambda x: x.workers):
        expected_ratio = None
        if (
            case.wave1_getters is not None
            and case.wave2_getters is not None
            and case.wave1_getters > 0
        ):
            expected_ratio = float(case.wave2_getters) / float(case.wave1_getters)
        wave1_util = None
        wave2_util = None
        if baseline_link_gibps > 0:
            if case.wave1_transfer_gibps_mean is not None:
                wave1_util = case.wave1_transfer_gibps_mean / baseline_link_gibps
            if case.wave2_transfer_gibps_mean is not None:
                wave2_util = case.wave2_transfer_gibps_mean / baseline_link_gibps
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.workers),
                    _render_int(case.wave1_getters),
                    _render_int(case.wave2_getters),
                    _render_float(expected_ratio),
                    _render_float(case.wave2_over_wave1_transfer_ratio),
                    _render_float(wave1_util),
                    _render_float(wave2_util),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | pass | detail |")
    lines.append("|---|---:|---|")
    lines.extend(
        [
            f"| {check.name} | {'PASS' if check.passed else 'FAIL'} | {check.detail} |"
            for check in checks
        ]
    )
    lines.append("")
    lines.append("## Selected Cases")
    lines.append("")
    lines.append(
        "| workers | case | l0 | p2p_ratio | get_success_rate | cluster_gibps_mean | wave1_gibps | wave2_gibps | wave2/wave1 |"
    )
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.extend(
        [
            "| "
            + " | ".join(
                [
                    str(case.workers),
                    case.case_name,
                    "PASS" if case.l0_pass else "FAIL",
                    _render_float(case.p2p_ratio),
                    _render_float(case.get_success_rate),
                    _render_float(case.cluster_gibps_mean),
                    _render_float(case.wave1_transfer_gibps_mean),
                    _render_float(case.wave2_transfer_gibps_mean),
                    _render_float(case.wave2_over_wave1_transfer_ratio),
                ]
            )
            + " |"
            for case in sorted(selected_cases, key=lambda x: x.workers)
        ]
    )
    return "\n".join(lines) + "\n"


def _render_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _render_int(value: int | None) -> str:
    if value is None:
        return "NA"
    return str(value)


def _evaluate_gate(
    *,
    selected_cases: list[FanoutSnapshot],
    min_p2p_ratio: float,
    min_get_success_rate: float,
    min_wave2_over_wave1: float,
    min_cluster_scale_ratio: float,
    baseline_link_gibps: float,
    min_baseline_ratio: float,
) -> list[GateCheck]:
    checks: list[GateCheck] = []
    for idx, case in enumerate(selected_cases, start=1):
        checks.append(
            GateCheck(
                name=f"round{idx}_l0",
                passed=case.l0_pass,
                detail=(
                    "l0_pass="
                    f"{case.l0_pass} p2p_ratio={_render_float(case.p2p_ratio)} "
                    f"get_success_rate={_render_float(case.get_success_rate)}"
                ),
            )
        )
        checks.append(
            GateCheck(
                name=f"round{idx}_p2p_ratio",
                passed=case.p2p_ratio is not None and case.p2p_ratio >= min_p2p_ratio,
                detail=(
                    f"actual={_render_float(case.p2p_ratio)} "
                    f"required>={min_p2p_ratio:.3f}"
                ),
            )
        )
        checks.append(
            GateCheck(
                name=f"round{idx}_get_success",
                passed=case.get_success_rate is not None
                and case.get_success_rate >= min_get_success_rate,
                detail=(
                    f"actual={_render_float(case.get_success_rate)} "
                    f"required>={min_get_success_rate:.3f}"
                ),
            )
        )
        checks.append(
            GateCheck(
                name=f"round{idx}_wave2_over_wave1",
                passed=case.wave2_over_wave1_transfer_ratio is not None
                and case.wave2_over_wave1_transfer_ratio >= min_wave2_over_wave1,
                detail=(
                    f"actual={_render_float(case.wave2_over_wave1_transfer_ratio)} "
                    f"required>={min_wave2_over_wave1:.3f}"
                ),
            )
        )
        if baseline_link_gibps > 0:
            link_floor = baseline_link_gibps * min_baseline_ratio
            link_peak = max(
                case.wave1_transfer_gibps_mean or 0.0,
                case.wave2_transfer_gibps_mean or 0.0,
            )
            checks.append(
                GateCheck(
                    name=f"round{idx}_micro_baseline_consistency",
                    passed=link_peak >= link_floor,
                    detail=(
                        f"actual_peak={link_peak:.3f} required>={link_floor:.3f} "
                        f"(baseline={baseline_link_gibps:.3f} ratio={min_baseline_ratio:.3f})"
                    ),
                )
            )

    round1 = selected_cases[0]
    round2 = selected_cases[1]
    if (round1.cluster_gibps_mean or 0.0) <= 0:
        checks.append(
            GateCheck(
                name="cluster_scale_ratio",
                passed=False,
                detail=(
                    f"round1 cluster_gibps_mean={_render_float(round1.cluster_gibps_mean)} "
                    "cannot compute ratio"
                ),
            )
        )
    else:
        ratio = (round2.cluster_gibps_mean or 0.0) / (round1.cluster_gibps_mean or 1.0)
        checks.append(
            GateCheck(
                name="cluster_scale_ratio",
                passed=ratio >= min_cluster_scale_ratio,
                detail=f"actual={ratio:.3f} required>={min_cluster_scale_ratio:.3f}",
            )
        )
    return checks


def _parse_iperf_single_link_ref(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return None
    return _safe_float(summary.get("single_link_ref_gibps"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate early scaleout gate after the first two fanout rounds."
    )
    parser.add_argument(
        "--fanout-dir",
        type=Path,
        required=True,
        help="Directory that contains fanout JSON outputs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run_id filter. When set, only case_name containing suite_<run_id>_ is used.",
    )
    parser.add_argument(
        "--target-size-mib",
        type=int,
        default=1024,
        help="Only fanout cases with this payload size are considered.",
    )
    parser.add_argument(
        "--round-workers",
        type=str,
        default="3,4",
        help="Worker counts to evaluate in order; the first two values are used as round1/round2.",
    )
    parser.add_argument(
        "--min-p2p-ratio",
        type=float,
        default=1.0,
        help="Per-round minimum p2p_ratio.",
    )
    parser.add_argument(
        "--min-get-success-rate",
        type=float,
        default=1.0,
        help="Per-round minimum get_success_rate.",
    )
    parser.add_argument(
        "--min-wave2-over-wave1",
        type=float,
        default=0.75,
        help="Per-round minimum wave2_over_wave1_transfer_ratio.",
    )
    parser.add_argument(
        "--min-cluster-scale-ratio",
        type=float,
        default=1.10,
        help="Minimum (round2.cluster_gibps_mean / round1.cluster_gibps_mean).",
    )
    parser.add_argument(
        "--baseline-link-gibps",
        type=float,
        default=0.0,
        help="Optional microbenchmark baseline single-link GiB/s.",
    )
    parser.add_argument(
        "--iperf-json",
        type=Path,
        default=None,
        help="Optional iperf3 probe json; used to auto-fill baseline when baseline-link-gibps<=0.",
    )
    parser.add_argument(
        "--min-baseline-ratio",
        type=float,
        default=0.60,
        help="When baseline-link-gibps>0, require max(wave1,wave2)>=baseline*ratio.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output path for gate JSON.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional output path for gate markdown.",
    )
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when gate fails.",
    )
    args = parser.parse_args()

    round_workers = _parse_round_workers(args.round_workers)
    if len(round_workers) < 2:
        raise SystemExit("--round-workers must include at least two worker counts")
    round_workers = round_workers[:2]

    case_by_workers = _collect_cases(
        fanout_dir=args.fanout_dir,
        run_id=args.run_id.strip(),
        target_size_mib=args.target_size_mib,
    )

    selected_cases: list[FanoutSnapshot] = []
    checks: list[GateCheck] = []

    iperf_single_link_ref = (
        _parse_iperf_single_link_ref(args.iperf_json)
        if args.iperf_json is not None
        else None
    )
    baseline_link_gibps = max(0.0, args.baseline_link_gibps)
    baseline_source = "arg"
    if baseline_link_gibps <= 0.0 and iperf_single_link_ref is not None:
        if iperf_single_link_ref > 0.0:
            baseline_link_gibps = iperf_single_link_ref
            baseline_source = "iperf_json"
    elif baseline_link_gibps > 0.0:
        baseline_source = "arg"
    else:
        baseline_source = "disabled"
    for idx, workers in enumerate(round_workers, start=1):
        case = case_by_workers.get(workers)
        found = case is not None
        checks.append(
            GateCheck(
                name=f"round{idx}_case_present",
                passed=found,
                detail=f"workers={workers} found={found}",
            )
        )
        if case is not None:
            selected_cases.append(case)

    if len(selected_cases) == 2:
        checks.extend(
            _evaluate_gate(
                selected_cases=selected_cases,
                min_p2p_ratio=args.min_p2p_ratio,
                min_get_success_rate=args.min_get_success_rate,
                min_wave2_over_wave1=args.min_wave2_over_wave1,
                min_cluster_scale_ratio=args.min_cluster_scale_ratio,
                baseline_link_gibps=baseline_link_gibps,
                min_baseline_ratio=args.min_baseline_ratio,
            )
        )

    passed = all(check.passed for check in checks)

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "fanout_dir": str(args.fanout_dir),
        "run_id": args.run_id.strip(),
        "target_size_mib": int(args.target_size_mib),
        "round_workers": round_workers,
        "thresholds": {
            "min_p2p_ratio": args.min_p2p_ratio,
            "min_get_success_rate": args.min_get_success_rate,
            "min_wave2_over_wave1": args.min_wave2_over_wave1,
            "min_cluster_scale_ratio": args.min_cluster_scale_ratio,
            "baseline_link_gibps": baseline_link_gibps,
            "min_baseline_ratio": args.min_baseline_ratio,
        },
        "baseline_source": baseline_source,
        "hardware_ceiling": {
            "iperf_json": str(args.iperf_json) if args.iperf_json is not None else None,
            "iperf_single_link_ref_gibps": iperf_single_link_ref,
        },
        "selected_cases": [asdict(case) for case in selected_cases],
        "checks": [asdict(check) for check in checks],
        "pass": passed,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            _build_markdown(
                passed=passed,
                checks=checks,
                selected_cases=selected_cases,
                round_workers=round_workers,
                target_size_mib=args.target_size_mib,
                min_cluster_scale_ratio=args.min_cluster_scale_ratio,
                baseline_link_gibps=baseline_link_gibps,
                min_baseline_ratio=args.min_baseline_ratio,
                baseline_source=baseline_source,
                iperf_single_link_ref_gibps=iperf_single_link_ref,
            ),
            encoding="utf-8",
        )

    print(
        f"[early-gate] pass={passed} run_id={args.run_id.strip() or 'ALL'} "
        f"round_workers={round_workers} out_json={args.out_json}"
    )
    if args.out_md is not None:
        print(f"[early-gate] out_md={args.out_md}")

    if not passed and args.fail_on_gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
