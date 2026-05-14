#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"json root is not object: {path}")
    return payload


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _infer_case_name(path: Path, payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        case_name = summary.get("case_name")
        if isinstance(case_name, str) and case_name.strip():
            return case_name.strip()
    params = payload.get("params", {})
    if isinstance(params, dict):
        case_name = params.get("case_name")
        if isinstance(case_name, str) and case_name.strip():
            return case_name.strip()
    return path.stem


def _infer_workers(payload: dict[str, Any]) -> int | None:
    params = payload.get("params", {})
    if not isinstance(params, dict):
        return None
    get_procs = params.get("get_procs")
    if isinstance(get_procs, list):
        return int(len(get_procs) + 1)
    receiver_procs = params.get("receiver_procs")
    if isinstance(receiver_procs, list):
        return int(len(receiver_procs) + 1)
    return None


_ROUND_RE = re.compile(r"(?:^|_)round(?P<round>\d+)(?:_|$)")
_ORDER_RE = re.compile(r"(?:^|_)o(?P<order>\d+)(?:_|$)")


def _infer_round_and_order(case_name: str) -> tuple[int | None, int | None]:
    round_match = _ROUND_RE.search(case_name)
    order_match = _ORDER_RE.search(case_name)
    round_index = int(round_match.group("round")) if round_match else None
    order_slot = int(order_match.group("order")) if order_match else None
    return round_index, order_slot


@dataclass(slots=True)
class FanoutCase:
    path: str
    case_name: str
    mode: str
    workers: int | None
    getters: int | None
    size_mib: int | None
    l0_pass: bool
    p2p_ratio: float | None
    all_get_complete: bool | None
    get_success_rate: float | None
    comm_errors_delta: int | None
    comm_bytes_mismatch_count: int | None
    wave2_over_wave1_transfer_ratio: float | None
    cluster_gibps_mean: float | None
    cluster_gibps_p90: float | None
    wave1_transfer_gibps_mean: float | None
    wave2_transfer_gibps_mean: float | None
    task_load_complete_sec_mean: float | None
    task_total_sec_mean: float | None
    completion_curve_auc_norm_mean: float | None
    completion_by_half_ratio_mean: float | None
    failure_infra: int
    failure_product: int
    failure_unknown: int
    transport_rows_error: str | None
    diffusion_unique_sources: int | None
    diffusion_top1_share: float | None
    diffusion_hhi: float | None


@dataclass(slots=True)
class TpCase:
    path: str
    case_name: str
    workers: int | None
    receivers: int | None
    round_index: int | None
    order_slot: int | None
    passed: bool | None
    l0_pass: bool
    group_probe_enabled: bool | None
    window_has_transports: bool | None
    requester_tagged_complete: bool | None
    grouped_transports: int | None
    grouping_present: bool | None
    group_contract_consistent: bool | None
    publish_to_apply_p95_s: float | None
    throughput_peak_active_gib_s: float | None
    throughput_p95_active_gib_s: float | None
    throughput_mean_active_gib_s: float | None
    diffusion_unique_sources: int | None
    diffusion_top1_share: float | None
    diffusion_hhi: float | None


def _parse_fanout_case(path: Path) -> FanoutCase | None:
    payload = _read_json(path)
    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in {"fanout", "cascade"}:
        return None

    case_name = _infer_case_name(path, payload)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    workers = _infer_workers(payload)
    getters = _safe_int(summary.get("getters"))
    if getters is not None:
        workers = getters + 1
    size_mib = _safe_int(summary.get("size_mib"))
    if size_mib is None:
        params = payload.get("params", {})
        if isinstance(params, dict):
            size_mib = _safe_int(params.get("size_mib"))

    p2p_ratio = _safe_float(summary.get("p2p_ratio"))
    all_get_complete = _safe_bool(summary.get("all_get_complete"))
    get_success_rate = _safe_float(summary.get("get_success_rate"))
    comm_errors_delta = _safe_int(summary.get("comm_errors_delta"))
    comm_bytes_mismatch_count = _safe_int(summary.get("comm_bytes_mismatch_count"))

    wave_ratio = _safe_float(summary.get("wave2_over_wave1_transfer_ratio"))
    cluster_gibps_mean = _safe_float(summary.get("cluster_gibps_mean"))
    cluster_gibps_p90 = _safe_float(summary.get("cluster_gibps_p90"))
    wave1_transfer = _safe_float(summary.get("wave1_transfer_gibps_mean"))
    wave2_transfer = _safe_float(summary.get("wave2_transfer_gibps_mean"))
    task_load_complete_sec_mean = _safe_float(
        summary.get("task_load_complete_sec_mean")
    )
    task_total_sec_mean = _safe_float(summary.get("task_total_sec_mean"))
    completion_curve_auc_norm_mean = _safe_float(
        summary.get("completion_curve_auc_norm_mean")
    )
    completion_by_half_ratio_mean = _safe_float(
        summary.get("completion_by_half_ratio_mean")
    )

    fail_counts = summary.get("failure_classification_counts", {})
    if not isinstance(fail_counts, dict):
        fail_counts = {}
    failure_infra = int(fail_counts.get("infra", 0) or 0)
    failure_product = int(fail_counts.get("product", 0) or 0)
    failure_unknown = int(fail_counts.get("unknown", 0) or 0)

    transport_rows_error = summary.get("transport_rows_error")
    if transport_rows_error is not None:
        transport_rows_error = str(transport_rows_error)

    diffusion = summary.get("replica_id_diffusion", {})
    if not isinstance(diffusion, dict):
        diffusion = {}
    diffusion_unique_sources = _safe_int(diffusion.get("unique_sources"))
    diffusion_top1_share = _safe_float(diffusion.get("top1_share"))
    diffusion_hhi = _safe_float(diffusion.get("hhi"))

    if mode == "cascade":
        comm_err_workers = summary.get("comm_error_workers", [])
        comm_bytes_workers = summary.get("comm_bytes_mismatch_workers", [])
        functional_chain_ok = _safe_bool(summary.get("functional_chain_ok"))
        if functional_chain_ok is None:
            functional_chain_ok = False
        l0_pass = bool(
            p2p_ratio is not None
            and p2p_ratio >= 0.999999
            and functional_chain_ok
            and isinstance(comm_err_workers, list)
            and len(comm_err_workers) == 0
            and isinstance(comm_bytes_workers, list)
            and len(comm_bytes_workers) == 0
        )
    else:
        l0_pass = bool(
            p2p_ratio is not None
            and p2p_ratio >= 0.999999
            and all_get_complete is True
            and comm_errors_delta == 0
            and comm_bytes_mismatch_count == 0
        )

    return FanoutCase(
        path=str(path),
        case_name=case_name,
        mode=mode,
        workers=workers,
        getters=getters,
        size_mib=size_mib,
        l0_pass=l0_pass,
        p2p_ratio=p2p_ratio,
        all_get_complete=all_get_complete,
        get_success_rate=get_success_rate,
        comm_errors_delta=comm_errors_delta,
        comm_bytes_mismatch_count=comm_bytes_mismatch_count,
        wave2_over_wave1_transfer_ratio=wave_ratio,
        cluster_gibps_mean=cluster_gibps_mean,
        cluster_gibps_p90=cluster_gibps_p90,
        wave1_transfer_gibps_mean=wave1_transfer,
        wave2_transfer_gibps_mean=wave2_transfer,
        task_load_complete_sec_mean=task_load_complete_sec_mean,
        task_total_sec_mean=task_total_sec_mean,
        completion_curve_auc_norm_mean=completion_curve_auc_norm_mean,
        completion_by_half_ratio_mean=completion_by_half_ratio_mean,
        failure_infra=failure_infra,
        failure_product=failure_product,
        failure_unknown=failure_unknown,
        transport_rows_error=transport_rows_error,
        diffusion_unique_sources=diffusion_unique_sources,
        diffusion_top1_share=diffusion_top1_share,
        diffusion_hhi=diffusion_hhi,
    )


def _parse_tp_case(path: Path) -> TpCase | None:
    payload = _read_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    if "transport_checks" not in summary or "distribution" not in summary:
        return None

    case_name = _infer_case_name(path, payload)
    round_index, order_slot = _infer_round_and_order(case_name)
    receiver_processes = summary.get("receiver_processes", [])
    receivers = (
        len(receiver_processes) if isinstance(receiver_processes, list) else None
    )
    workers = receivers + 1 if isinstance(receivers, int) else _infer_workers(payload)

    passed = _safe_bool(summary.get("passed"))
    transport_checks = summary.get("transport_checks", {})
    if not isinstance(transport_checks, dict):
        transport_checks = {}
    probe = transport_checks.get("group_metadata_probe", {})
    if not isinstance(probe, dict):
        probe = {}

    group_probe_enabled = _safe_bool(probe.get("enabled"))
    window_has_transports = _safe_bool(probe.get("window_has_transports"))
    requester_tagged_complete = _safe_bool(probe.get("requester_tagged_complete"))
    grouped_transports = _safe_int(probe.get("grouped_transports"))
    grouping_present = _safe_bool(probe.get("grouping_present"))
    group_contract_consistent = _safe_bool(probe.get("group_contract_consistent"))

    distribution = summary.get("distribution", {})
    if not isinstance(distribution, dict):
        distribution = {}
    diffusion = distribution.get("transport_diffusion", {})
    if not isinstance(diffusion, dict):
        diffusion = {}

    performance = summary.get("performance", {})
    if not isinstance(performance, dict):
        performance = {}
    publish_to_apply = performance.get("publish_to_apply_s", {})
    if not isinstance(publish_to_apply, dict):
        publish_to_apply = {}
    throughput = performance.get("transport_throughput_gib_s", {})
    if not isinstance(throughput, dict):
        throughput = {}

    group_gate_ok = bool(
        window_has_transports is True
        and requester_tagged_complete is True
        and grouped_transports is not None
        and grouped_transports > 0
        and grouping_present is True
        and group_contract_consistent is True
    )

    l0_pass = bool(passed is True and group_gate_ok)

    return TpCase(
        path=str(path),
        case_name=case_name,
        workers=workers,
        receivers=receivers,
        round_index=round_index,
        order_slot=order_slot,
        passed=passed,
        l0_pass=l0_pass,
        group_probe_enabled=group_probe_enabled,
        window_has_transports=window_has_transports,
        requester_tagged_complete=requester_tagged_complete,
        grouped_transports=grouped_transports,
        grouping_present=grouping_present,
        group_contract_consistent=group_contract_consistent,
        publish_to_apply_p95_s=_safe_float(publish_to_apply.get("p95")),
        throughput_peak_active_gib_s=_safe_float(
            throughput.get("peak_active_throughput_gib_s")
        ),
        throughput_p95_active_gib_s=_safe_float(
            throughput.get("p95_active_throughput_gib_s")
        ),
        throughput_mean_active_gib_s=_safe_float(
            throughput.get("mean_active_throughput_gib_s")
        ),
        diffusion_unique_sources=_safe_int(diffusion.get("unique_sources")),
        diffusion_top1_share=_safe_float(diffusion.get("top1_share")),
        diffusion_hhi=_safe_float(diffusion.get("hhi")),
    )


def _render_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _render_int(value: int | None) -> str:
    if value is None:
        return "NA"
    return str(value)


def _build_markdown(
    fanout_cases: list[FanoutCase],
    tp_cases: list[TpCase],
) -> str:
    lines: list[str] = []
    lines.append(f"# Scaleout Summary ({datetime.now(tz=timezone.utc).isoformat()})")
    lines.append("")

    lines.append("## Phase A (Fanout/Cascade)")
    lines.append("")
    if not fanout_cases:
        lines.append("No fanout/cascade result found.")
    else:
        fanout_l0_all = all(case.l0_pass for case in fanout_cases)
        fanout_ratio_ok = all(
            case.mode != "fanout"
            or case.wave2_over_wave1_transfer_ratio is None
            or case.wave2_over_wave1_transfer_ratio >= 1.0
            for case in fanout_cases
        )
        lines.append(
            f"- cases={len(fanout_cases)} l0_all_pass={fanout_l0_all} "
            f"wave2_over_wave1_all_ge_1={fanout_ratio_ok}"
        )
        lines.append("")
        lines.append(
            "| case | mode | workers | size_mib | L0 | get_success_rate | task_load_s | auc_norm | by_half | wave2/wave1 | cluster_gibps_mean | top1_share | hhi |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        lines.extend(
            [
                "| "
                + " | ".join(
                    [
                        case.case_name,
                        case.mode,
                        _render_int(case.workers),
                        _render_int(case.size_mib),
                        "PASS" if case.l0_pass else "FAIL",
                        _render_float(case.get_success_rate),
                        _render_float(case.task_load_complete_sec_mean),
                        _render_float(case.completion_curve_auc_norm_mean),
                        _render_float(case.completion_by_half_ratio_mean),
                        _render_float(case.wave2_over_wave1_transfer_ratio),
                        _render_float(case.cluster_gibps_mean),
                        _render_float(case.diffusion_top1_share),
                        _render_float(case.diffusion_hhi),
                    ]
                )
                + " |"
                for case in sorted(
                    fanout_cases,
                    key=lambda x: (
                        x.workers if x.workers is not None else 0,
                        x.mode,
                        x.size_mib if x.size_mib is not None else 0,
                        x.case_name,
                    ),
                )
            ]
        )

    lines.append("")
    lines.append("## Phase B (TP)")
    lines.append("")
    if not tp_cases:
        lines.append("No TP result found.")
    else:
        tp_l0_all = all(case.l0_pass for case in tp_cases)
        lines.append(f"- cases={len(tp_cases)} l0_all_pass={tp_l0_all}")
        lines.append("")
        lines.append(
            "| case | workers | round | order | passed | L0 | grouped_transports | p95_publish_to_apply_s | top1_share | hhi |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.extend(
            [
                "| "
                + " | ".join(
                    [
                        case.case_name,
                        _render_int(case.workers),
                        _render_int(case.round_index),
                        _render_int(case.order_slot),
                        str(case.passed),
                        "PASS" if case.l0_pass else "FAIL",
                        _render_int(case.grouped_transports),
                        _render_float(case.publish_to_apply_p95_s),
                        _render_float(case.diffusion_top1_share),
                        _render_float(case.diffusion_hhi),
                    ]
                )
                + " |"
                for case in sorted(
                    tp_cases,
                    key=lambda x: (
                        x.workers if x.workers is not None else 0,
                        x.round_index if x.round_index is not None else -1,
                        x.order_slot if x.order_slot is not None else -1,
                        x.case_name,
                    ),
                )
            ]
        )

    return "\n".join(lines) + "\n"


def _collect_fanout_cases(fanout_dir: Path) -> list[FanoutCase]:
    cases: list[FanoutCase] = []
    if not fanout_dir.exists():
        return cases
    for path in sorted(fanout_dir.glob("*.json")):
        try:
            parsed = _parse_fanout_case(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip fanout file {path}: {exc}")
            continue
        if parsed is not None:
            cases.append(parsed)
    return cases


def _collect_tp_cases(tp_dir: Path) -> list[TpCase]:
    cases: list[TpCase] = []
    if not tp_dir.exists():
        return cases
    for path in sorted(tp_dir.rglob("*.json")):
        try:
            parsed = _parse_tp_case(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip tp file {path}: {exc}")
            continue
        if parsed is None:
            continue
        # Ignore per-role summaries and retain only case-level outputs.
        if parsed.case_name.endswith("_summary"):
            continue
        cases.append(parsed)
    dedup: dict[str, TpCase] = {}
    for case in cases:
        dedup[case.path] = case
    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multihost fanout scaleout and TP group experiment results."
    )
    parser.add_argument(
        "--fanout-dir",
        type=Path,
        default=Path("/data/tc_cross_20260226/results_multi_host_scaleout"),
        help="Directory that contains fanout/cascade JSON outputs.",
    )
    parser.add_argument(
        "--tp-dir",
        type=Path,
        default=Path("/data/tc_cross_20260226/results_0083_tp4"),
        help="Directory that contains TP case JSON outputs.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("/data/tc_cross_20260226/report/scaleout_summary.json"),
        help="Output path for merged JSON summary.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("/data/tc_cross_20260226/report/scaleout_summary.md"),
        help="Output path for markdown report snippet.",
    )
    args = parser.parse_args()

    fanout_cases = _collect_fanout_cases(args.fanout_dir)
    tp_cases = _collect_tp_cases(args.tp_dir)

    payload = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "fanout_dir": str(args.fanout_dir),
        "tp_dir": str(args.tp_dir),
        "fanout": {
            "case_count": len(fanout_cases),
            "l0_all_pass": all(case.l0_pass for case in fanout_cases)
            if fanout_cases
            else None,
            "cases": [asdict(case) for case in fanout_cases],
        },
        "tp": {
            "case_count": len(tp_cases),
            "l0_all_pass": all(case.l0_pass for case in tp_cases) if tp_cases else None,
            "cases": [asdict(case) for case in tp_cases],
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    markdown = _build_markdown(
        fanout_cases,
        tp_cases,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown)

    print(f"[summary] fanout_cases={len(fanout_cases)} tp_cases={len(tp_cases)}")
    print(f"[summary] json={args.out_json}")
    print(f"[summary] md={args.out_md}")


if __name__ == "__main__":
    main()
