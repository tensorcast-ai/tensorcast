#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_summary_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "cross_host" / "summarize_scaleout_suite.py"
    spec = importlib.util.spec_from_file_location(
        "summarize_scaleout_suite",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load summarize_scaleout_suite.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_tp_case(
    path: Path,
    *,
    case_name: str,
    progressive_enabled: bool,
    failure_injection_enabled: bool = False,
    failure_injection_status: str = "disabled",
) -> None:
    payload = {
        "summary": {
            "case_name": case_name,
            "passed": True,
            "receiver_processes": ["rx0", "rx1"],
            "progressive": {
                "enabled": progressive_enabled,
            },
            "performance": {
                "publish_to_apply_s": {
                    "p95": 4.0 if progressive_enabled else 6.0,
                },
                "transport_throughput_gib_s": {
                    "peak_active_throughput_gib_s": (
                        9.0 if progressive_enabled else 7.0
                    ),
                },
            },
            "distribution": {
                "transport_diffusion": {
                    "unique_sources": 2,
                    "top1_share": 0.45 if progressive_enabled else 0.9,
                    "hhi": 0.35 if progressive_enabled else 0.82,
                },
            },
            "transport_checks": {
                "group_metadata_probe": {
                    "enabled": True,
                    "window_has_transports": True,
                    "requester_tagged_complete": True,
                    "grouped_transports": 3,
                    "grouping_present": True,
                    "group_contract_consistent": True,
                },
            },
            "failures": {
                "failure_injection": {
                    "enabled": failure_injection_enabled,
                    "status": failure_injection_status,
                },
            },
        },
        "params": {
            "case_name": case_name,
            "enable_progressive_replication": progressive_enabled,
            "failure_injection_mode": (
                "stop-daemon" if failure_injection_enabled else "none"
            ),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tp_summary_groups_baseline_progressive_and_failure_lanes(
    tmp_path: Path,
) -> None:
    module = _load_summary_module()
    baseline = tmp_path / "baseline.json"
    progressive = tmp_path / "progressive.json"
    failure = tmp_path / "failure.json"
    _write_tp_case(
        baseline,
        case_name="suite_x_r2_group_realization",
        progressive_enabled=False,
    )
    _write_tp_case(
        progressive,
        case_name="suite_x_r2_group_realization_progressive",
        progressive_enabled=True,
    )
    _write_tp_case(
        failure,
        case_name="suite_x_failure_injection_r2",
        progressive_enabled=True,
        failure_injection_enabled=True,
        failure_injection_status="injected",
    )

    cases = module._collect_tp_cases(tmp_path)
    by_name = {case.case_name: case for case in cases}
    assert by_name["suite_x_r2_group_realization"].lane == "baseline"
    assert by_name["suite_x_r2_group_realization_progressive"].lane == "progressive"
    assert by_name["suite_x_failure_injection_r2"].lane == (
        "progressive_failure_injection"
    )

    lane_summary = {row["lane"]: row for row in module._build_tp_lane_summary(cases)}
    assert lane_summary["baseline"]["case_count"] == 1
    assert lane_summary["progressive"]["diffusion_top1_share_mean"] == 0.45
    assert lane_summary["progressive_failure_injection"]["failure_injection_count"] == 1

    markdown = module._build_markdown([], cases)
    assert "TP Lane Summary" in markdown
    assert "progressive_failure_injection" in markdown
