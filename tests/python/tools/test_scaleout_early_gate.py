# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_case(
    *,
    out_dir: Path,
    run_id: str,
    workers: int,
    cluster_gibps_mean: float,
    wave_ratio: float,
) -> None:
    getters = workers - 1
    case_name = f"suite_{run_id}_small_fanout_{workers}n_c20b16w16_g0_s1024"
    payload = {
        "mode": "fanout",
        "summary": {
            "case_name": case_name,
            "getters": getters,
            "size_mib": 1024,
            "p2p_ratio": 1.0,
            "all_get_complete": True,
            "get_success_rate": 1.0,
            "comm_errors_delta": 0,
            "comm_bytes_mismatch_count": 0,
            "wave2_over_wave1_transfer_ratio": wave_ratio,
            "cluster_gibps_mean": cluster_gibps_mean,
            "wave1_transfer_gibps_mean": 2.2,
            "wave2_transfer_gibps_mean": 2.4,
        },
    }
    path = out_dir / f"{case_name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _script_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "examples" / "cross_host" / "scaleout_early_gate.py"


def test_scaleout_early_gate_pass(tmp_path: Path) -> None:
    run_id = "testrun-pass"
    fanout_dir = tmp_path / "fanout"
    fanout_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "gate.json"

    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=3,
        cluster_gibps_mean=0.35,
        wave_ratio=1.05,
    )
    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=4,
        cluster_gibps_mean=0.48,
        wave_ratio=1.10,
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--fanout-dir",
            str(fanout_dir),
            "--run-id",
            run_id,
            "--round-workers",
            "3,4",
            "--target-size-mib",
            "1024",
            "--out-json",
            str(out_json),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["pass"] is True


def test_scaleout_early_gate_fail_on_cluster_scale_ratio(tmp_path: Path) -> None:
    run_id = "testrun-fail"
    fanout_dir = tmp_path / "fanout"
    fanout_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "gate_fail.json"

    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=3,
        cluster_gibps_mean=0.50,
        wave_ratio=1.02,
    )
    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=4,
        cluster_gibps_mean=0.45,
        wave_ratio=1.01,
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--fanout-dir",
            str(fanout_dir),
            "--run-id",
            run_id,
            "--round-workers",
            "3,4",
            "--target-size-mib",
            "1024",
            "--out-json",
            str(out_json),
            "--min-cluster-scale-ratio",
            "1.10",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["pass"] is False


def test_scaleout_early_gate_autofills_baseline_from_iperf_json(
    tmp_path: Path,
) -> None:
    run_id = "testrun-iperf"
    fanout_dir = tmp_path / "fanout"
    fanout_dir.mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "gate_iperf.json"
    iperf_json = tmp_path / "iperf.json"

    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=3,
        cluster_gibps_mean=0.35,
        wave_ratio=1.05,
    )
    _write_case(
        out_dir=fanout_dir,
        run_id=run_id,
        workers=4,
        cluster_gibps_mean=0.48,
        wave_ratio=1.10,
    )

    iperf_json.write_text(
        json.dumps({"summary": {"single_link_ref_gibps": 2.5}}),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--fanout-dir",
            str(fanout_dir),
            "--run-id",
            run_id,
            "--round-workers",
            "3,4",
            "--target-size-mib",
            "1024",
            "--baseline-link-gibps",
            "0",
            "--iperf-json",
            str(iperf_json),
            "--out-json",
            str(out_json),
            "--no-fail-on-gate",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["baseline_source"] == "iperf_json"
    assert payload["thresholds"]["baseline_link_gibps"] == 2.5
