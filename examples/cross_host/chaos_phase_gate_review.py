#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

REQUIRED_PHASES = ("small", "medium", "large")


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def load_phase_review(run_dir: Path) -> dict[str, Any]:
    review_path = run_dir / "gate_review.json"
    if not review_path.exists():
        raise FileNotFoundError(
            f"missing gate review: {review_path} (run chaos_gate_review.py first)"
        )
    review = read_json_object(review_path)
    review["gate_review_path"] = review_path.as_posix()
    return review


def build_phase_result(phase: str, run_dir: Path) -> dict[str, Any]:
    error = ""
    passed = False
    review_path = (run_dir / "gate_review.json").as_posix()
    gate_count = 0
    try:
        review = load_phase_review(run_dir)
        passed = bool(review.get("passed", False))
        gate_rows = review.get("gates", [])
        gate_count = len(gate_rows) if isinstance(gate_rows, list) else 0
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return {
        "phase": phase,
        "run_dir": run_dir.as_posix(),
        "gate_review_path": review_path,
        "passed": bool(passed),
        "gate_count": int(gate_count),
        "error": error,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Chaos Phase Gate Review")
    lines.append("")
    lines.append(f"- generated_epoch: `{payload['generated_epoch']}`")
    lines.append(f"- passed: `{payload['passed']}`")
    lines.append("")
    lines.append("## Phase Checklist")
    lines.append("")
    for row in payload["phases"]:
        marker = "x" if bool(row.get("passed")) else " "
        lines.append(
            f"- [{marker}] `{row['phase']}` "
            f"run_dir=`{row['run_dir']}` "
            f"gate_review=`{row['gate_review_path']}` "
            f"error=`{row['error']}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate phase-level gate reviews (small/medium/large) into a "
            "single pass/fail report."
        )
    )
    parser.add_argument("--small-run-dir", required=True)
    parser.add_argument("--medium-run-dir", required=True)
    parser.add_argument("--large-run-dir", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="output JSON path (default: <small-run-dir>/../phase_gate_review.json)",
    )
    parser.add_argument(
        "--markdown-output",
        default="",
        help="output markdown path (default: <small-run-dir>/../phase_gate_review.md)",
    )
    parser.add_argument(
        "--allow-failure-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase_run_dirs = {
        "small": Path(str(args.small_run_dir)).resolve(),
        "medium": Path(str(args.medium_run_dir)).resolve(),
        "large": Path(str(args.large_run_dir)).resolve(),
    }
    phase_results = [
        build_phase_result(phase, phase_run_dirs[phase]) for phase in REQUIRED_PHASES
    ]
    passed = all(bool(row.get("passed")) for row in phase_results)
    payload = {
        "generated_epoch": float(time.time()),
        "phases": phase_results,
        "passed": bool(passed),
    }

    default_parent = phase_run_dirs["small"].parent
    output_path = (
        Path(str(args.output)).resolve()
        if str(args.output).strip()
        else default_parent / "phase_gate_review.json"
    )
    markdown_path = (
        Path(str(args.markdown_output)).resolve()
        if str(args.markdown_output).strip()
        else default_parent / "phase_gate_review.md"
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"PHASE_GATE_REVIEW {json.dumps(payload, ensure_ascii=False)}", flush=True)
    print(f"PHASE_GATE_REVIEW_JSON {output_path}", flush=True)
    print(f"PHASE_GATE_REVIEW_MD {markdown_path}", flush=True)

    if bool(payload["passed"]) or bool(args.allow_failure_exit):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
