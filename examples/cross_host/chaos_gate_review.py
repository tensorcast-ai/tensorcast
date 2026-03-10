#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CLASSIFICATION_KEYS = ("infra", "product", "unknown")
REQUIRED_CASE_FILES = ("result.json", "metrics.json", "classification.json")


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    expected: str
    observed: Any
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": bool(self.passed),
            "expected": self.expected,
            "observed": self.observed,
            "detail": self.detail,
        }


def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def discover_case_dirs(run_dir: Path) -> list[Path]:
    case_root = run_dir / "cases"
    if not case_root.exists() or not case_root.is_dir():
        return []
    return sorted(path for path in case_root.iterdir() if path.is_dir())


def collect_missing_case_files(case_dirs: list[Path]) -> list[str]:
    missing: list[str] = []
    for case_dir in case_dirs:
        for file_name in REQUIRED_CASE_FILES:
            file_path = case_dir / file_name
            if not file_path.exists():
                missing.append(file_path.as_posix())
    return missing


def count_case_outcomes(case_dirs: list[Path]) -> tuple[int, int]:
    positive = 0
    negative = 0
    for case_dir in case_dirs:
        result_path = case_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = read_json_object(result_path)
        except Exception:  # noqa: BLE001
            continue
        outcome = str(result.get("expected_outcome", "")).strip().lower()
        if outcome == "success":
            positive += 1
        elif outcome == "failure":
            negative += 1
    return positive, negative


def build_gate(
    *,
    gate: str,
    passed: bool,
    expected: str,
    observed: Any,
    detail: str,
) -> GateResult:
    return GateResult(
        gate=gate,
        passed=bool(passed),
        expected=expected,
        observed=observed,
        detail=detail,
    )


def evaluate_run(
    *,
    run_dir: Path,
    max_recover_time_sec: float,
    require_all_get_complete: bool,
    require_source_cardinality: bool,
    require_expected_failure_pass: bool,
    require_comm_errors_zero: bool,
) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    missing_run_files = [
        path.as_posix() for path in (summary_path, events_path) if not path.exists()
    ]

    summary: dict[str, Any] = {}
    summary_error = ""
    if summary_path.exists():
        try:
            summary = read_json_object(summary_path)
        except Exception as exc:  # noqa: BLE001
            summary_error = str(exc)
    else:
        summary_error = "summary.json missing"

    case_dirs = discover_case_dirs(run_dir)
    missing_case_files = collect_missing_case_files(case_dirs)
    positive_cases, negative_cases = count_case_outcomes(case_dirs)

    gates: list[GateResult] = []
    gates.append(
        build_gate(
            gate="run_artifacts_present",
            passed=not missing_run_files and not summary_error,
            expected="summary.json/events.jsonl must exist and summary is valid JSON",
            observed={
                "missing_run_files": missing_run_files,
                "summary_error": summary_error,
            },
            detail="run-level artifacts and JSON readability",
        )
    )
    gates.append(
        build_gate(
            gate="case_artifacts_present",
            passed=bool(case_dirs) and not missing_case_files,
            expected=(
                "at least one case dir and each case has "
                "result.json/metrics.json/classification.json"
            ),
            observed={
                "case_count": len(case_dirs),
                "missing_case_files": missing_case_files,
            },
            detail="case-level result package completeness",
        )
    )

    unexpected_failures = summary.get("unexpected_failures", [])
    unexpected_successes = summary.get("unexpected_successes", [])
    has_no_unexpected = (
        isinstance(unexpected_failures, list)
        and isinstance(unexpected_successes, list)
        and len(unexpected_failures) == 0
        and len(unexpected_successes) == 0
    )
    gates.append(
        build_gate(
            gate="no_unexpected_case_status",
            passed=has_no_unexpected,
            expected="unexpected_failures/unexpected_successes must be empty",
            observed={
                "unexpected_failures": unexpected_failures,
                "unexpected_successes": unexpected_successes,
            },
            detail="case outcome stability",
        )
    )

    expected_failure_pass = bool(summary.get("expected_failure_pass", False))
    if require_expected_failure_pass and negative_cases > 0:
        gates.append(
            build_gate(
                gate="expected_failure_pass",
                passed=expected_failure_pass,
                expected="all expected-failure cases should pass expected_failure gate",
                observed=expected_failure_pass,
                detail="negative-case gate behavior",
            )
        )
    else:
        gates.append(
            build_gate(
                gate="expected_failure_pass",
                passed=True,
                expected="requirement disabled or no negative cases",
                observed=expected_failure_pass,
                detail="skipped",
            )
        )

    all_get_complete = bool(summary.get("all_get_complete", False))
    if require_all_get_complete and positive_cases > 0:
        gates.append(
            build_gate(
                gate="all_get_complete",
                passed=all_get_complete,
                expected="all positive cases should end with all_get_complete=true",
                observed=all_get_complete,
                detail="fanout eventual completion gate",
            )
        )
    else:
        gates.append(
            build_gate(
                gate="all_get_complete",
                passed=True,
                expected="requirement disabled or no positive cases",
                observed=all_get_complete,
                detail="skipped",
            )
        )

    source_cardinality_timeline = summary.get("source_cardinality_timeline", [])
    source_timeline_present = (
        isinstance(source_cardinality_timeline, list)
        and len(source_cardinality_timeline) > 0
    )
    if require_source_cardinality and positive_cases > 0:
        gates.append(
            build_gate(
                gate="source_cardinality_timeline_present",
                passed=source_timeline_present,
                expected=(
                    "source_cardinality_timeline should be non-empty for positive cases"
                ),
                observed={
                    "count": len(source_cardinality_timeline)
                    if isinstance(source_cardinality_timeline, list)
                    else -1
                },
                detail="fanout diffusion observability gate",
            )
        )
    else:
        gates.append(
            build_gate(
                gate="source_cardinality_timeline_present",
                passed=True,
                expected="requirement disabled or no positive cases",
                observed={
                    "count": len(source_cardinality_timeline)
                    if isinstance(source_cardinality_timeline, list)
                    else -1
                },
                detail="skipped",
            )
        )

    recover_time_sec = float(summary.get("recover_time_sec", 0.0))
    gates.append(
        build_gate(
            gate="recover_time_bound",
            passed=recover_time_sec <= float(max_recover_time_sec),
            expected=f"recover_time_sec <= {float(max_recover_time_sec):.3f}",
            observed=float(recover_time_sec),
            detail="recovery SLO envelope",
        )
    )

    comm_errors_delta = int(summary.get("comm_errors_delta", 0))
    if require_comm_errors_zero:
        gates.append(
            build_gate(
                gate="comm_errors_zero",
                passed=comm_errors_delta == 0,
                expected="comm_errors_delta == 0",
                observed=int(comm_errors_delta),
                detail="data-plane transport error gate",
            )
        )
    else:
        gates.append(
            build_gate(
                gate="comm_errors_zero",
                passed=True,
                expected="requirement disabled",
                observed=int(comm_errors_delta),
                detail="skipped",
            )
        )

    raw_classification = summary.get("failure_classification_counts", {})
    classification_valid = isinstance(raw_classification, dict) and all(
        key in raw_classification for key in REQUIRED_CLASSIFICATION_KEYS
    )
    gates.append(
        build_gate(
            gate="classification_shape",
            passed=classification_valid,
            expected=(
                "failure_classification_counts contains infra/product/unknown keys"
            ),
            observed=raw_classification,
            detail="classification contract coverage",
        )
    )

    retry_reason_buckets = summary.get("retry_reason_buckets", {})
    budget_exit_reason_buckets = summary.get("budget_exit_reason_buckets", {})
    gates.append(
        build_gate(
            gate="budget_bucket_presence",
            passed=isinstance(retry_reason_buckets, dict)
            and isinstance(budget_exit_reason_buckets, dict),
            expected="retry_reason_buckets and budget_exit_reason_buckets are objects",
            observed={
                "retry_reason_buckets_type": type(retry_reason_buckets).__name__,
                "budget_exit_reason_buckets_type": type(
                    budget_exit_reason_buckets
                ).__name__,
            },
            detail="budget trace surface shape",
        )
    )

    gate_rows = [item.to_dict() for item in gates]
    passed = all(bool(item["passed"]) for item in gate_rows)
    return {
        "run_dir": run_dir.as_posix(),
        "generated_epoch": float(time.time()),
        "max_recover_time_sec": float(max_recover_time_sec),
        "positive_case_count": int(positive_cases),
        "negative_case_count": int(negative_cases),
        "passed": bool(passed),
        "gates": gate_rows,
    }


def render_markdown(review: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Chaos Gate Review")
    lines.append("")
    lines.append(f"- run_dir: `{review['run_dir']}`")
    lines.append(f"- generated_epoch: `{review['generated_epoch']}`")
    lines.append(f"- passed: `{review['passed']}`")
    lines.append(f"- positive_case_count: `{review['positive_case_count']}`")
    lines.append(f"- negative_case_count: `{review['negative_case_count']}`")
    lines.append("")
    lines.append("## Gate Checklist")
    lines.append("")
    for gate in review["gates"]:
        marker = "x" if bool(gate.get("passed")) else " "
        lines.append(
            f"- [{marker}] `{gate.get('gate')}`"
            f" expected=`{gate.get('expected')}`"
            f" observed=`{gate.get('observed')}`"
            f" detail=`{gate.get('detail')}`"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a chaos run output directory against phase gate criteria "
            "and emit gate_review.json / gate_review.md."
        )
    )
    parser.add_argument("--run-dir", required=True, help="chaos run directory")
    parser.add_argument(
        "--output",
        default="",
        help="output path for gate review JSON (default: <run-dir>/gate_review.json)",
    )
    parser.add_argument(
        "--markdown-output",
        default="",
        help="output path for gate review markdown (default: <run-dir>/gate_review.md)",
    )
    parser.add_argument("--max-recover-time-sec", type=float, default=180.0)
    parser.add_argument(
        "--require-all-get-complete",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-source-cardinality",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-expected-failure-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-comm-errors-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-failure-exit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return 0 even if gate review fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(str(args.run_dir)).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError(f"run dir not found: {run_dir}")

    review = evaluate_run(
        run_dir=run_dir,
        max_recover_time_sec=float(args.max_recover_time_sec),
        require_all_get_complete=bool(args.require_all_get_complete),
        require_source_cardinality=bool(args.require_source_cardinality),
        require_expected_failure_pass=bool(args.require_expected_failure_pass),
        require_comm_errors_zero=bool(args.require_comm_errors_zero),
    )

    output_path = (
        Path(str(args.output)).resolve()
        if str(args.output).strip()
        else run_dir / "gate_review.json"
    )
    output_path.write_text(
        json.dumps(review, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown_path = (
        Path(str(args.markdown_output)).resolve()
        if str(args.markdown_output).strip()
        else run_dir / "gate_review.md"
    )
    markdown_path.write_text(render_markdown(review), encoding="utf-8")

    print(f"GATE_REVIEW {json.dumps(review, ensure_ascii=False)}", flush=True)
    print(f"GATE_REVIEW_JSON {output_path}", flush=True)
    print(f"GATE_REVIEW_MD {markdown_path}", flush=True)

    if bool(review.get("passed")) or bool(args.allow_failure_exit):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
