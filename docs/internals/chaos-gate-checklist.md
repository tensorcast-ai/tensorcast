# Multi-Host Chaos Gate Checklist

Use this checklist to complete Phase 5.3 gate review for `docs/plans/0081-multihost-put-get-chaos-stability.md`.

## 1. Required Inputs

- `small` run directory: `<out_dir>/<run_id_small>`
- `medium` run directory: `<out_dir>/<run_id_medium>`
- `large` run directory: `<out_dir>/<run_id_large>`

Each run directory must include:

- `summary.json`
- `events.jsonl`
- `cases/<case_name>/result.json`
- `cases/<case_name>/metrics.json`
- `cases/<case_name>/classification.json`

## 2. Per-Phase Gate Review

Run gate review for each phase:

```bash
source .venv/bin/activate

python examples/cross_host/chaos_gate_review.py \
  --run-dir <run_dir> \
  --max-recover-time-sec 180
```

Outputs:

- `<run_dir>/gate_review.json`
- `<run_dir>/gate_review.md`

## 3. Signoff Checklist

- [ ] `small` phase gate review passed (`gate_review.json.passed=true`)
- [ ] `medium` phase gate review passed (`gate_review.json.passed=true`)
- [ ] `large` phase gate review passed (`gate_review.json.passed=true`)
- [ ] `all_get_complete` gate passed for positive cases
- [ ] `source_cardinality_timeline_present` gate passed
- [ ] `recover_time_bound` gate passed
- [ ] `comm_errors_zero` gate passed
- [ ] `expected_failure_pass` gate passed for negative cases
- [ ] `no_unexpected_case_status` gate passed
- [ ] `classification_shape` gate passed (`infra/product/unknown`)

Consolidated review command:

```bash
source .venv/bin/activate

python examples/cross_host/chaos_phase_gate_review.py \
  --small-run-dir <run_dir_small> \
  --medium-run-dir <run_dir_medium> \
  --large-run-dir <run_dir_large>
```

## 4. Handoff Attachments

- `small` / `medium` / `large` phase `gate_review.json`
- `small` / `medium` / `large` phase `gate_review.md`
- One consolidated chaos report based on `docs/internals/chaos-report-template.md`
