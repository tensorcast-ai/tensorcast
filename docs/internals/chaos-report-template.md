# Chaos Report Template

Use this template for each multi-host chaos run package.

## 1. Run Metadata

- `run_id`:
- `phase`: `small` / `medium` / `large`
- `chaos_profile`: `none` / `single_fault` / `combo_fault`
- `case_schema`:
- `global_store_profile`: `fast_failover` / `slow_cleanup`
- `orchestratorctl_meta`:

## 2. Summary Gate Checklist

- `all_get_complete`:
- `source_cardinality_timeline` present:
- `recover_time_sec`:
- `comm_bytes_delta`:
- `comm_errors_delta`:
- `retry_reason_buckets`:
- `budget_exit_reason_buckets`:
- `expected_failure_pass`:
- `failure_classification_counts`:

## 3. Event Timeline

Attach `events.jsonl` and include a short readout:

- first chaos event timestamp:
- first request failure timestamp:
- recovery-complete timestamp:
- correlation ids used:

## 4. Budget Trace

Include aggregated budget fields and one example per case:

```json
{
  "retry_reason_buckets": {
    "transport_timeout": 0
  },
  "budget_exit_reason_buckets": {
    "success": 0
  },
  "case_examples": [
    {
      "case_name": "example",
      "retry_reason_buckets": {},
      "budget_exit_reason_buckets": {}
    }
  ]
}
```

## 5. Failure Classification

Provide both aggregate counts and representative events:

```json
{
  "infra": 0,
  "product": 0,
  "unknown": 0
}
```

## 6. Expected Failure Cases

For each negative case, record:

- `expected_outcome=failure`
- `expected_error_pattern`
- actual return code
- matched/not matched
- final status (`expected_failure_pass` / `unexpected_failure` / `unexpected_success`)

## 7. Handoff Attachments

- `summary.json`
- `events.jsonl`
- `cases/<case_name>/result.json`
- `cases/<case_name>/metrics.json`
- `cases/<case_name>/classification.json`
- `meta/orchestratorctl_steps.jsonl`
