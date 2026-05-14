# 0117 Group Realization Cutover Evidence

Date: 2026-05-14

Status: Cutover evidence for `docs/designs/0117-group-realization-transaction.md`.

## Scope

This note records the final `0117` validation used to remove relaxed
operation-id grouping and make staged `group_realization` the WeightPublisher
model-parallel coherence path.

The current cutover goal is correctness first: all TP members stage against one
semantic `GroupVersionSet`, Global Store admits publish only after required
members are prepared, and receivers acquire staged values only after the publish
barrier.

## Real CUDA Smoke

Environment:

- host: current 4-GPU worker
- visible CUDA devices: 4
- publish device: `cpu`
- materialize device: `cuda:0`
- daemon config: `examples/config/store_daemon_config_cross_host_bench.yaml`
- output JSON:
  `/data/tc/0117-group-realization/single-host-real-smoke-unified.json`

Command shape:

```bash
source .venv/bin/activate
LD_LIBRARY_PATH=/data/cuda/cuda-12.8/lib64:${LD_LIBRARY_PATH:-} \
  python -m tensorcast.tools.weight_publisher_e2e single-host \
  --daemon-config-path examples/config/store_daemon_config_cross_host_bench.yaml \
  --daemon-ready-timeout-s 180 \
  --num-versions 1 \
  --keep-last 1 \
  --pre-publish-trim-margin 0 \
  --publish-interval-s 0 \
  --receiver-timeout-s 90 \
  --payload-mode tp_ranked \
  --tp-world-size 1 \
  --tp-total-bytes 1024 \
  --receiver-apply-mode tp_bind_into_swap \
  --transport-group-kind group_realization_transport \
  --transport-group-namespace smoke-0117-acquire \
  --transport-group-total-parts 1 \
  --transport-group-receiver-index 0 \
  --tp-materialize-deadline-s 90 \
  --publish-device cpu \
  --materialize-device cuda:0 \
  --output-json /data/tc/0117-group-realization/single-host-real-smoke-unified.json
```

Observed result:

| Field | Value |
| --- | --- |
| versions published | `1` |
| versions received | `1` |
| receiver apply mode | `tp_bind_into_swap` |
| receiver apply operation | `stage_acquire` |
| pointer stable | `true` |
| publish latency | `0.0976s` |
| materialize/apply latency | `1.5614s` |
| payload bytes | `1024` |

Interpretation:

- The receiver did not use live in-place `bind_into`; it staged, waited for the
  group publish barrier, and acquired through `stage_acquire`.
- The old non-group TP receive branch was removed before this run; the smoke
  therefore validates the unified staged path.
- The daemon and Global Store configs used by local and benchmark flows now
  enable group realization and staged acquire by default.
- Teardown emitted Global Store unregister and CUDA unloading warnings after
  successful completion. The process exit code was `0` and the JSON result
  contains the successful staged acquire.

## Issues Found During Cutover

The smoke exposed three integration gaps that were fixed before recording the
passing run:

- WeightPublisher could issue work while the daemon startup gate was still in
  progress. The tool now waits for daemon `READY` after `tc.init(...)`.
- The smoke initially used a stale daemon binary. `//daemon:tensorcast_daemon`
  was rebuilt before rerunning.
- Benchmark/default configs did not enable all final-stage features. Global
  Store group realization and daemon staged acquire/group realization are now
  enabled in the relevant example configs.

## Test Coverage

Python coverage exercised SDK operation-id cleanup, mapped binding behavior,
WeightPublisher grouped materialization, cross-host runner planning, and Global
Store strict transport semantics.

```bash
source .venv/bin/activate && pytest tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/global_store/test_pending_transport_request_repository.py -q
source .venv/bin/activate && pytest tests/python/api/test_call_context.py tests/python/api/test_serving_binding_reference_consumer.py tests/python/global_store/test_group_realization.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/global_store/test_pending_transport_request_repository.py tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py -q
source .venv/bin/activate && pytest tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py -q
```

C++ coverage exercised daemon staged/acquire behavior, binding registries,
target publication, materialization policy utilities, and operation RPCs.

```bash
bazel test //daemon:materialization_policy_utils_test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
bazel test //daemon:binding_registry_test //daemon:target_publication_registry_test //daemon:materialization_policy_utils_test //daemon:owned_binding_service_test //daemon:grpc_service_impl_operation_rpc_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Static checks:

```bash
source .venv/bin/activate && python -m py_compile tensorcast/api/store/binding.py tensorcast/api/store/artifact.py tensorcast/tools/weight_publisher_e2e.py examples/cross_host/cross_host_weight_publisher_runner.py examples/cross_host/summarize_scaleout_suite.py
source .venv/bin/activate && ruff check tensorcast/api/store/binding.py tensorcast/api/store/artifact.py tensorcast/tools/weight_publisher_e2e.py examples/cross_host/cross_host_weight_publisher_runner.py examples/cross_host/summarize_scaleout_suite.py tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/global_store/test_pending_transport_request_repository.py
source .venv/bin/activate && ruff format --check tensorcast/api/store/binding.py tensorcast/api/store/artifact.py tensorcast/tools/weight_publisher_e2e.py examples/cross_host/cross_host_weight_publisher_runner.py examples/cross_host/summarize_scaleout_suite.py tests/python/test_binding.py tests/python/api/test_mapped_binding.py tests/python/tools/test_weight_publisher_e2e_tp_bind_retry.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/global_store/test_pending_transport_request_repository.py
```

## Cross-Host Status

The cross-host suite was updated so the active/default path is
`group_realization`. Historical grouped/non-group comparisons remain archived in
the 0083 benchmark note, but they are no longer the current suite default.

No remote worker was launched for this cutover because the user did not require
a hard performance threshold and the local 4-GPU worker supplied the needed
real-CUDA correctness signal. Larger fanout pressure data should be gathered
with the updated cross-host suite and tracked under `0119` if it reveals
control-plane scaling work.
