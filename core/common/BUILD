load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

sc_header_only_library(
    name = "device_types_lib",
    hdrs = ["device_types.h"],
    visibility = ["//visibility:public"],
)

# Logging initialization
sc_cc_library(
    name = "logging_init_lib",
    srcs = ["logging_init.cc"],
    hdrs = ["logging_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":otel_log_sink_lib",
        "//proto/tensorcast/config/v1:config_cc",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/log:vlog_is_on",
    ],
)

# Replica verification utilities
sc_cc_library(
    name = "artifact_verification_lib",
    srcs = ["artifact_verification.cc"],
    hdrs = ["artifact_verification.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":artifact_hash_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "//core/cuda:error_handling_lib",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

# Content-addressed hashing utilities (RFC-0007)
sc_cc_library(
    name = "artifact_hash_lib",
    srcs = [
        "artifact_hash_cpu.cc",
        "artifact_hash_gpu.cc",
    ],
    hdrs = [
        "artifact_hash.h",
        "artifact_hash_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@boringssl//:crypto",
        "@folly//folly/executors:cpu_thread_pool_executor",
        "@folly//folly/futures:core",
    ],
)

sc_cc_library(
    name = "artifact_identity_lib",
    srcs = ["artifact_identity.cc"],
    hdrs = ["artifact_identity.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "artifact_identity_test",
    size = "small",
    srcs = ["artifact_identity_test.cc"],
    deps = [
        ":artifact_identity_lib",
        "@catch2//:catch2_main",
    ],
)

sc_cc_library(
    name = "selection_identity_lib",
    srcs = ["selection_identity.cc"],
    hdrs = ["selection_identity.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":artifact_hash_lib",
        "//proto/tensorcast/common/v1:common_cc",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "selection_identity_test",
    size = "small",
    srcs = ["selection_identity_test.cc"],
    deps = [
        ":selection_identity_lib",
        "@abseil-cpp//absl/strings",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_hash_test",
    size = "small",
    srcs = ["artifact_hash_test.cc"],
    deps = [
        ":artifact_hash_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

# Memory management - CUDA memory utilities
sc_cc_library(
    name = "cuda_memory_lib",
    srcs = ["memory/cuda_memory.cc"],
    hdrs = ["memory/cuda_memory.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/cuda:error_handling_lib",
    ],
)

# Memory management - Pinned memory pool (must come before pinned_memory)
sc_cc_library(
    name = "pinned_buffer_pool_lib",
    srcs = ["memory/pinned_buffer_pool.cc"],
    hdrs = [
        "memory/pinned_buffer_pool.h",
    ],
    includes = ["memory"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/cuda:cuda_api_lib",
        "//core/cuda:error_handling_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
    ],
)

cc_test(
    name = "pinned_buffer_pool_test",
    size = "small",
    srcs = ["memory/pinned_buffer_pool_test.cc"],
    deps = [
        ":pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "@catch2//:catch2_main",
    ],
)

# Memory management - Pinned memory
sc_cc_library(
    name = "pinned_memory_lib",
    srcs = ["memory/pinned_memory.cc"],
    hdrs = ["memory/pinned_memory.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
    deps = [
        ":pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:error_handling_lib",
    ],
)

sc_cc_library(
    name = "pinned_memory_authority_lib",
    srcs = ["memory/pinned_memory_authority.cc"],
    hdrs = ["memory/pinned_memory_authority.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pinned_buffer_pool_lib",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@folly//folly/executors:cpu_thread_pool_executor",
        "@folly//folly/futures:core",
    ],
)

# Memory location enum header
sc_header_only_library(
    name = "memory_location_lib",
    hdrs = ["memory/memory_location.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
)

sc_cc_library(
    name = "host_memory_lib",
    srcs = ["memory/host_memory.cc"],
    hdrs = ["memory/host_memory.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "host_memory_test",
    size = "small",
    srcs = ["memory/host_memory_test.cc"],
    deps = [
        ":host_memory_lib",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

# Memory management - Streaming pinned buffer
sc_cc_library(
    name = "streaming_pinned_buffer_lib",
    srcs = ["memory/streaming_pinned_buffer.cc"],
    hdrs = ["memory/streaming_pinned_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:error_handling_lib",
    ],
)

sc_cc_library(
    name = "streaming_chunk_guard_lib",
    srcs = ["memory/streaming_chunk_guard.cc"],
    hdrs = ["memory/streaming_chunk_guard.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":streaming_pinned_buffer_lib",
    ],
)

# Async runtime / executors (Folly-based)
sc_cc_library(
    name = "async_runtime_lib",
    srcs = ["async_runtime.cc"],
    hdrs = ["async_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@folly//folly/executors:cpu_thread_pool_executor",
        "@folly//folly/executors:serial_executor",
        "@folly//folly/futures:core",
        "@folly//folly/io/async:request_context",
        "@glog",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "ready_signal_lib",
    hdrs = ["ready_signal.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@folly//folly/futures:shared_promise",
    ],
)

# Canonical granularity constants (artifact chunk, tx slice, hash leaf)
sc_header_only_library(
    name = "const_granularity_lib",
    hdrs = ["const/granularity.h"],
    visibility = ["//visibility:public"],
)

# Memory management - Distributed virtual memory pool
## distributed_virtual_memory_pool_lib removed in UMA V3 cutover

## Metrics - OpenTelemetry wrappers (header-only)
## metric_objects_lib removed; use OpenTelemetry API directly

# Tracing - RequestContext bridge (TraceManager TLS ↔ folly::RequestContext)
sc_header_only_library(
    name = "trace_request_data_lib",
    hdrs = ["trace/trace_request_data.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":trace_manager_lib",
        "@abseil-cpp//absl/synchronization",
        "@folly//folly/io/async:request_context",
    ],
)

# OpenTelemetry - gRPC propagation helpers (header-only)
sc_header_only_library(
    name = "otel_grpc_propagation_lib",
    hdrs = ["otel/grpc_propagation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@opentelemetry-cpp//api:api",
    ],
)

# OpenTelemetry - build-time config header (header-only)

# OpenTelemetry - TraceScope bridge (header-only)
sc_header_only_library(
    name = "otel_trace_scope_bridge_hdr",
    hdrs = ["otel/trace_scope_bridge.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "otel_init_lib",
    srcs = ["otel/init.cc"],
    hdrs = ["otel/init.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/config/v1:config_cc",
        "@opentelemetry-cpp//api:api",
        # Explicitly depend on SDK trace/resource to avoid relying on exporter transitive deps
        "@opentelemetry-cpp//sdk/src/trace:trace",
        "@opentelemetry-cpp//sdk/src/logs:logs",
        "@opentelemetry-cpp//sdk/src/resource:resource",
        "@opentelemetry-cpp//sdk/src/metrics:metrics",
        "@opentelemetry-cpp//exporters/otlp:otlp_http_exporter",
        "@opentelemetry-cpp//exporters/otlp:otlp_grpc_exporter",
        "@opentelemetry-cpp//exporters/otlp:otlp_grpc_metric_exporter",
        "@opentelemetry-cpp//exporters/otlp:otlp_http_metric_exporter",
    ],
)

# OpenTelemetry - absl LogSink injecting trace/span context (optional)
sc_cc_library(
    name = "otel_log_sink_lib",
    srcs = ["otel/logging_sink.cc"],
    hdrs = ["otel/logging_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/config/v1:config_cc",
        "@abseil-cpp//absl/log:vlog_is_on",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@abseil-cpp//absl/types:span",
        "@opentelemetry-cpp//api:api",
    ],
)

# Tracing - CUDA async functions
sc_header_only_library(
    name = "trace_cuda_async_fn_lib",
    hdrs = ["trace/trace_cuda_async_fn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":trace_manager_lib",
        "//core/cuda:cuda_api_lib",
        "@folly//folly:executor",
    ],
)

# Async Copy Manager (minimal RFC-0018 core)
sc_cc_library(
    name = "async_copy_manager_lib",
    srcs = ["async_copy_manager.cc"],
    hdrs = ["async_copy_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":async_runtime_lib",
        ":trace_cuda_async_fn_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_runtime_lib",
        "//core/cuda:device_guard_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

# Tracing - Macros
sc_header_only_library(
    name = "trace_macros_lib",
    hdrs = ["trace/trace_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":otel_trace_scope_bridge_hdr",
        ":trace_manager_lib",
        ":trace_request_data_lib",
        ":trace_scope_lib",
    ],
)

# Tracing - Manager
sc_cc_library(
    name = "trace_manager_lib",
    srcs = ["trace/trace_manager.cc"],
    hdrs = ["trace/trace_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ],
)

# Tracing - Scope
sc_cc_library(
    name = "trace_scope_lib",
    srcs = ["trace/trace_scope.cc"],
    hdrs = ["trace/trace_scope.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":trace_manager_lib",
        "@abseil-cpp//absl/time",
    ],
)

cc_test(
    name = "streaming_pinned_buffer_test",
    srcs = ["memory/streaming_pinned_buffer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":pinned_buffer_pool_lib",
        ":streaming_pinned_buffer_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "streaming_chunk_guard_test",
    srcs = ["memory/streaming_chunk_guard_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":pinned_buffer_pool_lib",
        ":streaming_chunk_guard_lib",
        ":streaming_pinned_buffer_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "async_runtime_test",
    srcs = ["async_runtime_test.cc"],
    deps = [
        ":async_runtime_lib",
        ":trace_manager_lib",
        ":trace_request_data_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@folly//folly/io/async:request_context",
    ],
)

## Legacy metrics test removed – metrics now use OTel API

# Aggregate target for backward compatibility
# This should be phased out in favor of specific dependencies
sc_cc_library(
    name = "common",
    visibility = ["//visibility:public"],
    deps = [
        ":artifact_hash_lib",
        ":artifact_verification_lib",
        ":async_copy_manager_lib",
        ":async_runtime_lib",
        ":cuda_memory_lib",
        ":device_types_lib",
        ":logging_init_lib",
        ":pinned_buffer_pool_lib",
        ":pinned_memory_lib",
        ":ready_signal_lib",
        ":streaming_pinned_buffer_lib",
        ":system_capabilities_lib",
        ":trace_cuda_async_fn_lib",
        ":trace_macros_lib",
        ":trace_manager_lib",
        ":trace_request_data_lib",
        ":trace_scope_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_runtime_lib",
        "//core/cuda:device_guard_lib",
        "//core/cuda:error_handling_lib",
        "@gsl",
    ],
    alwayslink = True,
)

sc_cc_library(
    name = "capability_token_lib",
    srcs = ["capability_token.cc"],
    hdrs = ["capability_token.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@boringssl//:crypto",
        "@protobuf",
    ],
)

sc_cc_library(
    name = "system_capabilities_lib",
    srcs = ["system_capabilities.cc"],
    hdrs = ["system_capabilities.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:absl_log",
    ],
)

# Unified runtime config loader for DaemonConfig
sc_cc_library(
    name = "daemon_config_io_lib",
    srcs = ["config/daemon_config_io.cc"],
    hdrs = ["config/daemon_config_io.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/communicator:config_io_lib",
        "//proto/tensorcast/config/v1:config_cc",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
        "@protobuf",
        "@yaml-cpp//:yaml-cpp",
    ],
)

cc_test(
    name = "daemon_config_io_test",
    size = "small",
    srcs = ["config/daemon_config_io_test.cc"],
    deps = [
        ":daemon_config_io_lib",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "capability_token_test",
    size = "small",
    srcs = ["capability_token_test.cc"],
    deps = [
        ":capability_token_lib",
        "@abseil-cpp//absl/time",
        "@catch2//:catch2_main",
    ],
)
