load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "cuda_conditional_deps", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# Core CUDA API abstraction
sc_cc_library(
    name = "cuda_api_lib",
    srcs = select({
        ":use_fake_cuda": ["cuda_fake.cc"],
        "//conditions:default": ["cuda_real.cc"],
    }),
    hdrs = ["cuda_api.h"],
    defines = select({
        ":use_fake_cuda": ["USE_FAKE_CUDA"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":error_handling_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@cublas",
        "@cublas//:cublasLt",
        "@cudart",
        "@cudnn",
        "@cufft",
        "@cupti",
        "@curand",
        "@cusparse",
        "@cusparselt",
        "@nccl",
        "@nvjitlink//:nvJitLink",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# Device management utilities
sc_header_only_library(
    name = "device_guard_lib",
    hdrs = ["device_guard.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
    ],
)

sc_header_only_library(
    name = "device_types_lib",
    hdrs = ["device_types.h"],
    visibility = ["//visibility:public"],
)

# Error handling utilities
sc_header_only_library(
    name = "error_handling_lib",
    hdrs = ["error_handling.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:check",
    ],
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
        ":cuda_api_lib",
        ":error_handling_lib",
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@boringssl//:crypto",
    ] + cuda_conditional_deps(
        fake_cuda_deps = [],
        real_cuda_deps = ["@nvrtc//:nvrtc"],
    ),
)

sc_cc_library(
    name = "artifact_identity_lib",
    srcs = ["artifact_identity.cc"],
    hdrs = ["artifact_identity.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings:strings",
    ],
)

# Memory management - CUDA memory utilities
sc_cc_library(
    name = "cuda_memory_lib",
    srcs = ["memory/cuda_memory.cc"],
    hdrs = ["memory/cuda_memory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
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
        ":cuda_api_lib",
        ":error_handling_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
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
        ":cuda_api_lib",
        ":error_handling_lib",
        ":pinned_buffer_pool_lib",
    ],
)

# Memory location enum header
sc_header_only_library(
    name = "memory_location_lib",
    hdrs = ["memory/memory_location.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
)

# Virtual Address Space (VS)
sc_cc_library(
    name = "virtual_address_space_lib",
    srcs = ["memory/virtual_address_space.cc"],
    hdrs = ["memory/virtual_address_space.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
        ":system_capabilities_lib",
        "//core/store/replica:chunk_meta",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

# Memory management - Streaming pinned buffer
sc_cc_library(
    name = "streaming_pinned_buffer_lib",
    srcs = ["memory/streaming_pinned_buffer.cc"],
    hdrs = ["memory/streaming_pinned_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
        ":pinned_buffer_pool_lib",
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

# Tracing - Context
sc_header_only_library(
    name = "trace_ctx_lib",
    hdrs = ["trace/trace_ctx.h"],
    visibility = ["//visibility:public"],
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
        ":cuda_api_lib",
        ":trace_manager_lib",
    ],
)

# Async Copy Manager (minimal RFC-0018 core)
sc_cc_library(
    name = "async_copy_manager_lib",
    srcs = ["async_copy_manager.cc"],
    hdrs = ["async_copy_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":trace_cuda_async_fn_lib",
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
        ":trace_ctx_lib",
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
        ":trace_ctx_lib",
        ":trace_manager_lib",
        "@abseil-cpp//absl/time",
    ],
)

# Tests for memory components
cc_test(
    name = "virtual_address_space_test",
    srcs = ["memory/virtual_address_space_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":virtual_address_space_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "cpu_va_region_handle_test",
    srcs = ["memory/cpu_va_region_handle_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":virtual_address_space_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
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

# Test for cuda_api
cc_test(
    name = "cuda_api_test",
    srcs = ["cuda_api_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":cuda_api_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
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
        ":cuda_api_lib",
        ":cuda_memory_lib",
        ":device_guard_lib",
        ":device_types_lib",
        ":error_handling_lib",
        ":logging_init_lib",
        ":pinned_buffer_pool_lib",
        ":pinned_memory_lib",
        ":streaming_pinned_buffer_lib",
        ":system_capabilities_lib",
        ":trace_ctx_lib",
        ":trace_cuda_async_fn_lib",
        ":trace_macros_lib",
        ":trace_manager_lib",
        ":trace_scope_lib",
        ":virtual_address_space_lib",
        "@gsl",
    ],
    alwayslink = True,
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
