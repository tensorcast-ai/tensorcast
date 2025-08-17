load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

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
        "@cuda//:cublas",
        "@cuda//:cublasLt",
        "@cuda//:cudart",
        "@cuda//:cufft",
        "@cuda//:curand",
        "@cuda//:cusparse",
        "@cuda//:nvJitLink",
        "@cudnn",
        "@cuparselt//:cusparselt",
        "@cupti",
        "@nccl",
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
        "@abseil-cpp//absl/log:absl_check",
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
        "@abseil-cpp//absl/log:initialize",
    ],
)

# Model verification utilities
sc_cc_library(
    name = "model_verification_lib",
    srcs = ["model_verification.cc"],
    hdrs = ["model_verification.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
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
    name = "pinned_memory_pool_lib",
    srcs = ["memory/pinned_memory_pool.cc"],
    hdrs = ["memory/pinned_memory_pool.h"],
    includes = ["memory"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
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
        ":pinned_memory_pool_lib",
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
        ":pinned_memory_pool_lib",
    ],
)

# Memory management - Distributed virtual memory pool
sc_cc_library(
    name = "distributed_virtual_memory_pool_lib",
    srcs = ["memory/distributed_virtual_memory_pool.cc"],
    hdrs = ["memory/distributed_virtual_memory_pool.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":error_handling_lib",
        ":metric_objects_lib",
        "//core/store/model:chunk_meta",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

# Metrics - Registry (must come first as metric_objects.h includes it)
sc_cc_library(
    name = "metrics_registry_lib",
    srcs = ["metrics/metrics_registry.cc"],
    hdrs = ["metrics/metrics_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

# Metrics - Metric objects
sc_header_only_library(
    name = "metric_objects_lib",
    hdrs = ["metrics/metric_objects.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":metrics_registry_lib",
        "@abseil-cpp//absl/time",
    ],
)

# Metrics - Export
sc_cc_library(
    name = "metrics_export_lib",
    srcs = ["metrics/metrics_export.cc"],
    hdrs = ["metrics/metrics_export.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":metrics_registry_lib",
        "@nlohmann_json//:json",
    ],
)

# Tracing - Context
sc_header_only_library(
    name = "trace_ctx_lib",
    hdrs = ["trace/trace_ctx.h"],
    visibility = ["//visibility:public"],
)

# Tracing - CUDA async functions
sc_header_only_library(
    name = "trace_cuda_async_fn_lib",
    hdrs = ["trace/trace_cuda_async_fn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
    ],
)

# Tracing - Macros
sc_header_only_library(
    name = "trace_macros_lib",
    hdrs = ["trace/trace_macros.h"],
    visibility = ["//visibility:public"],
    deps = [
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
    name = "distributed_virtual_memory_pool_test",
    srcs = ["memory/distributed_virtual_memory_pool_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":distributed_virtual_memory_pool_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "dvmp_region_handle_test",
    srcs = ["memory/dvmp_region_handle_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":distributed_virtual_memory_pool_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

# Test for cuda_api
cc_test(
    name = "cuda_api_test",
    srcs = ["cuda_api_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":cuda_api_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

# Test for metrics
cc_test(
    name = "metrics_test",
    srcs = ["metrics/metrics_test.cc"],
    deps = [
        ":metric_objects_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

# Aggregate target for backward compatibility
# This should be phased out in favor of specific dependencies
sc_cc_library(
    name = "common",
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":cuda_memory_lib",
        ":device_guard_lib",
        ":device_types_lib",
        ":distributed_virtual_memory_pool_lib",
        ":error_handling_lib",
        ":logging_init_lib",
        ":metric_objects_lib",
        ":metrics_export_lib",
        ":metrics_registry_lib",
        ":model_verification_lib",
        ":pinned_memory_lib",
        ":pinned_memory_pool_lib",
        ":streaming_pinned_buffer_lib",
        ":trace_ctx_lib",
        ":trace_cuda_async_fn_lib",
        ":trace_macros_lib",
        ":trace_manager_lib",
        ":trace_scope_lib",
        "@gsl",
    ],
    alwayslink = True,
)
