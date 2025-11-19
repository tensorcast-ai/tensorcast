load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library")

package(default_visibility = ["//visibility:public"])

sc_cc_library(
    name = "materialization_service",
    srcs = ["materialization_service.cc"],
    hdrs = ["materialization_service.h"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_types",
        "//core/store:replica_registry",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/replica",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

sc_cc_library(
    name = "materialization_coordinator",
    srcs = ["materialization_coordinator.cc"],
    hdrs = ["materialization_coordinator.h"],
    deps = [
        ":materialization_service",
        "//core/common:memory_location_lib",
        "//core/store:device_types",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/control:materialization_backend",
        "//core/store/materialization/control:materialize_orchestrator",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

sc_cc_library(
    name = "ingestion_runtime",
    srcs = ["ingestion_runtime.cc"],
    hdrs = ["ingestion_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":materialization_coordinator",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:communication_manager",
        "//core/store:device_manager",
        "//core/store:metrics_collector",
        "//core/store:replica_registry",
        "//core/store:store_engine_options",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/replica",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)

cc_test(
    name = "materialization_service_test",
    srcs = ["materialization_service_test.cc"],
    deps = [
        ":materialization_service",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_manager",
        "//core/store:replica_registry",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/replica",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "ingestion_runtime_test",
    srcs = ["ingestion_runtime_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":ingestion_runtime",
        "//core/store:store_engine_options",
        "//core/store/runtime:runtime_env",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/ingestion/testing",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_check",
        "@catch2//:catch2_main",
    ],
)
