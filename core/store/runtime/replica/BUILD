load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library")

sc_cc_library(
    name = "replica_promotion_manager",
    srcs = ["replica_promotion_manager.cc"],
    hdrs = ["replica_promotion_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:communication_types",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/runtime:ingestion_events",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
    ],
)

sc_cc_library(
    name = "replica_runtime",
    srcs = ["replica_runtime.cc"],
    hdrs = [
        "replica_info.h",
        "replica_runtime.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store:communication_manager",
        "//core/store:communication_types",
        "//core/store:device_manager",
        "//core/store:eviction_service",
        "//core/store:metrics_collector",
        "//core/store:replica_registry",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:memory_state",
        "//core/store/replica:replica_placement",
        "//core/store/runtime:ingestion_events",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/context:runtime_context_events",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)

cc_test(
    name = "replica_runtime_test",
    srcs = ["replica_runtime_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":replica_runtime",
        "//core/store:device_types",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/store/runtime/context:runtime_context",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/status:status",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
