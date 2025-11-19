load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

sc_header_only_library(
    name = "ingestion_events",
    hdrs = ["ingestion_events.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
    ],
)

sc_cc_library(
    name = "runtime_env",
    srcs = ["runtime_env.cc"],
    hdrs = ["runtime_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:store_engine_options",
        "//core/store:worker_identity",
        "//core/store/runtime/context:runtime_context",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "runtime_services_test",
    srcs = ["runtime_services_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_check",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
