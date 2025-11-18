load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library")

sc_cc_library(
    name = "view_hash_utils",
    srcs = ["view_hash_utils.cc"],
    hdrs = ["view_hash_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_memory_lib",
        "//core/common:memory_location_lib",
        "//core/store/materialization/dataplane:source_hash",
        "//core/store/materialization/dataplane:verification_utils",
        "//core/store/materialization/dataplane:view_plan_source",
        "//core/store/materialization/dataplane:view_transform_executor",
        "//core/store/replica",
        "@gsl",
    ],
)

cc_test(
    name = "view_hash_utils_test",
    srcs = ["view_hash_utils_test.cc"],
    deps = [
        ":view_hash_utils",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
