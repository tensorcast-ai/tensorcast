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

sc_cc_library(
    name = "piece_view_state_utils",
    srcs = ["piece_view_state_utils.cc"],
    hdrs = ["piece_view_state_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:artifact_hash_lib",
        "//core/common:memory_location_lib",
        "//core/store:view_utils",
        "//core/store/materialization/dataplane:verification_utils",
        "//core/store/materialization/dataplane:view_ingest_executor",
        "//core/store/materialization/dataplane:view_planner",
        "//proto/tensorcast/global_store/v1:global_store_cc",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "view_registration_normalization_utils",
    srcs = ["view_registration_normalization_utils.cc"],
    hdrs = ["view_registration_normalization_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":piece_view_state_utils",
        "//core/store:view_utils",
        "//core/store/materialization/dataplane:view_identity",
        "//core/store/materialization/dataplane:view_planner",
        "//core/store/runtime/metadata:metadata_types",
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
