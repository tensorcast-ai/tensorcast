load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_cc_library(
    name = "catch_main",
    srcs = ["catch_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:logging_init_lib",
        "@abseil-cpp//absl/debugging:failure_signal_handler",
        "@catch2//:catch2_main",
    ],
)

sc_cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_api_lib",
        "//core/store/loader:disk_dir_hash",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "test_helpers",
    srcs = ["test_helpers.cc"],
    hdrs = ["test_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/communicator",
        "@catch2//:catch2_main",
    ],
)

sc_header_only_library(
    name = "concurrency_utils",
    hdrs = ["concurrency_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/synchronization",
    ],
)
