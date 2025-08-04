load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:defs.bzl", "cc_library")

# Helper library for test utilities
cc_library(
    name = "test_helpers",
    srcs = ["test_helpers.cc"],
    hdrs = ["test_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

# GPU Tensor Registration Tests
cc_binary(
    name = "communicator_test",
    srcs = glob(["*.cc"]),
    copts = [],
    tags = ["gpu"],
    deps = [
        ":test_helpers",
        "//core/communicator",
        "//core/store",
    ],
)
