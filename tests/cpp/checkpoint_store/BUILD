# Copyright (c) 2025, STEP AI. All rights reserved.
#
# BUILD file for CheckpointStore comprehensive test suite

load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "concurrency_test",
    srcs = [
        "concurrency_test.cc",
        "concurrency_utils.h",
    ],
    tags = [
        "checkpoint_store",
        "concurrency",
        "requires_cuda",
    ],
    deps = [
        "//core/store",
        "//tests/cpp:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_gpu_test",
    srcs = [
        "concurrency_utils.h",
        "multi_gpu_test.cc",
    ],
    tags = [
        "checkpoint_store",
        "multi_gpu",
    ],
    deps = [
        "//core/store",
        "//tests/cpp:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "stress_test",
    timeout = "long",  # Allow more time for stress tests
    srcs = [
        "concurrency_utils.h",
        "stress_test.cc",
    ],
    linkopts = ["-lcuda"],
    tags = [
        "checkpoint_store",
        "manual",
        "stress",
    ],  # manual tag for long-running tests
    deps = [
        "//core/store",
        "//tests/cpp:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "edge_test",
    srcs = [
        "concurrency_utils.h",
        "edge_test.cc",
    ],
    linkopts = ["-lcuda"],
    tags = [
        "checkpoint_store",
        "edge",
    ],
    deps = [
        "//core/store",
        "//tests/cpp:common",
        "//tests/cpp/communicator:test_helpers",
        "@catch2//:catch2_main",
    ],
)

# Test suite that runs all checkpoint_store tests
test_suite(
    name = "all_tests",
    tests = [
        ":concurrency_test",
        ":edge_test",
        ":multi_gpu_test",
        ":stress_test",
    ],
)

# Quick test suite (excludes stress tests)
test_suite(
    name = "quick_tests",
    tests = [
        ":concurrency_test",
        ":edge_test",
        ":multi_gpu_test",
    ],
)
