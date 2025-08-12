load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "core",
    deps = [
        "//core/checkpoint",
        "//core/store",
    ],
    alwayslink = True,
)

# cc_binary(
#     name = "cpu_ce_test",
#     srcs = [
#         "common.cc",
#         "common.h",
#         "cpu_ce_test.cc",
#     ],
#     deps = [
#         "//core/communicator",
#     ],
# )

# cc_binary(
#     name = "gpu_ce_test",
#     srcs = [
#         "common.cc",
#         "common.h",
#         "gpu_ce_test.cc",
#     ],
#     deps = [
#         "//core/communicator",
#     ],
#     # linkopts = ["-lcuda"],
# )

cc_library(
    name = "common",
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    visibility = ["//tests/cpp/checkpoint_store:__pkg__"],
    deps = [
        "//core/store",
        "@abseil-cpp//absl/log:log",
    ],
)

cc_test(
    name = "checkpoint_test",
    srcs = [
        "checkpoint/duplicate_buffer_test.cc",
    ],
    tags = ["requires_cuda"],
    deps = [
        "//core/checkpoint",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "model_p2p_registration_test",
    srcs = [
        "model_p2p_registration_test.cc",
        # "common.cc", # Add if common helpers are used
        # "common.h",
    ],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "-lcuda",
            "-lrt",
        ],
    }),
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "model_p2p_transfer_test",
    srcs = [
        "model_p2p_transfer_test.cc",
    ],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "-lcuda",
            "-lrt",
        ],
    }),
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:initialize",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_store_test",
    srcs = [
        "checkpoint_store_test.cc",
    ],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":common",
        "//core/common",
        "//core/store",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_store_p2p_loader_test",
    srcs = [
        "checkpoint_store_p2p_loader_test.cc",
    ],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    deps = [
        ":common",
        "//core/common",
        "//core/store",
        "//tests/cpp/communicator:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_cuda_api",
    srcs = ["test_cuda_api.cc"],
    # The `tags` attribute cannot be made configurable. Use a static tag list instead.
    tags = ["requires_cuda"],
    deps = [
        "//core/common",
        "@catch2//:catch2_main",
    ],
)

# Comprehensive CheckpointStore test suite
test_suite(
    name = "checkpoint_store_all_tests",
    tests = [
        ":checkpoint_store_p2p_loader_test",
        ":checkpoint_store_test",
        "//tests/cpp/checkpoint_store:all_tests",
    ],
)

cc_test(
    name = "model_memory_coordinator_test",
    srcs = ["model_memory_coordinator_test.cc"],
    deps = [
        "//core/store/model:model_memory_coordinator",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@catch2//:catch2_main",
    ],
)

cc_library(
    name = "test_common",
    srcs = [
        "common.cc",
    ],
    hdrs = [
        "common.h",
    ],
    deps = [
        "//core/common",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "catch_main",
    testonly = 1,
    srcs = ["catch_main.cc"],
    deps = [
        "//core/common",
        "@abseil-cpp//absl/debugging:failure_signal_handler",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:initialize",
        "@catch2//:catch2_main",
    ],
)

config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)
