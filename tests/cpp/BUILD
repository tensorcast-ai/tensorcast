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
    name = "model_disk_test",
    srcs = [
        "model_disk/aligned_cpu_test.cc",
        "model_disk/basic_cpu_test.cc",
        "model_disk/cpu_to_gpu_test.cc",
        "model_disk/disk_loader_streaming_buffer_test.cc",
        "model_disk/error_test.cc",
        "model_disk/gpu_auto_release_test.cc",
        "model_disk/model_verification_system_test.cc",
        "model_disk/multi_gpu_test.cc",
        "model_disk/streaming_buffer_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        ":common",
        "//core/common",
        "//core/communicator",
        "//core/store",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_test",
    srcs = [
        "checkpoint/duplicate_buffer_test.cc",
    ],
    env = {
        "LD_LIBRARY_PATH": "$$LD_LIBRARY_PATH:external/+_repo_rules2+libtorch/lib:/data/cuda/cuda-12.6/lib64",
    },
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
    linkopts = [
        "-lcuda",  # Link against CUDA runtime
        "-lrt",  # Needed for filesystem or time?
    ],
    deps = [
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
    linkopts = [
        "-lcuda",  # Link against CUDA runtime
        "-lrt",
    ],
    deps = [
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
    linkopts = [
        "-lcuda",
    ],
    deps = [
        ":common",
        "//core/store",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_store_p2p_loader_test",
    srcs = [
        "checkpoint_store_p2p_loader_test.cc",
    ],
    linkopts = [
        "-lcuda",
    ],
    deps = [
        ":common",
        "//core/store",
        "//tests/cpp/communicator:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_store_eviction_test",
    srcs = [
        "checkpoint_store_eviction_test.cc",
    ],
    linkopts = [
        "-lcuda",
    ],
    deps = [
        "//core/store",
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
        ":checkpoint_store_eviction_test",
        ":checkpoint_store_p2p_loader_test",
        ":checkpoint_store_test",
        "//tests/cpp/checkpoint_store:all_tests",
    ],
)

# Quick CheckpointStore test suite (excludes stress tests)
test_suite(
    name = "checkpoint_store_quick_tests",
    tests = [
        ":checkpoint_store_eviction_test",
        ":checkpoint_store_p2p_loader_test",
        ":checkpoint_store_test",
        "//tests/cpp/checkpoint_store:quick_tests",
    ],
)

# loader_unit_tests moved to tests/cpp/unit/BUILD

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
