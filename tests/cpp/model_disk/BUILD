load("@rules_cc//cc:defs.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "disk_loader_streaming_buffer_test",
    srcs = [
        "disk_loader_streaming_buffer_test.cc",
    ],
    includes = [
        "..",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:loading_spec",
        "//core/store/loader:disk_loader",
        "//core/store/loader:source",
        "//core/store/model:memory_manager",
        "//tests/cpp:test_common",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "aligned_cpu_test",
    srcs = [
        "aligned_cpu_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "basic_cpu_test",
    srcs = [
        "basic_cpu_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "cpu_to_gpu_test",
    srcs = [
        "cpu_to_gpu_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "error_test",
    srcs = [
        "error_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_auto_release_test",
    srcs = [
        "gpu_auto_release_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "model_verification_system_test",
    srcs = [
        "model_verification_system_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_gpu_test",
    srcs = [
        "multi_gpu_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "streaming_buffer_test",
    srcs = [
        "streaming_buffer_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "model_verifier_direct_test",
    srcs = [
        "model_verifier_direct_test.cc",
    ],
    linkopts = [
        "-lm",
        "-lstdc++",
    ],
    deps = [
        "//core/common",
        "//core/communicator",
        "//core/store",
        "//tests/cpp:test_common",
        "@catch2//:catch2_main",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        ":aligned_cpu_test",
        ":basic_cpu_test",
        ":cpu_to_gpu_test",
        ":disk_loader_streaming_buffer_test",
        ":error_test",
        ":gpu_auto_release_test",
        ":model_verification_system_test",
        ":model_verifier_direct_test",
        ":multi_gpu_test",
        ":streaming_buffer_test",
    ],
)
