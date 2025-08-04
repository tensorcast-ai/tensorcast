load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "comm_engine_tcp_cc",
    srcs = ["comm_engine_tcp.cc"],
    deps = [
        "//core/communicator",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@catch2//:catch2_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "comm_engine_rdma_cc",
    srcs = ["comm_engine_rdma.cc"],
    deps = [
        "//core/communicator",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@catch2//:catch2_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "test_metrics_cc",
    srcs = ["test_metrics.cc"],
    deps = [
        "//core/common",
        "@catch2//:catch2_main",
    ],
    alwayslink = 1,
)

cc_library(
    name = "catch_main",
    testonly = 1,
    srcs = ["catch_main.cc"],
    deps = [
        "@abseil-cpp//absl/debugging:failure_signal_handler",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:initialize",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "comm_engine_tcp_test",
    srcs = [],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        ":comm_engine_tcp_cc",
    ],
)

cc_test(
    name = "comm_engine_rdma_test",
    srcs = [],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        ":comm_engine_rdma_cc",
    ],
)

cc_test(
    name = "test_metrics_test",
    srcs = [],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        ":test_metrics_cc",
    ],
)

cc_test(
    name = "test_distributed_memory_pool",
    srcs = ["test_distributed_memory_pool.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/common:distributed_memory_pool",
        "//core/store/model:chunk_meta",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_chunk_state_machine",
    srcs = ["test_chunk_state_machine.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/store/model:chunk_meta",
        "@catch2//:catch2_main",
    ],
)
