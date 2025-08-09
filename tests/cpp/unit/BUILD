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
        "//core/common",
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

# New loader abstraction tests
cc_test(
    name = "test_pump",
    srcs = ["test_pump.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/store/loader:buffer_pool",
        "//core/store/loader:pump",
        "//core/store/loader:sink",
        "//core/store/loader:source",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_pump_direct",
    srcs = ["test_pump_direct.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/store:direct_write",
        "//core/store/loader:buffer_pool",
        "//core/store/loader:pump",
        "//core/store/loader:sink",
        "//core/store/loader:source",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_file_partition_source",
    srcs = ["test_file_partition_source.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/store/loader:file_partition_source",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_gpu_memory_sink",
    srcs = ["test_gpu_memory_sink.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/common",
        "//core/common:device_guard",
        "//core/store/loader:gpu_memory_sink",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_mux_seekable_source",
    srcs = ["mux_seekable_source_test.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/store/loader:mux_seekable_source",
        "//core/store/loader:source",
        "@abseil-cpp//absl/status:statusor",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "test_dvmp_region_handle",
    srcs = ["test_dvmp_region_handle.cc"],
    linkopts = [
        "-lm",
        "-lstdc++",
        "-g",
    ],
    linkstatic = 1,
    deps = [
        ":catch_main",
        "//core/common:distributed_memory_pool",
        "@catch2//:catch2_main",
    ],
)
