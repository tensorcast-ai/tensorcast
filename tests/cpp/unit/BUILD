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
        ":comm_engine_tcp_cc",
        "//tests/cpp:catch_main",
    ],
)

# cc_test(
#     name = "comm_engine_rdma_test",
#     srcs = [],
#     linkopts = [
#         "-lm",
#         "-lstdc++",
#         "-g",
#     ],
#     linkstatic = 1,
#     deps = [
#         "//tests/cpp:catch_main",
#         ":comm_engine_rdma_cc",
#     ],
# )

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
        ":test_metrics_cc",
        "//tests/cpp:catch_main",
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
        "//core/common:distributed_memory_pool",
        "//core/store/model:chunk_meta",
        "//tests/cpp:catch_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
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
        "//core/store/model:chunk_meta",
        "//tests/cpp:catch_main",
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
        "//core/store/loader:buffer_pool",
        "//core/store/loader:pump",
        "//core/store/loader:sink",
        "//core/store/loader:source",
        "//tests/cpp:catch_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
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
        "//core/store:direct_write",
        "//core/store/loader:buffer_pool",
        "//core/store/loader:pump",
        "//core/store/loader:sink",
        "//core/store/loader:source",
        "//tests/cpp:catch_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/types:span",
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
        "//core/store/loader:file_partition_source",
        "//tests/cpp:catch_main",
        "@abseil-cpp//absl/status",
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
        "//core/common",
        "//core/common:device_guard",
        "//core/store/loader:gpu_memory_sink",
        "//tests/cpp:catch_main",
        "@abseil-cpp//absl/status",
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
        "//core/store/loader:mux_seekable_source",
        "//core/store/loader:source",
        "//tests/cpp:catch_main",
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
        "//core/common:distributed_memory_pool",
        "//tests/cpp:catch_main",
        "@catch2//:catch2_main",
    ],
)
