load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "loader",
    hdrs = ["loader.h"],
    deps = [
        "//core/store:loading_spec",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "disk_loader",
    srcs = ["disk_loader.cc"],
    hdrs = ["disk_loader.h"],
    deps = [
        ":file_partition_reader",
        ":loader",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/model:memory_manager",
        "//core/store/model:memory_state",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "p2p_loader",
    srcs = ["p2p_loader.cc"],
    hdrs = ["p2p_loader.h"],
    deps = [
        ":loader",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store:communication_types",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/model:memory_manager",
        "//core/store/model:memory_state",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "file_partition_reader",
    srcs = ["file_partition_reader.cc"],
    hdrs = ["file_partition_reader.h"],
    deps = [
        "//core/common",
        "@abseil-cpp//absl/status",
    ],
)
