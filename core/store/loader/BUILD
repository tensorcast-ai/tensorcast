load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Base interfaces
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
    name = "source",
    hdrs = ["source.h"],
    deps = [
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "sink",
    hdrs = ["sink.h"],
    deps = [
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "buffer_pool",
    hdrs = ["buffer_pool.h"],
    deps = [
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

# Pump orchestration
cc_library(
    name = "pump",
    srcs = ["pump.cc"],
    hdrs = ["pump.h"],
    deps = [
        ":buffer_pool",
        ":sink",
        ":source",
        ":streaming_buffer_adapter",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
    ],
)

# Source implementations
cc_library(
    name = "file_partition_source",
    srcs = ["file_partition_source.cc"],
    hdrs = ["file_partition_source.h"],
    deps = [
        ":source",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "remote_key_source",
    srcs = ["remote_key_source.cc"],
    hdrs = ["remote_key_source.h"],
    deps = [
        ":source",
        "//core/communicator",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
    ],
)

cc_library(
    name = "mux_seekable_source",
    srcs = ["mux_seekable_source.cc"],
    hdrs = ["mux_seekable_source.h"],
    deps = [
        ":source",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status:statusor",
    ],
)

# Sink implementations
cc_library(
    name = "gpu_memory_sink",
    srcs = ["gpu_memory_sink.cc"],
    hdrs = ["gpu_memory_sink.h"],
    deps = [
        ":sink",
        "//core/common",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "dvmp_region_sink",
    srcs = ["dvmp_region_sink.cc"],
    hdrs = ["dvmp_region_sink.h"],
    deps = [
        ":sink",
        "//core/common:distributed_memory_pool",
        "//core/store/model:memory_manager",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "dvmp_mapped_sink",
    srcs = ["dvmp_mapped_sink.cc"],
    hdrs = ["dvmp_mapped_sink.h"],
    deps = [
        ":sink",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

cc_library(
    name = "unified_memory_sink",
    srcs = ["unified_memory_sink.cc"],
    hdrs = ["unified_memory_sink.h"],
    deps = [
        ":sink",
        "//core/store/model:chunk_meta",
        "//core/store/model:memory_manager",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
    ],
)

# Buffer pool adapter
cc_library(
    name = "streaming_buffer_adapter",
    srcs = ["streaming_buffer_adapter.cc"],
    hdrs = ["streaming_buffer_adapter.h"],
    deps = [
        ":buffer_pool",
        "//core/common",
        "@abseil-cpp//absl/log",
    ],
)

# Main loaders (refactored to use new abstractions)
cc_library(
    name = "disk_loader",
    srcs = ["disk_loader.cc"],
    hdrs = ["disk_loader.h"],
    deps = [
        ":dvmp_region_sink",
        ":file_partition_source",
        ":gpu_memory_sink",
        ":loader",
        ":pump",
        ":streaming_buffer_adapter",
        ":unified_memory_sink",
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
        ":dvmp_region_sink",
        ":file_partition_source",
        ":gpu_memory_sink",
        ":loader",
        ":mux_seekable_source",
        ":pump",
        ":remote_key_source",
        ":streaming_buffer_adapter",
        ":unified_memory_sink",
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

# FilePartitionReader removed - replaced by file_partition_source and pump pipeline
