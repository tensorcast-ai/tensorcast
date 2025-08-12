load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Base interfaces
sc_header_only_library(
    name = "loader",
    hdrs = ["loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:loading_spec",
        "//core/store/model:model_location",
    ],
)

sc_header_only_library(
    name = "source",
    hdrs = ["source.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:direct_write",
    ],
)

sc_header_only_library(
    name = "sink",
    hdrs = ["sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:direct_write",
        "@abseil-cpp//absl/types:span",
    ],
)

sc_header_only_library(
    name = "buffer_pool",
    hdrs = ["buffer_pool.h"],
    visibility = ["//visibility:public"],
)

# Pump orchestration
sc_cc_library(
    name = "pump",
    srcs = ["pump.cc"],
    hdrs = ["pump.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_pool",
        ":sink",
        ":source",
        ":streaming_buffer_adapter",
        "//core/store:direct_write",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
    ],
)

# Source implementations
sc_cc_library(
    name = "file_partition_source",
    srcs = ["file_partition_source.cc"],
    hdrs = ["file_partition_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "remote_key_source",
    srcs = ["remote_key_source.cc"],
    hdrs = ["remote_key_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/communicator:engine_lib",
    ],
)

sc_cc_library(
    name = "mux_seekable_source",
    srcs = ["mux_seekable_source.cc"],
    hdrs = ["mux_seekable_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/common:error_handling_lib",
        "//core/common:metric_objects_lib",
    ],
)

# Sink implementations
sc_cc_library(
    name = "gpu_memory_sink",
    srcs = ["gpu_memory_sink.cc"],
    hdrs = ["gpu_memory_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sink",
        "//core/common:cuda_api_lib",
        "//core/common:device_guard_lib",
    ],
)

sc_cc_library(
    name = "dvmp_region_sink",
    srcs = ["dvmp_region_sink.cc"],
    hdrs = ["dvmp_region_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sink",
        "//core/common:distributed_virtual_memory_pool_lib",
        "@abseil-cpp//absl/types:span",
    ],
)

# Buffer pool adapter
sc_cc_library(
    name = "streaming_buffer_adapter",
    srcs = ["streaming_buffer_adapter.cc"],
    hdrs = ["streaming_buffer_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_pool",
        "//core/common:streaming_pinned_buffer_lib",
    ],
)

# Main loaders (refactored to use new abstractions)
sc_cc_library(
    name = "disk_loader",
    srcs = ["disk_loader.cc"],
    hdrs = ["disk_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":loader",
        "//core/common:error_handling_lib",
        "//core/common:model_verification_lib",
        "//core/store:loading_spec",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "p2p_loader",
    srcs = ["p2p_loader.cc"],
    hdrs = ["p2p_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":loader",
        ":mux_seekable_source",
        ":remote_key_source",
        "//core/common:error_handling_lib",
        "//core/common:metric_objects_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:loading_spec",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/strings",
    ],
)
