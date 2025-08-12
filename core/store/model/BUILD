load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Basic type definitions
sc_header_only_library(
    name = "chunk_meta",
    hdrs = ["chunk_meta.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "model_location",
    hdrs = ["model_location.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "memory_state",
    hdrs = ["memory_state.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "chunk_directory",
    hdrs = ["chunk_directory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
    ],
)

sc_header_only_library(
    name = "model_config",
    hdrs = ["model_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:device_types_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
    ],
)

# Model memory coordinator
sc_cc_library(
    name = "model_memory_coordinator",
    srcs = ["model_memory_coordinator.cc"],
    hdrs = ["model_memory_coordinator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":model_location",
        "//core/common:cuda_memory_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:error_handling_lib",
        "//core/store:device_registry",
        "//core/store:device_types",
        "//core/store:direct_write",
        "//core/store:loading_spec",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

# Transfer helpers
sc_cc_library(
    name = "transfer_helpers",
    srcs = ["transfer_helpers.cc"],
    hdrs = ["transfer_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":model_location",
        ":model_memory_coordinator",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/store:device_types",
        "@abseil-cpp//absl/log:absl_check",
    ],
)

# Transfer constants
sc_header_only_library(
    name = "transfer_constants",
    hdrs = ["transfer_constants.h"],
    visibility = ["//visibility:public"],
)

# Transfer service
sc_cc_library(
    name = "transfer_service",
    srcs = ["transfer_service.cc"],
    hdrs = ["transfer_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_memory_coordinator",
        ":transfer_constants",
        ":transfer_helpers",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store/loader:dvmp_region_sink",
        "//core/store/loader:file_partition_source",
        "//core/store/loader:gpu_memory_sink",
        "//core/store/loader:pump",
        "//core/store/loader:streaming_buffer_adapter",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

# Chunk export service
sc_cc_library(
    name = "chunk_export_service",
    srcs = ["chunk_export_service.cc"],
    hdrs = ["chunk_export_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_memory_coordinator",
        ":transfer_constants",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

# Memory manager
sc_cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_export_service",
        ":chunk_meta",
        ":memory_state",
        ":model_location",
        ":model_memory_coordinator",
        ":transfer_helpers",
        ":transfer_service",
        "//core/common:cuda_api_lib",
        "//core/common:device_types_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/communicator:engine_lib",
        "//core/store:device_types",
        "//core/store:direct_write",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader:dvmp_region_sink",
        "//core/store/loader:gpu_memory_sink",
        "//core/store/loader:pump",
        "//core/store/loader:streaming_buffer_adapter",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
    ],
)

# Main model library
sc_cc_library(
    name = "model",
    srcs = ["model.cc"],
    hdrs = [
        "chunk_directory.h",
        "model.h",
        "model_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":memory_manager",
        ":memory_state",
        ":model_location",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
    ],
)
