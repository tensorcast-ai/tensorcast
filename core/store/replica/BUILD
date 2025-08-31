load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# Basic type definitions
sc_header_only_library(
    name = "chunk_meta",
    hdrs = ["chunk_meta.h"],
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
    name = "replica_config",
    hdrs = ["replica_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:device_types_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
    ],
)

# Replica memory coordinator
sc_cc_library(
    name = "replica_memory_coordinator",
    srcs = ["replica_memory_coordinator.cc"],
    hdrs = ["replica_memory_coordinator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        "//core/common:cuda_memory_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:error_handling_lib",
        "//core/common:memory_location_lib",
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
        ":replica_memory_coordinator",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:memory_location_lib",
        "//core/common:otel_config_lib",
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
        ":replica_memory_coordinator",
        ":transfer_constants",
        ":transfer_helpers",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store/loader:dvmp_region_sink",
        "//core/store/loader:file_partition_source",
        "//core/store/loader:gpu_memory_sink",
        "//core/store/loader:multi_safetensors_source",
        "//core/store/loader:pump",
        "//core/store/loader:safetensors_source",
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
        ":replica_memory_coordinator",
        ":transfer_constants",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:uma_lease_provider",
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
        ":replica_memory_coordinator",
        ":transfer_helpers",
        ":transfer_service",
        "//core/common:cuda_api_lib",
        "//core/common:device_types_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:memory_location_lib",
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

sc_cc_library(
    name = "replica",
    srcs = ["replica.cc"],
    hdrs = [
        "chunk_directory.h",
        "replica.h",
        "replica_config.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":memory_manager",
        ":memory_state",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:memory_location_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:inline_buffer_loader",
        "//core/store/loader:p2p_loader",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
    ],
)

# Tests
cc_test(
    name = "chunk_state_machine_test",
    srcs = ["chunk_state_machine_test.cc"],
    deps = [
        ":chunk_meta",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_memory_coordinator_test",
    srcs = ["replica_memory_coordinator_test.cc"],
    deps = [
        ":replica_memory_coordinator",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_p2p_registration_test",
    srcs = ["replica_p2p_registration_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "-lcuda",
            "-lrt",
        ],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common",
        "//core/communicator:engine_lib",
        "//core/store:store_engine",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_p2p_transfer_test",
    srcs = ["replica_p2p_transfer_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common",
        "//core/communicator:engine_lib",
        "//core/store:store_engine",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)
