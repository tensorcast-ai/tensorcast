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
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:virtual_address_space_lib",
        "//core/store:loading_spec",
    ],
)

# Direct write grant types (moved from //core/store)
sc_header_only_library(
    name = "direct_write",
    hdrs = ["types/direct_write_grant.h"],
    visibility = ["//visibility:public"],
)

## Unified Memory Authority (canonical UMA target)
sc_cc_library(
    name = "unified_memory_authority",
    srcs = ["unified_memory_authority.cc"],
    hdrs = ["unified_memory_authority.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":direct_write",
        "//core/common:cuda_memory_lib",
        "//core/common:error_handling_lib",
        "//core/common:memory_location_lib",
        "//core/common:virtual_address_space_lib",
        "//core/store:device_registry",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@gsl",
        "@opentelemetry-cpp//api:api",
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
        ":unified_memory_authority",
        "//core/common:async_copy_manager_lib",
        "//core/common:cuda_api_lib",
        "//core/common:memory_location_lib",
        "//core/common:streaming_chunk_guard_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/common:virtual_address_space_lib",
        "//core/store:device_types",
        "@opentelemetry-cpp//api:api",
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
        ":transfer_constants",
        ":transfer_helpers",
        ":unified_memory_authority",
        "//core/common:cuda_api_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:virtual_address_space_lib",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store/loader:cpu_va_sink",
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
    name = "memory_export_registry",
    srcs = ["memory_export_registry.cc"],
    hdrs = ["memory_export_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":unified_memory_authority",
        ":transfer_constants",
        "//core/common:cuda_api_lib",
        "//core/common:virtual_address_space_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        # UMA lease provider bridge removed; UMA keepalive is held within ExportRegistration
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

# Replica orchestrator (renamed from memory_manager)
sc_cc_library(
    name = "replica_load_controller",
    srcs = ["replica_load_controller.cc"],
    hdrs = ["replica_load_controller.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_meta",
        ":direct_write",
        ":memory_export_registry",
        ":memory_state",
        ":transfer_helpers",
        ":transfer_service",
        ":unified_memory_authority",
        "//core/common:async_copy_manager_lib",
        "//core/common:cuda_api_lib",
        "//core/common:device_types_lib",
        "//core/common:memory_location_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/common:virtual_address_space_lib",
        "//core/communicator:engine_lib",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader:cpu_va_sink",
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
        ":memory_state",
        ":replica_load_controller",
        "//core/common:cuda_api_lib",
        "//core/common:memory_location_lib",
        "//core/common:virtual_address_space_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:inline_buffer_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/loader:view_plan_source",
        "//core/store/loader:view_transform_executor",
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
    name = "unified_memory_authority_plan_commit_test",
    srcs = ["unified_memory_authority_plan_commit_test.cc"],
    deps = [
        ":unified_memory_authority",
        "//core/common:virtual_address_space_lib",
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
        "@gsl",
    ],
)

cc_test(
    name = "replica_p2p_transfer_test",
    srcs = ["replica_p2p_transfer_test.cc"],
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

# UMA unit test
cc_test(
    name = "unified_memory_authority_test",
    srcs = ["unified_memory_authority_test.cc"],
    deps = [
        ":unified_memory_authority",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:virtual_address_space_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)
