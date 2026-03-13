load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

sc_header_only_library(
    name = "memory_state",
    hdrs = ["memory_state.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "chunk_state",
    hdrs = ["chunk_state.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "chunk_directory",
    hdrs = ["chunk_directory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_state",
    ],
)

sc_header_only_library(
    name = "replica_config",
    hdrs = ["replica_config.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:async_runtime_lib",
        "//core/common:device_types_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
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
        ":chunk_state",
        ":direct_write",
        "//core/common:const_granularity_lib",
        "//core/common:cuda_memory_lib",
        "//core/common:memory_location_lib",
        "//core/common:system_capabilities_lib",
        "//core/cuda:error_handling_lib",
        "//core/store:device_registry",
        "//core/store:device_types",
        "//core/store:memory_tier_budget",
        "//core/store/materialization/contracts:loading_spec",
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
        ":chunk_state",
        ":unified_memory_authority",
        "//core/common:async_copy_manager_lib",
        "//core/common:memory_location_lib",
        "//core/common:streaming_chunk_guard_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/cuda:cuda_api_lib",
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
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:cpu_va_sink",
        "//core/store/materialization/dataplane:file_partition_source",
        "//core/store/materialization/dataplane:gpu_memory_sink",
        "//core/store/materialization/dataplane:multi_safetensors_source",
        "//core/store/materialization/dataplane:pump",
        "//core/store/materialization/dataplane:safetensors_source",
        "//core/store/materialization/dataplane:streaming_buffer_adapter",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@folly//folly:executor",
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
        "//core/cuda:cuda_api_lib",
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
    name = "collective_disk_loader",
    srcs = ["collective_disk_loader.cc"],
    hdrs = ["collective_disk_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_memory_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:error_handling_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:disk_artifact_context",
        "//core/store/materialization/dataplane:multi_safetensors_source",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@nccl",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "replica_load_controller",
    srcs = ["replica_load_controller.cc"],
    hdrs = ["replica_load_controller.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_state",
        ":collective_disk_loader",
        ":direct_write",
        ":memory_export_registry",
        ":memory_state",
        ":transfer_helpers",
        ":transfer_service",
        ":unified_memory_authority",
        "//core/common:async_copy_manager_lib",
        "//core/common:device_types_lib",
        "//core/common:memory_location_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/communicator:engine_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store:device_types",
        "//core/store:memory_types",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:cpu_va_sink",
        "//core/store/materialization/dataplane:gpu_memory_sink",
        "//core/store/materialization/dataplane:pump",
        "//core/store/materialization/dataplane:streaming_buffer_adapter",
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
        ":chunk_state",
        ":memory_state",
        ":replica_load_controller",
        "//core/common:memory_location_lib",
        "//core/communicator:engine_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store:communication_types",
        "//core/store:device_registry",
        "//core/store:device_types",
        "//core/store:memory_types",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:byte_range_map_builder",
        "//core/store/materialization/dataplane:byte_range_mapped_source",
        "//core/store/materialization/dataplane:byte_range_program",
        "//core/store/materialization/dataplane:disk_loader",
        "//core/store/materialization/dataplane:inline_buffer_loader",
        "//core/store/materialization/dataplane:loader",
        "//core/store/materialization/dataplane:p2p_loader",
        "//core/store/materialization/dataplane:view_plan_source",
        "//core/store/materialization/dataplane:view_transform_executor",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

# Tests

cc_test(
    name = "unified_memory_authority_plan_commit_test",
    srcs = ["unified_memory_authority_plan_commit_test.cc"],
    deps = [
        ":unified_memory_authority",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_load_controller_device_key_test",
    srcs = ["replica_load_controller_device_key_test.cc"],
    deps = [
        ":replica_load_controller",
        "//core/common:const_granularity_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_types",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_p2p_registration_test",
    srcs = ["replica_p2p_registration_test.cc"],
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
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)
