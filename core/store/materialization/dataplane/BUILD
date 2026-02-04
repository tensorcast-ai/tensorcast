load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Base interfaces
sc_header_only_library(
    name = "loader",
    hdrs = ["contracts/loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/store/materialization/contracts:loading_spec",
    ],
)

sc_header_only_library(
    name = "source",
    hdrs = ["contracts/source.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/replica:direct_write",
    ],
)

sc_header_only_library(
    name = "sink",
    hdrs = ["contracts/sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:async_copy_manager_lib",
        "//core/store/replica:direct_write",
        "@abseil-cpp//absl/types:span",
    ],
)

sc_header_only_library(
    name = "buffer_pool",
    hdrs = ["contracts/buffer_pool.h"],
    visibility = ["//visibility:public"],
)

# Pump orchestration
sc_cc_library(
    name = "pump",
    srcs = ["runtime/pump.cc"],
    hdrs = ["runtime/pump.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_pool",
        ":sink",
        ":source",
        ":streaming_buffer_adapter",
        "//core/store/replica:direct_write",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
        "@folly//folly:executor",
        "@opentelemetry-cpp//api:api",
    ],
)

# Source implementations
sc_cc_library(
    name = "file_partition_source",
    srcs = ["sources/file_partition_source.cc"],
    hdrs = ["sources/file_partition_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "disk_dir_hash",
    srcs = ["metadata/disk_dir_hash.cc"],
    hdrs = ["metadata/disk_dir_hash.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":source_hash",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "canonical_index",
    srcs = ["metadata/canonical_index.cc"],
    hdrs = ["metadata/canonical_index.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "safetensors_util",
    srcs = ["metadata/safetensors_util.cc"],
    hdrs = ["metadata/safetensors_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":canonical_index",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "index_reader",
    srcs = ["metadata/index_reader.cc"],
    hdrs = ["metadata/index_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":canonical_index",
        ":safetensors_util",
        "//core/common:artifact_hash_lib",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "safetensors_source",
    srcs = ["sources/safetensors_source.cc"],
    hdrs = ["sources/safetensors_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":safetensors_util",
        ":source",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "multi_safetensors_source",
    srcs = ["sources/multi_safetensors_source.cc"],
    hdrs = ["sources/multi_safetensors_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":safetensors_util",
        ":source",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "remote_key_source",
    srcs = ["sources/remote_key_source.cc"],
    hdrs = ["sources/remote_key_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/communicator:engine_lib",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "byte_range_map_builder",
    srcs = ["sources/byte_range_map_builder.cc"],
    hdrs = ["sources/byte_range_map_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:byte_range_map",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "byte_range_program",
    srcs = ["sources/byte_range_program.cc"],
    hdrs = ["sources/byte_range_program.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/common:artifact_hash_lib",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:byte_range_map",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "byte_range_mapped_source",
    srcs = ["sources/byte_range_mapped_source.cc"],
    hdrs = ["sources/byte_range_mapped_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":byte_range_program",
        ":source",
        "//core/store/materialization/contracts:byte_range_map",
        "@abseil-cpp//absl/synchronization",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "source_window_scheduler",
    srcs = ["sources/source_window_scheduler.cc"],
    hdrs = ["sources/source_window_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sink",
        ":source",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/store/materialization/contracts:byte_range_map",
        "@abseil-cpp//absl/types:span",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "memory_source",
    srcs = ["sources/memory_source.cc"],
    hdrs = ["sources/memory_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/cuda:cuda_api_lib",
        "@gsl",
    ],
)

sc_cc_library(
    name = "mux_seekable_source",
    srcs = ["sources/mux_seekable_source.cc"],
    hdrs = ["sources/mux_seekable_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":source",
        "//core/cuda:error_handling_lib",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

# Unified hashing over SeekableSource
sc_cc_library(
    name = "source_hash",
    srcs = ["metadata/source_hash.cc"],
    hdrs = ["metadata/source_hash.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":memory_source",
        ":source",
        "//core/common:artifact_hash_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

sc_cc_library(
    name = "verification_utils",
    srcs = ["verification/verification_utils.cc"],
    hdrs = ["verification/verification_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":safetensors_util",
        ":source_hash",
        "//core/common:artifact_hash_lib",
        "//core/common:artifact_verification_lib",
        "//core/common:memory_location_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "view_planner",
    srcs = ["view/view_planner.cc"],
    hdrs = ["view/view_planner.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":canonical_index",
        "//core/store/materialization/contracts:byte_range_map",
        "//core/store/materialization/contracts:view_plan",
        "//core/store/materialization/contracts:view_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "view_plan_source",
    srcs = ["view/view_plan_source.cc"],
    hdrs = ["view/view_plan_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":byte_range_mapped_source",
        ":byte_range_program",
        ":source",
        ":view_planner",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:view_plan",
        "//core/store/materialization/contracts:view_spec",
        "@gsl",
    ],
)

sc_cc_library(
    name = "view_transform_executor",
    srcs = ["view/view_transform_executor.cc"],
    hdrs = ["view/view_transform_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":view_planner",
        "//core/common:memory_location_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:view_plan",
        "@abseil-cpp//absl/strings",
        "@libtorch//:torch",
    ],
)

sc_cc_library(
    name = "view_ingest_executor",
    srcs = ["view/view_ingest_executor.cc"],
    hdrs = ["view/view_ingest_executor.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":view_planner",
        "//core/common:memory_location_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:view_plan",
        "@abseil-cpp//absl/strings",
        "@libtorch//:torch",
    ],
)

# Sink implementations
sc_cc_library(
    name = "gpu_memory_sink",
    srcs = ["sinks/gpu_memory_sink.cc"],
    hdrs = ["sinks/gpu_memory_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sink",
        "//core/common:async_copy_manager_lib",
        "//core/common:cuda_memory_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "@abseil-cpp//absl/time",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "target_layout_gpu_sink",
    srcs = ["sinks/target_layout_gpu_sink.cc"],
    hdrs = ["sinks/target_layout_gpu_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_memory_sink",
        ":sink",
        "@gsl",
    ],
)

## Removed separate in-memory source targets in favor of local classes in source_hash.cc

sc_cc_library(
    name = "cpu_va_sink",
    srcs = ["sinks/cpu_va_sink.cc"],
    hdrs = ["sinks/cpu_va_sink.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":sink",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica:unified_memory_authority",
        "@abseil-cpp//absl/types:span",
    ],
)

# Buffer pool adapter
sc_cc_library(
    name = "streaming_buffer_adapter",
    srcs = ["runtime/streaming_buffer_adapter.cc"],
    hdrs = ["runtime/streaming_buffer_adapter.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":buffer_pool",
        "//core/common:streaming_pinned_buffer_lib",
        "@gsl",
    ],
)

# Main loaders (refactored to use new abstractions)
sc_cc_library(
    name = "disk_loader",
    srcs = ["loaders/disk_loader.cc"],
    hdrs = ["loaders/disk_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":loader",
        ":multi_safetensors_source",
        ":safetensors_source",
        ":safetensors_util",
        "//core/common:artifact_verification_lib",
        "//core/common:memory_location_lib",
        "//core/cuda:error_handling_lib",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "p2p_loader",
    srcs = ["loaders/p2p_loader.cc"],
    hdrs = ["loaders/p2p_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":loader",
        ":mux_seekable_source",
        ":remote_key_source",
        "//core/common:memory_location_lib",
        "//core/communicator:engine_lib",
        "//core/cuda:error_handling_lib",
        "//core/store:communication_types",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/strings",
    ],
)

# Minimal loader for memory-only path (InlineBufferSource)
sc_header_only_library(
    name = "inline_buffer_loader",
    hdrs = ["contracts/inline_buffer_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":loader",
        "//core/common:memory_location_lib",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
    ],
)

# Tests
cc_test(
    name = "disk_mmap_test",
    srcs = ["loaders/tests/disk_mmap_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:pinned_buffer_pool_lib",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_cpu_load_test",
    srcs = ["loaders/tests/disk_cpu_load_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:pinned_buffer_pool_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_cpu_gpu_transfer_test",
    srcs = ["loaders/tests/disk_cpu_gpu_transfer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_loader_streaming_buffer_test",
    srcs = ["loaders/tests/disk_loader_streaming_buffer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":disk_loader",
        ":source",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica:replica_load_controller",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_loader_partition_order_test",
    srcs = ["loaders/tests/disk_loader_partition_order_test.cc"],
    deps = [
        ":disk_loader",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_error_test",
    srcs = ["loaders/tests/disk_error_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:pinned_buffer_pool_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "file_partition_source_test",
    srcs = ["sources/tests/file_partition_source_test.cc"],
    deps = [
        ":file_partition_source",
        "//core/checkpoint:aligned_buffer_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "safetensors_source_test",
    srcs = ["sources/tests/safetensors_source_test.cc"],
    deps = [
        ":safetensors_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_safetensors_source_test",
    srcs = ["sources/tests/multi_safetensors_source_test.cc"],
    deps = [
        ":multi_safetensors_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "byte_range_map_builder_test",
    srcs = ["sources/tests/byte_range_map_builder_test.cc"],
    deps = [
        ":byte_range_map_builder",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "source_window_scheduler_test",
    srcs = ["sources/tests/source_window_scheduler_test.cc"],
    deps = [
        ":source_window_scheduler",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "index_reader_test",
    srcs = ["metadata/tests/index_reader_test.cc"],
    deps = [
        ":index_reader",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "disk_dir_hash_test",
    srcs = ["metadata/tests/disk_dir_hash_test.cc"],
    deps = [
        ":disk_dir_hash",
        ":source_hash",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "verification_utils_test",
    srcs = ["verification/tests/verification_utils_test.cc"],
    deps = [
        ":source_hash",
        ":verification_utils",
        "//core/common:artifact_hash_lib",
        "//core/common:artifact_verification_lib",
        "//core/cuda:cuda_api_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
        "@gsl",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "disk_loader_safetensors_test",
    srcs = ["loaders/tests/disk_loader_safetensors_test.cc"],
    deps = [
        ":disk_loader",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_gpu_auto_release_test",
    srcs = ["runtime/tests/artifact_gpu_auto_release_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "view_planner_test",
    srcs = ["view/tests/view_planner_test.cc"],
    deps = [
        ":canonical_index",
        ":view_planner",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "view_plan_source_test",
    srcs = ["view/tests/view_plan_source_test.cc"],
    deps = [
        ":view_plan_source",
        ":view_planner",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "view_transform_executor_test",
    srcs = ["view/tests/view_transform_executor_test.cc"],
    deps = [
        ":view_planner",
        ":view_transform_executor",
        "//core/common:memory_location_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "view_ingest_executor_test",
    srcs = ["view/tests/view_ingest_executor_test.cc"],
    deps = [
        ":view_ingest_executor",
        ":view_planner",
        "//core/common:memory_location_lib",
        "//core/cuda:cuda_api_lib",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "gpu_memory_sink_test",
    srcs = ["sinks/tests/gpu_memory_sink_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "target_layout_gpu_sink_test",
    srcs = ["sinks/tests/target_layout_gpu_sink_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":target_layout_gpu_sink",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "artifact_verification_test",
    srcs = ["verification/tests/artifact_verification_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_verifier_direct_test",
    srcs = ["verification/tests/artifact_verifier_direct_test.cc"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_multi_gpu_test",
    srcs = ["loaders/tests/disk_multi_gpu_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "mux_seekable_source_test",
    srcs = ["sources/tests/mux_seekable_source_test.cc"],
    deps = [
        ":mux_seekable_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pump_direct_test",
    srcs = ["runtime/tests/pump_direct_test.cc"],
    deps = [
        ":pump",
        "//core/common:async_runtime_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pump_test",
    srcs = ["runtime/tests/pump_test.cc"],
    deps = [
        ":pump",
        "//core/common:async_runtime_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pump_async_test",
    size = "small",
    srcs = [
        "runtime/tests/pump_async_test.cc",
    ],
    deps = [
        ":buffer_pool",
        ":gpu_memory_sink",
        ":pump",
        ":sink",
        ":source",
        ":streaming_buffer_adapter",
        "//core/common:async_copy_manager_lib",
        "//core/common:async_runtime_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "artifact_streaming_buffer_test",
    srcs = ["runtime/tests/artifact_streaming_buffer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":streaming_buffer_adapter",
        "//core/common:artifact_hash_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "p2p_loader_tcp_test",
    srcs = ["loaders/tests/p2p_loader_tcp_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":p2p_loader",
        "//core/common:artifact_verification_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/communicator:engine_lib",
        "//core/store/replica:replica_load_controller",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
