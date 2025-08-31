load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Base interfaces
sc_header_only_library(
    name = "loader",
    hdrs = ["loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/store:loading_spec",
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
    name = "disk_dir_hash",
    srcs = ["disk_dir_hash.cc"],
    hdrs = ["disk_dir_hash.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":source_hash",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "canonical_index",
    srcs = ["canonical_index.cc"],
    hdrs = ["canonical_index.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "safetensors_util",
    srcs = ["safetensors_util.cc"],
    hdrs = ["safetensors_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "safetensors_source",
    srcs = ["safetensors_source.cc"],
    hdrs = ["safetensors_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":safetensors_util",
        ":source",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "multi_safetensors_source",
    srcs = ["multi_safetensors_source.cc"],
    hdrs = ["multi_safetensors_source.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":safetensors_util",
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
        "//core/common:otel_config_lib",
        "//core/common:otel_link_deps_lib",
        "//core/communicator:engine_lib",
        "@opentelemetry-cpp//api:api",
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

# Unified hashing over SeekableSource
sc_cc_library(
    name = "source_hash",
    srcs = ["source_hash.cc"],
    hdrs = ["source_hash.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":file_partition_source",
        ":source",
        "//core/common:artifact_hash_lib",
        "//core/common:cuda_api_lib",
        "@abseil-cpp//absl/types:span",
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

## Removed separate in-memory source targets in favor of local classes in source_hash.cc

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
        ":multi_safetensors_source",
        ":safetensors_source",
        ":safetensors_util",
        "//core/common:artifact_verification_lib",
        "//core/common:error_handling_lib",
        "//core/common:memory_location_lib",
        "//core/store:loading_spec",
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
        "//core/common:memory_location_lib",
        "//core/common:metric_objects_lib",
        "//core/communicator:engine_lib",
        "//core/store:communication_types",
        "//core/store:loading_spec",
        "@abseil-cpp//absl/strings",
    ],
)

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# Minimal loader for memory-only path (InlineBufferSource)
sc_header_only_library(
    name = "inline_buffer_loader",
    hdrs = ["inline_buffer_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":loader",
        "//core/common:memory_location_lib",
        "//core/store:loading_spec",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
    ],
)

# Tests
cc_test(
    name = "disk_mmap_test",
    srcs = ["disk_mmap_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common:pinned_memory_pool_lib",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_cpu_load_test",
    srcs = ["disk_cpu_load_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_cpu_gpu_transfer_test",
    srcs = ["disk_cpu_gpu_transfer_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_loader_streaming_buffer_test",
    srcs = ["disk_loader_streaming_buffer_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":disk_loader",
        ":source",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica:memory_manager",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_error_test",
    srcs = ["disk_error_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "file_partition_source_test",
    srcs = ["file_partition_source_test.cc"],
    deps = [
        ":file_partition_source",
        "//core/checkpoint:aligned_buffer_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "safetensors_source_test",
    srcs = ["safetensors_source_test.cc"],
    deps = [
        ":safetensors_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_safetensors_source_test",
    srcs = ["multi_safetensors_source_test.cc"],
    deps = [
        ":multi_safetensors_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_loader_safetensors_test",
    srcs = ["disk_loader_safetensors_test.cc"],
    deps = [
        ":disk_loader",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_gpu_auto_release_test",
    srcs = ["artifact_gpu_auto_release_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_memory_sink_test",
    srcs = ["gpu_memory_sink_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_verification_test",
    srcs = ["artifact_verification_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_verifier_direct_test",
    srcs = ["artifact_verifier_direct_test.cc"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_multi_gpu_test",
    srcs = ["disk_multi_gpu_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":file_partition_source",
        ":gpu_memory_sink",
        ":pump",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "mux_seekable_source_test",
    srcs = ["mux_seekable_source_test.cc"],
    deps = [
        ":mux_seekable_source",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pump_direct_test",
    srcs = ["pump_direct_test.cc"],
    deps = [
        ":pump",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pump_test",
    srcs = ["pump_test.cc"],
    deps = [
        ":pump",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_streaming_buffer_test",
    srcs = ["artifact_streaming_buffer_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":streaming_buffer_adapter",
        "//core/common:artifact_hash_lib",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "p2p_loader_tcp_test",
    srcs = ["p2p_loader_tcp_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":p2p_loader",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/communicator:engine_lib",
        "//core/store/replica:memory_manager",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)
