load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# ========== Type definitions ==========

# Header-only token and VA range types for direct-write capability
# moved to //core/store/replica:direct_write to avoid subpackage file reference

# Type header libraries
sc_header_only_library(
    name = "communication_types",
    hdrs = ["communication_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        "//core/common:memory_location_lib",
        "//core/communicator:engine_hdrs",
        "@gsl",
    ],
)

sc_header_only_library(
    name = "device_types",
    hdrs = ["device_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:device_types_lib",
        "//core/common:memory_location_lib",
        "@abseil-cpp//absl/hash",
    ],
)

sc_header_only_library(
    name = "memory_types",
    hdrs = ["memory_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
    ],
)

# UMA target is canonical: depend on //core/store/replica:unified_memory_authority

# ========== Core store components ==========

sc_cc_library(
    name = "device_registry",
    srcs = ["device_registry.cc"],
    hdrs = ["device_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "store_engine_options",
    hdrs = ["store_engine_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:const_granularity_lib",
    ],
)

# ========== Loading specifications ==========

alias(
    name = "loading_spec",
    actual = "//core/store/materialization/contracts:loading_spec",
    visibility = ["//visibility:public"],
)

# ========== Components subdirectory ==========

sc_cc_library(
    name = "global_store_client",
    srcs = ["components/global_store_client.cc"],
    hdrs = ["components/global_store_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        "//core/common:memory_location_lib",
        "//core/common:otel_grpc_propagation_lib",
        "//core/communicator:utils_lib",
        "//core/store/replica:chunk_state",
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "device_manager",
    srcs = ["components/device_manager.cc"],
    hdrs = ["components/device_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_api_lib",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "communication_manager",
    srcs = ["components/communication_manager.cc"],
    hdrs = ["components/communication_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":communication_types",
        "//core/communicator:engine_lib",
        "//proto/tensorcast/communicator/v1:communicator_cc",
    ],
)

sc_cc_library(
    name = "replica_registry",
    srcs = ["components/replica_registry.cc"],
    hdrs = ["components/replica_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        ":memory_types",
        "//core/store/replica",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "metrics_collector",
    srcs = ["components/metrics_collector.cc"],
    hdrs = ["components/metrics_collector.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_manager",
        ":memory_types",
        ":replica_registry",
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "worker_identity",
    hdrs = ["components/worker_identity.h"],
    visibility = ["//visibility:public"],
)

sc_cc_library(
    name = "eviction_service",
    srcs = ["components/eviction_service.cc"],
    hdrs = ["components/eviction_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_manager",
        ":metrics_collector",
        ":replica_registry",
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_types",
        "//core/store/replica",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "eviction_service_test",
    srcs = ["components/eviction_service_test.cc"],
    deps = [
        ":eviction_service",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_gpu_verification_race_test",
    srcs = ["multi_gpu_verification_race_test.cc"],
    deps = [
        "//core/store/materialization/dataplane:verification_utils",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

sc_cc_library(
    name = "view_utils",
    srcs = ["view_utils.cc"],
    hdrs = ["view_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/dataplane:view_planner",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

# ========== Main Store Engine ==========

sc_cc_library(
    name = "store_engine",
    srcs = ["store_engine.cc"],
    hdrs = ["store_engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":communication_manager",
        ":communication_types",
        ":device_types",
        ":eviction_service",
        ":global_store_client",
        ":store_engine_options",
        ":view_utils",
        ":worker_identity",
        "//core/checkpoint:checkpoint_lib",
        "//core/common:artifact_hash_lib",
        "//core/common:artifact_identity_lib",
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/common:trace_macros_lib",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/materialization/control:replica_registration_helper",
        "//core/store/materialization/dataplane:canonical_index",
        "//core/store/materialization/dataplane:disk_loader",
        "//core/store/materialization/dataplane:index_reader",
        "//core/store/materialization/dataplane:loader",
        "//core/store/materialization/dataplane:p2p_loader",
        "//core/store/materialization/dataplane:segment_plan_source",
        "//core/store/materialization/dataplane:source_hash",
        "//core/store/materialization/dataplane:verification_utils",
        "//core/store/materialization/dataplane:view_ingest_executor",
        "//core/store/materialization/dataplane:view_plan_source",
        "//core/store/materialization/dataplane:view_planner",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/replica",
        "//core/store/replica:memory_state",
        "//core/store/replica:replica_config",
        "//core/store/runtime:runtime_env",
        "//core/store/runtime/ingestion:ingestion_runtime",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@gsl",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "@cudart",
            "@libtorch//:torch",
        ],
    }),
)

# Tests for store_engine
cc_test(
    name = "store_engine_test",
    srcs = ["store_engine_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":store_engine",
        "//core/common",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "store_engine_view_test",
    srcs = ["store_engine_view_test.cc"],
    deps = [
        ":store_engine",
        "//core/common",
        "//core/store/materialization/dataplane:source_hash",
        "//core/store/materialization/dataplane:view_planner",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
        "@gsl",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "registration_memory_replica_test",
    srcs = ["registration_memory_replica_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":store_engine",
        "//core/common",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "concurrency_test",
    srcs = ["concurrency_test.cc"],
    tags = [
        "store_engine",
        "stress",
    ],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "//core/testing:concurrency_utils",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "edge_test",
    srcs = ["edge_test.cc"],
    tags = ["store_engine"],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "//core/testing:concurrency_utils",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "stress_test",
    srcs = ["stress_test.cc"],
    tags = [
        "manual",
        "store_engine",
        "stress",
    ],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "//core/testing:concurrency_utils",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "multi_gpu_test",
    srcs = ["multi_gpu_test.cc"],
    shard_count = 3,
    tags = [
        "multi_gpu",
        "requires_cuda",
        "store_engine",
    ],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:concurrency_utils",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "store_engine_verification_default_test",
    srcs = ["store_engine_verification_default_test.cc"],
    tags = ["store_engine"],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "store_engine_safetensors_backfill_test",
    srcs = ["store_engine_safetensors_backfill_test.cc"],
    tags = ["store_engine"],
    deps = [
        ":store_engine",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

# Aggregate target removed; use fine-grained deps in dependents.

# Concurrency utilities are provided by //core/testing:concurrency_utils
