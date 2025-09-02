load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# ========== Type definitions ==========

# Header-only token and VA range types for direct-write capability
sc_header_only_library(
    name = "direct_write",
    hdrs = ["direct_write.h"],
    visibility = ["//visibility:public"],
)

# Type header libraries
sc_header_only_library(
    name = "communication_types",
    hdrs = ["communication_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        "//core/common:memory_location_lib",
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
)

# ========== Loading specifications ==========

sc_header_only_library(
    name = "loading_spec",
    hdrs = ["loading/loading_spec.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":communication_types",
        ":device_types",
        "//core/store/replica:memory_state",
        "@abseil-cpp//absl/hash",
    ],
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
        "//core/store/replica:chunk_meta",
        "//proto:common_grpc_cpp",
        "//proto:global_store_grpc_cpp",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
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
        ":uma_lease_provider",
        "//core/communicator:engine_lib",
        "//proto:communicator_config_cc_lib",
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
        "//core/common:pinned_memory_pool_lib",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

# ========== Loading orchestration ==========

# materialize_orchestrator is now part of store_engine target to avoid circular dependency

sc_cc_library(
    name = "chunk_aware_loading_strategy",
    srcs = ["loading/chunk_aware_loading_strategy.cc"],
    hdrs = ["loading/chunk_aware_loading_strategy.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":global_store_client",
        ":loading_spec",
        "//core/common:cuda_api_lib",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/replica:chunk_meta",
        "//core/store/replica:memory_manager",
        "//core/store/replica:replica_memory_coordinator",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "replica_registration_helper",
    srcs = ["loading/replica_registration_helper.cc"],
    hdrs = ["loading/replica_registration_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        ":global_store_client",
    ],
)

# ========== Main Store Engine ==========

sc_cc_library(
    name = "store_engine",
    srcs = [
        "loading/materialize_orchestrator.cc",
        "store_engine.cc",
    ],
    hdrs = [
        "loading/materialize_orchestrator.h",
        "store_engine.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":communication_manager",
        ":communication_types",
        ":device_manager",
        ":device_registry",
        ":device_types",
        ":global_store_client",
        ":loading_spec",
        ":metrics_collector",
        ":replica_registration_helper",
        ":replica_registry",
        ":store_engine_options",
        "//core/checkpoint:checkpoint_lib",
        "//core/common:artifact_hash_lib",
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/common:trace_macros_lib",
        "//core/store/loader",
        "//core/store/loader:canonical_index",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/loader:source_hash",
        "//core/store/replica",
        "//core/store/replica:memory_state",
        "//core/store/replica:replica_config",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    name = "store_engine_p2p_loader_test",
    srcs = ["store_engine_p2p_loader_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":store_engine",
        "//core/common",
        "//core/communicator:engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "registration_memory_replica_test",
    srcs = ["registration_memory_replica_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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

sc_cc_library(
    name = "uma_lease_provider",
    srcs = ["components/uma_lease_provider.cc"],
    hdrs = ["components/uma_lease_provider.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        "//core/communicator:dram_stager_lib",
        "//core/store/replica:replica_memory_coordinator",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "uma_lease_provider_test",
    srcs = ["components/uma_lease_provider_test.cc"],
    deps = [
        ":uma_lease_provider",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/store/replica:replica_memory_coordinator",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)
