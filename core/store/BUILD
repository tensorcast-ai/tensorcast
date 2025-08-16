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
        "//core/store/model:model_location",
    ],
)

sc_header_only_library(
    name = "device_types",
    hdrs = ["device_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:device_types_lib",
        "//core/store/model:model_location",
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
    name = "checkpoint_store_options",
    hdrs = ["checkpoint_store_options.h"],
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
        "//core/store/model:memory_state",
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
        "//core/store/model:chunk_meta",
        "//core/store/model:model_location",
        "//proto:global_store_grpc_cpp",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

sc_cc_library(
    name = "device_manager",
    srcs = ["components/device_manager.cc"],
    hdrs = ["components/device_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_api_lib",
        "//core/common:metric_objects_lib",
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
    ],
)

sc_cc_library(
    name = "model_registry",
    srcs = ["components/model_registry.cc"],
    hdrs = ["components/model_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":device_types",
        ":memory_types",
        "//core/store/model",
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
        ":model_registry",
        "//core/common:metric_objects_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
    ],
)

# ========== Loading orchestration ==========

# prepare_orchestrator is now part of checkpoint_store target to avoid circular dependency

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
        "//core/store/model:chunk_meta",
        "//core/store/model:memory_manager",
        "//core/store/model:model_memory_coordinator",
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

# ========== Main checkpoint store ==========

sc_cc_library(
    name = "checkpoint_store",
    srcs = [
        "checkpoint_store.cc",
        "loading/prepare_orchestrator.cc",
    ],
    hdrs = [
        "checkpoint_store.h",
        "loading/prepare_orchestrator.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_store_options",
        ":communication_manager",
        ":communication_types",
        ":device_manager",
        ":device_registry",
        ":device_types",
        ":global_store_client",
        ":loading_spec",
        ":metrics_collector",
        ":model_registry",
        ":replica_registration_helper",
        "//core/checkpoint:checkpoint_lib",
        "//core/common:cuda_api_lib",
        "//core/common:distributed_virtual_memory_pool_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/common:trace_macros_lib",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/model",
        "//core/store/model:memory_state",
        "//core/store/model:model_config",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "@cuda//:cudart",
            "@libtorch//:torch",
        ],
    }),
)

# Tests for checkpoint_store
cc_test(
    name = "checkpoint_store_test",
    srcs = ["checkpoint_store_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":checkpoint_store",
        "//core/common",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "checkpoint_store_p2p_loader_test",
    srcs = ["checkpoint_store_p2p_loader_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":checkpoint_store",
        "//core/common",
        "//core/communicator",
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
        ":checkpoint_store",
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
        "checkpoint_store",
        "stress",
    ],
    deps = [
        ":checkpoint_store",
        ":concurrency_utils",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "edge_test",
    srcs = ["edge_test.cc"],
    tags = ["checkpoint_store"],
    deps = [
        ":checkpoint_store",
        ":concurrency_utils",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "stress_test",
    srcs = ["stress_test.cc"],
    tags = [
        "checkpoint_store",
        "manual",
        "stress",
    ],
    deps = [
        ":checkpoint_store",
        ":concurrency_utils",
        "//core/testing:catch_main",
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
        "checkpoint_store",
        "multi_gpu",
        "requires_cuda",
    ],
    deps = [
        ":checkpoint_store",
        ":concurrency_utils",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

# Aggregate target for backward compatibility
sc_cc_library(
    name = "store",
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_store",
        ":checkpoint_store_options",
        ":chunk_aware_loading_strategy",
        ":communication_manager",
        ":communication_types",
        ":device_manager",
        ":device_registry",
        ":device_types",
        ":direct_write",
        ":global_store_client",
        ":loading_spec",
        ":memory_types",
        ":metrics_collector",
        ":model_registry",
        ":replica_registration_helper",
        "//core/store/loader",
        "//core/store/model",
    ],
    alwayslink = True,
)

# Concurrency utilities header for tests
sc_header_only_library(
    name = "concurrency_utils",
    hdrs = ["concurrency_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//core/testing:common",
        "@abseil-cpp//absl/synchronization",
    ],
)
