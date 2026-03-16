load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using mock IBV backend
config_setting(
    name = "use_mock_ibv",
    define_values = {"use_mock_ibv": "true"},
)

# ========== Base definitions ==========

sc_header_only_library(
    name = "constants_lib",
    hdrs = ["base/constants.h"],
    visibility = ["//visibility:public"],
)

# ========== Misc utilities ==========

sc_header_only_library(
    name = "common_lib",
    hdrs = ["misc/common.h"],
    visibility = ["//core/communicator:__subpackages__"],  # Visible to subpackages
)

sc_cc_library(
    name = "utils_lib",
    srcs = ["misc/utils.cc"],
    hdrs = ["misc/utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
    ],
)

sc_cc_library(
    name = "epoll_wrap_lib",
    srcs = ["misc/epoll_wrap.cc"],
    hdrs = ["misc/epoll_wrap.h"],
    includes = ["."],
    visibility = ["//core/communicator:__subpackages__"],
)

sc_header_only_library(
    name = "ibv_mock_lib",
    hdrs = ["misc/ibv_mock.h"],
    visibility = ["//core/communicator:__subpackages__"],
)

sc_cc_library(
    name = "ibv_wrap_lib",
    srcs = [
        "misc/ibv_wrap.cc",
        "misc/mlx5_warp.cc",
    ],
    hdrs = [
        "misc/ibv_wrap.h",
        "misc/mlx5_warp.h",
    ],
    defines = select({
        ":use_mock_ibv": ["USE_MOCK_IBV"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
    ] + select({
        ":use_mock_ibv": [":ibv_mock_lib"],
        "//conditions:default": [],
    }),
)

sc_header_only_library(
    name = "map_lib",
    hdrs = ["misc/map.h"],
    visibility = ["//core/communicator:__subpackages__"],
)

sc_header_only_library(
    name = "queue_lib",
    hdrs = ["misc/queue.h"],
    visibility = ["//core/communicator:__subpackages__"],
    deps = [
        ":common_lib",
    ],
)

sc_cc_library(
    name = "metric_lib",
    srcs = ["misc/metric.cc"],
    hdrs = ["misc/metric.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils_lib",
        "@abseil-cpp//absl/synchronization",
    ],
)

# ========== Transport layer ==========

sc_cc_library(
    name = "base_transport_lib",
    srcs = ["transport/base_transport.cc"],
    hdrs = ["transport/base_transport.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
    ],
)

sc_cc_library(
    name = "transport_message_lib",
    srcs = ["transport/transport_message.cc"],
    hdrs = ["transport/transport_message.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils_lib",
    ],
)

sc_cc_library(
    name = "request_lib",
    srcs = ["transport/request.cc"],
    hdrs = ["transport/request.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":metric_lib",
        ":partition_tensor_lib",
    ],
)

cc_test(
    name = "request_test",
    srcs = ["transport/request_test.cc"],
    deps = [
        ":constants_lib",
        ":partition_tensor_lib",
        ":request_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "mtcp_transport_lane_test",
    srcs = ["transport/mtcp_transport_lane_test.cc"],
    deps = [
        ":mtcp_transport_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_binary(
    name = "cpu_ce_test_binary",
    srcs = ["engine/cpu_ce_test.cc"],
    deps = [
        ":engine_lib",
        ":utils_lib",
        "//core/testing:common",
        "//core/testing:test_helpers",
    ],
)

cc_binary(
    name = "gpu_ce_test_binary",
    srcs = ["engine/gpu_ce_test.cc"],
    deps = [
        ":engine_lib",
        ":utils_lib",
        "//core/cuda:cuda_api_lib",
        "//core/testing:common",
        "//core/testing:test_helpers",
    ],
)

cc_binary(
    name = "communicator_bench_binary",
    srcs = ["bench/communicator_bench.cc"],
    deps = [
        ":engine_lib",
        ":rdma_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

# Combined net_dev and partition_tensor to avoid circular dependency
sc_cc_library(
    name = "partition_tensor_lib",
    srcs = [
        "transport/net_dev.cc",
        "transport/partition_tensor.cc",
    ],
    hdrs = [
        "transport/net_dev.h",
        "transport/partition_tensor.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
        ":ibv_wrap_lib",
        ":metric_lib",
        ":queue_lib",
        ":utils_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
    ],
)

sc_header_only_library(
    name = "net_dev_lib",
    hdrs = ["transport/net_dev.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":partition_tensor_lib",
    ],
)

# TCP transport (context and transport combined)
sc_cc_library(
    name = "tcp_lib",
    srcs = [
        "transport/tcp_context.cc",
        "transport/tcp_transport.cc",
    ],
    hdrs = [
        "transport/tcp_context.h",
        "transport/tcp_transport.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base_transport_lib",
        ":epoll_wrap_lib",
        ":map_lib",
        ":protocol_lib",
        ":transport_message_lib",
        ":utils_lib",
        "@abseil-cpp//absl/log:check",
    ],
)

# MTCP transport (multi-connection TCP)
# We'll define it later with the engine components

# RDMA transport (context and transport combined)
sc_cc_library(
    name = "rdma_lib",
    srcs = [
        "transport/rdma_context.cc",
        "transport/rdma_transport.cc",
    ],
    hdrs = [
        "transport/rdma_context.h",
        "transport/rdma_transport.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":base_transport_lib",
        ":constants_lib",
        ":ibv_wrap_lib",
        ":net_dev_lib",
        ":request_lib",
        ":utils_lib",
        "//core/cuda:cuda_api_lib",
    ],
)

# ========== Engine layer ==========

sc_header_only_library(
    name = "protocol_lib",
    hdrs = ["engine/protocol.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ibv_wrap_lib",
        ":rdma_lib",
    ],
)

sc_cc_library(
    name = "message_lib",
    srcs = ["engine/message.cc"],
    hdrs = ["engine/message.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
        ":protocol_lib",
        ":transport_message_lib",
        ":utils_lib",
    ],
)

sc_cc_library(
    name = "store_lib",
    srcs = ["engine/store.cc"],
    hdrs = ["engine/store.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":map_lib",
        ":partition_tensor_lib",
        "@abseil-cpp//absl/synchronization",
    ],
)

# MTCP transport
sc_cc_library(
    name = "mtcp_transport_lib",
    srcs = ["transport/mtcp_transport.cc"],
    hdrs = ["transport/mtcp_transport.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_transport_lib",
        ":constants_lib",
        ":memory_stager_lib",
        ":partition_tensor_lib",
        ":staging_flow_controller_lib",
        ":tcp_lib",
        "//core/common:async_copy_manager_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:streaming_chunk_guard_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "//core/cuda:error_handling_lib",
        "@opentelemetry-cpp//api:api",
    ],
)

# Unified memory stager interface
sc_header_only_library(
    name = "memory_stager_lib",
    hdrs = ["engine/memory_stager.h"],
    visibility = ["//visibility:public"],
)

sc_cc_library(
    name = "staging_flow_controller_lib",
    srcs = ["engine/staging_flow_controller.cc"],
    hdrs = ["engine/staging_flow_controller.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ibv_wrap_lib",
        ":memory_stager_lib",
        ":partition_tensor_lib",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@gsl",
    ],
)

# Host-pinned GPU stager (header-only)
sc_header_only_library(
    name = "host_pinned_gpu_stager_lib",
    hdrs = ["engine/host_pinned_gpu_stager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_stager_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:streaming_chunk_guard_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/cuda:cuda_api_lib",
    ],
)

# Host-pinned CPU stager implementation
sc_cc_library(
    name = "host_pinned_cpu_stager_lib",
    srcs = ["engine/host_pinned_cpu_stager.cc"],
    hdrs = ["engine/host_pinned_cpu_stager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_stager_lib",
        ":partition_tensor_lib",
        "//core/common:pinned_buffer_pool_lib",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

# GPU VRAM staged-RDMA stager
sc_cc_library(
    name = "gpu_vram_rdma_stager_lib",
    srcs = ["engine/gpu_vram_rdma_stager.cc"],
    hdrs = ["engine/gpu_vram_rdma_stager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":constants_lib",
        ":memory_stager_lib",
        ":partition_tensor_lib",
        "//core/cuda:cuda_api_lib",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
    ],
)

sc_cc_library(
    name = "channel_lib",
    srcs = ["engine/channel.cc"],
    hdrs = ["engine/channel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":mtcp_transport_lib",
        ":rdma_lib",
        ":staging_flow_controller_lib",
    ],
)

sc_cc_library(
    name = "mtcp_transfer_completion_tracker_lib",
    srcs = ["engine/mtcp_transfer_completion_tracker.cc"],
    hdrs = ["engine/mtcp_transfer_completion_tracker.h"],
    visibility = ["//visibility:public"],
)

sc_cc_library(
    name = "engine_lib",
    srcs = ["engine/engine.cc"],
    hdrs = ["engine/engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":channel_lib",
        ":constants_lib",
        ":gpu_vram_rdma_stager_lib",
        ":host_pinned_cpu_stager_lib",
        ":host_pinned_gpu_stager_lib",
        ":message_lib",
        ":mr_cache_lib",
        ":mtcp_transfer_completion_tracker_lib",
        ":mtcp_transport_lib",
        ":protocol_lib",
        ":rdma_lib",
        ":staging_flow_controller_lib",
        ":store_lib",
        ":tcp_lib",
        ":utils_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:system_capabilities_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "//core/cuda:error_handling_lib",
        "//proto/tensorcast/communicator/v1:communicator_cc",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/strings",
        "@cudart",
        "@opentelemetry-cpp//api:api",
    ],
)

# MR cache for pinned buffers per PD
sc_cc_library(
    name = "mr_cache_lib",
    srcs = ["engine/mr_cache.cc"],
    hdrs = ["engine/mr_cache.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ibv_wrap_lib",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
    ],
)

## Removed: communicator_config_lib (no adapter; use protobuf directly)

# YAML/JSON loader for communicator config
sc_cc_library(
    name = "config_io_lib",
    srcs = ["config_io.cc"],
    hdrs = ["config_io.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/communicator/v1:communicator_cc",
        "@nlohmann_json//:json",
        "@protobuf",
        "@yaml-cpp//:yaml-cpp",
    ],
)

# Aggregate target removed; depend on specific libs instead.

# Tests
cc_test(
    name = "rdma_engine_test",
    srcs = ["engine/rdma_engine_test.cc"],
    tags = ["rdma"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
    ],
)

cc_test(
    name = "rdma_vmm_test",
    srcs = ["engine/rdma_vmm_test.cc"],
    tags = [
        "manual",
        "rdma",
        "requires_cuda",
    ],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "rdma_vmm_ub_test",
    srcs = ["engine/rdma_vmm_ub_test.cc"],
    tags = [
        "manual",
        "rdma",
        "requires_cuda",
    ],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "staging_flow_controller_test",
    srcs = ["engine/staging_flow_controller_test.cc"],
    deps = [
        ":constants_lib",
        ":memory_stager_lib",
        ":partition_tensor_lib",
        ":staging_flow_controller_lib",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "mtcp_transfer_completion_tracker_test",
    srcs = ["engine/mtcp_transfer_completion_tracker_test.cc"],
    deps = [
        ":mtcp_transfer_completion_tracker_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "rdma_stage_fn_test",
    srcs = ["engine/rdma_stage_fn_test.cc"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "tcp_engine_test",
    srcs = ["engine/tcp_engine_test.cc"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "unregister_drain_test",
    srcs = ["engine/unregister_drain_test.cc"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "store_test",
    srcs = ["engine/store_test.cc"],
    deps = [
        ":constants_lib",
        ":store_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "config_io_test",
    srcs = ["config_io_test.cc"],
    deps = [
        ":config_io_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_tensor_registration_test",
    srcs = ["engine/gpu_tensor_registration_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "tcp_transfer_test",
    srcs = ["engine/tcp_transfer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "tcp_concurrent_test",
    srcs = ["engine/tcp_concurrent_test.cc"],
    tags = [
        "requires_cuda",
        "stress",
    ],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "tcp_error_handling_test",
    srcs = ["engine/tcp_error_handling_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "stager_mr_normalization_test",
    srcs = ["engine/stager_mr_normalization_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":host_pinned_cpu_stager_lib",
        ":host_pinned_gpu_stager_lib",
        ":memory_stager_lib",
        ":partition_tensor_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_api_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_vram_rdma_stager_test",
    srcs = ["engine/gpu_vram_rdma_stager_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":gpu_vram_rdma_stager_lib",
        ":partition_tensor_lib",
        "//core/cuda:cuda_api_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

sc_header_only_library(
    name = "engine_hdrs",
    hdrs = ["engine/engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":channel_lib",
        ":constants_lib",
        ":host_pinned_cpu_stager_lib",
        ":ibv_wrap_lib",
        ":message_lib",
        ":mr_cache_lib",
        ":rdma_lib",
        ":request_lib",
        ":store_lib",
        ":tcp_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//proto/tensorcast/communicator/v1:communicator_cc",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
    ],
)
