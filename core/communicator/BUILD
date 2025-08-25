load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
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
    name = "envs_lib",
    srcs = ["misc/envs.cc"],
    hdrs = ["misc/envs.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
)

sc_cc_library(
    name = "utils_lib",
    srcs = ["misc/utils.cc"],
    hdrs = ["misc/utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
        ":envs_lib",
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
    srcs = ["misc/ibv_wrap.cc"],
    hdrs = ["misc/ibv_wrap.h"],
    defines = select({
        ":use_fake_cuda": ["USE_FAKE_CUDA"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":common_lib",
    ] + select({
        ":use_fake_cuda": [":ibv_mock_lib"],
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
        ":envs_lib",
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
        ":envs_lib",
        ":epoll_wrap_lib",
        ":map_lib",
        ":protocol_lib",
        ":transport_message_lib",
        ":utils_lib",
        "@abseil-cpp//absl/log:check",
    ],
)

# MTCP is part of the engine layer since it depends on gpu_tcp_stager
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
        ":envs_lib",
        ":ibv_wrap_lib",
        ":net_dev_lib",
        ":request_lib",
        ":utils_lib",
        "//core/common:cuda_api_lib",
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

# MTCP transport (without gpu_tcp_stager which is now in engine subpackage)
sc_cc_library(
    name = "mtcp_transport_lib",
    srcs = ["transport/mtcp_transport.cc"],
    hdrs = ["transport/mtcp_transport.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_transport_lib",
        ":constants_lib",
        ":gpu_tcp_stager_lib",
        ":partition_tensor_lib",
        ":tcp_lib",
        "//core/common:cuda_api_lib",
        "//core/common:device_guard_lib",
        "//core/common:error_handling_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
    ],
)

# GPU TCP Stager
sc_cc_library(
    name = "gpu_tcp_stager_lib",
    srcs = ["engine/gpu_tcp_stager.cc"],
    hdrs = ["engine/gpu_tcp_stager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":base_transport_lib",
        ":constants_lib",
        ":partition_tensor_lib",
        ":tcp_lib",
        "//core/common:cuda_api_lib",
        "//core/common:device_guard_lib",
        "//core/common:error_handling_lib",
        "//core/common:pinned_memory_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
    ],
)

sc_cc_library(
    name = "channel_lib",
    srcs = ["engine/channel.cc"],
    hdrs = ["engine/channel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_tcp_stager_lib",
        ":mtcp_transport_lib",
        ":rdma_lib",
    ],
)

sc_cc_library(
    name = "engine_lib",
    srcs = ["engine/engine.cc"],
    hdrs = ["engine/engine.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":channel_lib",
        ":constants_lib",
        ":envs_lib",
        ":message_lib",
        ":mtcp_transport_lib",
        ":protocol_lib",
        ":rdma_lib",
        ":store_lib",
        ":tcp_lib",
        ":utils_lib",
        "//core/common:cuda_api_lib",
        "//core/common:device_guard_lib",
        "//core/common:error_handling_lib",
        "//core/common:pinned_memory_pool_lib",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": ["@cudart"],
    }),
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
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "tcp_engine_test",
    srcs = ["engine/tcp_engine_test.cc"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_tcp_stager_test",
    srcs = ["engine/gpu_tcp_stager_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":gpu_tcp_stager_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "gpu_tensor_registration_test",
    srcs = ["engine/gpu_tensor_registration_test.cc"],
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
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
    linkopts = select({
        ":use_fake_cuda": [],
        "//conditions:default": ["-lcuda"],
    }),
    tags = ["requires_cuda"],
    deps = [
        ":engine_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@catch2//:catch2_main",
    ],
)
