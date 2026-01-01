load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# Aligned buffer utilities
sc_cc_library(
    name = "aligned_buffer_lib",
    srcs = ["aligned_buffer.cc"],
    hdrs = ["aligned_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:check",
    ],
)

# Progress bar for streaming operations
sc_header_only_library(
    name = "progress_bar_lib",
    hdrs = ["progress_bar.h"],
    visibility = ["//:__pkg__"],  # Package-private
)

# Base tensor writer interface
sc_cc_library(
    name = "tensor_writer_lib",
    srcs = ["tensor_writer.cc"],
    hdrs = ["tensor_writer.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":aligned_buffer_lib",
    ],
)

# Streaming tensor writer implementation
sc_cc_library(
    name = "streaming_tensor_writer_lib",
    srcs = ["streaming_tensor_writer.cc"],
    hdrs = ["streaming_tensor_writer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_writer_lib",
        "//core/common:async_copy_manager_lib",
        "//core/common:cuda_api_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/synchronization",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "@libtorch",
        ],
    }),
)

# Main checkpoint library
sc_cc_library(
    name = "checkpoint_lib",
    srcs = ["checkpoint.cc"],
    hdrs = ["checkpoint.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":aligned_buffer_lib",
        ":tensor_writer_lib",
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store/materialization/dataplane:safetensors_util",
        "@abseil-cpp//absl/log:check",
        "@libtorch",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "@cublas",
            "@cublas//:cublasLt",
        ],
    }),
)

# Aggregate target for backward compatibility
sc_cc_library(
    name = "checkpoint",
    visibility = ["//visibility:public"],
    deps = [
        ":aligned_buffer_lib",
        ":checkpoint_lib",
        ":streaming_tensor_writer_lib",
        ":tensor_writer_lib",
    ],
    alwayslink = True,
)

# Tests
cc_test(
    name = "duplicate_buffer_test",
    srcs = ["duplicate_buffer_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":checkpoint",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)
