load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

# Core CUDA API abstraction
sc_cc_library(
    name = "cuda_api_lib",
    srcs = [
        "cuda_backend.cc",
        "cuda_backend_fake.cc",
        "cuda_backend_real.cc",
        "cuda_driver_api.cc",
        "dynamic_library.cc",
        "lazy_nvrtc.cc",
    ],
    hdrs = [
        "cuda_api.h",
        "cuda_backend.h",
        "cuda_driver_api.h",
        "dynamic_library.h",
        "lazy_nvrtc.h",
    ],
    linkopts = ["-ldl"],
    visibility = ["//visibility:public"],
    deps = [
        ":error_handling_lib",
        "//third_party/cuda_compat:nv_target",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@cublas",
        "@cublas//:cublasLt",
        "@cudart",
        "@cudnn",
        "@cufft",
        "@cupti",
        "@curand",
        "@cusparse",
        "@cusparselt",
        "@nccl",
        "@nvjitlink//:nvJitLink",
        "@nvrtc//:nvrtc_headers",
    ],
    alwayslink = True,
)

# Device management utilities
sc_header_only_library(
    name = "device_guard_lib",
    hdrs = ["device_guard.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        "@abseil-cpp//absl/log:check",
    ],
)

# CUDA runtime facade (streams/events/guards)
sc_cc_library(
    name = "cuda_runtime_lib",
    srcs = [
        "cuda_event.cc",
        "cuda_runtime.cc",
        "cuda_stream.cc",
    ],
    hdrs = [
        "cuda_event.h",
        "cuda_runtime.h",
        "cuda_stream.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_api_lib",
        ":device_guard_lib",
        ":error_handling_lib",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

# Error handling utilities
sc_header_only_library(
    name = "error_handling_lib",
    hdrs = ["error_handling.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/log:check",
    ],
)

cc_test(
    name = "cuda_api_test",
    srcs = ["cuda_api_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":cuda_api_lib",
        ":cuda_runtime_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "cuda_backend_selection_test",
    srcs = ["cuda_backend_selection_test.cc"],
    deps = [
        ":cuda_api_lib",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)
