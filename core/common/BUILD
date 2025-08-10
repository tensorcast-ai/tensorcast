load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

cc_library(
    name = "common",
    srcs = glob(
        [
            "**/*.cc",
            "**/*.h",
        ],
        exclude = [
            "cuda_api.h",
            "cuda_real.cc",
            "cuda_fake.cc",
            "memory/distributed_memory_pool.cc",
            "memory/distributed_memory_pool.h",
        ],
    ) + select({
        ":use_fake_cuda": ["cuda_fake.cc"],
        "//conditions:default": ["cuda_real.cc"],
    }),
    hdrs = [
        "cuda_api.h",
        "device_guard.h",
        "error_handling.h",
        "logging_init.h",
        "model_verification.h",
    ],
    defines = select({
        ":use_fake_cuda": ["USE_FAKE_CUDA"],
        "//conditions:default": [],
    }),
    includes = ["."],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@nlohmann_json//:json",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": [
            "@cuda//:cublas",
            "@cuda//:cublasLt",
            "@cuda//:cudart",
            "@cuda//:cufft",
            "@cuda//:cupti",
            "@cuda//:curand",
            "@cuda//:cusparse",
            # "@cuda//:nvToolsExt",
            "@cuda//:nvJitLink",
            "@cudnn",
            "@cuparselt//:cusparselt",
            "@nccl",
        ],
    }),
    alwayslink = True,
)

cc_library(
    name = "distributed_memory_pool",
    srcs = ["memory/distributed_memory_pool.cc"],
    hdrs = ["memory/distributed_memory_pool.h"],
    deps = [
        "//core/common",
        "//core/store/model:chunk_meta",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "device_guard",
    hdrs = [
        "device_guard.h",
    ],
    deps = [
        "//core/common",
    ],
)
