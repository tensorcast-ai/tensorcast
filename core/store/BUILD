load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Type header libraries
cc_library(
    name = "communication_types",
    hdrs = ["communication_types.h"],
    deps = [
        ":device_types",
        "//core/store/model:model_location",
    ],
)

cc_library(
    name = "device_types",
    hdrs = ["device_types.h"],
    deps = [
        "//core/common",
        "//core/store/model:model_location",
        "@abseil-cpp//absl/hash",
    ],
)

cc_library(
    name = "memory_types",
    hdrs = ["memory_types.h"],
    deps = [
        ":concurrent_queue",
        ":concurrent_vector",
        ":device_types",
    ],
)

cc_library(
    name = "concurrent_queue",
    hdrs = ["concurrent_queue.h"],
    deps = [],
)

cc_library(
    name = "concurrent_vector",
    hdrs = ["concurrent_vector.h"],
    deps = [],
)

# Global Store Client library
cc_library(
    name = "global_store_client",
    srcs = ["components/global_store_client.cc"],
    hdrs = ["components/global_store_client.h"],
    deps = [
        ":device_types",
        "//core/store/model:chunk_meta",
        "//core/store/model:model_location",
        "//proto:global_store_grpc_cpp",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

# Single unified store target including all components
cc_library(
    name = "store",
    srcs = glob(["**/*.cc"]),
    hdrs = glob(["**/*.h"]),
    includes = [
        ".",
        "..",
    ],
    deps = [
        "//core/checkpoint",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store/model",
        "//proto:global_store_grpc_cpp",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
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
    alwayslink = True,
)
