load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvrtc",
    srcs = [
        "nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
    ],
    hdrs = glob([
        "nvidia/cuda_nvrtc/include/**/*.h",
    ]),
    includes = ["nvidia/cuda_nvrtc/include"],
    deps = [
        ":nvrtc_builtins",
    ],
)

cc_library(
    name = "nvrtc_headers",
    hdrs = glob([
        "nvidia/cuda_nvrtc/include/**/*.h",
    ]),
    includes = ["nvidia/cuda_nvrtc/include"],
)

cc_library(
    name = "nvrtc_builtins",
    srcs = [
        "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.8",
    ],
    hdrs = glob([
        "nvidia/cuda_nvrtc/include/**/*.h",
    ]),
    includes = ["nvidia/cuda_nvrtc/include"],
)
