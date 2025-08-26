load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart",
    srcs = [
        "nvidia/cuda_runtime/lib/libcudart.so.12",
    ],
    hdrs = glob([
        "nvidia/cuda_runtime/include/**/*.h",
        "nvidia/cuda_runtime/include/**/*.hpp",
    ]),
    includes = ["nvidia/cuda_runtime/include/"],
    deps = [
        "@triton//:cudart",
    ],
)
