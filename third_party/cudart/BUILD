load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# cu12 ships nvidia/cuda_runtime/{lib,include}/...; cu13 ships nvidia/cu13/{lib,include}/...
cc_library(
    name = "cudart",
    srcs = glob(
        [
            "nvidia/cuda_runtime/lib/libcudart.so.*",
            "nvidia/cu13/lib/libcudart.so.*",
        ],
        allow_empty = True,
    ) or fail("cudart: no libcudart.so.* found under nvidia/cuda_runtime/lib or nvidia/cu13/lib; wheel layout unrecognized"),
    hdrs = glob(
        [
            "nvidia/cuda_runtime/include/**/*.h",
            "nvidia/cuda_runtime/include/**/*.hpp",
            "nvidia/cu13/include/**/*.h",
            "nvidia/cu13/include/**/*.hpp",
        ],
        allow_empty = True,
    ),
    includes = [
        "nvidia/cu13/include/",
        "nvidia/cuda_runtime/include/",
    ],
    deps = [
        "@triton//:cudart",
    ],
)
