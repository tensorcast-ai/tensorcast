load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# cu12 ships nvidia/cuda_nvrtc/{lib,include}/...; cu13 ships nvidia/cu13/{lib,include}/...
cc_library(
    name = "nvrtc",
    srcs = glob(
        [
            "nvidia/cuda_nvrtc/lib/libnvrtc.so.*",
            "nvidia/cu13/lib/libnvrtc.so.*",
        ],
        allow_empty = True,
    ) or fail("nvrtc: no libnvrtc.so.* found under nvidia/cuda_nvrtc/lib or nvidia/cu13/lib; wheel layout unrecognized"),
    hdrs = glob(
        [
            "nvidia/cuda_nvrtc/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ) or fail("nvrtc: no headers found under nvidia/cuda_nvrtc/include or nvidia/cu13/include"),
    includes = [
        "nvidia/cu13/include",
        "nvidia/cuda_nvrtc/include",
    ],
    deps = [
        ":nvrtc_builtins",
    ],
)

cc_library(
    name = "nvrtc_headers",
    hdrs = glob(
        [
            "nvidia/cuda_nvrtc/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ) or fail("nvrtc_headers: no headers found under nvidia/cuda_nvrtc/include or nvidia/cu13/include"),
    includes = [
        "nvidia/cu13/include",
        "nvidia/cuda_nvrtc/include",
    ],
)

cc_library(
    name = "nvrtc_builtins",
    srcs = glob(
        [
            "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.*",
            "nvidia/cu13/lib/libnvrtc-builtins.so.*",
        ],
        allow_empty = True,
    ) or fail("nvrtc_builtins: no libnvrtc-builtins.so.* found under nvidia/cuda_nvrtc/lib or nvidia/cu13/lib"),
    hdrs = glob(
        [
            "nvidia/cuda_nvrtc/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ),
    includes = [
        "nvidia/cu13/include",
        "nvidia/cuda_nvrtc/include",
    ],
)
