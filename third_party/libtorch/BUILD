load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "libtorch",
    deps = [
        ":torch",
    ],
)

cc_library(
    name = "torch",
    srcs = [
        "torch/lib/libgomp.so.1",
        "torch/lib/libtorch.so",
        "torch/lib/libtorch_cpu.so",
        "torch/lib/libtorch_cuda.so",
        "torch/lib/libtorch_global_deps.so",
    ],
    hdrs = glob(
        [
            "torch/include/torch/**/*.h",
            "torch/include/torch/**/*.hpp",
        ],
        exclude = [
            "torch/include/torch/csrc/api/include/**/*.h",
            "torch/include/torch/csrc/api/include/**/*.hpp",
        ],
    ) + glob(
        [
            "torch/include/torch/csrc/api/include/**/*.h",
            "torch/include/torch/csrc/api/include/**/*.hpp",
        ],
        allow_empty = True,
    ),
    includes = [
        "torch/include",
        "torch/include/torch/csrc/api/include/",
    ],
    deps = [
        ":ATen",
        ":c10_cuda",
    ],
)

cc_library(
    name = "c10_cuda",
    srcs = ["torch/lib/libc10_cuda.so"],
    hdrs = glob([
        "torch/include/c10/**/*.h",
    ]),
    strip_include_prefix = "torch/include",
    deps = [
        ":c10",
    ],
)

cc_library(
    name = "c10",
    srcs = ["torch/lib/libc10.so"],
    hdrs = glob([
        "torch/include/c10/**/*.h",
    ]),
    strip_include_prefix = "torch/include",
)

cc_library(
    name = "ATen",
    hdrs = glob([
        "torch/include/ATen/**/*.h",
    ]),
    strip_include_prefix = "torch/include",
)

cc_library(
    name = "caffe2",
    srcs = [
        "torch/lib/libcaffe2_nvrtc.so",
    ],
    hdrs = glob([
        "torch/include/caffe2/**/*.h",
    ]),
    strip_include_prefix = "torch/include",
)
