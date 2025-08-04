load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

cc_library(
    name = "cudart",
    srcs = select({
        ":windows": [
            "lib/x64/cudart.lib",
        ],
        "//conditions:default": [
            "lib64/libcudart.so.12",
        ],
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
        "include/**/*",
    ]),
    includes = ["include/"],
)

cc_library(
    name = "nvToolsExt",
    srcs = select({
        ":windows": [
            "bin/nvToolsExt64_1.dll",
        ],
        "//conditions:default": [
            "lib64/libnvToolsExt.so.1",
        ],
    }),
)

cc_library(
    name = "cuda",
    srcs = select({
        ":windows": [
            "bin/*.dll",
        ],
        "//conditions:default": glob([
            "lib64/**/lib*.so",
        ]),
    }),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
        "include/**/*",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublas",
    srcs = [
        "lib64/libcublas.so.12",
    ],
    hdrs = glob([
        "include/**/*cublas*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
        "include/**/*",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cupti",
    srcs = [
        "extras/CUPTI/lib64/libcupti.so.12",
    ],
)

cc_library(
    name = "cusparse",
    srcs = [
        "lib64/libcusparse.so.12",
    ],
)

cc_library(
    name = "curand",
    srcs = [
        "lib64/libcurand.so.10",
    ],
)

cc_library(
    name = "cufft",
    srcs = [
        "lib64/libcufft.so.11",
    ],
)

cc_library(
    name = "cublasLt",
    srcs = [
        "lib64/libcublasLt.so.12",
    ],
)

cc_library(
    name = "nvJitLink",
    srcs = [
        "lib64/libnvJitLink.so.12",
    ],
)

cc_library(
    name = "nvrtc",
    srcs = [
        "lib64/libnvrtc.so.12",
    ],
)
