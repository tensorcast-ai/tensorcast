load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cublas",
    srcs = [
        "nvidia/cublas/lib/libcublas.so.12",
    ],
    hdrs = glob([
        "nvidia/cublas/include/**/*.h",
    ]),
    includes = ["nvidia/cublas/include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublasLt",
    srcs = [
        "nvidia/cublas/lib/libcublasLt.so.12",
    ],
)
