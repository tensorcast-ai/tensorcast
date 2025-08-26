load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudart",
    # srcs = [
    #     "nvidia/lib/libcudart.so.12",
    # ],
    hdrs = glob([
        "triton/backends/nvidia/include/**/*.h",
        "triton/backends/nvidia/include/**/*.hpp",
    ]),
    includes = ["triton/backends/nvidia/include/"],
)
