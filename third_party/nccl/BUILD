load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "nccl_lib",
    shared_library = "nvidia/nccl/lib/libnccl.so.2",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nccl",
    hdrs = glob([
        "nvidia/nccl/include/**/*.h",
    ]),
    includes = ["nvidia/nccl/include/"],
    linkopts = ["-Wl,-rpath,lib/"],
    visibility = ["//visibility:public"],
    deps = [
        "nccl_lib",
    ],
)
