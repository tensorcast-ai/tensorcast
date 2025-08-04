load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_import(
    name = "nccl_lib",
    shared_library = "lib/libnccl.so.2",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nccl",
    visibility = ["//visibility:public"],
    deps = [
        "nccl_lib",
    ],
)
