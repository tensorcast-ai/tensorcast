load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudnn_headers",
    hdrs = glob(["nvidia/cudnn/include/cudnn*.h"]),
    includes = ["nvidia/cudnn/include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "cudnn_lib",
    shared_library = "nvidia/cudnn/lib/libcudnn.so.9",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    visibility = ["//visibility:public"],
    deps = [
        "cudnn_headers",
        "cudnn_lib",
    ],
)
