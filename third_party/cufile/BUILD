load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cufile",
    srcs = [
        "nvidia/cufile/lib/libcufile.so.0",
        "nvidia/cufile/lib/libcufile_rdma.so.1",
    ],
    hdrs = glob([
        "nvidia/cufile/include/**/*.h",
    ]),
    includes = ["nvidia/cufile/include/"],
)
