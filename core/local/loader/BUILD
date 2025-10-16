load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

sc_header_only_library(
    name = "chunk_loader",
    hdrs = ["chunk_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

sc_cc_library(
    name = "disk_chunk_loader",
    srcs = ["disk_chunk_loader.cc"],
    hdrs = ["disk_chunk_loader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":chunk_loader",
        "//core/local/chunk:data_chunk_lib",
    ],
)
