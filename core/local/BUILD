load("//core:BUILD.bzl", "sc_cc_library")

sc_cc_library(
    name = "local_meta_lib",
    srcs = [
        "artifact/artifact.cc",
        "chunk/chunk.cc",
        "view/view.cc",
    ],
    hdrs = [
        "artifact/artifact.h",
        "chunk/chunk.h",
        "view/view.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/local:local_data_lib",
        "//core/store:store_engine",
    ],
)

sc_cc_library(
    name = "local_data_lib",
    srcs = [
        "chunk/cpu_data_chunk.cc",
        "chunk/data_chunk.cc",
        "loader/disk_chunk_loader.cc",
    ],
    hdrs = [
        "chunk/chunk.h",
        "chunk/data_chunk.h",
        "loader/chunk_loader.h",
        "loader/disk_chunk_loader.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:store_engine",
    ],
)
