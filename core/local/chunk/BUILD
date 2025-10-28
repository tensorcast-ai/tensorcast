load("//core:BUILD.bzl", "sc_cc_library")

sc_cc_library(
    name = "chunk_lib",
    srcs = [
        "chunk.cc",
        "cpu_data_chunk.cc",
        "data_chunk.cc",
    ],
    hdrs = [
        "chunk.h",
        "data_chunk.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:store_engine",
        "//core/store/replica:chunk_meta",
        "//core/store/replica",
        "@abseil-cpp//absl/log:absl_log",
        # For forward-declared ChunkLoader usage in signature types only (no include here)
        "//core/local/loader:chunk_loader",
    ],
)
