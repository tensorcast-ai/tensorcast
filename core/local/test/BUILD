load("@rules_cc//cc:defs.bzl", "cc_test")

cc_test(
    name = "data_chunk_test",
    srcs = ["data_chunk_test.cc"],
    deps = [
        "//core/local:local_data_lib",
        "//core/local:local_meta_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "disk_chunk_loader_test",
    srcs = ["disk_chunk_loader_test.cc"],
    deps = [
        "//core/local:local_data_lib",
        "//core/local:local_meta_lib",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@catch2//:catch2_main",
    ],
)
