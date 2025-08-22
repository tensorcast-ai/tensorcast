load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "trace_demo",
    srcs = ["trace_demo.cc"],
    deps = [
        "//core/store:store_engine",
        "@abseil-cpp//absl/time",
    ],
)
