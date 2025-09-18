load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "trace_demo",
    srcs = ["trace_demo.cc"],
    deps = [
        "//core/store:store_engine",
        "@abseil-cpp//absl/time",
    ],
)

cc_binary(
    name = "gpu_hash_benchmark",
    srcs = ["gpu_hash_benchmark.cc"],
    linkopts = select({
        "//core/store/loader:use_fake_cuda": [],
        "//conditions:default": [
            "-L/usr/local/nvidia/lib64",
            "-lcuda",
        ],
    }),
    deps = [
        "//core/common:artifact_hash_lib",
        "//core/common:cuda_api_lib",
        "//core/store/loader:source_hash",
        "@abseil-cpp//absl/cleanup",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)
