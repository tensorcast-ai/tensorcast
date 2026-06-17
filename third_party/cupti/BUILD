load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cupti",
    srcs = glob(
        [
            "nvidia/cuda_cupti/lib/libcupti.so.*",
            "nvidia/cu13/lib/libcupti.so.*",
        ],
        allow_empty = True,
    ) or fail("cupti: no libcupti.so.* found under nvidia/cuda_cupti/lib or nvidia/cu13/lib"),
)
