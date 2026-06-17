load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "curand",
    srcs = glob(
        [
            "nvidia/curand/lib/libcurand.so.*",
            "nvidia/cu13/lib/libcurand.so.*",
        ],
        allow_empty = True,
    ) or fail("curand: no libcurand.so.* found under nvidia/curand/lib or nvidia/cu13/lib"),
)
