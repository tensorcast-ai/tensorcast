load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# SONAME bumped: cu12 -> libcufft.so.11; cu13 -> libcufft.so.12.
cc_library(
    name = "cufft",
    srcs = glob(
        [
            "nvidia/cufft/lib/libcufft.so.*",
            "nvidia/cu13/lib/libcufft.so.*",
        ],
        allow_empty = True,
    ) or fail("cufft: no libcufft.so.* found under nvidia/cufft/lib or nvidia/cu13/lib"),
)
