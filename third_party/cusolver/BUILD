load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cusolver",
    srcs = glob(
        [
            "nvidia/cusolver/lib/libcusolver*.so*",
            "nvidia/cu13/lib/libcusolver*.so*",
        ],
        allow_empty = True,
    ) or fail("cusolver: no libcusolver*.so* found under nvidia/cusolver/lib or nvidia/cu13/lib"),
    hdrs = glob(
        [
            "nvidia/cusolver/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ),
    includes = [
        "nvidia/cu13/include/",
        "nvidia/cusolver/include/",
    ],
    linkopts = ["-Wl,-rpath,lib/"],
)
