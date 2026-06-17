load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# SONAME unchanged across cu12/cu13 (libcusparse.so.12 in both); only the path differs.
cc_library(
    name = "cusparse",
    srcs = glob(
        [
            "nvidia/cusparse/lib/libcusparse.so.*",
            "nvidia/cu13/lib/libcusparse.so.*",
        ],
        allow_empty = True,
    ) or fail("cusparse: no libcusparse.so.* found under nvidia/cusparse/lib or nvidia/cu13/lib"),
    hdrs = glob(
        [
            "nvidia/cusparse/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ),
    includes = [
        "nvidia/cu13/include/",
        "nvidia/cusparse/include/",
    ],
    linkopts = ["-Wl,-rpath,lib/"],
)
