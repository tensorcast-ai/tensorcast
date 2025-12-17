load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cusolver",
    srcs = glob([
        "nvidia/cusolver/lib/libcusolver*.so*",
    ]),
    hdrs = glob([
        "nvidia/cusolver/include/**/*.h",
    ]),
    includes = ["nvidia/cusolver/include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
