load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cusparse",
    srcs = [
        "nvidia/cusparse/lib/libcusparse.so.12",
    ],
    hdrs = glob([
        "nvidia/cusparse/include/**/*.h",
    ]),
    includes = ["nvidia/cusparse/include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
