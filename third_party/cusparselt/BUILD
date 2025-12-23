load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cusparselt",
    srcs = [
        "nvidia/cusparselt/lib/libcusparseLt.so.0",
    ],
    hdrs = glob([
        "nvidia/cusparselt/include/**/*.h",
    ]),
    includes = ["nvidia/cusparselt/include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)
