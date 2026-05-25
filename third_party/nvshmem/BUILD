load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvshmem",
    srcs = [
        "nvidia/nvshmem/lib/libnvshmem_host.so.3",
    ],
    hdrs = glob([
        "nvidia/nvshmem/include/**/*.h",
    ]),
    includes = ["nvidia/nvshmem/include/"],
)
