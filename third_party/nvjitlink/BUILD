load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "nvJitLink",
    srcs = glob(
        [
            "nvidia/nvjitlink/lib/libnvJitLink.so.*",
            "nvidia/cu13/lib/libnvJitLink.so.*",
        ],
        allow_empty = True,
    ) or fail("nvJitLink: no libnvJitLink.so.* found under nvidia/nvjitlink/lib or nvidia/cu13/lib"),
)
