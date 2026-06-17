load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# NVTX renamed: cu12 ships libnvToolsExt.so.1; cu13 ships libnvtx3interop.so.1.
# Bazel target name stays "nvToolsExt" for callsite stability.
cc_library(
    name = "nvToolsExt",
    srcs = glob(
        [
            "nvidia/nvtx/lib/libnvToolsExt.so.*",
            "nvidia/cu13/lib/libnvtx3interop.so.*",
        ],
        allow_empty = True,
    ) or fail("nvToolsExt: neither libnvToolsExt.so.* (cu12) nor libnvtx3interop.so.* (cu13) found"),
)
