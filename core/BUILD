load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# Aggregate library that includes all core components
cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        "//core/checkpoint",
        "//core/common",
        "//core/communicator",
        "//core/store",
    ],
    alwayslink = True,
)

# Main shared library target
cc_binary(
    name = "libscstore.so",
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":core",
    ],
)
