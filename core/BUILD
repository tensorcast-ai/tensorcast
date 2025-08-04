load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

cc_library(
    name = "core",
    deps = [
        "//core/checkpoint",
        "//core/communicator",
        "//core/store",
    ],
    alwayslink = True,
)

cc_binary(
    name = "libscstore.so",
    linkshared = True,
    linkstatic = True,
    deps = [
        ":core",
    ],
)
