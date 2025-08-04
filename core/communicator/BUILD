load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

config_setting(
    name = "use_pre_cxx11_abi",
    values = {
        "define": "abi=pre_cxx11_abi",
    },
)

# Config for using fake CUDA backend
config_setting(
    name = "use_fake_cuda",
    define_values = {"use_fake_cuda": "true"},
)

cc_library(
    name = "communicator",
    srcs = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    hdrs = ["engine/engine.h"],
    includes = ["."],
    deps = [
        "//core/common",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ] + select({
        ":use_fake_cuda": [],
        "//conditions:default": ["@cuda//:cudart"],
    }),
    alwayslink = True,
)
