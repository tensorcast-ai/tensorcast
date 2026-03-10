load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")

proto_library(
    name = "plan_proto",
    srcs = ["plan.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_proto",
        "//proto/tensorcast/daemon/v2:daemon_proto_lib",
    ],
)

buf_lint_test(
    name = "plan_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":plan_proto"],
)
