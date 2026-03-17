load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "plan_proto",
    srcs = ["plan.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_proto",
    ],
)

cpp_proto_library(
    name = "plan_cc",
    protos = [":plan_proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
    ],
)

buf_lint_test(
    name = "plan_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":plan_proto"],
)
