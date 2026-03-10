load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "common_proto",
    srcs = [
        "capability_token.proto",
        "common.proto",
    ],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = ["@protobuf//:timestamp_proto"],
)

buf_lint_test(
    name = "common_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":common_proto"],
)

cpp_proto_library(
    name = "common_cc",
    protos = [":common_proto"],
    visibility = ["//visibility:public"],
)
