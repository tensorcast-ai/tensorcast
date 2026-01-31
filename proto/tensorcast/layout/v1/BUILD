load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "layout_proto",
    srcs = ["layout.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
)

cpp_proto_library(
    name = "layout_cc",
    protos = [":layout_proto"],
    visibility = ["//visibility:public"],
)

buf_lint_test(
    name = "layout_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":layout_proto"],
)
