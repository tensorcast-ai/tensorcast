load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "publication_proto",
    srcs = ["publication.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
)

cpp_proto_library(
    name = "publication_cc",
    protos = [":publication_proto"],
    visibility = ["//visibility:public"],
)

buf_lint_test(
    name = "publication_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":publication_proto"],
)
