load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_grpc_library", "cpp_proto_library")

proto_library(
    name = "global_store_proto_lib",
    srcs = ["global_store.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_proto",
        "//proto/tensorcast/layout/v1:layout_proto",
        "//proto/tensorcast/operation/v1:operation_proto",
        "@protobuf//:duration_proto",
        "@protobuf//:timestamp_proto",
        "@protobuf//:wrappers_proto",
    ],
)

cpp_proto_library(
    name = "global_store_cc",
    protos = [":global_store_proto_lib"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/layout/v1:layout_cc",
        "//proto/tensorcast/operation/v1:operation_cc",
    ],
)

cpp_grpc_library(
    name = "global_store_grpc_cc",
    protos = [":global_store_proto_lib"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/layout/v1:layout_cc",
        "//proto/tensorcast/operation/v1:operation_cc",
    ],
)

buf_lint_test(
    name = "global_store_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":global_store_proto_lib"],
)
