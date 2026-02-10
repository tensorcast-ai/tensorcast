load("@rules_buf//buf:defs.bzl", "buf_format")

package(default_visibility = ["//visibility:public"])

exports_files(["buf.yaml"])

# Format all .proto files in this package using Buf
buf_format(
    name = "buf_format",
)
