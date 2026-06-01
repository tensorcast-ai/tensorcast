load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# cu12 ships nvidia/cublas/{lib,include}/...; cu13 ships nvidia/cu13/{lib,include}/...
# Each http_archive only extracts one wheel, so exactly one path set is populated.
# allow_empty=True is required because the non-matching layout's glob is empty by design.
# `or fail(...)` aborts loading if BOTH layouts are missing — the genuine error case.

cc_library(
    name = "cublas",
    srcs = glob(
        [
            "nvidia/cublas/lib/libcublas.so.*",
            "nvidia/cu13/lib/libcublas.so.*",
        ],
        allow_empty = True,
    ) or fail("cublas: no libcublas.so.* found under nvidia/cublas/lib or nvidia/cu13/lib; wheel layout unrecognized"),
    hdrs = glob(
        [
            "nvidia/cublas/include/**/*.h",
            "nvidia/cu13/include/**/*.h",
        ],
        allow_empty = True,
    ),
    includes = [
        "nvidia/cu13/include/",
        "nvidia/cublas/include/",
    ],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublasLt",
    srcs = glob(
        [
            "nvidia/cublas/lib/libcublasLt.so.*",
            "nvidia/cu13/lib/libcublasLt.so.*",
        ],
        allow_empty = True,
    ) or fail("cublasLt: no libcublasLt.so.* found under nvidia/cublas/lib or nvidia/cu13/lib"),
)
