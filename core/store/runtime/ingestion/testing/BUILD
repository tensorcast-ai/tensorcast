load("//core:BUILD.bzl", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_header_only_library(
    name = "testing",
    testonly = True,
    hdrs = [
        "fake_ingestion_pipeline.h",
        "scoped_ingestion_runtime_test_harness.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/runtime:ingestion_events",
        "//core/store/runtime:runtime_env",
        "//core/store/runtime/ingestion:ingestion_runtime",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status:status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
    ],
)
