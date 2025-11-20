load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "p2p_ingestion_test",
    srcs = ["p2p_ingestion_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/communicator:engine_lib",
        "//core/store:store_engine_options",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_log",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "p2p_verification_fail_test",
    srcs = ["p2p_verification_fail_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/common:cuda_api_lib",
        "//core/communicator:engine_lib",
        "//core/store:store_engine_options",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)
