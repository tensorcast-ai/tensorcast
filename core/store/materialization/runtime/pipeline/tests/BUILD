load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(default_visibility = ["//visibility:public"])

cc_test(
    name = "p2p_ingestion_test",
    srcs = ["p2p_ingestion_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        "//core/common:artifact_verification_lib",
        "//core/communicator:engine_lib",
        "//core/cuda:cuda_api_lib",
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
        "//core/communicator:engine_lib",
        "//core/cuda:cuda_api_lib",
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

cc_test(
    name = "source_adapter_test",
    srcs = ["source_adapter_test.cc"],
    env = {
        "TENSORCAST_CUDA_BACKEND": "fake",
    },
    deps = [
        "//core/communicator:routing_context_lib",
        "//core/store:communication_manager",
        "//core/store:store_engine_options",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/runtime/context:runtime_context",
        "//core/testing:catch_main",
        "//proto/tensorcast/communicator/v1:communicator_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "verification_stage_policy_test",
    srcs = ["verification_stage_policy_test.cc"],
    deps = [
        "//core/store:store_engine_options",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/testing:catch_main",
        "@catch2//:catch2_main",
    ],
)
