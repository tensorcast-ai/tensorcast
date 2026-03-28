load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_cc_library(
    name = "ingestion_event_hub",
    srcs = ["ingestion_event_hub.cc"],
    hdrs = ["ingestion_event_hub.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/runtime:ingestion_events",
        "//core/store/runtime/context:runtime_context_events",
        "@abseil-cpp//absl/functional:any_invocable",
    ],
)

sc_header_only_library(
    name = "artifact_truth",
    hdrs = ["artifact_truth.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
    ],
)

sc_header_only_library(
    name = "artifact_lowering_plan",
    hdrs = ["artifact_lowering_plan.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":artifact_truth",
        "//core/common:selection_identity_lib",
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:loader",
        "@abseil-cpp//absl/time",
    ],
)

sc_header_only_library(
    name = "materialization_strategy_types",
    hdrs = ["materialization_strategy_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:byte_range_map",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/contracts:representation_contract",
        "//core/store/materialization/contracts:view_plan",
    ],
)

sc_cc_library(
    name = "materialization_service",
    srcs = ["materialization_service.cc"],
    hdrs = ["materialization_service.h"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/store:device_types",
        "//core/store:replica_registry",
        "//core/store:store_engine_options",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/replica",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

sc_cc_library(
    name = "materialization_facade",
    srcs = ["materialization_facade.cc"],
    hdrs = ["materialization_facade.h"],
    deps = [
        ":artifact_lowering_plan",
        ":ingestion_event_hub",
        ":materialization_service",
        ":materialization_strategy_types",
        "//core/common:artifact_hash_lib",
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:trace_macros_lib",
        "//core/store:device_types",
        "//core/store:endpoint_id",
        "//core/store:replica_registry",
        "//core/store:seal_assembly_result",
        "//core/store:store_engine_options",
        "//core/store/materialization/control:materialization_backend",
        "//core/store/materialization/control:materialize_orchestrator",
        "//core/store/materialization/dataplane:byte_range_map_builder",
        "//core/store/materialization/dataplane:byte_range_mapped_source",
        "//core/store/materialization/dataplane:byte_range_program",
        "//core/store/materialization/dataplane:disk_loader",
        "//core/store/materialization/dataplane:memory_source",
        "//core/store/materialization/dataplane:p2p_loader",
        "//core/store/materialization/dataplane:pump",
        "//core/store/materialization/dataplane:source_window_scheduler",
        "//core/store/materialization/dataplane:streaming_buffer_adapter",
        "//core/store/materialization/dataplane:target_layout_gpu_sink",
        "//core/store/materialization/dataplane:view_plan_source",
        "//core/store/materialization/dataplane:view_transform_executor",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/replica:collective_disk_loader",
        "//core/store/runtime:ingestion_events",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@gsl",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "ingestion_runtime",
    srcs = ["ingestion_runtime.cc"],
    hdrs = ["ingestion_runtime.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":artifact_lowering_plan",
        ":materialization_facade",
        ":materialization_strategy_types",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:byte_range_map",
        "//core/store/runtime/context:runtime_context",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "@gsl",
    ],
)

cc_test(
    name = "ingestion_event_hub_test",
    srcs = ["ingestion_event_hub_test.cc"],
    deps = [
        ":ingestion_event_hub",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "materialization_service_test",
    srcs = ["materialization_service_test.cc"],
    deps = [
        ":materialization_service",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_manager",
        "//core/store:replica_registry",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/replica",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "ingestion_runtime_test",
    srcs = ["ingestion_runtime_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":ingestion_runtime",
        "//core/store:store_engine_options",
        "//core/store/runtime:runtime_env",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/ingestion/testing",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_check",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "materialization_facade_test",
    srcs = ["materialization_facade_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":materialization_facade",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/store:worker_identity",
        "//core/store/replica",
        "//core/store/runtime:runtime_env",
        "//core/store/runtime/context:runtime_context_events",
        "//core/store/runtime/ingestion/testing",
        "//core/store/runtime/metadata:metadata_gateway",
        "//core/store/runtime/replica:replica_runtime",
        "//core/testing:catch_main",
        "//core/testing:common",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
