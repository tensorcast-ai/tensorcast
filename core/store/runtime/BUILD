load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

sc_cc_library(
    name = "component_catalog",
    srcs = ["component_catalog.cc"],
    hdrs = ["component_catalog.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:communication_manager",
        "//core/store:device_manager",
        "//core/store:global_store_client",
        "//core/store:metrics_collector",
        "//core/store:replica_registry",
        "//core/store:store_engine_options",
        "//core/store:worker_identity",
        "//core/store/materialization/common:view_hash_utils",
    ],
)

sc_cc_library(
    name = "replica_runtime",
    srcs = ["replica_runtime.cc"],
    hdrs = [
        "replica_info.h",
        "replica_runtime.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":component_catalog",
        ":ingestion_events",
        ":runtime_event_hub",
        "//core/common:cuda_api_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:communication_manager",
        "//core/store:communication_types",
        "//core/store:device_manager",
        "//core/store:eviction_service",
        "//core/store:metrics_collector",
        "//core/store:replica_registry",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:memory_state",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)

sc_header_only_library(
    name = "ingestion_events",
    hdrs = ["ingestion_events.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
    ],
)

sc_cc_library(
    name = "global_metadata_gateway",
    srcs = ["global_metadata_gateway.cc"],
    hdrs = ["global_metadata_gateway.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":component_catalog",
        ":ingestion_events",
        ":replica_runtime",
        ":runtime_event_hub",
        "//core/store:global_store_client",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/control:replica_registration_helper",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)

sc_cc_library(
    name = "runtime_event_hub",
    srcs = ["runtime_event_hub.cc"],
    hdrs = ["runtime_event_hub.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ingestion_events",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "runtime_env",
    srcs = ["runtime_env.cc"],
    hdrs = ["runtime_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":component_catalog",
        ":runtime_event_hub",
        "//core/store:store_engine_options",
        "//core/store:worker_identity",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "artifact_ingress_manager",
    srcs = ["artifact_ingress_manager.cc"],
    hdrs = ["artifact_ingress_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":global_metadata_gateway",
        ":replica_runtime",
        ":runtime_env",
        ":runtime_event_hub",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:communication_manager",
        "//core/store:device_manager",
        "//core/store:metrics_collector",
        "//core/store:registration_facade",
        "//core/store:replica_registry",
        "//core/store:store_engine_options",
        "//core/store/materialization/control:materialization_coordinator",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "//core/store/replica",
        "@gsl",
    ],
)

cc_test(
    name = "runtime_services_test",
    srcs = ["runtime_services_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":component_catalog",
        ":replica_runtime",
        ":runtime_event_hub",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_check",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "global_metadata_gateway_test",
    srcs = ["global_metadata_gateway_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":component_catalog",
        ":global_metadata_gateway",
        ":replica_runtime",
        ":runtime_event_hub",
        "//core/store:store_engine_options",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica",
        "//core/store/replica:replica_config",
        "//core/testing:catch_main",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/status:status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "artifact_ingress_manager_test",
    srcs = ["artifact_ingress_manager_test.cc"],
    tags = ["requires_cuda"],
    deps = [
        ":artifact_ingress_manager",
        ":global_metadata_gateway",
        ":replica_runtime",
        ":runtime_env",
        ":runtime_event_hub",
        "//core/store:store_engine_options",
        "//core/testing:catch_main",
        "//core/testing:common",
        "//core/testing:test_helpers",
        "@abseil-cpp//absl/log:absl_check",
        "@catch2//:catch2_main",
    ],
)
