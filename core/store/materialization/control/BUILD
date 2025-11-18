load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_header_only_library(
    name = "materialization_backend",
    hdrs = ["materialization_backend.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ],
)

sc_cc_library(
    name = "replica_registration_helper",
    srcs = ["replica_registration_helper.cc"],
    hdrs = ["replica_registration_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:memory_location_lib",
        "//core/store:device_types",
        "//core/store:global_store_client",
        "@abseil-cpp//absl/log:absl_log",
        "@gsl",
    ],
)

sc_cc_library(
    name = "materialize_orchestrator",
    srcs = ["materialize_orchestrator.cc"],
    hdrs = ["materialize_orchestrator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":materialization_backend",
        "//core/common:memory_location_lib",
        "//core/store:device_types",
        "//core/store:global_store_client",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

sc_cc_library(
    name = "materialization_service",
    srcs = ["materialization_service.cc"],
    hdrs = ["materialization_service.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":materialize_orchestrator",
        ":replica_registration_helper",
        "//core/common:memory_location_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:device_types",
        "//core/store:replica_registry",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/contracts:materialization_request",
        "//core/store/replica",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

sc_cc_library(
    name = "materialization_coordinator",
    srcs = ["materialization_coordinator.cc"],
    hdrs = ["materialization_coordinator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":materialization_backend",
        ":materialization_service",
        ":materialize_orchestrator",
        ":replica_registration_helper",
        "//core/common:memory_location_lib",
        "//core/store:component_catalog",
        "//core/store:device_types",
        "//core/store:global_store_publisher",
        "//core/store:replica_service",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/runtime/pipeline:ingestion_pipeline",
        "@abseil-cpp//absl/strings",
        "@gsl",
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
