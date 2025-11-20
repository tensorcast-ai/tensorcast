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
