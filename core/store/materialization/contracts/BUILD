load("@rules_cc//cc:defs.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_header_only_library(
    name = "view_spec",
    hdrs = ["view/view_spec.h"],
    visibility = ["//visibility:public"],
)

sc_header_only_library(
    name = "view_plan",
    hdrs = ["view/view_plan.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":view_spec",
    ],
)

sc_header_only_library(
    name = "view_id",
    hdrs = ["view/view_id.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":view_plan",
        ":view_spec",
    ],
)

sc_header_only_library(
    name = "loading_spec",
    hdrs = ["loading_spec.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":view_id",
        "//core/common:ready_signal_lib",
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store/replica:memory_state",
        "@abseil-cpp//absl/hash",
        "@folly//folly/futures:core",
        "@glog",
    ],
)

sc_cc_library(
    name = "materialization_request",
    srcs = ["materialization_request.cc"],
    hdrs = ["materialization_request.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":loading_spec",
        "//core/common:artifact_identity_lib",
        "//core/common:memory_location_lib",
        "//core/store:device_manager",
        "//core/store:device_registry",
        "//core/store:device_types",
        "@abseil-cpp//absl/strings",
    ],
)

sc_header_only_library(
    name = "registry",
    hdrs = ["registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":loading_spec",
        ":materialization_request",
        "//core/store:device_types",
    ],
)

sc_header_only_library(
    name = "registry_fakes",
    hdrs = ["tests/registry_fakes.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":registry",
    ],
)

cc_test(
    name = "materialization_request_test",
    srcs = ["materialization_request_test.cc"],
    deps = [
        ":materialization_request",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)
