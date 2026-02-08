load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_header_only_library(
    name = "safe_sys_hdr",
    hdrs = ["common/safe_sys.h"],
    deps = [
        "@abseil-cpp//absl/status",
    ],
)

sc_cc_library(
    name = "status_utils",
    hdrs = ["util/status_utils.h"],
)

sc_cc_library(
    name = "grpc_peer_utils_lib",
    srcs = ["util/grpc_peer_utils.cc"],
    hdrs = ["util/grpc_peer_utils.h"],
)

sc_cc_library(
    name = "path_utils_lib",
    srcs = ["util/path_utils.cc"],
    hdrs = ["util/path_utils.h"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "identity_utils_lib",
    srcs = ["util/identity_utils.cc"],
    hdrs = ["util/identity_utils.h"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "grpc_service_impl",
    srcs = ["service/grpc_service_impl.cc"],
    hdrs = ["service/grpc_service_impl.h"],
    copts = [],
    deps = [
        ":background_scheduler_hdr",
        ":deadline_utils_hdr",
        ":device_resolver_hdr",
        ":grpc_metrics_hdr",
        ":grpc_span_hdr",
        ":handle_lease_registry_lib",
        ":ipc_region_registry_lib",
        ":key_mapping_controller_lib",
        ":lease_controller_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":local_handle_server_lib",
        ":materialization_controller_lib",
        ":path_utils_lib",
        ":persistence_rpc_controller_lib",
        ":ref_tracker_hdr",
        ":registration_controller_lib",
        ":registration_manager_lib",
        ":replica_listing_hdr",
        ":replica_session_controller_lib",
        ":rpc_context_hdr",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        ":shutdown_signal_hdr",
        ":status_controller_hdr",
        ":status_utils",
        ":sweep_tasks_hdr",
        ":transport_controller_lib",
        ":transport_lock_manager_hdr",
        ":types_hdr",
        ":verification_tracker_hdr",
        "//core/common:artifact_hash_lib",
        "//core/common:otel_grpc_propagation_lib",
        "//core/store:global_store_client",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:source_hash",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "daemon_app_lib",
    srcs = ["app/daemon_app.cc"],
    hdrs = ["app/daemon_app.h"],
    deps = [
        ":daemon_kernel_lib",
        ":daemon_options_hdr",
        ":grpc_service_impl",
        ":key_mapping_controller_lib",
        ":lease_controller_lib",
        ":local_handle_server_lib",
        ":persistence_rpc_controller_lib",
        ":replica_session_controller_lib",
        ":worker_lifecycle",
        ":worker_lifecycle_ports_hdr",
        "//core/common:async_runtime_lib",
        "//core/store:global_store_client",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
    ],
)

sc_cc_library(
    name = "daemon_service_harness_lib",
    srcs = ["testing/daemon_service_harness.cc"],
    hdrs = ["testing/daemon_service_harness.h"],
    deps = [
        ":daemon_kernel_lib",
        ":daemon_options_hdr",
        ":grpc_service_impl",
        ":key_mapping_controller_lib",
        ":lease_controller_lib",
        ":local_handle_server_lib",
        ":materialization_controller_lib",
        ":persistence_rpc_controller_lib",
        ":registration_controller_lib",
        ":replica_session_controller_lib",
        ":status_controller_hdr",
        ":transport_controller_lib",
        "//core/common:async_runtime_lib",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time",
    ],
)

sc_cc_library(
    name = "cuda_ipc_spawn_helper_lib",
    srcs = ["testing/cuda_ipc_spawn_helper.cc"],
    hdrs = ["testing/cuda_ipc_spawn_helper.h"],
)

sc_cc_library(
    name = "startup_memory_preflight_lib",
    srcs = ["app/startup_memory_preflight.cc"],
    hdrs = ["app/startup_memory_preflight.h"],
    deps = [
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "startup_memory_preflight_test",
    srcs = ["app/startup_memory_preflight_test.cc"],
    deps = [
        ":startup_memory_preflight_lib",
        "@catch2//:catch2_main",
    ],
)

cc_binary(
    name = "cuda_ipc_helper",
    srcs = ["testing/cuda_ipc_helper.cc"],
    deps = [
        "//core/cuda:cuda_api_lib",
        "//core/cuda:device_guard_lib",
        "@abseil-cpp//absl/status",
    ],
)

sc_cc_library(
    name = "handle_lease_registry_lib",
    srcs = ["state/handle_lease_registry.cc"],
    hdrs = ["state/handle_lease_registry.h"],
    deps = [
        ":session_lifecycle_lib",
        "//core/common:memory_location_lib",
        "//core/store:store_engine",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica:unified_memory_authority",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random:random",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@abseil-cpp//absl/types:span",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "placement_lease_tokens_lib",
    srcs = ["state/placement_lease_tokens.cc"],
    hdrs = ["state/placement_lease_tokens.h"],
    deps = [
        ":session_lifecycle_lib",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random:random",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

sc_cc_library(
    name = "retention_registry_lib",
    srcs = ["state/retention_registry.cc"],
    hdrs = ["state/retention_registry.h"],
    deps = [
        ":session_lifecycle_lib",
        ":store_policy_resolver_lib",
        "//core/common:capability_token_lib",
        "//core/store:device_types",
        "//core/store:stable_dram_cache_policy",
        "//core/store:store_engine",
        "//core/store/materialization/contracts:loading_spec",
        "//proto/tensorcast/common/v1:common_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/random:random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "local_handle_server_lib",
    srcs = ["state/local_handle_server.cc"],
    hdrs = ["state/local_handle_server.h"],
    deps = [
        ":handle_lease_registry_lib",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "store_policy_resolver_lib",
    srcs = ["state/store_policy_resolver.cc"],
    hdrs = ["state/store_policy_resolver.h"],
    deps = [
        "//core/store:stable_dram_cache_policy",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
    ],
)

sc_cc_library(
    name = "persistence_manager_lib",
    srcs = ["state/persistence_manager.cc"],
    hdrs = ["state/persistence_manager.h"],
    deps = [
        ":background_scheduler_hdr",
        ":lip_manager_lib",
        ":store_policy_resolver_lib",
        "//core/common:artifact_hash_lib",
        "//core/common:async_runtime_lib",
        "//core/store:global_store_client",
        "//core/store:store_engine",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica:memory_state",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "//proto/tensorcast/global_store/v1:global_store_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:optional",
        "@abseil-cpp//absl/types:span",
        "@folly//folly/futures:core",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "replica_listing_hdr",
    hdrs = ["service/replica_listing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ref_tracker_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@nlohmann_json//:json",
    ],
)

sc_header_only_library(
    name = "verification_tracker_hdr",
    hdrs = ["state/verification_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:ready_signal_lib",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@folly//folly:executor",
        "@folly//folly/futures:core",
        "@glog",
    ],
)

sc_cc_library(
    name = "registration_manager_lib",
    hdrs = ["state/registration_manager.h"],
    deps = [
        ":store_policy_resolver_lib",
        ":types_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "lip_manager_lib",
    srcs = [
        "state/lip_manager.cc",
        "state/lip_metadata_utils.cc",
    ],
    hdrs = [
        "state/lip_manager.h",
        "state/lip_metadata_utils.h",
    ],
    deps = [
        ":ipc_region_registry_lib",
        ":types_hdr",
        "//core/common:artifact_hash_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/cuda:device_guard_lib",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:byte_range_map_builder",
        "//core/store/materialization/dataplane:canonical_index",
        "//core/store/materialization/dataplane:source_hash",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "target_write_registry_lib",
    srcs = ["state/target_write_registry.cc"],
    hdrs = ["state/target_write_registry.h"],
    deps = [
        ":types_hdr",
        "//proto/tensorcast/common/v1:common_cc",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

sc_cc_library(
    name = "local_disk_import_catalog_lib",
    srcs = ["state/local_disk_import_catalog.cc"],
    hdrs = ["state/local_disk_import_catalog.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

sc_cc_library(
    name = "ipc_region_registry_lib",
    srcs = ["state/ipc_region_registry.cc"],
    hdrs = ["state/ipc_region_registry.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random:random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

cc_test(
    name = "ipc_region_registry_test",
    srcs = ["state/ipc_region_registry_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_binary(
    name = "tensorcast_daemon",
    srcs = ["app/server_main.cc"],
    deps = [
        ":daemon_app_lib",
        ":identity_utils_lib",
        ":startup_memory_preflight_lib",
        "//core/common:async_runtime_lib",
        "//core/common:daemon_config_io_lib",
        "//core/common:logging_init_lib",
        "//core/common:otel_init_lib",
        "//core/common:otel_log_sink_lib",
        "//core/common:trace_manager_lib",
        "//core/communicator:config_io_lib",
        "//core/store:store_engine_options",
        "//proto/tensorcast/config/v1:config_cc",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@gsl",
    ],
)

cc_test(
    name = "lip_metadata_utils_test",
    srcs = ["state/lip_metadata_utils_test.cc"],
    deps = [
        ":lip_manager_lib",
        "@catch2//:catch2_main",
    ],
)

# Header-only alias for the session manager to avoid leaking absl deps to binaries unintentionally.
sc_header_only_library(
    name = "session_manager_hdr",
    hdrs = ["state/replica_session_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:ready_signal_lib",
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@folly//folly/futures:core",
    ],
)

sc_header_only_library(
    name = "transport_lock_manager_hdr",
    hdrs = ["state/transport_lock_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "worker_lifecycle_ports_hdr",
    hdrs = ["ha/worker_lifecycle_ports.h"],
    deps = [
        ":retire_gates_lib",
        ":shutdown_signal_hdr",
        ":worker_identity_store_lib",
        "//core/common:async_runtime_lib",
    ],
)

sc_cc_library(
    name = "worker_lifecycle",
    srcs = ["ha/worker_lifecycle_manager.cc"],
    hdrs = ["ha/worker_lifecycle_manager.h"],
    deps = [
        ":identity_utils_lib",
        ":worker_lifecycle_ports_hdr",
        "//core/communicator:utils_lib",
        "//core/store:store_engine",
        "//core/store/runtime/context:runtime_context_events",
        "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/crc:crc32c",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time:time",
        "@folly//folly/futures:core",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "ref_tracker_hdr",
    hdrs = ["state/ref_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "types_hdr",
    hdrs = ["state/types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ipc_region_registry_lib",
        "//core/common:artifact_identity_lib",
        "//core/cuda:cuda_ipc_lib",
    ],
)

sc_header_only_library(
    name = "background_scheduler_hdr",
    hdrs = ["state/background_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:trace_request_data_lib",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@folly//folly/io/async:request_context",
    ],
)

sc_header_only_library(
    name = "grpc_span_hdr",
    hdrs = ["service/grpc_span.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:otel_grpc_propagation_lib",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "grpc_metrics_hdr",
    hdrs = ["service/grpc_metrics.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:trace_manager_lib",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "sweep_tasks_hdr",
    hdrs = ["state/sweep_tasks.h"],
    deps = [
        ":background_scheduler_hdr",
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        ":transport_lock_manager_hdr",
        ":verification_tracker_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "pid_monitor_lib",
    srcs = ["state/pid_monitor.cc"],
    hdrs = ["state/pid_monitor.h"],
    deps = [
        ":safe_sys_hdr",
    ],
)

sc_cc_library(
    name = "session_lifecycle_lib",
    srcs = ["state/session_lifecycle.cc"],
    hdrs = ["state/session_lifecycle.h"],
    deps = [
        ":lip_manager_lib",
        ":pid_monitor_lib",
        ":ref_tracker_hdr",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/replica:memory_state",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "daemon_options_hdr",
    hdrs = ["state/daemon_options.h"],
    deps = [
        "//core/common:capability_token_lib",
        "@abseil-cpp//absl/time",
    ],
)

sc_header_only_library(
    name = "shutdown_signal_hdr",
    hdrs = ["state/shutdown_signal.h"],
)

sc_cc_library(
    name = "worker_identity_store_lib",
    srcs = ["state/worker_identity_store.cc"],
    hdrs = ["state/worker_identity_store.h"],
    deps = [
        ":persistence_manager_lib",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "retire_gates_lib",
    srcs = ["state/retire_gates.cc"],
    hdrs = ["state/retire_gates.h"],
    deps = [
        ":ref_tracker_hdr",
        ":session_lifecycle_lib",
        ":transport_lock_manager_hdr",
        "//core/store/materialization/contracts:loading_spec",
    ],
)

sc_cc_library(
    name = "daemon_kernel_lib",
    srcs = ["state/daemon_kernel.cc"],
    hdrs = ["state/daemon_kernel.h"],
    deps = [
        ":background_scheduler_hdr",
        ":daemon_options_hdr",
        ":device_resolver_hdr",
        ":handle_lease_registry_lib",
        ":ipc_region_registry_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":local_disk_import_catalog_lib",
        ":persistence_manager_lib",
        ":pid_monitor_lib",
        ":placement_lease_tokens_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":retention_registry_lib",
        ":retire_gates_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        ":sessions_service_hdr",
        ":shutdown_signal_hdr",
        ":sweep_tasks_hdr",
        ":transport_lock_manager_hdr",
        ":verification_tracker_hdr",
        ":worker_identity_store_lib",
        "//core/common:async_runtime_lib",
        "//core/common:capability_token_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time",
    ],
)

# RFC-0016 helpers and controllers
sc_header_only_library(
    name = "rpc_context_hdr",
    hdrs = ["service/rpc_context.h"],
    deps = [
        ":grpc_metrics_hdr",
        ":grpc_span_hdr",
        "//core/common:trace_request_data_lib",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@folly//folly/io/async:request_context",
    ],
)

sc_header_only_library(
    name = "deadline_utils_hdr",
    hdrs = ["util/deadline_utils.h"],
    deps = [],
)

sc_header_only_library(
    name = "device_resolver_hdr",
    hdrs = ["state/device_resolver.h"],
    deps = [
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

sc_header_only_library(
    name = "sessions_service_hdr",
    hdrs = ["state/sessions_service.h"],
    deps = [
        ":background_scheduler_hdr",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        ":verification_tracker_hdr",
        "//core/common:ready_signal_lib",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time:time",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "status_controller_hdr",
    hdrs = [
        "service/controllers/status_assembler.h",
        "service/controllers/status_controller.h",
    ],
    deps = [
        ":ref_tracker_hdr",
        ":replica_listing_hdr",
        ":rpc_context_hdr",
        ":shutdown_signal_hdr",
        ":worker_identity_store_lib",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "lease_controller_lib",
    srcs = ["service/controllers/lease_controller.cc"],
    hdrs = ["service/controllers/lease_controller.h"],
    deps = [
        ":placement_lease_tokens_lib",
        ":retention_registry_lib",
        ":rpc_context_hdr",
        ":session_lifecycle_lib",
        ":shutdown_signal_hdr",
        ":status_utils",
        "//core/common:capability_token_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/time",
    ],
)

sc_cc_library(
    name = "replica_session_controller_lib",
    srcs = ["service/controllers/replica_session_controller.cc"],
    hdrs = ["service/controllers/replica_session_controller.h"],
    deps = [
        ":rpc_context_hdr",
        ":session_lifecycle_lib",
        ":sessions_service_hdr",
        ":status_utils",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@folly//folly/futures:core",
    ],
)

sc_cc_library(
    name = "key_mapping_controller_lib",
    srcs = ["service/controllers/key_mapping_controller.cc"],
    hdrs = ["service/controllers/key_mapping_controller.h"],
    deps = [
        ":rpc_context_hdr",
        ":shutdown_signal_hdr",
        ":status_utils",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
    ],
)

sc_cc_library(
    name = "persistence_rpc_controller_lib",
    srcs = ["service/controllers/persistence_rpc_controller.cc"],
    hdrs = ["service/controllers/persistence_rpc_controller.h"],
    deps = [
        ":persistence_manager_lib",
        ":rpc_context_hdr",
        ":shutdown_signal_hdr",
        ":status_utils",
        ":store_policy_resolver_lib",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
    ],
)

sc_cc_library(
    name = "lip_bridge_lib",
    srcs = ["state/lip_bridge.cc"],
    hdrs = ["state/lip_bridge.h"],
    deps = [
        ":lip_manager_lib",
        # no direct StoreEngine dependency in header; uses loading_spec
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "materialization_controller_lib",
    srcs = ["service/controllers/materialization_controller.cc"],
    hdrs = ["service/controllers/materialization_controller.h"],
    deps = [
        ":daemon_options_hdr",
        ":deadline_utils_hdr",
        ":device_resolver_hdr",
        ":grpc_peer_utils_lib",
        ":handle_lease_registry_lib",
        ":lip_bridge_lib",
        ":local_disk_import_catalog_lib",
        ":path_utils_lib",
        ":ref_tracker_hdr",
        ":rpc_context_hdr",
        ":session_lifecycle_lib",
        ":sessions_service_hdr",
        ":shutdown_signal_hdr",
        ":status_utils",
        ":target_write_registry_lib",
        ":worker_identity_store_lib",
        "//core/common:capability_token_lib",
        "//core/common:selection_identity_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/store:store_engine",
        "//core/store:view_utils",
        "//core/store/materialization/contracts:byte_range_map",
        "//core/store/materialization/dataplane:disk_dir_hash",
        "//core/store/materialization/dataplane:source_hash",
        "//core/store/materialization/dataplane:view_planner",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@opentelemetry-cpp//api:api",
        "@protobuf",
    ],
)

sc_cc_library(
    name = "registration_controller_lib",
    srcs = ["service/controllers/registration_controller.cc"],
    hdrs = ["service/controllers/registration_controller.h"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":rpc_context_hdr",
        ":session_lifecycle_lib",
        ":status_utils",
        ":store_policy_resolver_lib",
        ":worker_identity_store_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/cuda:device_guard_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:byte_range_map_builder",
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "transport_controller_lib",
    srcs = ["service/controllers/transport_controller.cc"],
    hdrs = ["service/controllers/transport_controller.h"],
    deps = [
        ":lip_manager_lib",
        ":rpc_context_hdr",
        ":status_utils",
        ":transport_lock_manager_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "grpc_peer_utils_test",
    srcs = ["util/grpc_peer_utils_test.cc"],
    deps = [
        ":grpc_peer_utils_lib",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "path_utils_test",
    srcs = ["util/path_utils_test.cc"],
    deps = [
        ":path_utils_lib",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_test",
    srcs = ["state/session_lifecycle_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_immediate_unload_test",
    srcs = ["state/session_lifecycle_immediate_unload_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_pid_unwatch_test",
    srcs = ["state/session_lifecycle_pid_unwatch_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_session_guard_test",
    srcs = ["state/session_lifecycle_session_guard_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/common:ready_signal_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pid_monitor_unwatch_integration_test",
    srcs = ["state/pid_monitor_unwatch_integration_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":pid_monitor_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "eviction_task_behavior_test",
    srcs = ["state/eviction_task_behavior_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        ":sweep_tasks_hdr",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_eviction_notify_test",
    srcs = ["state/session_lifecycle_eviction_notify_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_load_test",
    srcs = ["state/session_lifecycle_load_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "transport_lock_manager_test",
    srcs = ["state/transport_lock_manager_test.cc"],
    deps = [
        ":transport_lock_manager_hdr",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "status_utils_test",
    srcs = ["util/status_utils_test.cc"],
    deps = [
        ":status_utils",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_registration_test",
    srcs = ["service/grpc_service_impl_registration_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/log",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "daemon_shutdown_drain_test",
    srcs = ["app/daemon_shutdown_drain_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/common:async_runtime_lib",
        "//core/store:store_engine",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_keepalive_revoke_test",
    srcs = ["service/grpc_service_impl_keepalive_revoke_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_lease_copy_test",
    srcs = ["service/grpc_service_impl_lease_copy_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":cuda_ipc_spawn_helper_lib",
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store/materialization/dataplane:source_hash",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "grpc_service_impl_lip_same_device_denial_test",
    srcs = ["service/grpc_service_impl_lip_same_device_denial_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":cuda_ipc_spawn_helper_lib",
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_lip_ttl_expiry_test",
    srcs = ["service/grpc_service_impl_lip_ttl_expiry_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":cuda_ipc_spawn_helper_lib",
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/cuda:cuda_api_lib",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_lip_piece_assembly_test",
    srcs = ["service/grpc_service_impl_lip_piece_assembly_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":cuda_ipc_spawn_helper_lib",
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/cuda:cuda_api_lib",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_parity_test",
    srcs = ["service/grpc_service_impl_parity_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_publish_replica_key_test",
    srcs = ["service/grpc_service_impl_publish_replica_key_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:testing_global_store_client_stub",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_publish_target_replica_test",
    srcs = ["service/grpc_service_impl_publish_target_replica_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/common:capability_token_lib",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_retire_published_replica_test",
    srcs = ["service/grpc_service_impl_retire_published_replica_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_disk_index_test",
    srcs = ["service/grpc_service_impl_disk_index_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_wait_for_shared_disk_test",
    srcs = ["service/grpc_service_impl_wait_for_shared_disk_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:testing_global_store_client_stub",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_cpu_memfd_e2e_test",
    srcs = ["service/grpc_service_impl_cpu_memfd_e2e_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_cpu_memfd_stable_budget_test",
    srcs = ["service/grpc_service_impl_cpu_memfd_stable_budget_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_cpu_memfd_fd_pressure_test",
    srcs = ["service/grpc_service_impl_cpu_memfd_fd_pressure_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "local_handle_lease_ttl_expiry_test",
    srcs = ["state/local_handle_lease_ttl_expiry_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "local_handle_unknown_token_test",
    srcs = ["state/local_handle_unknown_token_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_replica_status_test",
    srcs = ["service/grpc_service_impl_replica_status_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/common:ready_signal_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_placement_lease_test",
    srcs = ["service/grpc_service_impl_placement_lease_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/common:capability_token_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "retention_registry_test",
    srcs = ["state/retention_registry_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":retention_registry_lib",
        ":session_lifecycle_lib",
        ":session_manager_hdr",
        "//core/common:capability_token_lib",
        "//core/store:device_types",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_no_lease_materialize_test",
    srcs = ["service/grpc_service_impl_no_lease_materialize_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_deregister_purge_shared_disk_test",
    srcs = ["service/grpc_service_impl_deregister_purge_shared_disk_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "replica_session_manager_join_semantics_test",
    srcs = ["state/replica_session_manager_join_semantics_test.cc"],
    deps = [
        ":session_manager_hdr",
        "//core/store:device_types",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "resolve_artifact_from_disk_test",
    srcs = ["service/resolve_artifact_from_disk_test.cc"],
    deps = [
        ":background_scheduler_hdr",
        ":device_resolver_hdr",
        ":ipc_region_registry_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":materialization_controller_lib",
        ":ref_tracker_hdr",
        ":rpc_context_hdr",
        ":session_manager_hdr",
        ":sessions_service_hdr",
        ":shutdown_signal_hdr",
        ":verification_tracker_hdr",
        "//core/common:artifact_hash_lib",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/time:time",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "materialize_into_target_validation_test",
    srcs = ["service/materialize_into_target_validation_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":background_scheduler_hdr",
        ":cuda_ipc_spawn_helper_lib",
        ":device_resolver_hdr",
        ":ipc_region_registry_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":materialization_controller_lib",
        ":ref_tracker_hdr",
        ":rpc_context_hdr",
        ":session_manager_hdr",
        ":sessions_service_hdr",
        ":shutdown_signal_hdr",
        ":verification_tracker_hdr",
        "//core/common:artifact_hash_lib",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/cuda:device_guard_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/time:time",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "materialize_into_mapped_target_test",
    srcs = ["service/materialize_into_mapped_target_test.cc"],
    data = [":cuda_ipc_helper"],
    deps = [
        ":background_scheduler_hdr",
        ":cuda_ipc_spawn_helper_lib",
        ":device_resolver_hdr",
        ":ipc_region_registry_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":materialization_controller_lib",
        ":ref_tracker_hdr",
        ":rpc_context_hdr",
        ":session_manager_hdr",
        ":sessions_service_hdr",
        ":shutdown_signal_hdr",
        ":verification_tracker_hdr",
        "//core/cuda:cuda_api_lib",
        "//core/cuda:cuda_ipc_lib",
        "//core/cuda:device_guard_lib",
        "//core/store:device_registry",
        "//core/store:store_engine",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_test",
    srcs = ["service/grpc_service_impl_status_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_multigpu_test",
    srcs = ["service/grpc_service_impl_status_multigpu_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_mixed_residency_test",
    srcs = ["service/grpc_service_impl_status_mixed_residency_test.cc"],
    deps = [
        ":daemon_service_harness_lib",
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "persistence_manager_test",
    srcs = ["state/persistence_manager_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":persistence_manager_lib",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/store:testing_recording_global_store_client",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "store_policy_resolver_test",
    srcs = ["state/store_policy_resolver_test.cc"],
    deps = [
        ":store_policy_resolver_lib",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "worker_lifecycle_manager_sync_test",
    srcs = ["ha/worker_lifecycle_manager_sync_test.cc"],
    deps = [
        ":daemon_kernel_lib",
        ":daemon_options_hdr",
        ":worker_lifecycle",
        ":worker_lifecycle_ports_hdr",
        "//core/store:global_store_client",
        "//core/store:store_engine",
        "//core/store:store_engine_options",
        "//core/testing:common",
        "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
        "//proto/tensorcast/memory_tier/v1:memory_tier_grpc_cc",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@catch2//:catch2_main",
        "@gsl",
        "@nlohmann_json//:json",
    ],
)
