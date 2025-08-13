
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <cstdint>
#include <vector>
#include "scstore/csrc/logging.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/logging_init.h"
#include "core/common/metrics/metrics_export.h"
#include "core/store/checkpoint_store.h"
#include "core/store/checkpoint_store_options.h"
#include "core/store/communication_types.h"
#include "core/store/components/communication_manager.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_location.h"

namespace py = pybind11;

// NOLINTBEGIN(google-build-using-namespace,fuchsia-statically-constructed-objects,misc-const-correctness,misc-use-anonymous-namespace,)
using namespace stepcast::store;
using stepcast::DeviceType; // Bring the top-level DeviceType enum into scope

// Helper function to convert time_point to Python timestamp
static double time_point_to_timestamp(const std::chrono::time_point<std::chrono::system_clock>& tp) {
  auto duration = tp.time_since_epoch();
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  return static_cast<double>(seconds.count());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Initialize logging only once across all modules
  ensure_logging_initialized();

#ifdef USE_FAKE_CUDA
  py::module_::import("warnings")
      .attr("warn")("CUDA not detected, running with FakeCuda backend. Only logical correctness is guaranteed.");
#endif
  // Bind ModelLocation enum
  py::enum_<ModelLocation>(m, "ModelLocation")
      .value("NONE", ModelLocation::NONE)
      .value("DISK", ModelLocation::DISK)
      .value("CPU", ModelLocation::PAGEABLE_CPU)
      .value("GPU", ModelLocation::GPU)
      .value("REMOTE", ModelLocation::REMOTE)
      .export_values();

  // Bind MemoryState enum
  py::enum_<MemoryState>(m, "MemoryState")
      .value("UNINITIALIZED", MemoryState::UNINITIALIZED)
      .value("UNALLOCATED", MemoryState::UNALLOCATED)
      .value("ALLOCATED", MemoryState::ALLOCATED)
      .value("LOADING", MemoryState::LOADING)
      .value("LOADED", MemoryState::LOADED)
      .value("FAILED", MemoryState::FAILED)
      .export_values();

  // Bind CommRegistrationInfo struct
  // Note: void* pointers are cast to uintptr_t for Python representation
  py::class_<CommRegistrationInfo>(m, "CommRegistrationInfo")
      .def(py::init<>())
      .def_readwrite("model_size", &CommRegistrationInfo::model_size)
      .def_readwrite("location", &CommRegistrationInfo::location)
      .def_readwrite("device_id", &CommRegistrationInfo::device_id)
      .def_readwrite("comm_dev_type", &CommRegistrationInfo::comm_dev_type)
      .def_property_readonly(
          "buffer_addresses",
          [](const CommRegistrationInfo& cri) {
            std::vector<uintptr_t> addresses;
            addresses.reserve(cri.buffer_addresses.size());
            for (const auto& addr : cri.buffer_addresses) {
              addresses.push_back(reinterpret_cast<uintptr_t>(addr));
            }
            return addresses;
          })
      .def_readwrite("buffer_sizes", &CommRegistrationInfo::buffer_sizes)
      .def_readwrite("remote_memory_keys", &CommRegistrationInfo::remote_memory_keys)
      .def("__repr__", [](const CommRegistrationInfo& cri) {
        std::string repr = "<CommRegistrationInfo model_size=" + std::to_string(cri.model_size) +
            ", location=" + std::to_string(static_cast<int>(cri.location)) +
            ", device_id=" + std::to_string(cri.device_id) + ", buffer_addresses=[";
        bool first = true;
        for (const auto& addr_ptr : cri.buffer_addresses) {
          auto addr = reinterpret_cast<uintptr_t>(addr_ptr);
          if (!first) {
            repr += ", ";
          }
          repr += std::to_string(addr);
          first = false;
        }
        repr += "], buffer_sizes=" + std::to_string(cri.buffer_sizes.size()) +
            ", remote_memory_keys=" + std::to_string(cri.remote_memory_keys.size()) + ">";
        return repr;
      });

  // Bind ModelInfo struct
  py::class_<CheckpointStore::ModelInfo>(m, "ModelInfo")
      .def(py::init<>())
      .def_readwrite("model_id", &CheckpointStore::ModelInfo::model_id)
      .def_readwrite("size_bytes", &CheckpointStore::ModelInfo::size_bytes)
      .def_readwrite("cpu_state", &CheckpointStore::ModelInfo::cpu_state)
      .def_readwrite("gpu_state", &CheckpointStore::ModelInfo::gpu_state)
      .def_readwrite("gpu_device_id", &CheckpointStore::ModelInfo::gpu_device_id)
      .def_readwrite("gpu_device_uuid", &CheckpointStore::ModelInfo::gpu_device_uuid)
      .def_readwrite("is_registered_for_comm", &CheckpointStore::ModelInfo::is_registered_for_comm)
      .def_property_readonly(
          "last_access_timestamp",
          [](const CheckpointStore::ModelInfo& info) { return time_point_to_timestamp(info.last_access_time); })
      .def_property_readonly(
          "load_timestamp",
          [](const CheckpointStore::ModelInfo& info) { return time_point_to_timestamp(info.load_time); })
      .def("__repr__", [](const CheckpointStore::ModelInfo& info) {
        std::string repr = "<ModelInfo model_id='" + info.model_id +
            "', size_bytes=" + std::to_string(info.size_bytes) +
            ", cpu_state=" + std::to_string(static_cast<int>(info.cpu_state)) +
            ", gpu_state=" + std::to_string(static_cast<int>(info.gpu_state));
        if (info.gpu_device_id >= 0) {
          repr += ", gpu_device_id=" + std::to_string(info.gpu_device_id);
        }
        if (!info.gpu_device_uuid.empty()) {
          repr += ", gpu_device_uuid='" + info.gpu_device_uuid + "'";
        }
        repr += ", is_registered_for_comm=" + std::string(info.is_registered_for_comm ? "True" : "False") + ">";
        return repr;
      });

  // Bind DeviceType enum
  py::enum_<DeviceType>(m, "DeviceType")
      .value("CPU", DeviceType::CPU)
      .value("GPU", DeviceType::GPU)
      .value("REMOTE", DeviceType::REMOTE)
      .value("DISK", DeviceType::DISK)
      .value("NONE", DeviceType::NONE)
      .export_values();

  // Bind PrepareMode enum
  py::enum_<CheckpointStore::PrepareMode>(m, "PrepareMode")
      .value("AUTO", CheckpointStore::PrepareMode::AUTO)
      .value("COPY_ONLY", CheckpointStore::PrepareMode::COPY_ONLY)
      .value("LOAD_ONLY", CheckpointStore::PrepareMode::LOAD_ONLY)
      .export_values();

  // Bind DeviceKey struct
  py::class_<DeviceKey>(m, "DeviceKey")
      .def(py::init<>())
      .def_readwrite("type", &DeviceKey::type)
      .def_readwrite("ordinal", &DeviceKey::ordinal)
      .def_readwrite("uuid", &DeviceKey::uuid)
      .def("__repr__", [](const DeviceKey& dk) {
        return "<DeviceKey type=" + std::to_string(static_cast<int>(dk.type)) +
            ", ordinal=" + std::to_string(dk.ordinal) + ", uuid='" + dk.uuid + "'>";
      });

  // NEW: Bind InstanceKey struct (multi-device identifier)
  py::class_<InstanceKey>(m, "InstanceKey")
      .def(py::init<>())
      .def_readwrite("model_id", &InstanceKey::model_id)
      .def_readwrite("device", &InstanceKey::device)
      .def_readwrite("replica", &InstanceKey::replica)
      .def("__repr__", [](const InstanceKey& ik) {
        return "<InstanceKey model_id='" + ik.model_id + "', device=" + ik.device.to_string() +
            ", replica=" + std::to_string(ik.replica) + ">";
      });

  // Minimal ModelHandle Python wrapper
  py::class_<ModelHandle>(m, "ModelHandle")
      .def(
          "wait_ready",
          [](ModelHandle& mh, int timeout_ms) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = mh.wait_ready(std::chrono::milliseconds(timeout_ms));
            }
            if (!st.ok()) {
              PyErr_SetString(PyExc_RuntimeError, st.ToString().c_str());
              throw py::error_already_set();
            }
          })
      .def_property_readonly(
          "gpu_ptr", [](const ModelHandle& mh) { return reinterpret_cast<uint64_t>(mh.gpu_base_ptr); })
      .def_property_readonly(
          "ipc_handle_bytes",
          [](const ModelHandle& mh) {
            return py::bytes(mh.cuda_ipc_handle.bytes.data(), mh.cuda_ipc_handle.bytes.size());
          })
      .def_property_readonly("instance_key", [](const ModelHandle& mh) {
        const auto& k = mh.instance_key;
        return std::string("InstanceKey{") + k.model_id + ", " + k.device.to_string() +
            ", replica=" + std::to_string(k.replica) + "}";
      });

  auto ckpt_cls = py::class_<CheckpointStore>(m, "CheckpointStore");
  ckpt_cls
      .def(
          "prepare",
          [](CheckpointStore& cs,
             const std::string& model_id,
             const py::object& target_device_obj,
             CheckpointStore::PrepareMode mode,
             const py::kwargs& kwargs) {
            // Build DeviceKey from python input
            DeviceKey dev_key;
            if (py::isinstance<DeviceKey>(target_device_obj)) {
              dev_key = target_device_obj.cast<DeviceKey>();
            } else if (py::isinstance<py::str>(target_device_obj)) {
              auto spec = target_device_obj.cast<std::string>();
              if (spec == "cpu") {
                dev_key.type = DeviceType::CPU;
                dev_key.ordinal = -1;
              } else if (spec.starts_with("gpu")) {
                dev_key.type = DeviceType::GPU;
                int ordinal = 0;
                size_t colon = spec.find(":");
                if (colon != std::string::npos) {
                  ordinal = std::stoi(spec.substr(colon + 1));
                }
                dev_key.ordinal = ordinal;
              } else {
                PyErr_SetString(PyExc_ValueError, "Unsupported device spec string");
                throw py::error_already_set();
              }
            } else if (py::isinstance<py::int_>(target_device_obj)) {
              dev_key.type = DeviceType::GPU;
              dev_key.ordinal = target_device_obj.cast<int>();
            } else {
              PyErr_SetString(PyExc_TypeError, "target_device must be DeviceKey, str, or int");
              throw py::error_already_set();
            }

            LoadingHints hints;
            if (kwargs.contains("pinned_timeout_ms") && !kwargs["pinned_timeout_ms"].is_none()) {
              int t = kwargs["pinned_timeout_ms"].cast<int>();
              if (t > 0) {
                hints.pinned_timeout = std::chrono::milliseconds(t);
              }
            }

            absl::StatusOr<ModelHandle> h_or;
            {
              py::gil_scoped_release release;
              h_or = cs.prepare(model_id, dev_key, mode, hints);
            }
            if (!h_or.ok()) {
              PyErr_SetString(PyExc_RuntimeError, h_or.status().ToString().c_str());
              throw py::error_already_set();
            }
            return h_or.value();
          },
          py::arg("model_id"),
          py::arg("target_device") = std::string("gpu:0"),
          py::arg("mode") = CheckpointStore::PrepareMode::AUTO,
          "Prepare a model instance on the specified device and return a ModelHandle.")
      .def("clear_mem", &CheckpointStore::clear_mem, "Clear all allocated memory.")
      .def("get_mem_pool_size", &CheckpointStore::get_mem_pool_size, "Get the memory pool size.")
      .def("get_chunk_size", &CheckpointStore::get_chunk_size, "Get the chunk size.")
      .def(
          "get_available_memory",
          &CheckpointStore::get_available_memory,
          "Get available memory in the pinned memory pool.")
      .def(
          "get_loaded_devices",
          [](CheckpointStore& cs, const std::string& model_id) {
            py::gil_scoped_release release;
            return cs.get_loaded_devices(model_id);
          },
          py::arg("model_id"),
          "Return a list of DeviceKey where the given model is loaded.")
      .def(
          "list_device_models",
          [](CheckpointStore& cs, const DeviceKey& device) {
            py::gil_scoped_release release;
            return cs.list_device_models(device);
          },
          py::arg("device"),
          "Return a list of InstanceKey for models resident on the given device.")
      .def(
          "wait_instance_ready",
          [](CheckpointStore& cs, const InstanceKey& key) {
            py::gil_scoped_release release;
            return cs.wait_instance_ready(key);
          },
          py::arg("instance_key"),
          "Block until the instance becomes ready. Returns 0 on success.")
      .def(
          "unload_instance",
          [](CheckpointStore& cs, const InstanceKey& key) {
            py::gil_scoped_release release;
            return cs.unload_instance(key);
          },
          py::arg("instance_key"),
          "Unload the specified model instance from memory.")
      .def(
          "get_instance_state",
          [](CheckpointStore& cs, const InstanceKey& key, DeviceType mem_type) {
            py::gil_scoped_release release;
            return cs.get_instance_state(key, mem_type);
          },
          py::arg("instance_key"),
          py::arg("memory_type"),
          "Get the MemoryState of the specified instance and memory type.")
      .def(
          "get_instance_gpu_ptr",
          [](CheckpointStore& cs, const InstanceKey& key) {
            absl::StatusOr<uint64_t> ptr_or;
            {
              py::gil_scoped_release release;
              ptr_or = cs.get_instance_gpu_ptr(key);
            }
            if (!ptr_or.ok()) {
              PyErr_SetString(PyExc_RuntimeError, ptr_or.status().ToString().c_str());
              throw py::error_already_set();
            }
            return ptr_or.value();
          },
          py::arg("instance_key"),
          "Return the base GPU address for the given instance.")
      .def(
          "enable_remote_instance_access",
          [](CheckpointStore& cs, const InstanceKey& key, ModelLocation loc) {
            absl::StatusOr<CommRegistrationInfo> info_or;
            {
              py::gil_scoped_release release;
              info_or = cs.enable_remote_instance_access(key, loc);
            }
            if (!info_or.ok()) {
              PyErr_SetString(PyExc_RuntimeError, info_or.status().ToString().c_str());
              throw py::error_already_set();
            }
            return info_or.value();
          },
          py::arg("instance_key"),
          py::arg("location"),
          "Enable remote memory access for the given instance and return registration info.")
      .def(
          "disable_remote_instance_access",
          [](CheckpointStore& cs, const InstanceKey& key, ModelLocation loc) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.disable_remote_instance_access(key, loc);
            }
            if (!st.ok()) {
              PyErr_SetString(PyExc_RuntimeError, st.ToString().c_str());
              throw py::error_already_set();
            }
            return true;
          },
          py::arg("instance_key"),
          py::arg("location"),
          "Disable remote memory access for the given instance.")
      .def(
          "lock_chunks",
          [](CheckpointStore& cs, const InstanceKey& key, const std::vector<uint32_t>& chunk_indices) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.lock_chunks(key, absl::MakeSpan(chunk_indices));
            }
            if (!st.ok()) {
              PyErr_SetString(PyExc_RuntimeError, st.ToString().c_str());
              throw py::error_already_set();
            }
            return 0; // Return 0 on success
          },
          py::arg("instance_key"),
          py::arg("chunk_indices"),
          "Lock chunks for H2D or P2P transfer to prevent concurrent eviction.")
      .def(
          "unlock_chunks",
          [](CheckpointStore& cs, const InstanceKey& key, const std::vector<uint32_t>& chunk_indices, bool copied_gpu) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.unlock_chunks(key, absl::MakeSpan(chunk_indices), copied_gpu);
            }
            if (!st.ok()) {
              PyErr_SetString(PyExc_RuntimeError, st.ToString().c_str());
              throw py::error_already_set();
            }
            return 0; // Return 0 on success
          },
          py::arg("instance_key"),
          py::arg("chunk_indices"),
          py::arg("copied_gpu"),
          "Unlock chunks after H2D or P2P transfer completion.")
      .def("__repr__", [](const CheckpointStore& /*cs*/) { return "<CheckpointStore>"; })
      .def(
          "get_all_models_info",
          &CheckpointStore::get_all_models_info,
          "Get detailed information about all loaded models.")
      .def(
          "get_gpu_memory_stats",
          [](CheckpointStore& /*cs*/) {
            // Query GPU memory stats via Python torch.cuda API.
            // Returns a list where each element is a (total, free) tuple for a device.
            namespace py = pybind11;
            py::list result;
            try {
              py::module_ torch = py::module_::import("torch");
              if (!torch.attr("cuda").attr("is_available")().cast<bool>()) {
                // No CUDA available
                return result;
              }
              int device_count = torch.attr("cuda").attr("device_count")().cast<int>();
              for (int device_id = 0; device_id < device_count; ++device_id) {
                // torch.cuda.mem_get_info returns (free, total)
                py::tuple mem_info = torch.attr("cuda").attr("mem_get_info")(device_id);
                if (py::len(mem_info) != 2) {
                  // Unexpected format; skip
                  continue;
                }
                auto free_bytes = mem_info[0].cast<size_t>();
                auto total_bytes = mem_info[1].cast<size_t>();
                // Convert to (total, free) to match Python expectation
                result.append(py::make_tuple(total_bytes, free_bytes));
              }
            } catch (const py::error_already_set& e) {
              // Torch might not be available; return empty list
              (void)e; // suppress unused variable warning
            }
            return result;
          },
          "Return a list of (total, free) GPU memory stats for each CUDA device.");

  // Add global metrics function
  m.def(
      "get_global_metrics_text",
      []() {
        const std::string text = stepcast::metrics::get_global_metrics_text();
        return py::bytes(text);
      },
      "Get the global metrics snapshot in OpenMetrics text format");

  // ------------------------------------------------------------------
  // Phase-2: factory function accepting a Python dict and constructing a
  // CheckpointStore via the new CheckpointStoreOptions structure.  This API
  // is opt-in so that existing callers continue to work without changes.
  // ------------------------------------------------------------------
  m.def(
      "create_checkpoint_store",
      [](py::dict cfg) {
        CheckpointStoreOptions opts;

        auto get_or = [&cfg](const char* key, auto default_val) {
          if (cfg.contains(key) && !cfg[key].is_none()) {
            return cfg[key].cast<decltype(default_val)>();
          }
          return default_val;
        };

        opts.storage_path = get_or("storage_path", opts.storage_path);
        opts.memory_pool_size = get_or("memory_pool_size", opts.memory_pool_size);
        opts.num_thread = get_or("num_thread", opts.num_thread);
        opts.chunk_size = get_or("chunk_size", opts.chunk_size);
        opts.pinned_memory_timeout = std::chrono::milliseconds(
            get_or("pinned_memory_timeout_ms", static_cast<int>(opts.pinned_memory_timeout.count())));
        opts.p2p_port = get_or("p2p_port", opts.p2p_port);
        opts.global_store_address = get_or("global_store_address", std::string(""));

        // Optional: external CommunicationManager for dependency injection
        if (cfg.contains("comm_manager") && !cfg["comm_manager"].is_none()) {
          try {
            auto mgr = cfg["comm_manager"].cast<std::shared_ptr<stepcast::store::CommunicationManager>>();
            if (mgr && mgr->is_enabled()) {
              opts.comm_manager = mgr;
            }
          } catch (const py::cast_error& /*e*/) {
            PyErr_SetString(PyExc_TypeError, "comm_manager must be a CommunicationManager instance");
            throw py::error_already_set();
          }
        }

        py::gil_scoped_release release;
        return std::make_unique<CheckpointStore>(opts);
      },
      py::arg("config"),
      R"pbdoc(Create a CheckpointStore from a configuration dict.

Expected keys (all optional):
    storage_path: str
    memory_pool_size: int (bytes)
    num_thread: int
    chunk_size: int (bytes)
    pinned_memory_timeout_ms: int
    p2p_port: int
    global_store_address: str

Missing keys fall back to sensible defaults.)pbdoc");

  // ------------------------------------------------------------------
  // Phase-3: Bind CommunicationManager so that Python can create and pass a
  // single communication instance to multiple CheckpointStore objects.
  // ------------------------------------------------------------------

  py::class_<stepcast::store::CommunicationManager, std::shared_ptr<stepcast::store::CommunicationManager>>(
      m, "CommunicationManager")
      .def(
          py::init([](const std::string& listen_addr, uint16_t port, bool enable_rdma) {
            auto mgr = std::make_shared<stepcast::store::CommunicationManager>();
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = mgr->initialize(listen_addr, port, enable_rdma);
            }
            if (!st.ok()) {
              PyErr_SetString(PyExc_RuntimeError, st.ToString().c_str());
              throw py::error_already_set();
            }
            return mgr;
          }),
          py::arg("listen_addr") = std::string("0.0.0.0"),
          py::arg("port") = 9090,
          py::arg("enable_rdma") = false,
          R"pbdoc(Create and initialize a CommunicationManager that wraps a
shared CommunicateEngine. Use this instance to inject the engine into
CheckpointStoreOptions so multiple CheckpointStore objects share the same
transport layer.)pbdoc")
      .def(
          "is_enabled",
          &stepcast::store::CommunicationManager::is_enabled,
          "Return True if the communication engine is initialized and enabled.");
}

// NOLINTEND(google-build-using-namespace,fuchsia-statically-constructed-objects,misc-const-correctness,misc-use-anonymous-namespace,)
