
// Copyright (c) 2025, StepCast Team. All rights reserved.

#include <pybind11/eval.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include <cstdint>
#include "scstore/csrc/logging.h"
#include "scstore/csrc/py_error_utils.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "core/common/logging_init.h"
#include "core/common/metrics/metrics_export.h"
#include "core/store/communication_types.h"
#include "core/store/components/communication_manager.h"
#include "core/store/device_types.h"
#include "core/store/loading/loading_spec.h"
#include "core/store/model/memory_state.h"
#include "core/store/model/model_location.h"
#include "core/store/store_engine.h"
#include "core/store/store_engine_options.h"

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
  py::class_<StoreEngine::ModelInfo>(m, "ModelInfo")
      .def(py::init<>())
      .def_readwrite("model_id", &StoreEngine::ModelInfo::model_id)
      .def_readwrite("size_bytes", &StoreEngine::ModelInfo::size_bytes)
      .def_readwrite("cpu_state", &StoreEngine::ModelInfo::cpu_state)
      .def_readwrite("gpu_state", &StoreEngine::ModelInfo::gpu_state)
      .def_readwrite("gpu_device_id", &StoreEngine::ModelInfo::gpu_device_id)
      .def_readwrite("gpu_device_uuid", &StoreEngine::ModelInfo::gpu_device_uuid)
      .def_readwrite("is_registered_for_comm", &StoreEngine::ModelInfo::is_registered_for_comm)
      .def_property_readonly(
          "last_access_timestamp",
          [](const StoreEngine::ModelInfo& info) { return time_point_to_timestamp(info.last_access_time); })
      .def_property_readonly(
          "load_timestamp", [](const StoreEngine::ModelInfo& info) { return time_point_to_timestamp(info.load_time); })
      .def("__repr__", [](const StoreEngine::ModelInfo& info) {
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
  py::enum_<StoreEngine::PrepareMode>(m, "PrepareMode")
      .value("AUTO", StoreEngine::PrepareMode::AUTO)
      .value("COPY_ONLY", StoreEngine::PrepareMode::COPY_ONLY)
      .value("LOAD_ONLY", StoreEngine::PrepareMode::LOAD_ONLY)
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
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
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

  auto ckpt_cls = py::class_<StoreEngine>(m, "StoreEngine");
  ckpt_cls
      .def(
          "prepare",
          [](StoreEngine& cs,
             const py::object& target_device_obj,
             StoreEngine::PrepareMode mode,
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
                PY_THROW_WITH_LOG(PyExc_ValueError, std::string("Unsupported device spec string"));
              }
            } else if (py::isinstance<py::int_>(target_device_obj)) {
              dev_key.type = DeviceType::GPU;
              dev_key.ordinal = target_device_obj.cast<int>();
            } else {
              PY_THROW_WITH_LOG(PyExc_TypeError, std::string("target_device must be DeviceKey, str, or int"));
            }

            LoadingHints hints;
            if (kwargs.contains("pinned_timeout_ms") && !kwargs["pinned_timeout_ms"].is_none()) {
              int t = kwargs["pinned_timeout_ms"].cast<int>();
              if (t > 0) {
                hints.pinned_timeout = std::chrono::milliseconds(t);
              }
            }
            if (kwargs.contains("model_id") && !kwargs["model_id"].is_none()) {
              hints.model_id = kwargs["model_id"].cast<std::string>();
            }
            if (kwargs.contains("disk_path") && !kwargs["disk_path"].is_none()) {
              hints.disk_path = kwargs["disk_path"].cast<std::string>();
            }

            absl::StatusOr<ModelHandle> h_or;
            {
              py::gil_scoped_release release;
              h_or = cs.prepare(dev_key, mode, hints);
            }
            if (!h_or.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, h_or.status().ToString());
            }
            return h_or.value();
          },
          py::arg("target_device") = std::string("gpu:0"),
          py::arg("mode") = StoreEngine::PrepareMode::AUTO,
          "Prepare a model instance on the specified device and return a ModelHandle.")
      .def("clear_mem", &StoreEngine::clear_mem, "Clear all allocated memory.")
      .def("get_mem_pool_size", &StoreEngine::get_mem_pool_size, "Get the memory pool size.")
      .def("get_chunk_size", &StoreEngine::get_chunk_size, "Get the chunk size.")
      .def(
          "get_available_memory", &StoreEngine::get_available_memory, "Get available memory in the pinned memory pool.")
      .def(
          "get_loaded_devices",
          [](StoreEngine& cs, const std::string& model_id) {
            py::gil_scoped_release release;
            return cs.get_loaded_devices(model_id);
          },
          py::arg("model_id"),
          "Return a list of DeviceKey where the given model is loaded.")
      .def(
          "list_device_models",
          [](StoreEngine& cs, const DeviceKey& device) {
            py::gil_scoped_release release;
            return cs.list_device_models(device);
          },
          py::arg("device"),
          "Return a list of InstanceKey for models resident on the given device.")
      .def(
          "wait_instance_ready",
          [](StoreEngine& cs, const InstanceKey& key) {
            py::gil_scoped_release release;
            return cs.wait_instance_ready(key);
          },
          py::arg("instance_key"),
          "Block until the instance becomes ready. Returns 0 on success.")
      .def(
          "unload_instance",
          [](StoreEngine& cs, const InstanceKey& key) {
            py::gil_scoped_release release;
            return cs.unload_instance(key);
          },
          py::arg("instance_key"),
          "Unload the specified model instance from memory.")
      .def(
          "get_instance_state",
          [](StoreEngine& cs, const InstanceKey& key, DeviceType mem_type) {
            py::gil_scoped_release release;
            return cs.get_instance_state(key, mem_type);
          },
          py::arg("instance_key"),
          py::arg("memory_type"),
          "Get the MemoryState of the specified instance and memory type.")
      .def(
          "get_instance_gpu_ptr",
          [](StoreEngine& cs, const InstanceKey& key) {
            absl::StatusOr<uint64_t> ptr_or;
            {
              py::gil_scoped_release release;
              ptr_or = cs.get_instance_gpu_ptr(key);
            }
            if (!ptr_or.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, ptr_or.status().ToString());
            }
            return ptr_or.value();
          },
          py::arg("instance_key"),
          "Return the base GPU address for the given instance.")
      .def(
          "enable_remote_instance_access",
          [](StoreEngine& cs, const InstanceKey& key, ModelLocation loc) {
            absl::StatusOr<CommRegistrationInfo> info_or;
            {
              py::gil_scoped_release release;
              info_or = cs.enable_remote_instance_access(key, loc);
            }
            if (!info_or.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, info_or.status().ToString());
            }
            return info_or.value();
          },
          py::arg("instance_key"),
          py::arg("location"),
          "Enable remote memory access for the given instance and return registration info.")
      .def(
          "disable_remote_instance_access",
          [](StoreEngine& cs, const InstanceKey& key, ModelLocation loc) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.disable_remote_instance_access(key, loc);
            }
            if (!st.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
            }
            return true;
          },
          py::arg("instance_key"),
          py::arg("location"),
          "Disable remote memory access for the given instance.")
      .def(
          "begin_register_tensor_dict",
          [](StoreEngine& cs, const py::dict& reg_dict) {
            StoreEngine::TensorDictRegistration reg;
            auto get_uint32 = [&](const char* key, uint32_t fb) -> uint32_t {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<uint32_t>();
              }
              return fb;
            };
            auto get_opt_str = [&](const char* key) -> std::optional<std::string> {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<std::string>();
              }
              return std::nullopt;
            };
            auto get_str = [&](const char* key, const char* fallback = "") -> std::string {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<std::string>();
              }
              return std::string(fallback);
            };
            auto get_uint64 = [&](const char* key, uint64_t fb) -> uint64_t {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<uint64_t>();
              }
              return fb;
            };
            auto get_int = [&](const char* key, int fb) -> int {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<int>();
              }
              return fb;
            };
            auto get_bool = [&](const char* key, bool fb) -> bool {
              if (reg_dict.contains(key) && !reg_dict[key].is_none()) {
                return reg_dict[key].cast<bool>();
              }
              return fb;
            };

            reg.model_id = get_str("model_id");
            reg.tensor_index_key = get_str("tensor_index_key");
            reg.tensor_index_data = get_opt_str("tensor_index_data");
            reg.schema_version = get_str("schema_version", "v2");
            reg.encoding = get_str("encoding", "json");
            reg.device_id = get_int("device_id", 0);
            reg.total_size_bytes = get_uint64("total_size_bytes", 0);
            reg.enable_p2p = get_bool("enable_p2p", true);
            reg.ttl_ms = get_uint32("ttl_ms", 0);

            absl::StatusOr<StoreEngine::RegistrationBeginResult> out_or;
            {
              py::gil_scoped_release release;
              out_or = cs.begin_register_tensor_dict(reg);
            }
            if (!out_or.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, out_or.status().ToString());
            }

            const auto& out = out_or.value();
            py::dict py_out;
            py_out["registration_id"] = out.registration_id;
            py_out["device_id"] = out.device_id;
            py_out["size_bytes"] = out.size_bytes;
            py_out["daemon_ipc_handle"] = py::bytes(
                reinterpret_cast<const char*>(out.cuda_ipc_handle_bytes.data()), out.cuda_ipc_handle_bytes.size());
            return py_out;
          },
          py::arg("registration"),
          "Begin registering an in-memory tensor dict and return CUDA IPC handle bytes.")
      .def(
          "commit_registered_tensor_dict",
          [](StoreEngine& cs, const std::string& registration_id) {
            absl::StatusOr<StoreEngine::RegistrationCommitResult> ok_or;
            {
              py::gil_scoped_release release;
              ok_or = cs.commit_registered_tensor_dict(registration_id);
            }
            if (!ok_or.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, ok_or.status().ToString());
            }
            const auto& r = ok_or.value();
            py::dict d;
            d["registration_id"] = r.registration_id;
            d["model_id"] = r.model_id;
            d["device_id"] = r.device_id;
            d["size_bytes"] = r.size_bytes;
            // RFC-0007: include descriptor components for callers
            d["index_multihash"] = r.index_multihash;
            d["data_multihash"] = r.data_multihash;
            d["schema_version"] = r.schema_version;
            d["encoding"] = r.encoding;
            return d;
          },
          py::arg("registration_id"),
          "Commit a pending tensor dict registration.")
      .def(
          "abort_registered_tensor_dict",
          [](StoreEngine& cs, const std::string& registration_id) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.abort_registered_tensor_dict(registration_id);
            }
            if (!st.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
            }
            return true;
          },
          py::arg("registration_id"),
          "Abort a pending tensor dict registration and release memory.")
      .def(
          "lock_chunks",
          [](StoreEngine& cs, const InstanceKey& key, const std::vector<uint32_t>& chunk_indices) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.lock_chunks(key, absl::MakeSpan(chunk_indices));
            }
            if (!st.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
            }
            return 0; // Return 0 on success
          },
          py::arg("instance_key"),
          py::arg("chunk_indices"),
          "Lock chunks for H2D or P2P transfer to prevent concurrent eviction.")
      .def(
          "unlock_chunks",
          [](StoreEngine& cs, const InstanceKey& key, const std::vector<uint32_t>& chunk_indices, bool copied_gpu) {
            absl::Status st;
            {
              py::gil_scoped_release release;
              st = cs.unlock_chunks(key, absl::MakeSpan(chunk_indices), copied_gpu);
            }
            if (!st.ok()) {
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
            }
            return 0; // Return 0 on success
          },
          py::arg("instance_key"),
          py::arg("chunk_indices"),
          py::arg("copied_gpu"),
          "Unlock chunks after H2D or P2P transfer completion.")
      .def("__repr__", [](const StoreEngine& /*cs*/) { return "<StoreEngine>"; })
      .def(
          "get_all_models_info", &StoreEngine::get_all_models_info, "Get detailed information about all loaded models.")
      .def(
          "get_gpu_memory_stats",
          [](StoreEngine& /*cs*/) {
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
  // StoreEngine via the new StoreEngineOptions structure.  This API
  // is opt-in so that existing callers continue to work without changes.
  // ------------------------------------------------------------------
  m.def(
      "create_store_engine",
      [](py::dict cfg) {
        StoreEngineOptions opts;

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
            PY_THROW_WITH_LOG(PyExc_TypeError, std::string("comm_manager must be a CommunicationManager instance"));
          }
        }

        py::gil_scoped_release release;
        return std::make_unique<StoreEngine>(opts);
      },
      py::arg("config"),
      R"pbdoc(Create a StoreEngine from a configuration dict.

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
  // single communication instance to multiple StoreEngine objects.
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
              PY_THROW_WITH_LOG(PyExc_RuntimeError, st.ToString());
            }
            return mgr;
          }),
          py::arg("listen_addr") = std::string("0.0.0.0"),
          py::arg("port") = 9090,
          py::arg("enable_rdma") = false,
          R"pbdoc(Create and initialize a CommunicationManager that wraps a
shared CommunicateEngine. Use this instance to inject the engine into
StoreEngineOptions so multiple StoreEngine objects share the same
transport layer.)pbdoc")
      .def(
          "is_enabled",
          &stepcast::store::CommunicationManager::is_enabled,
          "Return True if the communication engine is initialized and enabled.");
}

// NOLINTEND(google-build-using-namespace,fuchsia-statically-constructed-objects,misc-const-correctness,misc-use-anonymous-namespace,)
