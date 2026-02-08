// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/assembly_operation_service.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "core/common/artifact_hash.h"
#include "core/store/materialization/contracts/loading_spec.h"
#include "core/store/materialization/dataplane/sources/memory_source.h"
#include "daemon/service/controllers/materialization_policy_utils.h"
#include "daemon/util/status_utils.h"
#include "folly/futures/Future.h"
#include "google/protobuf/util/time_util.h"
#include "gsl/pointers"
#include "nlohmann/json.hpp"

namespace tensorcast::daemon {

using ::grpc::Status;
using ::grpc::StatusCode;
using materialization_policy::spec_includes_transpose;
using status_utils::to_grpc_status;

namespace {

class OperationLeaseGuard {
 public:
  OperationLeaseGuard(
      std::shared_ptr<store::components::IGlobalStoreClient> client,
      std::string lease_token,
      std::string operation_id)
      : client_(std::move(client)), lease_token_(std::move(lease_token)), operation_id_(std::move(operation_id)) {}

  ~OperationLeaseGuard() {
    release();
  }

  void release() {
    if (released_ || client_ == nullptr || lease_token_.empty()) {
      released_ = true;
      return;
    }
    tensorcast::operation::v1::ReleaseOperationLeaseRequest release_req;
    release_req.set_lease_token(lease_token_);
    auto release_or = client_->release_operation_lease(release_req);
    if (!release_or.ok()) {
      LOG(WARNING) << "release_operation_lease failed for op=" << operation_id_ << ": " << release_or.status();
    }
    released_ = true;
  }

 private:
  std::shared_ptr<store::components::IGlobalStoreClient> client_;
  std::string lease_token_;
  std::string operation_id_;
  bool released_{false};
};

std::string compute_seal_operation_id(std::string_view assembly_id) {
  const std::string payload = absl::StrCat("seal_assembly:", assembly_id);
  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  const std::vector<uint8_t> digest = tensorcast::common::sha256_digest_bytes(bytes);
  return tensorcast::common::multibase_multihash_sha256(digest);
}

std::string owner_id_for_operation(const WorkerIdentityStore& identity) {
  auto daemon_id = identity.daemon_id();
  if (!daemon_id.empty()) {
    return daemon_id;
  }
  auto worker_id = identity.worker_id();
  if (!worker_id.empty()) {
    return worker_id;
  }
  return "unknown";
}

bool retryable_status(const absl::Status& st) {
  return absl::IsUnavailable(st) || absl::IsDeadlineExceeded(st) || absl::IsAborted(st) || absl::IsInternal(st) ||
      absl::IsUnknown(st);
}

constexpr uint64_t kProofChunkBytesV1 = 4ULL * 1024 * 1024;

struct TensorInterval {
  std::string tensor_name;
  uint64_t offset{0};
  uint64_t size_bytes{0};
};

absl::StatusOr<std::vector<TensorInterval>> parse_tensor_intervals(std::string_view canonical_index_json) {
  if (canonical_index_json.empty()) {
    return absl::InvalidArgumentError("canonical_index_json must not be empty");
  }
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(canonical_index_json, nullptr, true);
  } catch (const std::exception& ex) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to parse canonical index JSON: ", ex.what()));
  }
  if (!j.is_object()) {
    return absl::InvalidArgumentError("canonical index JSON must be an object");
  }

  std::vector<TensorInterval> out;
  out.reserve(j.size());
  for (auto it = j.begin(); it != j.end(); ++it) {
    const std::string tensor_name = it.key();
    const auto& arr = it.value();
    if (!arr.is_array() || arr.size() < 2) {
      continue;
    }
    TensorInterval interval;
    interval.tensor_name = tensor_name;
    interval.offset = arr[0].get<uint64_t>();
    interval.size_bytes = arr[1].get<uint64_t>();
    out.push_back(std::move(interval));
  }

  std::sort(
      out.begin(), out.end(), [](const TensorInterval& a, const TensorInterval& b) { return a.offset < b.offset; });
  return out;
}

std::vector<uint8_t> compute_view_meta_digest(const store::components::ViewInfo& view) {
  std::vector<store::components::CanonicalRange> ranges = view.canonical_ranges;
  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    if (a.offset != b.offset) {
      return a.offset < b.offset;
    }
    return a.length < b.length;
  });

  std::string payload;
  payload.reserve(256 + ranges.size() * 32);
  absl::StrAppend(&payload, "view_id=", view.view_id, ";");
  absl::StrAppend(&payload, "view_data_hash=", view.view_data_hash.value_or(""), ";");
  absl::StrAppend(&payload, "view_size_bytes=", view.view_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_size_bytes=", view.canonical_size_bytes, ";");
  absl::StrAppend(&payload, "canonical_bytes_covered=", view.canonical_bytes_covered, ";");
  for (const auto& range : ranges) {
    absl::StrAppend(&payload, range.offset, ":", range.length, ";");
  }

  const auto bytes = absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
  return tensorcast::common::sha256_digest_bytes(bytes);
}

absl::StatusOr<bool> check_post_seal_view_reuse_safe(
    store::components::IGlobalStoreClient& client,
    std::string_view assembly_id,
    std::string_view mi2_id) {
  if (assembly_id.empty() || mi2_id.empty()) {
    return absl::InvalidArgumentError("check_post_seal_view_reuse_safe requires assembly_id and mi2_id");
  }

  auto layouts_or = client.list_artifact_layouts(mi2_id);
  if (!layouts_or.ok()) {
    return layouts_or.status();
  }
  if (layouts_or->empty()) {
    return true;
  }

  absl::flat_hash_map<std::string, absl::flat_hash_set<std::string>> tensors_by_schema;
  for (const auto& layout_id : *layouts_or) {
    if (layout_id.empty()) {
      continue;
    }
    auto spec_or = client.get_layout_spec(layout_id);
    if (!spec_or.ok()) {
      return spec_or.status();
    }
    const auto& layout_spec = spec_or->layout();
    const std::string schema_version = layout_spec.proof_schema_version();
    for (const auto& entry : layout_spec.tensors()) {
      if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
        if (schema_version.empty()) {
          return absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
        }
        tensors_by_schema[schema_version].insert(entry.first);
      }
    }
  }

  if (tensors_by_schema.empty()) {
    return true;
  }

  for (const auto& [schema_version, tensors] : tensors_by_schema) {
    if (tensors.empty()) {
      continue;
    }
    tensorcast::global_store::v1::CheckProofCommitmentsMatchRequest req;
    req.set_assembly_id(std::string(assembly_id));
    req.set_mi2_id(std::string(mi2_id));
    req.set_proof_schema_version(schema_version);
    for (const auto& name : tensors) {
      if (!name.empty()) {
        req.add_tensor_names(name);
      }
    }
    auto resp_or = client.check_proof_commitments_match(req);
    if (!resp_or.ok()) {
      return resp_or.status();
    }
    if (!resp_or->match()) {
      return false;
    }
  }
  return true;
}

} // namespace

AssemblyOperationService::AssemblyOperationService(Dep d)
    : d_(std::move(d)), seal_operation_tracker_(std::make_shared<SealOperationTracker>()) {}

grpc::Status AssemblyOperationService::seal_assembly(
    RpcContext& rctx,
    const v2::SealAssemblyRequest& req,
    v2::SealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }

  auto result_or = d_.engine.seal_assembly(req.assembly_id(), req.publish_canonical());
  if (!result_or.ok()) {
    return to_grpc_status(result_or.status());
  }
  const auto& result = *result_or;
  resp.set_sealed_artifact_id(result.sealed_artifact_id);
  resp.set_already_sealed(result.already_sealed);
  auto* desc = resp.mutable_descriptor_();
  desc->set_artifact_id(result.sealed_artifact_id);
  if (!result.index_multihash.empty()) {
    desc->set_index_multihash(result.index_multihash);
  }
  if (!result.data_multihash.empty()) {
    desc->set_data_multihash(result.data_multihash);
  }
  if (!result.schema_version.empty()) {
    desc->set_schema_version(result.schema_version);
  }
  if (!result.encoding.empty()) {
    desc->set_encoding(result.encoding);
  }
  if (result.total_size > 0) {
    desc->set_total_size(result.total_size);
  }
  desc->set_id_kind(tensorcast::common::v1::ArtifactIdKind::ARTIFACT_ID_KIND_MI2);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::start_seal_assembly(
    RpcContext& rctx,
    const v2::StartSealAssemblyRequest& req,
    v2::StartSealAssemblyResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.artifact.id", req.assembly_id());

  if (req.assembly_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "assembly_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const std::string operation_id = compute_seal_operation_id(req.assembly_id());
  auto* out_ref = resp.mutable_operation();
  out_ref->set_operation_id(operation_id);
  out_ref->set_kind("seal_assembly");
  out_ref->set_target_artifact_id(req.assembly_id());

  tensorcast::operation::v1::AcquireOperationLeaseRequest lease_req;
  lease_req.set_operation_id(operation_id);
  lease_req.set_kind("seal_assembly");
  lease_req.set_target_artifact_id(req.assembly_id());
  lease_req.set_owner_id(owner_id_for_operation(d_.identity));
  // Let Global Store apply defaults and clamp to limits.
  lease_req.set_ttl_ms(0);

  auto lease_or = d_.global_store_client->acquire_operation_lease(lease_req);
  if (!lease_or.ok()) {
    if (absl::IsAlreadyExists(lease_or.status())) {
      rctx.mark_success();
      return Status::OK;
    }
    return to_grpc_status(lease_or.status());
  }
  const auto& lease_resp = *lease_or;
  if (!lease_resp.acquired()) {
    rctx.mark_success();
    return Status::OK;
  }

  const auto lease = lease_resp.lease();
  const uint64_t lease_generation = lease.lease_generation();
  const std::string lease_token = lease.lease_token();
  const std::string assembly_id = req.assembly_id();
  const std::string layout_id = req.layout_id();

  auto seal_tracker = seal_operation_tracker_;
  bool should_start = false;
  {
    absl::MutexLock lock(&seal_tracker->mu);
    should_start = seal_tracker->active_operations.insert(operation_id).second;
  }

  if (should_start) {
    auto client_sp = d_.global_store_client;
    auto executor = d_.async_runtime.blocking_executor();
    auto* async_runtime = &d_.async_runtime;
    auto* engine = &d_.engine;
    auto* devices = &d_.devices;
    auto* identity = &d_.identity;
    const DaemonOptions::PostSealPolicy post_seal_policy = d_.post_seal_policy;
    executor->add(
        [seal_tracker,
         client_sp = std::move(client_sp),
         async_runtime,
         engine,
         devices,
         identity,
         post_seal_policy,
         operation_id,
         assembly_id,
         layout_id,
         lease_generation,
         lease_token]() mutable -> void {
          if (client_sp == nullptr) {
            return;
          }
          absl::Status final_status = absl::OkStatus();
          OperationLeaseGuard lease_guard(client_sp, lease_token, operation_id);
          auto cleanup = absl::MakeCleanup([seal_tracker, operation_id]() {
            absl::MutexLock lock(&seal_tracker->mu);
            seal_tracker->active_operations.erase(operation_id);
          });

          auto keepalive_stop = std::make_shared<std::atomic<bool>>(false);
          auto keepalive_exec = async_runtime->blocking_executor();
          auto keepalive = std::make_shared<std::function<void()>>();
          std::weak_ptr<std::function<void()>> keepalive_weak = keepalive;
          *keepalive = [client_sp,
                        keepalive_stop,
                        keepalive_exec,
                        keepalive_weak,
                        &timekeeper = async_runtime->timekeeper(),
                        lease_token]() mutable {
            if (keepalive_stop->load(std::memory_order_relaxed)) {
              return;
            }
            timekeeper.after(std::chrono::milliseconds(5000))
                .via(keepalive_exec)
                .thenValue([client_sp, keepalive_stop, lease_token, keepalive_weak](folly::Unit) mutable {
                  if (keepalive_stop->load(std::memory_order_relaxed)) {
                    return;
                  }
                  tensorcast::operation::v1::KeepaliveOperationLeaseRequest req;
                  req.set_lease_token(lease_token);
                  req.set_ttl_ms(0);
                  auto resp_or = client_sp->keepalive_operation_lease(req);
                  if (!resp_or.ok()) {
                    LOG(WARNING) << "keepalive_operation_lease failed for op=" << lease_token << ": "
                                 << resp_or.status();
                  }
                  auto next = keepalive_weak.lock();
                  if (next != nullptr) {
                    (*next)();
                  }
                });
          };
          (*keepalive)();

          tensorcast::daemon::v2::SealAssemblySnapshot snapshot_msg;
          google::protobuf::Any snapshot_any;
          bool snapshot_loaded = false;
          {
            tensorcast::operation::v1::GetOperationRequest get_req;
            get_req.set_operation_id(operation_id);
            auto existing_or = client_sp->get_operation(get_req);
            if (existing_or.ok()) {
              const auto& existing_snapshot = existing_or->snapshot();
              const bool has_snapshot = !existing_snapshot.type_url().empty() || !existing_snapshot.value().empty();
              if (has_snapshot) {
                snapshot_any = existing_snapshot;
                snapshot_loaded = snapshot_any.UnpackTo(&snapshot_msg);
                if (!snapshot_loaded) {
                  final_status = absl::FailedPreconditionError("unsupported seal snapshot payload");
                }
              }
            } else {
              LOG(WARNING) << "get_operation failed while loading seal snapshot (op=" << operation_id
                           << "): " << existing_or.status();
            }
          }

          if (!snapshot_loaded && final_status.ok()) {
            snapshot_msg.set_assembly_id(assembly_id);
            if (!layout_id.empty()) {
              snapshot_msg.set_layout_id(layout_id);
              snapshot_msg.set_assembly_layout_binding_version(0);
            } else {
              auto binding_or = client_sp->get_assembly_layout_binding(assembly_id);
              if (binding_or.ok()) {
                snapshot_msg.set_layout_id(binding_or->layout_id());
                snapshot_msg.set_assembly_layout_binding_version(binding_or->binding_version());
              }
            }

            auto views_or = client_sp->list_views(assembly_id);
            if (!views_or.ok()) {
              final_status = views_or.status();
            } else {
              std::vector<store::components::ViewInfo> views = std::move(*views_or);
              std::sort(views.begin(), views.end(), [](const auto& a, const auto& b) { return a.view_id < b.view_id; });
              for (const auto& view : views) {
                if (view.view_id.empty()) {
                  continue;
                }
                auto* out = snapshot_msg.add_views();
                out->set_view_id(view.view_id);
                auto digest = compute_view_meta_digest(view);
                out->set_meta_digest(digest.data(), static_cast<int>(digest.size()));
              }
            }

            if (final_status.ok()) {
              snapshot_any.PackFrom(snapshot_msg);
              snapshot_loaded = true;
            }
          }

          tensorcast::operation::v1::UpdateOperationRequest running;
          running.set_operation_id(operation_id);
          running.set_lease_generation(lease_generation);
          auto* status = running.mutable_status();
          status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
          status->set_message("sealing");
          status->set_progress(0.0);
          *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
          if (snapshot_loaded) {
            running.mutable_snapshot()->CopyFrom(snapshot_any);
          }
          final_status = client_sp->update_operation(running);
          if (!final_status.ok()) {
            LOG(WARNING) << "update_operation(RUNNING) failed for op=" << operation_id << ": " << final_status;
            keepalive_stop->store(true, std::memory_order_relaxed);
            return;
          }

          std::vector<std::string> allowed_view_ids;
          allowed_view_ids.reserve(static_cast<size_t>(snapshot_msg.views_size()));
          for (const auto& view : snapshot_msg.views()) {
            if (!view.view_id().empty()) {
              allowed_view_ids.push_back(view.view_id());
            }
          }

          if (snapshot_msg.views_size() > 0) {
            auto current_views_or = client_sp->list_views(assembly_id);
            if (!current_views_or.ok()) {
              final_status = current_views_or.status();
            } else {
              absl::flat_hash_map<std::string, std::string> expected;
              expected.reserve(static_cast<size_t>(snapshot_msg.views_size()));
              for (const auto& view : snapshot_msg.views()) {
                expected.emplace(view.view_id(), view.meta_digest());
              }
              for (const auto& view : *current_views_or) {
                auto it = expected.find(view.view_id);
                if (it == expected.end()) {
                  continue;
                }
                auto digest = compute_view_meta_digest(view);
                const std::string computed(reinterpret_cast<const char*>(digest.data()), digest.size());
                if (computed != it->second) {
                  final_status = absl::FailedPreconditionError(
                      absl::StrCat("seal snapshot view metadata mismatch for view_id=", view.view_id));
                  break;
                }
                expected.erase(it);
              }
              if (final_status.ok() && !expected.empty()) {
                final_status = absl::FailedPreconditionError("seal snapshot view missing from current view set");
              }
            }
          }

          if (!final_status.ok()) {
            keepalive_stop->store(true, std::memory_order_relaxed);
            // Fall through to FAILED status update below.
          }

          auto last_progress_ms = std::make_shared<std::atomic<int64_t>>(0);
          auto max_hashed = std::make_shared<std::atomic<uint64_t>>(0);
          auto enable_updates = std::make_shared<std::atomic<bool>>(true);
          store::runtime::ingestion::MaterializationFacade::SealProgressCallback progress_cb =
              [client_sp, operation_id, lease_generation, last_progress_ms, max_hashed, enable_updates](
                  uint64_t hashed_leaf_count, uint64_t total_hash_leaves) mutable {
                if (!enable_updates->load(std::memory_order_relaxed) || total_hash_leaves == 0) {
                  return;
                }
                const uint64_t prev_max = max_hashed->load(std::memory_order_relaxed);
                if (hashed_leaf_count <= prev_max && hashed_leaf_count != total_hash_leaves) {
                  return;
                }
                max_hashed->store(std::max(prev_max, hashed_leaf_count), std::memory_order_relaxed);

                const int64_t now_ms = absl::ToUnixMillis(absl::Now());
                const int64_t last_ms = last_progress_ms->load(std::memory_order_relaxed);
                if (hashed_leaf_count != total_hash_leaves && last_ms != 0 && now_ms - last_ms < 1000) {
                  return;
                }
                last_progress_ms->store(now_ms, std::memory_order_relaxed);

                tensorcast::operation::v1::UpdateOperationRequest update;
                update.set_operation_id(operation_id);
                update.set_lease_generation(lease_generation);
                auto* status = update.mutable_status();
                status->set_state(tensorcast::operation::v1::OPERATION_STATE_RUNNING);
                status->set_message(absl::StrCat("hashing ", hashed_leaf_count, "/", total_hash_leaves));
                status->set_progress(static_cast<double>(hashed_leaf_count) / static_cast<double>(total_hash_leaves));
                *status->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
                absl::Status st = client_sp->update_operation(update);
                if (!st.ok()) {
                  enable_updates->store(false, std::memory_order_relaxed);
                  LOG(WARNING) << "update_operation(progress) failed for op=" << operation_id << ": " << st;
                }
              };

          const std::vector<std::string>* allowed_ptr = snapshot_loaded ? &allowed_view_ids : nullptr;
          auto seal_or = final_status.ok()
              ? engine->seal_assembly(assembly_id, /*publish_canonical=*/true, std::move(progress_cb), allowed_ptr)
              : absl::StatusOr<store::SealAssemblyResult>(final_status);
          if (!seal_or.ok()) {
            final_status = seal_or.status();
          } else {
            const std::string sealed_artifact_id = seal_or->sealed_artifact_id;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              final_status = client_sp->attach_layout_to_artifact(sealed_artifact_id, snapshot_msg.layout_id());
            }

            std::optional<tensorcast::layout::v1::LayoutSpec> layout_spec_for_post_seal;
            if (final_status.ok() && !snapshot_msg.layout_id().empty()) {
              auto layout_or = client_sp->get_layout_spec(snapshot_msg.layout_id());
              if (!layout_or.ok()) {
                final_status = layout_or.status();
              } else {
                layout_spec_for_post_seal = layout_or->layout();
                const auto& layout_spec = *layout_spec_for_post_seal;
                const std::string proof_schema_version = layout_spec.proof_schema_version();
                absl::flat_hash_set<std::string> replicated_tensors;
                replicated_tensors.reserve(layout_spec.tensors_size());
                for (const auto& entry : layout_spec.tensors()) {
                  if (entry.second.overlap_mode() == tensorcast::layout::v1::OVERLAP_MODE_REPLICATE_EQUAL) {
                    replicated_tensors.insert(entry.first);
                  }
                }

                if (!replicated_tensors.empty()) {
                  if (proof_schema_version.empty()) {
                    final_status =
                        absl::FailedPreconditionError("proof_schema_version required for replicated tensors");
                  } else if (proof_schema_version != "v1") {
                    final_status = absl::UnimplementedError("unsupported proof_schema_version");
                  } else {
                    auto index_or = client_sp->get_artifact_index_by_id(sealed_artifact_id);
                    if (!index_or.ok()) {
                      final_status = index_or.status();
                    } else {
                      auto intervals_or = parse_tensor_intervals(*index_or);
                      if (!intervals_or.ok()) {
                        final_status = intervals_or.status();
                      } else {
                        auto resident_devices = engine->get_resident_devices(sealed_artifact_id);
                        auto gpu_it = std::find_if(
                            resident_devices.begin(), resident_devices.end(), [](const store::DeviceKey& d) {
                              return d.type == DeviceType::GPU;
                            });
                        if (gpu_it == resident_devices.end()) {
                          final_status =
                              absl::FailedPreconditionError("sealed artifact GPU replica unavailable for proofs");
                        } else {
                          store::loading::ReplicaKey replica_key;
                          replica_key.artifact_id = sealed_artifact_id;
                          replica_key.view_id = std::nullopt;
                          replica_key.device = *gpu_it;
                          replica_key.replica = 0;

                          auto size_or = engine->get_replica_size(replica_key);
                          auto ptr_or = engine->get_replica_gpu_ptr(replica_key);
                          if (!size_or.ok()) {
                            final_status = size_or.status();
                          } else if (!ptr_or.ok()) {
                            final_status = ptr_or.status();
                          } else {
                            store::loader::GpuMemorySource src(
                                gsl::not_null<void*>{reinterpret_cast<void*>(*ptr_or)},
                                /*device_id=*/gpu_it->ordinal,
                                *size_or);

                            std::vector<tensorcast::global_store::v1::TensorProofCommitmentWrite> writes;
                            for (const auto& interval : *intervals_or) {
                              if (interval.size_bytes == 0) {
                                continue;
                              }
                              if (!replicated_tensors.contains(interval.tensor_name)) {
                                continue;
                              }
                              const uint64_t expected_chunks =
                                  (interval.size_bytes + kProofChunkBytesV1 - 1) / kProofChunkBytesV1;
                              for (uint64_t chunk_idx = 0; chunk_idx < expected_chunks; ++chunk_idx) {
                                const uint64_t local_start = chunk_idx * kProofChunkBytesV1;
                                const uint64_t local_end =
                                    std::min<uint64_t>(interval.size_bytes, local_start + kProofChunkBytesV1);
                                if (local_end <= local_start) {
                                  continue;
                                }
                                const uint64_t abs_start = interval.offset + local_start;
                                const uint64_t read_len = local_end - local_start;
                                if (read_len > std::numeric_limits<size_t>::max()) {
                                  final_status = absl::OutOfRangeError("proof chunk exceeds host memory limits");
                                  break;
                                }
                                std::vector<uint8_t> buffer(static_cast<size_t>(read_len));
                                auto read_or = src.read_at(abs_start, buffer.data(), static_cast<size_t>(read_len));
                                if (!read_or.ok()) {
                                  final_status = read_or.status();
                                  break;
                                }
                                if (*read_or != buffer.size()) {
                                  final_status = absl::DataLossError("short read while computing MI2 proof digest");
                                  break;
                                }
                                std::vector<uint8_t> digest =
                                    tensorcast::common::sha256_digest_bytes(absl::MakeSpan(buffer));
                                if (digest.size() != 32) {
                                  final_status = absl::InternalError("sha256 digest size mismatch");
                                  break;
                                }
                                tensorcast::global_store::v1::TensorProofCommitmentWrite write;
                                write.set_tensor_name(interval.tensor_name);
                                write.set_proof_chunk_idx(chunk_idx);
                                write.set_digest(digest.data(), static_cast<int>(digest.size()));
                                writes.push_back(std::move(write));
                              }
                              if (!final_status.ok()) {
                                break;
                              }
                            }

                            if (final_status.ok() && !writes.empty()) {
                              constexpr size_t kBatchEntries = 1024;
                              for (size_t i = 0; i < writes.size(); i += kBatchEntries) {
                                tensorcast::global_store::v1::WriteTensorProofCommitmentsRequest write_req;
                                write_req.set_mi2_id(sealed_artifact_id);
                                write_req.set_proof_schema_version(proof_schema_version);
                                const size_t end = std::min(writes.size(), i + kBatchEntries);
                                for (size_t j = i; j < end; ++j) {
                                  *write_req.add_commitments() = writes[j];
                                }
                                auto write_resp_or = client_sp->write_tensor_proof_commitments(write_req);
                                if (!write_resp_or.ok()) {
                                  final_status = write_resp_or.status();
                                  break;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            if (final_status.ok()) {
              const auto& policy = post_seal_policy;
              const bool allow_migration = policy.migrate_views;
              const bool allow_retire = policy.retire_pieces;
              if (allow_retire && !policy.migrate_views && !policy.reuse_views_if_safe) {
                LOG(WARNING) << "post-seal retire_pieces enabled without migrate_views or reuse_views_if_safe; "
                             << "view reads may fail after seal";
              }

              if (allow_migration) {
                auto views_or = client_sp->list_views(assembly_id);
                if (!views_or.ok()) {
                  LOG(WARNING) << "post-seal migrate_views list_views failed for assembly=" << assembly_id << ": "
                               << views_or.status();
                } else {
                  absl::flat_hash_set<std::string> allowed_set;
                  if (!allowed_view_ids.empty()) {
                    allowed_set.reserve(allowed_view_ids.size());
                    for (const auto& id : allowed_view_ids) {
                      if (!id.empty()) {
                        allowed_set.insert(id);
                      }
                    }
                  }
                  absl::flat_hash_set<std::string> expected_set;
                  if (layout_spec_for_post_seal.has_value()) {
                    const auto& expected = layout_spec_for_post_seal->expected_view_ids();
                    expected_set.reserve(static_cast<size_t>(expected.size()));
                    for (const auto& id : expected) {
                      if (!id.empty()) {
                        expected_set.insert(id);
                      }
                    }
                  }

                  const std::vector<std::string>* allowed_ptr = allowed_view_ids.empty() ? nullptr : &allowed_view_ids;
                  if (engine->get_num_gpus() == 0) {
                    LOG(WARNING) << "post-seal migrate_views skipped: no GPU devices available";
                  } else {
                    for (const auto& view : *views_or) {
                      if (view.view_id.empty()) {
                        continue;
                      }
                      if (!allowed_set.empty() && !allowed_set.contains(view.view_id)) {
                        continue;
                      }
                      if (!expected_set.empty() && !expected_set.contains(view.view_id)) {
                        continue;
                      }
                      if (view.view_spec_json.empty()) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (missing view_spec_json)";
                        continue;
                      }
                      if (view.view_size_bytes == 0) {
                        LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                     << " (view_size_bytes=0)";
                        continue;
                      }
                      if (policy.migrate_transpose_only) {
                        auto spec_or = store::view::parse_view_spec_json(view.view_spec_json);
                        if (!spec_or.ok()) {
                          LOG(WARNING) << "post-seal migrate_views skipping view_id=" << view.view_id
                                       << " (invalid view_spec_json): " << spec_or.status();
                          continue;
                        }
                        if (!spec_includes_transpose(*spec_or)) {
                          continue;
                        }
                      }

                      const store::DeviceKey target_device = devices->DefaultGpu();
                      auto handle_or = engine->materialize_view_from_assembly(
                          assembly_id,
                          sealed_artifact_id,
                          view.view_id,
                          view.view_spec_json,
                          target_device,
                          store::loading::TransformPlacement::kServer,
                          allowed_ptr);
                      if (!handle_or.ok()) {
                        LOG(WARNING) << "post-seal migrate_views failed for view_id=" << view.view_id << ": "
                                     << handle_or.status();
                        continue;
                      }

                      auto publish_status = engine->register_replica_with_global_store(handle_or->replica_key, {});
                      if (!publish_status.ok() && !absl::IsAlreadyExists(publish_status)) {
                        LOG(WARNING) << "post-seal migrate_views register_replica failed for view_id=" << view.view_id
                                     << ": " << publish_status;
                      }

                      store::components::ViewStateUpdate update;
                      update.artifact_id = sealed_artifact_id;
                      update.view_id = view.view_id;
                      update.view_spec_json = view.view_spec_json;
                      update.view_size_bytes = view.view_size_bytes;
                      if (view.view_data_hash.has_value()) {
                        update.view_data_hash = view.view_data_hash;
                      }
                      update.mark_verified = view.verified_at.has_value();
                      update.canonical_size_bytes = view.canonical_size_bytes;
                      update.canonical_bytes_covered = view.canonical_bytes_covered;
                      update.canonical_ranges = view.canonical_ranges;
                      auto view_status = client_sp->update_artifact_view_state(update);
                      if (!view_status.ok()) {
                        LOG(WARNING) << "post-seal migrate_views update_view_state failed for view_id=" << view.view_id
                                     << ": " << view_status;
                      }
                    }
                  }
                }
              }

              if (allow_retire) {
                const std::string worker_id = identity->worker_id();
                if (!worker_id.empty()) {
                  auto unreg_status = client_sp->unregister_replica_by_worker(assembly_id, worker_id);
                  if (!unreg_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unregister_replica_by_worker failed for assembly="
                                 << assembly_id << ": " << unreg_status;
                  }
                } else {
                  LOG(WARNING) << "post-seal retire_pieces skipped unregister_replica_by_worker: worker_id unavailable";
                }

                std::vector<store::loading::ReplicaKey> to_unload;
                for (const auto& info : engine->get_all_replicas_info()) {
                  if (info.key.artifact_id == assembly_id) {
                    to_unload.push_back(info.key);
                  }
                }
                for (const auto& key : to_unload) {
                  auto unload_status = engine->unload_replica_status(key);
                  if (!unload_status.ok()) {
                    LOG(WARNING) << "post-seal retire_pieces unload_replica failed for key=" << key << ": "
                                 << unload_status;
                  }
                }
              }
            }

            tensorcast::daemon::v2::SealAssemblyResult result_msg;
            auto* artifact = result_msg.mutable_artifact();
            artifact->set_artifact_id(sealed_artifact_id);
            if (!seal_or->index_multihash.empty()) {
              artifact->set_index_multihash(seal_or->index_multihash);
            }
            if (!seal_or->data_multihash.empty()) {
              artifact->set_data_multihash(seal_or->data_multihash);
            }
            if (!seal_or->schema_version.empty()) {
              artifact->set_schema_version(seal_or->schema_version);
            }
            if (!seal_or->encoding.empty()) {
              artifact->set_encoding(seal_or->encoding);
            }
            if (seal_or->total_size > 0) {
              artifact->set_total_size(seal_or->total_size);
            }

            tensorcast::operation::v1::UpdateOperationRequest success;
            success.set_operation_id(operation_id);
            success.set_lease_generation(lease_generation);
            auto* out = success.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_SUCCESS);
            out->set_message("sealed");
            out->set_progress(1.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            out->mutable_result()->PackFrom(result_msg);
            if (final_status.ok()) {
              final_status = client_sp->update_operation(success);
            }
          }

          if (!final_status.ok()) {
            tensorcast::operation::v1::UpdateOperationRequest failed;
            failed.set_operation_id(operation_id);
            failed.set_lease_generation(lease_generation);
            auto* out = failed.mutable_status();
            out->set_state(tensorcast::operation::v1::OPERATION_STATE_FAILED);
            out->set_message("seal failed");
            out->set_progress(0.0);
            *out->mutable_as_of() = google::protobuf::util::TimeUtil::GetCurrentTime();
            auto* err = out->mutable_error();
            err->set_status_code(absl::StatusCodeToString(final_status.code()));
            err->set_message(std::string(final_status.message()));
            err->set_retryable(retryable_status(final_status));
            absl::Status update_st = client_sp->update_operation(failed);
            if (!update_st.ok()) {
              LOG(WARNING) << "update_operation(FAILED) failed for op=" << operation_id << ": " << update_st;
            }
          }

          keepalive_stop->store(true, std::memory_order_relaxed);
          lease_guard.release();
        });
  }

  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::get_operation(
    RpcContext& rctx,
    const tensorcast::operation::v1::GetOperationRequest& req,
    tensorcast::operation::v1::GetOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  auto op_or = d_.global_store_client->get_operation(req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp = std::move(*op_or);
  rctx.mark_success();
  return Status::OK;
}

grpc::Status AssemblyOperationService::wait_operation(
    RpcContext& rctx,
    const v2::WaitOperationRequest& req,
    v2::WaitOperationResponse& resp) {
  auto& span = rctx.span();
  span->SetAttribute("tc.operation.id", req.operation_id());

  if (req.operation_id().empty()) {
    return {StatusCode::INVALID_ARGUMENT, "operation_id is required"};
  }
  if (d_.shutdown_signal.is_shutting_down()) {
    return {StatusCode::UNAVAILABLE, "daemon is shutting down"};
  }
  if (!d_.global_store_client || !d_.global_store_client->is_connected()) {
    return {StatusCode::FAILED_PRECONDITION, "GlobalStoreClient not connected"};
  }

  const uint64_t timeout_ms = req.timeout_ms();
  const absl::Time start = absl::Now();
  const absl::Time deadline = timeout_ms > 0 ? start + absl::Milliseconds(timeout_ms) : absl::InfiniteFuture();

  absl::Duration sleep = absl::Milliseconds(50);
  tensorcast::operation::v1::GetOperationRequest op_req;
  op_req.set_operation_id(req.operation_id());

  while (absl::Now() < deadline) {
    auto op_or = d_.global_store_client->get_operation(op_req);
    if (!op_or.ok()) {
      return to_grpc_status(op_or.status());
    }
    const auto state = op_or->status().state();
    resp.mutable_operation()->Swap(&(*op_or));
    if (state == tensorcast::operation::v1::OPERATION_STATE_SUCCESS ||
        state == tensorcast::operation::v1::OPERATION_STATE_FAILED ||
        state == tensorcast::operation::v1::OPERATION_STATE_CANCELLED) {
      rctx.mark_success();
      return Status::OK;
    }
    absl::SleepFor(sleep);
    sleep = std::min(sleep * 12 / 10, absl::Milliseconds(500));
  }

  auto op_or = d_.global_store_client->get_operation(op_req);
  if (!op_or.ok()) {
    return to_grpc_status(op_or.status());
  }
  resp.mutable_operation()->Swap(&(*op_or));
  rctx.mark_success();
  return Status::OK;
}

} // namespace tensorcast::daemon
