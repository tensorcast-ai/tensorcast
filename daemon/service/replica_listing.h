// Copyright (c) 2025-2026, TensorCast Team.

// Helper to build get_loaded_replicas response with optional cursor pagination.
// Keeps grpc_service_impl thin.

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/store/device_types.h"
#include "core/store/store_engine.h"
#include "daemon/state/ref_tracker.h"
#include "nlohmann/json.hpp"
#include "tensorcast/daemon/v2/store_daemon.pb.h"

namespace tensorcast::daemon::listing {

inline void FillLoadedReplicas(
    store::StoreEngine& engine,
    const RefTracker& refs,
    const v2::GetLoadedReplicasRequest& req,
    v2::GetLoadedReplicasResponse& resp,
    bool use_cursor_pagination) {
  struct Entry {
    std::string artifact_id;
    int device_id;
    int32_t ref_count;
    std::vector<int32_t> pids;
    uint64_t size_bytes;
    int64_t last_access_ts;
  };

  std::vector<Entry> entries;
  entries.reserve(64);
  for (const auto& info : engine.get_all_replicas_info()) {
    int device_id = -1;
    if (info.gpu_state != common::memory::MemoryLocation::NONE) {
      device_id = info.gpu_device_id;
    }
    if (req.has_artifact_id_filter() && info.artifact_id.find(req.artifact_id_filter()) == std::string::npos)
      continue;
    if (req.has_device_id_filter() && device_id != req.device_id_filter())
      continue;

    const store::loading::ReplicaKey& key = info.key;

    Entry e;
    e.artifact_id = info.artifact_id;
    e.device_id = device_id;
    e.ref_count = static_cast<int32_t>(refs.ref_count(key));
    for (int32_t pid : refs.pids(key))
      e.pids.push_back(pid);
    e.size_bytes = info.size_bytes;
    e.last_access_ts =
        std::chrono::duration_cast<std::chrono::seconds>(info.last_access_time.time_since_epoch()).count();
    entries.push_back(std::move(e));
  }

  const uint32_t page_size =
      req.has_pagination() && req.pagination().has_page_size() ? req.pagination().page_size() : 100;

  auto* pi = resp.mutable_page_info();
  pi->set_total_size(static_cast<uint32_t>(entries.size()));

  if (!use_cursor_pagination) {
    uint32_t start = 0;
    if (req.has_pagination() && req.pagination().has_page_token()) {
      try {
        start = static_cast<uint32_t>(std::stoul(req.pagination().page_token()));
      } catch (...) {
        start = 0;
      }
    }
    const uint32_t end = std::min<uint32_t>(start + page_size, static_cast<uint32_t>(entries.size()));
    for (uint32_t i = start; i < end; ++i) {
      const auto& e = entries[i];
      auto* out = resp.add_replicas();
      out->set_artifact_id(e.artifact_id);
      out->set_device_id(e.device_id);
      out->set_ref_count(e.ref_count);
      for (int32_t pid : e.pids)
        out->add_pids(pid);
      out->set_size_bytes(static_cast<int64_t>(e.size_bytes));
      auto* ts = out->mutable_last_access_ts();
      ts->set_seconds(e.last_access_ts);
      ts->set_nanos(0);
    }
    if (end < entries.size()) {
      pi->set_next_page_token(std::to_string(end));
    } else {
      pi->set_next_page_token("");
    }
    return;
  }

  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    if (a.artifact_id != b.artifact_id)
      return a.artifact_id < b.artifact_id;
    return a.device_id < b.device_id;
  });

  struct CursorKey {
    std::string artifact_id;
    int device_id;
  };

  std::optional<CursorKey> cursor;
  if (req.has_pagination() && req.pagination().has_page_token()) {
    const auto& tok = req.pagination().page_token();
    try {
      nlohmann::json j = nlohmann::json::parse(tok);
      if (j.contains("artifact_id") && j.contains("device_id")) {
        cursor = CursorKey{.artifact_id = j["artifact_id"].get<std::string>(), .device_id = j["device_id"].get<int>()};
      }
    } catch (...) {
      try {
        auto start = static_cast<uint32_t>(std::stoul(tok));
        if (start < entries.size()) {
          cursor = CursorKey{.artifact_id = entries[start].artifact_id, .device_id = entries[start].device_id};
        }
      } catch (...) {
      }
    }
  }
  auto after_cursor = [&](const Entry& e) {
    if (!cursor.has_value())
      return true;
    if (e.artifact_id > cursor->artifact_id)
      return true;
    if (e.artifact_id < cursor->artifact_id)
      return false;
    return e.device_id > cursor->device_id;
  };
  uint32_t emitted = 0;
  Entry last_emitted{};
  bool any = false;
  for (const auto& e : entries) {
    if (!after_cursor(e))
      continue;
    auto* out = resp.add_replicas();
    out->set_artifact_id(e.artifact_id);
    out->set_device_id(e.device_id);
    out->set_ref_count(e.ref_count);
    for (int32_t pid : e.pids)
      out->add_pids(pid);
    out->set_size_bytes(static_cast<int64_t>(e.size_bytes));
    auto* ts = out->mutable_last_access_ts();
    ts->set_seconds(e.last_access_ts);
    ts->set_nanos(0);
    last_emitted = e;
    any = true;
    if (++emitted >= page_size)
      break;
  }
  if (any && emitted == page_size) {
    nlohmann::json j;
    j["artifact_id"] = last_emitted.artifact_id;
    j["device_id"] = last_emitted.device_id;
    pi->set_next_page_token(j.dump());
  } else {
    pi->set_next_page_token("");
  }
}

} // namespace tensorcast::daemon::listing
