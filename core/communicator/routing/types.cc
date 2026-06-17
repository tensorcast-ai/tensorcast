// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/routing/types.h"

#include <algorithm>
#include <format>
#include <limits>
#include <vector>

namespace tensorcast::communicator::routing {
namespace {

HealthState derive_health_from_counts(
    uint64_t success_count,
    uint64_t failure_count,
    absl::Time last_success,
    absl::Time last_failure) {
  if (success_count == 0 && failure_count == 0) {
    return HealthState::kUnknown;
  }
  if (success_count == 0 && failure_count > 0) {
    return HealthState::kUnhealthy;
  }
  if (failure_count == 0) {
    return HealthState::kHealthy;
  }
  if (last_failure > last_success) {
    return HealthState::kDegraded;
  }
  return HealthState::kHealthy;
}

bool add_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs > std::numeric_limits<uint64_t>::max() - rhs;
}

absl::Status invalid_plan(std::string message) {
  return absl::InvalidArgumentError(std::move(message));
}

struct DestinationSpan {
  uint64_t begin = 0;
  uint64_t end = 0;
  size_t slice_index = 0;
};

struct SourceCoverageSpan {
  uint64_t begin = 0;
  uint64_t end = 0;
  size_t slice_index = 0;
};

} // namespace

std::string_view to_string(ConnectionProtocol protocol) {
  switch (protocol) {
    case ConnectionProtocol::kAuto:
      return "AUTO";
    case ConnectionProtocol::kRdma:
      return "RDMA";
    case ConnectionProtocol::kTcp:
      return "TCP";
    case ConnectionProtocol::kMtcp:
      return "MTCP";
    case ConnectionProtocol::kNvlink:
      return "NVLINK";
    case ConnectionProtocol::kPcie:
      return "PCIE";
    case ConnectionProtocol::kShm:
      return "SHM";
    default:
      return "UNKNOWN";
  }
}

std::string_view to_string(ConnectionType type) {
  switch (type) {
    case ConnectionType::kForward:
      return "FORWARD";
    case ConnectionType::kP2P:
      return "P2P";
    case ConnectionType::kSwitch:
      return "SWITCH";
    default:
      return "UNKNOWN";
  }
}

bool EndpointBinding::has_network_address() const {
  return !ip.empty() && port != 0;
}

HealthState derive_health(const ConnectionStats& stats) {
  return derive_health_from_counts(stats.success_count, stats.failure_count, stats.last_success, stats.last_failure);
}

HealthState derive_health(const LinkStats& stats) {
  return derive_health_from_counts(stats.success_count, stats.failure_count, stats.last_success, stats.last_failure);
}

absl::Status validate_read_plan(const ReadPlan& plan) {
  if (plan.slices.empty()) {
    if (plan.local_regions.empty() && plan.source_slices.empty()) {
      return absl::OkStatus();
    }
    return invalid_plan("read plan with no slices must not declare local regions or source slices");
  }
  if (plan.local_regions.empty()) {
    return invalid_plan("read plan requires at least one local region");
  }
  if (plan.source_slices.empty()) {
    return invalid_plan("read plan requires at least one source slice");
  }

  const SourceSlice& first_source = plan.source_slices.front();
  if (first_source.authority_id.empty()) {
    return invalid_plan("read plan source_slices[0] missing authority_id");
  }
  if (first_source.route.local_endpoint_id.empty() || first_source.route.remote_endpoint_id.empty()) {
    return invalid_plan("read plan source_slices[0] missing route endpoint ids");
  }

  for (size_t source_index = 0; source_index < plan.source_slices.size(); ++source_index) {
    const SourceSlice& source = plan.source_slices[source_index];
    if (source.authority_id.empty()) {
      return invalid_plan(std::format("read plan source_slices[{}] missing authority_id", source_index));
    }
    if (source.tensor_key.empty()) {
      return invalid_plan(std::format("read plan source_slices[{}] missing tensor_key", source_index));
    }
    if (source.route.local_endpoint_id.empty() || source.route.remote_endpoint_id.empty()) {
      return invalid_plan(std::format("read plan source_slices[{}] missing route endpoint ids", source_index));
    }
    if (source.route.rail_id < -1) {
      return invalid_plan(
          std::format("read plan source_slices[{}] has invalid rail_id {}", source_index, source.route.rail_id));
    }
    if (source.bytes == 0) {
      return invalid_plan(std::format("read plan source_slices[{}] has zero bytes", source_index));
    }
    if (add_overflows(source.remote_offset, source.bytes)) {
      return invalid_plan(std::format("read plan source_slices[{}] remote range overflows", source_index));
    }
    if (source.authority_id != first_source.authority_id) {
      return invalid_plan(
          std::format(
              "read plan mixes authority ids: source_slices[0]={} source_slices[{}]={}",
              first_source.authority_id,
              source_index,
              source.authority_id));
    }
    if (!(source.route == first_source.route)) {
      return invalid_plan(
          std::format("read plan mixes route context between source_slices[0] and source_slices[{}]", source_index));
    }
  }

  const LocalRegion& first_region = plan.local_regions.front();
  if (first_region.dev_type != base::COMMUNICATE_ENGINE_DEV_CPU) {
    return absl::FailedPreconditionError("read plan currently supports CPU local regions only");
  }

  for (size_t region_index = 0; region_index < plan.local_regions.size(); ++region_index) {
    const LocalRegion& region = plan.local_regions[region_index];
    if (region.bytes == 0) {
      return invalid_plan(std::format("read plan local_regions[{}] has zero bytes", region_index));
    }
    if (add_overflows(region.addr, region.bytes)) {
      return invalid_plan(std::format("read plan local_regions[{}] address range overflows", region_index));
    }
    if (region.dev_type != base::COMMUNICATE_ENGINE_DEV_CPU) {
      return absl::FailedPreconditionError("read plan currently supports CPU local regions only");
    }
    if (region.dev_id != first_region.dev_id) {
      return invalid_plan(
          std::format(
              "read plan mixes local dev_id values: local_regions[0]={} local_regions[{}]={}",
              first_region.dev_id,
              region_index,
              region.dev_id));
    }
  }

  std::vector<DestinationSpan> destination_spans;
  destination_spans.reserve(plan.slices.size());
  std::vector<std::vector<SourceCoverageSpan>> source_coverages(plan.source_slices.size());
  for (size_t slice_index = 0; slice_index < plan.slices.size(); ++slice_index) {
    const ReadPlanSlice& slice = plan.slices[slice_index];
    if (slice.bytes == 0) {
      return invalid_plan(std::format("read plan slices[{}] has zero bytes", slice_index));
    }
    if (slice.source_slice_index >= plan.source_slices.size()) {
      return invalid_plan(std::format("read plan slices[{}] has invalid source_slice_index", slice_index));
    }
    if (slice.local_region_index >= plan.local_regions.size()) {
      return invalid_plan(std::format("read plan slices[{}] has invalid local_region_index", slice_index));
    }

    const SourceSlice& source = plan.source_slices[slice.source_slice_index];
    const LocalRegion& region = plan.local_regions[slice.local_region_index];
    if (add_overflows(slice.source_slice_offset, slice.bytes) ||
        slice.source_slice_offset + slice.bytes > source.bytes) {
      return invalid_plan(std::format("read plan slices[{}] exceeds source slice bounds", slice_index));
    }
    if (add_overflows(slice.local_region_offset, slice.bytes) ||
        slice.local_region_offset + slice.bytes > region.bytes) {
      return invalid_plan(std::format("read plan slices[{}] exceeds local region bounds", slice_index));
    }
    if (add_overflows(region.addr, slice.local_region_offset) ||
        add_overflows(region.addr + slice.local_region_offset, slice.bytes)) {
      return invalid_plan(std::format("read plan slices[{}] destination range overflows", slice_index));
    }

    const uint64_t begin = region.addr + slice.local_region_offset;
    destination_spans.push_back(
        DestinationSpan{
            .begin = begin,
            .end = begin + slice.bytes,
            .slice_index = slice_index,
        });
    source_coverages[slice.source_slice_index].push_back(
        SourceCoverageSpan{
            .begin = slice.source_slice_offset,
            .end = slice.source_slice_offset + slice.bytes,
            .slice_index = slice_index,
        });
  }

  std::sort(
      destination_spans.begin(), destination_spans.end(), [](const DestinationSpan& lhs, const DestinationSpan& rhs) {
        if (lhs.begin != rhs.begin) {
          return lhs.begin < rhs.begin;
        }
        return lhs.end < rhs.end;
      });

  for (size_t index = 1; index < destination_spans.size(); ++index) {
    const DestinationSpan& previous = destination_spans[index - 1];
    const DestinationSpan& current = destination_spans[index];
    if (current.begin < previous.end) {
      return invalid_plan(
          std::format(
              "read plan destination overlap between slices[{}] and slices[{}]",
              previous.slice_index,
              current.slice_index));
    }
  }

  for (size_t source_index = 0; source_index < source_coverages.size(); ++source_index) {
    auto& coverages = source_coverages[source_index];
    if (coverages.empty()) {
      return invalid_plan(std::format("read plan source_slices[{}] has no covering slices", source_index));
    }
    std::sort(coverages.begin(), coverages.end(), [](const SourceCoverageSpan& lhs, const SourceCoverageSpan& rhs) {
      if (lhs.begin != rhs.begin) {
        return lhs.begin < rhs.begin;
      }
      return lhs.end < rhs.end;
    });

    uint64_t expected_begin = 0;
    for (const auto& coverage : coverages) {
      if (coverage.begin != expected_begin) {
        return invalid_plan(
            std::format(
                "read plan source_slices[{}] coverage gap or overlap at offset {}", source_index, expected_begin));
      }
      expected_begin = coverage.end;
    }
    if (expected_begin != plan.source_slices[source_index].bytes) {
      return invalid_plan(
          std::format(
              "read plan source_slices[{}] coverage ends at {} but expected {}",
              source_index,
              expected_begin,
              plan.source_slices[source_index].bytes));
    }
  }

  return absl::OkStatus();
}

} // namespace tensorcast::communicator::routing
