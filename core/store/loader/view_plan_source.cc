// Copyright (c) 2025, TensorCast Team.

#include "core/store/loader/view_plan_source.h"

#include <algorithm>
#include <cstring>

#include "absl/status/status.h"

namespace tensorcast::store::loader {

namespace {

uint64_t range_end(const SelectionPlan::Range& range) {
  return range.dst_offset + range.length;
}

class OwningViewPlanSource final : public SeekableSource {
 public:
  OwningViewPlanSource(std::unique_ptr<SeekableSource> base, SelectionPlan plan)
      : base_(std::move(base)), adapter_(gsl::not_null<SeekableSource*>{base_.get()}, std::move(plan)) {}

  absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
    return adapter_.read(dst, max_bytes);
  }

  absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
    return adapter_.read_at(offset, dst, bytes);
  }

  [[nodiscard]] bool supports_direct_write() const override {
    return false;
  }

  absl::StatusOr<size_t> read_into(uint64_t, size_t, const DirectWriteGrant&) override {
    return absl::UnimplementedError("direct write not supported for view plan sources");
  }

 private:
  std::unique_ptr<SeekableSource> base_;
  ViewPlanSource adapter_;
};

} // namespace

ViewPlanSource::ViewPlanSource(gsl::not_null<SeekableSource*> base, SelectionPlan plan)
    : base_(base),
      ranges_(plan.ranges.begin(), plan.ranges.end()),
      total_bytes_(plan.total_bytes),
      alias_eligible_(plan.is_segment_aligned),
      requires_materialization_(plan.requires_materialization) {
  std::sort(ranges_.begin(), ranges_.end(), [](const SelectionPlan::Range& a, const SelectionPlan::Range& b) {
    return a.dst_offset < b.dst_offset;
  });
}

absl::StatusOr<size_t> ViewPlanSource::copy_from_range(
    const SelectionPlan::Range& range,
    uint64_t range_offset,
    uint8_t* dst,
    size_t bytes) const {
  if (bytes == 0) {
    return static_cast<size_t>(0);
  }
  if (range.kind == SelectionPlan::Range::Kind::kPad) {
    std::memset(dst, 0, bytes);
    return bytes;
  }
  const uint64_t source_offset = range.src_offset + range_offset;
  auto read_or = base_->read_at(source_offset, dst, bytes);
  if (!read_or.ok()) {
    return read_or.status();
  }
  if (*read_or != bytes) {
    return absl::InternalError("short read while executing view selection plan");
  }
  return *read_or;
}

absl::StatusOr<size_t> ViewPlanSource::read(void* dst, size_t max_bytes) {
  auto bytes_or = read_at(cursor_, dst, max_bytes);
  if (!bytes_or.ok()) {
    return bytes_or;
  }
  cursor_ += *bytes_or;
  return bytes_or;
}

absl::StatusOr<size_t> ViewPlanSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  if (offset >= total_bytes_ || bytes == 0) {
    return static_cast<size_t>(0);
  }
  const uint64_t remaining_bytes = total_bytes_ - offset;
  size_t to_copy = static_cast<size_t>(std::min<uint64_t>(bytes, remaining_bytes));
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t copied = 0;
  uint64_t cursor = offset;

  for (const auto& range : ranges_) {
    if (cursor >= range_end(range)) {
      continue;
    }
    if (cursor < range.dst_offset) {
      return absl::InternalError("selection plan contains uncovered gaps");
    }
    const uint64_t local_offset = cursor - range.dst_offset;
    const size_t available = static_cast<size_t>(range.length - local_offset);
    const size_t chunk = std::min(to_copy - copied, available);
    auto copied_or = copy_from_range(range, local_offset, out + copied, chunk);
    if (!copied_or.ok()) {
      return copied_or.status();
    }
    copied += *copied_or;
    cursor += *copied_or;
    if (copied == to_copy) {
      break;
    }
  }
  return copied;
}

std::unique_ptr<SeekableSource> make_view_plan_source(std::unique_ptr<SeekableSource> base, SelectionPlan plan) {
  if (!base) {
    return nullptr;
  }
  if (plan.total_bytes == 0 || plan.ranges.empty()) {
    return base;
  }
  return std::make_unique<OwningViewPlanSource>(std::move(base), std::move(plan));
}

} // namespace tensorcast::store::loader
