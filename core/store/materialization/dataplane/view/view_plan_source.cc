// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_plan_source.h"

#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "core/store/materialization/dataplane/sources/byte_range_program.h"

namespace tensorcast::store::loader {

namespace {

std::shared_ptr<SeekableSource> alias_unowned_source(SeekableSource* ptr) {
  return std::shared_ptr<SeekableSource>(ptr, [](SeekableSource*) {});
}

} // namespace

ViewPlanSource::ViewPlanSource(
    gsl::not_null<SeekableSource*> base,
    SelectionPlan plan,
    StoreEngineOptions::ByteMappingConfig config)
    : base_(base),
      total_bytes_(plan.map.total_bytes),
      alias_eligible_(plan.is_segment_aligned),
      requires_materialization_(plan.requires_materialization) {
  ByteRangeCompiler compiler(config, "view");
  auto program_or = compiler.Compile(plan.map);
  ABSL_CHECK_OK(program_or.status());

  std::vector<std::shared_ptr<SeekableSource>> sources;
  sources.emplace_back(alias_unowned_source(base_.get()));
  ByteRangeMappedSource::Options options;
  options.path = "view";
  options.enable_direct_write_at = config.enable_direct_write_at;
  auto mapped_or = ByteRangeMappedSource::Create(plan.map, *program_or, std::move(sources), std::move(options));
  ABSL_CHECK_OK(mapped_or.status());
  mapped_source_ = std::move(*mapped_or);
}

absl::StatusOr<size_t> ViewPlanSource::read(void* dst, size_t max_bytes) {
  return mapped_source_->read(dst, max_bytes);
}

absl::StatusOr<size_t> ViewPlanSource::read_at(uint64_t offset, void* dst, size_t bytes) {
  return mapped_source_->read_at(offset, dst, bytes);
}

bool ViewPlanSource::supports_direct_write_at() const {
  return mapped_source_->supports_direct_write_at();
}

absl::StatusOr<size_t> ViewPlanSource::read_into_at(
    uint64_t src_offset,
    uint64_t dest_va_offset,
    size_t bytes,
    const DirectWriteGrant& grant) {
  return mapped_source_->read_into_at(src_offset, dest_va_offset, bytes, grant);
}

std::unique_ptr<SeekableSource> make_view_plan_source(
    std::unique_ptr<SeekableSource> base,
    SelectionPlan plan,
    StoreEngineOptions::ByteMappingConfig config) {
  if (!base) {
    return nullptr;
  }
  if (plan.map.total_bytes == 0 || plan.map.segments.empty()) {
    return base;
  }

  class OwningViewPlanSource final : public SeekableSource {
   public:
    OwningViewPlanSource(
        std::unique_ptr<SeekableSource> base,
        SelectionPlan plan,
        StoreEngineOptions::ByteMappingConfig config)
        : base_(std::move(base)), adapter_(gsl::not_null<SeekableSource*>{base_.get()}, std::move(plan), config) {}

    [[nodiscard]] uint64_t total_bytes() const override {
      return adapter_.total_bytes();
    }

    absl::StatusOr<size_t> read(void* dst, size_t max_bytes) override {
      return adapter_.read(dst, max_bytes);
    }

    absl::StatusOr<size_t> read_at(uint64_t offset, void* dst, size_t bytes) override {
      return adapter_.read_at(offset, dst, bytes);
    }

    [[nodiscard]] bool supports_direct_write_at() const override {
      return adapter_.supports_direct_write_at();
    }

    absl::StatusOr<size_t> read_into_at(
        uint64_t src_offset,
        uint64_t dest_va_offset,
        size_t bytes,
        const DirectWriteGrant& grant) override {
      return adapter_.read_into_at(src_offset, dest_va_offset, bytes, grant);
    }

   private:
    std::unique_ptr<SeekableSource> base_;
    ViewPlanSource adapter_;
  };

  return std::make_unique<OwningViewPlanSource>(std::move(base), std::move(plan), config);
}

} // namespace tensorcast::store::loader
