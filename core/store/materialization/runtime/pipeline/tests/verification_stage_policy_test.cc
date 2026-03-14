// Copyright (c) 2025-2026, TensorCast Team.

#include "core/store/materialization/runtime/pipeline/verification_stage.h"

#include "catch2/catch_test_macros.hpp"
#include "core/store/store_engine_options.h"

namespace tensorcast::store::materialization::runtime::pipeline {

TEST_CASE("Full digest policy skips trusted safetensors re-hash under read-only source", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = true;
  ctx.disk.existing_data_multihash = "mh:trusted";
  ctx.hints.verify = loading::MaterializeHints::Verify::CHECKSUM;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadOnly;

  const FullDigestDecision decision = resolve_full_digest_decision(ctx);
  REQUIRE(decision.should_compute == false);
  REQUIRE(decision.trusted_existing_data_multihash == true);
  REQUIRE(decision.forced_by_safetensors == false);
}

TEST_CASE("Full digest policy keeps safetensors force when trusted hash is missing", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = true;
  ctx.hints.verify = loading::MaterializeHints::Verify::CHECKSUM;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadOnly;

  const FullDigestDecision decision = resolve_full_digest_decision(ctx);
  REQUIRE(decision.should_compute == true);
  REQUIRE(decision.trusted_existing_data_multihash == false);
  REQUIRE(decision.forced_by_safetensors == true);
}

TEST_CASE("Read-only disk tensor_dict path skips verification stage work", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = true;
  ctx.disk.existing_data_multihash = "mh:trusted";
  ctx.hints.verify = loading::MaterializeHints::Verify::CHECKSUM;
  ctx.hints.export_policy = loading::ExportPolicy::kNever;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadOnly;

  REQUIRE(should_skip_disk_verification(ctx) == true);
}

TEST_CASE("Explicit full digest disables read-only disk verification fast path", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = true;
  ctx.disk.existing_data_multihash = "mh:trusted";
  ctx.hints.verify = loading::MaterializeHints::Verify::FULL_DIGEST;
  ctx.hints.export_policy = loading::ExportPolicy::kNever;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadOnly;

  REQUIRE(should_skip_disk_verification(ctx) == false);
}

TEST_CASE("Full digest policy honors explicit FULL_DIGEST hint", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = false;
  ctx.hints.verify = loading::MaterializeHints::Verify::FULL_DIGEST;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadOnly;

  const FullDigestDecision decision = resolve_full_digest_decision(ctx);
  REQUIRE(decision.should_compute == true);
  REQUIRE(decision.forced_by_hint == true);
}

TEST_CASE("Full digest policy honors engine force option", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;
  options.force_full_digest_on_load = true;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = false;
  ctx.hints.verify = loading::MaterializeHints::Verify::CHECKSUM;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadWrite;

  const FullDigestDecision decision = resolve_full_digest_decision(ctx);
  REQUIRE(decision.should_compute == true);
  REQUIRE(decision.forced_by_engine_option == true);
}

TEST_CASE("Full digest policy stays disabled for non-safetensors checksum mode", "[pipeline][verification]") {
  IngestionContext ctx;
  StoreEngineOptions options;

  ctx.options = &options;
  ctx.source_type = SourceType::kDisk;
  ctx.disk.is_safetensors = false;
  ctx.hints.verify = loading::MaterializeHints::Verify::CHECKSUM;
  ctx.hints.source_mutation_policy = loading::SourceMutationPolicy::kReadWrite;

  const FullDigestDecision decision = resolve_full_digest_decision(ctx);
  REQUIRE(decision.should_compute == false);
}

} // namespace tensorcast::store::materialization::runtime::pipeline
