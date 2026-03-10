// Copyright (c) 2026, TensorCast Team.

#include "core/store/materialization/dataplane/view/view_identity.h"

#include <string>

#include <catch2/catch_test_macros.hpp>

namespace {

using tensorcast::store::loader::ViewSpec;
using tensorcast::store::materialization::view::NarrowOp;
using tensorcast::store::materialization::view::TensorViewOps;
using tensorcast::store::materialization::view::TransposeOp;
using tensorcast::store::materialization::view::ViewOp;

std::string make_canonical_index_json() {
  return R"({"a":[0,16,[2,2],[2,1],"torch.float32",0],"b":[16,16,[2,2],[2,1],"torch.float32",0]})";
}

ViewSpec make_view_spec_b_then_a() {
  ViewSpec spec;

  TensorViewOps b_ops;
  b_ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 1, .start = 0, .length = 1}));
  spec.tensors.emplace("b", std::move(b_ops));

  TensorViewOps a_ops;
  a_ops.ops.push_back(ViewOp::Transpose(TransposeOp{.dim0 = 0, .dim1 = 1}));
  spec.tensors.emplace("a", std::move(a_ops));

  return spec;
}

ViewSpec make_view_spec_a_then_b() {
  ViewSpec spec;

  TensorViewOps a_ops;
  a_ops.ops.push_back(ViewOp::Transpose(TransposeOp{.dim0 = 0, .dim1 = 1}));
  spec.tensors.emplace("a", std::move(a_ops));

  TensorViewOps b_ops;
  b_ops.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 1, .start = 0, .length = 1}));
  spec.tensors.emplace("b", std::move(b_ops));

  return spec;
}

} // namespace

TEST_CASE("View identity canonicalization is insertion-order stable", "[view_identity]") {
  const ViewSpec spec_b_then_a = make_view_spec_b_then_a();
  const ViewSpec spec_a_then_b = make_view_spec_a_then_b();
  const std::string canonical_index_json = make_canonical_index_json();

  const std::string bytes_one = tensorcast::store::loader::canonicalize_view_spec_for_identity(spec_b_then_a);
  const std::string bytes_two = tensorcast::store::loader::canonicalize_view_spec_for_identity(spec_a_then_b);
  REQUIRE(bytes_one == bytes_two);

  auto view_id_one_or = tensorcast::store::loader::compute_view_id_from_spec(spec_b_then_a, canonical_index_json);
  auto view_id_two_or = tensorcast::store::loader::compute_view_id_from_spec(spec_a_then_b, canonical_index_json);
  REQUIRE(view_id_one_or.ok());
  REQUIRE(view_id_two_or.ok());
  REQUIRE(*view_id_one_or == *view_id_two_or);
  REQUIRE(*view_id_one_or == "bciqd3js2h75e5gw6cagg5kp3d6xc7laeqca263wrnmirudisjlwuwwa");
}

TEST_CASE("View identity is sensitive to op ordering", "[view_identity]") {
  ViewSpec spec_one;
  TensorViewOps ops_one;
  ops_one.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 1, .start = 0, .length = 1}));
  ops_one.ops.push_back(ViewOp::Transpose(TransposeOp{.dim0 = 0, .dim1 = 1}));
  spec_one.tensors.emplace("a", std::move(ops_one));

  ViewSpec spec_two;
  TensorViewOps ops_two;
  ops_two.ops.push_back(ViewOp::Transpose(TransposeOp{.dim0 = 0, .dim1 = 1}));
  ops_two.ops.push_back(ViewOp::Narrow(NarrowOp{.dim = 1, .start = 0, .length = 1}));
  spec_two.tensors.emplace("a", std::move(ops_two));

  const std::string canonical_index_json = make_canonical_index_json();
  auto view_id_one_or = tensorcast::store::loader::compute_view_id_from_spec(spec_one, canonical_index_json);
  auto view_id_two_or = tensorcast::store::loader::compute_view_id_from_spec(spec_two, canonical_index_json);
  REQUIRE(view_id_one_or.ok());
  REQUIRE(view_id_two_or.ok());
  REQUIRE(*view_id_one_or != *view_id_two_or);
}

TEST_CASE("View identity rejects empty view specs", "[view_identity]") {
  const ViewSpec empty_spec;
  auto view_id_or = tensorcast::store::loader::compute_view_id_from_spec(empty_spec, make_canonical_index_json());
  REQUIRE_FALSE(view_id_or.ok());
}
