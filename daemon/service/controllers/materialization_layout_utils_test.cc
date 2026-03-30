// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/service/controllers/materialization_layout_utils.h"

#include <catch2/catch_test_macros.hpp>

namespace {

using tensorcast::daemon::materialization_layout::dtype_element_size;

TEST_CASE("dtype_element_size supports float8 families", "[daemon][materialization_layout]") {
  CHECK(dtype_element_size("torch.float8_e4m3fn").value() == 1);
  CHECK(dtype_element_size("torch.float8_e5m2").value() == 1);
  CHECK(dtype_element_size("torch.float8_e4m3fnuz").value() == 1);
  CHECK(dtype_element_size("torch.float8_e5m2fnuz").value() == 1);
  CHECK(dtype_element_size("torch.float8_e8m0fnu").value() == 1);
}

} // namespace
