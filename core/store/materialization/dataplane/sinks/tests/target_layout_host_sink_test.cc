// Copyright (c) 2025-2026, TensorCast Team.

#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstddef>
#include <cstdint>

#include "core/store/materialization/dataplane/sinks/target_layout_host_sink.h"

using tensorcast::store::VaRange;
using tensorcast::store::loader::HostTargetStorage;
using tensorcast::store::loader::TargetLayoutHostSink;

namespace {

TargetLayoutHostSink::Options make_options(std::array<std::byte, 4>& first, std::array<std::byte, 4>& second) {
  TargetLayoutHostSink::Options options;
  options.storages = {
      HostTargetStorage{first.data(), first.size()},
      HostTargetStorage{second.data(), second.size()},
  };
  return options;
}

} // namespace

TEST_CASE("TargetLayoutHostSink routes writes across host storages", "[target_layout_host_sink][host_shared]") {
  std::array<std::byte, 4> first{};
  std::array<std::byte, 4> second{};
  TargetLayoutHostSink sink(make_options(first, second));

  const std::array<std::byte, 6> payload{
      std::byte{'A'},
      std::byte{'B'},
      std::byte{'C'},
      std::byte{'D'},
      std::byte{'E'},
      std::byte{'F'},
  };

  REQUIRE(sink.write_at(2, payload.data(), payload.size()).ok());
  REQUIRE(sink.close().ok());

  REQUIRE(first[0] == std::byte{0});
  REQUIRE(first[1] == std::byte{0});
  REQUIRE(first[2] == std::byte{'A'});
  REQUIRE(first[3] == std::byte{'B'});
  REQUIRE(second[0] == std::byte{'C'});
  REQUIRE(second[1] == std::byte{'D'});
  REQUIRE(second[2] == std::byte{'E'});
  REQUIRE(second[3] == std::byte{'F'});
}

TEST_CASE(
    "TargetLayoutHostSink plans direct-write windows across host storages",
    "[target_layout_host_sink][host_shared]") {
  std::array<std::byte, 4> first{};
  std::array<std::byte, 4> second{};
  TargetLayoutHostSink sink(make_options(first, second));

  const std::array<VaRange, 1> ranges{VaRange{2, 4}};
  auto grant_or = sink.plan_direct_write(ranges);
  REQUIRE(grant_or.ok());
  REQUIRE(grant_or->windows.size() == 2);
  REQUIRE(grant_or->windows[0].va_offset == 2);
  REQUIRE(grant_or->windows[0].length == 2);
  REQUIRE(grant_or->windows[0].local_addr == reinterpret_cast<uint64_t>(first.data() + 2));
  REQUIRE(grant_or->windows[1].va_offset == 4);
  REQUIRE(grant_or->windows[1].length == 2);
  REQUIRE(grant_or->windows[1].local_addr == reinterpret_cast<uint64_t>(second.data()));
}
