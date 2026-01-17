// Copyright (c) 2025-2026, TensorCast Team.

#include "daemon/util/identity_utils.h"

#include <cctype>
#include <fstream>
#include <optional>
#include <ranges>
#include <string>

#include <unistd.h>

#include "absl/strings/str_cat.h"

namespace tensorcast::daemon {
namespace {

std::optional<std::string> read_machine_id() {
  std::ifstream in("/etc/machine-id");
  if (!in.is_open()) {
    return std::nullopt;
  }
  std::string line;
  std::getline(in, line);
  const auto trimmed = [&line]() {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    auto begin = std::ranges::find_if(line, not_space);
    auto end = std::ranges::find_if(std::ranges::reverse_view(line), not_space).base();
    if (begin >= end) {
      return std::string();
    }
    return std::string(begin, end);
  }();
  if (trimmed.empty()) {
    return std::nullopt;
  }
  return std::make_optional(trimmed);
}

std::string sanitize_component(std::string value) {
  for (char& c : value) {
    if (c == '/' || c == '\\') {
      c = '_';
    }
  }
  return value;
}

} // namespace

std::string derive_host_id() {
  char hostname[256];
  if (::gethostname(hostname, sizeof(hostname)) != 0) {
    return "unknown";
  }
  hostname[sizeof(hostname) - 1] = '\0';
  std::string host = hostname[0] ? hostname : "unknown";
  if (auto machine_id = read_machine_id(); machine_id.has_value()) {
    host = absl::StrCat(host, "-", *machine_id);
  }
  return sanitize_component(std::move(host));
}

std::string derive_node_id() {
  char hostname[256];
  if (::gethostname(hostname, sizeof(hostname)) == 0) {
    return hostname;
  }
  return "unknown";
}

} // namespace tensorcast::daemon
