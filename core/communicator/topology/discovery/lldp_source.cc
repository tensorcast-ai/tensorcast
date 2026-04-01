// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/topology/discovery/lldp_source.h"

#include <fstream>
#include <regex>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"

namespace tensorcast::communicator::topology::discovery {
namespace {

constexpr const char* kPciBdfPattern = "^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\\.[0-7]$";

absl::StatusOr<LldpNicRecord> parse_lldp_line(const std::string& line, int line_no) {
  const std::string trimmed = std::string(absl::StripAsciiWhitespace(line));
  if (trimmed.empty() || trimmed[0] == '#') {
    return absl::CancelledError("skip");
  }

  const size_t equal_pos = trimmed.find('=');
  if (equal_pos == std::string::npos || equal_pos == 0 || equal_pos + 1 >= trimmed.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("line ", line_no, ": expected '<if_name>=<pci_bdf>,<nic_name>,<rail_id>'"));
  }

  LldpNicRecord record;
  record.if_name = std::string(absl::StripAsciiWhitespace(trimmed.substr(0, equal_pos)));
  const std::string value = std::string(absl::StripAsciiWhitespace(trimmed.substr(equal_pos + 1)));

  const size_t first_comma = value.find(',');
  const size_t second_comma = value.find(',', first_comma == std::string::npos ? first_comma : first_comma + 1);
  if (first_comma == std::string::npos || second_comma == std::string::npos || second_comma + 1 >= value.size()) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": expected '<pci_bdf>,<nic_name>,<rail_id>'"));
  }

  record.pci_bdf = std::string(absl::StripAsciiWhitespace(value.substr(0, first_comma)));
  record.nic_name =
      std::string(absl::StripAsciiWhitespace(value.substr(first_comma + 1, second_comma - first_comma - 1)));
  const std::string rail_text = std::string(absl::StripAsciiWhitespace(value.substr(second_comma + 1)));

  if (record.if_name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": if_name is empty"));
  }
  if (record.nic_name.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": nic_name is empty"));
  }
  if (!std::regex_match(record.pci_bdf, std::regex(kPciBdfPattern))) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": invalid PCI BDF '", record.pci_bdf, "'"));
  }

  try {
    record.rail_id = std::stoi(rail_text);
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": invalid rail_id '", rail_text, "'"));
  }
  if (record.rail_id <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("line ", line_no, ": rail_id must be positive, got ", record.rail_id));
  }

  return record;
}

absl::Status handle_parse_error(const absl::Status& status, bool strict) {
  if (strict) {
    return status;
  }
  LOG(WARNING) << "Skipping LLDP record: " << status;
  return absl::OkStatus();
}

} // namespace

absl::StatusOr<std::vector<LldpNicRecord>> load_lldp_records(const std::string& file_path, LldpParseOptions options) {
  std::ifstream input(file_path);
  if (!input.good()) {
    return absl::NotFoundError(absl::StrCat("LLDP file not found: ", file_path));
  }

  std::vector<LldpNicRecord> records;
  absl::flat_hash_map<std::string, LldpNicRecord> by_nic;
  std::string line;
  int line_no = 0;

  while (std::getline(input, line)) {
    line_no += 1;
    auto record_or = parse_lldp_line(line, line_no);
    if (!record_or.ok()) {
      if (record_or.status().code() == absl::StatusCode::kCancelled) {
        continue;
      }
      absl::Status parse_status = handle_parse_error(record_or.status(), options.strict);
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    const LldpNicRecord& record = record_or.value();
    auto it = by_nic.find(record.nic_name);
    if (it == by_nic.end()) {
      by_nic.emplace(record.nic_name, record);
      records.push_back(record);
      continue;
    }

    if (it->second == record) {
      continue;
    }

    const absl::Status conflict = absl::InvalidArgumentError(
        absl::StrCat(
            "line ",
            line_no,
            ": conflicting LLDP entries for nic '",
            record.nic_name,
            "' (existing rail=",
            it->second.rail_id,
            ", new rail=",
            record.rail_id,
            ")"));
    absl::Status conflict_status = handle_parse_error(conflict, options.strict);
    if (!conflict_status.ok()) {
      return conflict_status;
    }
  }

  return records;
}

absl::StatusOr<absl::flat_hash_map<std::string, LldpNicRecord>> load_lldp_records_by_nic(
    const std::string& file_path,
    LldpParseOptions options) {
  auto records_or = load_lldp_records(file_path, options);
  if (!records_or.ok()) {
    return records_or.status();
  }

  absl::flat_hash_map<std::string, LldpNicRecord> by_nic;
  by_nic.reserve(records_or->size());
  for (const auto& record : *records_or) {
    by_nic.emplace(record.nic_name, record);
  }
  return by_nic;
}

} // namespace tensorcast::communicator::topology::discovery
