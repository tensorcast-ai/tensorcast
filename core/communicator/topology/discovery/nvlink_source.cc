// Copyright (c) 2026, TensorCast Team.

#include "core/communicator/topology/discovery/nvlink_source.h"

#include <sys/wait.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

namespace tensorcast::communicator::topology::discovery {
namespace {

struct EdgeAccumulator {
  std::string src_gpu_uuid;
  std::string dst_gpu_uuid;
  int link_count = 0;
  double bandwidth_hint_gbps = 0.0;
};

struct TopologyMatrix {
  std::vector<std::string> header_gpu_labels;
  absl::flat_hash_map<std::string, absl::flat_hash_map<std::string, std::string>> cell_by_src_dst;
};

absl::Status handle_parse_error(const absl::Status& status, bool strict, const std::string& source_label) {
  if (strict) {
    return status;
  }
  LOG(WARNING) << "Skipping " << source_label << ": " << status;
  return absl::OkStatus();
}

std::string make_edge_key(const std::string& a, const std::string& b) {
  if (a <= b) {
    return absl::StrCat(a, "|", b);
  }
  return absl::StrCat(b, "|", a);
}

bool is_decimal_number(const std::string& text) {
  if (text.empty()) {
    return false;
  }
  for (char ch : text) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  return true;
}

bool is_gpu_label(const std::string& token) {
  if (!absl::StartsWith(token, "GPU")) {
    return false;
  }
  const std::string suffix = token.substr(3);
  return is_decimal_number(suffix);
}

std::string gpu_label_from_index(int gpu_index) {
  return absl::StrCat("GPU", gpu_index);
}

std::vector<std::string> split_whitespace(const std::string& text) {
  std::istringstream input(text);
  std::vector<std::string> tokens;
  std::string token;
  while (input >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::string strip_ansi_escape_sequences(std::string_view text) {
  std::string cleaned;
  cleaned.reserve(text.size());
  size_t cursor = 0;
  while (cursor < text.size()) {
    if (text[cursor] != '\x1b' || cursor + 1 >= text.size() || text[cursor + 1] != '[') {
      cleaned.push_back(text[cursor]);
      cursor += 1;
      continue;
    }

    cursor += 2;
    while (cursor < text.size()) {
      const unsigned char ch = static_cast<unsigned char>(text[cursor]);
      cursor += 1;
      // ANSI CSI final bytes are in [0x40, 0x7E].
      if (ch >= 0x40 && ch <= 0x7E) {
        break;
      }
    }
  }
  return cleaned;
}

absl::StatusOr<std::string> run_command_capture_stdout(const std::string& command) {
  errno = 0;
  FILE* raw = popen(command.c_str(), "r");
  if (raw == nullptr) {
    return absl::ErrnoToStatus(errno, absl::StrCat("popen failed for command: ", command));
  }

  std::array<char, 4096> buffer{};
  std::string output;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), raw) != nullptr) {
    output.append(buffer.data());
  }

  errno = 0;
  const int close_status = pclose(raw);
  if (close_status == -1) {
    return absl::ErrnoToStatus(errno, absl::StrCat("pclose failed for command: ", command));
  }
  if (!WIFEXITED(close_status) || WEXITSTATUS(close_status) != 0) {
    return absl::UnavailableError(absl::StrCat("command failed: ", command, " (status=", close_status, ")"));
  }

  return output;
}

absl::StatusOr<NvlinkGpuRecord> parse_gpu_row(const std::vector<std::string>& cols, int line_no) {
  if (cols.size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": gpu row expects 3 columns"));
  }
  NvlinkGpuRecord record;
  record.gpu_uuid = std::string(absl::StripAsciiWhitespace(cols[1]));
  if (record.gpu_uuid.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": gpu_uuid is empty"));
  }

  const std::string index_text = std::string(absl::StripAsciiWhitespace(cols[2]));
  try {
    record.gpu_index = std::stoi(index_text);
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": invalid gpu_index '", index_text, "'"));
  }
  if (record.gpu_index < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("line ", line_no, ": gpu_index must be >= 0, got ", record.gpu_index));
  }
  return record;
}

absl::StatusOr<NvlinkEdge> parse_edge_row(const std::vector<std::string>& cols, int line_no) {
  if (cols.size() != 5) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": edge row expects 5 columns"));
  }

  NvlinkEdge edge;
  edge.src_gpu_uuid = std::string(absl::StripAsciiWhitespace(cols[1]));
  edge.dst_gpu_uuid = std::string(absl::StripAsciiWhitespace(cols[2]));
  if (edge.src_gpu_uuid.empty() || edge.dst_gpu_uuid.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": edge gpu_uuid is empty"));
  }
  if (edge.src_gpu_uuid == edge.dst_gpu_uuid) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": self-loop edge is not allowed"));
  }

  const std::string link_count_text = std::string(absl::StripAsciiWhitespace(cols[3]));
  const std::string bandwidth_text = std::string(absl::StripAsciiWhitespace(cols[4]));
  try {
    edge.link_count = std::stoi(link_count_text);
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": invalid link_count '", link_count_text, "'"));
  }
  if (edge.link_count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("line ", line_no, ": link_count must be positive, got ", edge.link_count));
  }

  try {
    edge.bandwidth_hint_gbps = std::stod(bandwidth_text);
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(
        absl::StrCat("line ", line_no, ": invalid bandwidth_hint_gbps '", bandwidth_text, "'"));
  }
  if (edge.bandwidth_hint_gbps < 0.0) {
    return absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": bandwidth_hint_gbps must be >= 0"));
  }

  return edge;
}

absl::StatusOr<std::vector<NvlinkGpuRecord>> parse_runtime_gpu_query_output(
    const std::string& gpu_query_output,
    bool strict) {
  absl::flat_hash_map<int, std::string> uuid_by_index;
  absl::flat_hash_map<std::string, int> index_by_uuid;

  std::istringstream input(gpu_query_output);
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    line_no += 1;
    const std::string trimmed = std::string(absl::StripAsciiWhitespace(line));
    if (trimmed.empty()) {
      continue;
    }

    std::vector<std::string> cols = absl::StrSplit(trimmed, ',');
    if (cols.size() < 2) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat("line ", line_no, ": expected '<gpu_index>,<gpu_uuid>' from runtime query")),
          strict,
          "NVLINK runtime GPU row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    const std::string index_text = std::string(absl::StripAsciiWhitespace(cols[0]));
    const std::string gpu_uuid = std::string(absl::StripAsciiWhitespace(cols[1]));

    // Accept accidental header rows if command format drifts.
    if (absl::EqualsIgnoreCase(index_text, "index") && absl::EqualsIgnoreCase(gpu_uuid, "uuid")) {
      continue;
    }

    int gpu_index = -1;
    try {
      gpu_index = std::stoi(index_text);
    } catch (const std::exception&) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat("line ", line_no, ": invalid GPU index '", index_text, "' in runtime query")),
          strict,
          "NVLINK runtime GPU row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    if (gpu_index < 0 || gpu_uuid.empty()) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat(
                  "line ",
                  line_no,
                  ": runtime GPU row requires gpu_index>=0 and non-empty gpu_uuid (index=",
                  gpu_index,
                  ", uuid='",
                  gpu_uuid,
                  "')")),
          strict,
          "NVLINK runtime GPU row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    auto uuid_by_index_it = uuid_by_index.find(gpu_index);
    if (uuid_by_index_it != uuid_by_index.end() && uuid_by_index_it->second != gpu_uuid) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat(
                  "line ",
                  line_no,
                  ": conflicting gpu_uuid for gpu_index ",
                  gpu_index,
                  " (existing='",
                  uuid_by_index_it->second,
                  "', new='",
                  gpu_uuid,
                  "')")),
          strict,
          "NVLINK runtime GPU row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    auto index_by_uuid_it = index_by_uuid.find(gpu_uuid);
    if (index_by_uuid_it != index_by_uuid.end() && index_by_uuid_it->second != gpu_index) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat(
                  "line ",
                  line_no,
                  ": conflicting gpu_index for gpu_uuid '",
                  gpu_uuid,
                  "' (existing=",
                  index_by_uuid_it->second,
                  ", new=",
                  gpu_index,
                  ")")),
          strict,
          "NVLINK runtime GPU row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    uuid_by_index[gpu_index] = gpu_uuid;
    index_by_uuid[gpu_uuid] = gpu_index;
  }

  if (uuid_by_index.empty()) {
    return absl::NotFoundError("runtime NVLINK probe returned no GPU records");
  }

  std::vector<NvlinkGpuRecord> gpus;
  gpus.reserve(uuid_by_index.size());
  for (const auto& [gpu_index, gpu_uuid] : uuid_by_index) {
    gpus.push_back(NvlinkGpuRecord{.gpu_uuid = gpu_uuid, .gpu_index = gpu_index});
  }
  std::sort(gpus.begin(), gpus.end(), [](const NvlinkGpuRecord& a, const NvlinkGpuRecord& b) {
    return a.gpu_uuid < b.gpu_uuid;
  });
  return gpus;
}

absl::StatusOr<TopologyMatrix> parse_runtime_topology_matrix_output(
    const std::string& topology_matrix_output,
    bool strict) {
  TopologyMatrix matrix;
  bool header_found = false;

  std::istringstream input(topology_matrix_output);
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    line_no += 1;
    const std::string trimmed = std::string(absl::StripAsciiWhitespace(strip_ansi_escape_sequences(line)));
    if (trimmed.empty()) {
      continue;
    }

    const std::vector<std::string> tokens = split_whitespace(trimmed);
    if (tokens.empty()) {
      continue;
    }

    if (!header_found) {
      size_t header_gpu_count = 0;
      while (header_gpu_count < tokens.size() && is_gpu_label(tokens[header_gpu_count])) {
        header_gpu_count += 1;
      }
      if (header_gpu_count == 0) {
        continue;
      }
      matrix.header_gpu_labels.assign(tokens.begin(), tokens.begin() + header_gpu_count);
      header_found = true;
      continue;
    }

    if (absl::StartsWith(tokens[0], "Legend")) {
      break;
    }
    const std::string& row_label = tokens[0];
    if (!is_gpu_label(row_label)) {
      continue;
    }

    const size_t expected_cols = matrix.header_gpu_labels.size() + 1;
    if (tokens.size() < expected_cols) {
      absl::Status parse_status = handle_parse_error(
          absl::InvalidArgumentError(
              absl::StrCat(
                  "line ",
                  line_no,
                  ": runtime topology row has ",
                  tokens.size(),
                  " columns, expected >=",
                  expected_cols,
                  " for row ",
                  row_label)),
          strict,
          "NVLINK runtime topology row");
      if (!parse_status.ok()) {
        return parse_status;
      }
      continue;
    }

    auto& row = matrix.cell_by_src_dst[row_label];
    for (size_t col = 0; col < matrix.header_gpu_labels.size(); ++col) {
      row[matrix.header_gpu_labels[col]] = tokens[col + 1];
    }
  }

  if (!header_found) {
    return absl::NotFoundError("runtime NVLINK topology matrix header not found");
  }
  return matrix;
}

absl::StatusOr<int> parse_nvlink_count_token(
    const std::string& token,
    const std::string& src_gpu_label,
    const std::string& dst_gpu_label) {
  const std::string normalized = absl::AsciiStrToUpper(std::string(absl::StripAsciiWhitespace(token)));
  if (normalized.empty() || normalized == "X" || normalized == "N/A") {
    return 0;
  }
  if (!absl::StartsWith(normalized, "NV")) {
    return 0;
  }

  std::string suffix = normalized.substr(2);
  if (suffix.empty() || suffix == "L") {
    return 1;
  }
  if (absl::StartsWith(suffix, "L")) {
    suffix = suffix.substr(1);
    if (suffix.empty()) {
      return 1;
    }
  }
  if (!is_decimal_number(suffix)) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "invalid NVLINK token '",
            token,
            "' between ",
            src_gpu_label,
            " and ",
            dst_gpu_label,
            " (expected NV<number> or NVL)"));
  }

  int link_count = 0;
  try {
    link_count = std::stoi(suffix);
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "invalid NVLINK token '", token, "' between ", src_gpu_label, " and ", dst_gpu_label, " (stoi failed)"));
  }
  if (link_count <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(
            "invalid non-positive NVLINK count '", token, "' between ", src_gpu_label, " and ", dst_gpu_label));
  }
  return link_count;
}

std::optional<std::string> lookup_matrix_cell(
    const TopologyMatrix& matrix,
    const std::string& src_label,
    const std::string& dst_label) {
  auto row_it = matrix.cell_by_src_dst.find(src_label);
  if (row_it == matrix.cell_by_src_dst.end()) {
    return std::nullopt;
  }
  auto col_it = row_it->second.find(dst_label);
  if (col_it == row_it->second.end()) {
    return std::nullopt;
  }
  return col_it->second;
}

void sort_snapshot(NvlinkSnapshot* snapshot) {
  std::sort(snapshot->gpus.begin(), snapshot->gpus.end(), [](const NvlinkGpuRecord& a, const NvlinkGpuRecord& b) {
    return a.gpu_uuid < b.gpu_uuid;
  });
  std::sort(snapshot->edges.begin(), snapshot->edges.end(), [](const NvlinkEdge& a, const NvlinkEdge& b) {
    if (a.src_gpu_uuid != b.src_gpu_uuid) {
      return a.src_gpu_uuid < b.src_gpu_uuid;
    }
    return a.dst_gpu_uuid < b.dst_gpu_uuid;
  });
}

} // namespace

absl::StatusOr<NvlinkSnapshot> load_nvlink_snapshot(const std::string& file_path, NvlinkSnapshotOptions options) {
  std::ifstream input(file_path);
  if (!input.good()) {
    return absl::NotFoundError(absl::StrCat("NVLINK snapshot not found: ", file_path));
  }

  absl::flat_hash_map<std::string, NvlinkGpuRecord> gpu_records;
  absl::flat_hash_map<std::string, EdgeAccumulator> edge_records;

  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    line_no += 1;
    const std::string trimmed = std::string(absl::StripAsciiWhitespace(line));
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    std::vector<std::string> cols = absl::StrSplit(trimmed, ',');
    if (cols.empty()) {
      continue;
    }

    const std::string type = std::string(absl::StripAsciiWhitespace(cols[0]));
    if (type == "gpu") {
      auto gpu_or = parse_gpu_row(cols, line_no);
      if (!gpu_or.ok()) {
        absl::Status parse_status = handle_parse_error(gpu_or.status(), options.strict, "NVLINK snapshot GPU row");
        if (!parse_status.ok()) {
          return parse_status;
        }
        continue;
      }
      const NvlinkGpuRecord& gpu = gpu_or.value();
      auto it = gpu_records.find(gpu.gpu_uuid);
      if (it == gpu_records.end()) {
        gpu_records.emplace(gpu.gpu_uuid, gpu);
        continue;
      }
      if (it->second.gpu_index != gpu.gpu_index) {
        const absl::Status conflict = absl::InvalidArgumentError(
            absl::StrCat(
                "line ",
                line_no,
                ": conflicting gpu_index for gpu_uuid '",
                gpu.gpu_uuid,
                "' (existing=",
                it->second.gpu_index,
                ", new=",
                gpu.gpu_index,
                ")"));
        absl::Status conflict_status = handle_parse_error(conflict, options.strict, "NVLINK snapshot GPU row");
        if (!conflict_status.ok()) {
          return conflict_status;
        }
      }
      continue;
    }

    if (type == "edge") {
      auto edge_or = parse_edge_row(cols, line_no);
      if (!edge_or.ok()) {
        absl::Status parse_status = handle_parse_error(edge_or.status(), options.strict, "NVLINK snapshot edge row");
        if (!parse_status.ok()) {
          return parse_status;
        }
        continue;
      }
      const NvlinkEdge& edge = edge_or.value();
      const std::string key = make_edge_key(edge.src_gpu_uuid, edge.dst_gpu_uuid);
      auto it = edge_records.find(key);
      if (it == edge_records.end()) {
        EdgeAccumulator acc;
        if (edge.src_gpu_uuid <= edge.dst_gpu_uuid) {
          acc.src_gpu_uuid = edge.src_gpu_uuid;
          acc.dst_gpu_uuid = edge.dst_gpu_uuid;
        } else {
          acc.src_gpu_uuid = edge.dst_gpu_uuid;
          acc.dst_gpu_uuid = edge.src_gpu_uuid;
        }
        acc.link_count = edge.link_count;
        acc.bandwidth_hint_gbps = edge.bandwidth_hint_gbps;
        edge_records.emplace(key, std::move(acc));
      } else {
        it->second.link_count += edge.link_count;
        it->second.bandwidth_hint_gbps = std::max(it->second.bandwidth_hint_gbps, edge.bandwidth_hint_gbps);
      }
      continue;
    }

    const absl::Status unknown =
        absl::InvalidArgumentError(absl::StrCat("line ", line_no, ": unknown row type '", type, "'"));
    absl::Status parse_status = handle_parse_error(unknown, options.strict, "NVLINK snapshot row");
    if (!parse_status.ok()) {
      return parse_status;
    }
  }

  NvlinkSnapshot snapshot;
  snapshot.gpus.reserve(gpu_records.size());
  for (const auto& [gpu_uuid, record] : gpu_records) {
    snapshot.gpus.push_back(record);
  }

  snapshot.edges.reserve(edge_records.size());
  for (const auto& [key, edge] : edge_records) {
    snapshot.edges.push_back(
        NvlinkEdge{edge.src_gpu_uuid, edge.dst_gpu_uuid, edge.link_count, edge.bandwidth_hint_gbps});
  }

  sort_snapshot(&snapshot);
  return snapshot;
}

absl::StatusOr<NvlinkSnapshot> parse_nvlink_runtime_probe_outputs(
    const std::string& gpu_query_output,
    const std::string& topology_matrix_output,
    NvlinkSnapshotOptions options) {
  auto gpus_or = parse_runtime_gpu_query_output(gpu_query_output, options.strict);
  if (!gpus_or.ok()) {
    return gpus_or.status();
  }
  auto matrix_or = parse_runtime_topology_matrix_output(topology_matrix_output, options.strict);
  if (!matrix_or.ok()) {
    return matrix_or.status();
  }

  NvlinkSnapshot snapshot;
  snapshot.gpus = std::move(gpus_or).value();

  std::vector<NvlinkGpuRecord> gpus_by_index = snapshot.gpus;
  std::sort(gpus_by_index.begin(), gpus_by_index.end(), [](const NvlinkGpuRecord& a, const NvlinkGpuRecord& b) {
    return a.gpu_index < b.gpu_index;
  });

  absl::flat_hash_map<std::string, EdgeAccumulator> edge_records;
  const TopologyMatrix& matrix = matrix_or.value();

  for (size_t i = 0; i < gpus_by_index.size(); ++i) {
    for (size_t j = i + 1; j < gpus_by_index.size(); ++j) {
      const NvlinkGpuRecord& src_gpu = gpus_by_index[i];
      const NvlinkGpuRecord& dst_gpu = gpus_by_index[j];
      const std::string src_label = gpu_label_from_index(src_gpu.gpu_index);
      const std::string dst_label = gpu_label_from_index(dst_gpu.gpu_index);

      const std::optional<std::string> forward_token = lookup_matrix_cell(matrix, src_label, dst_label);
      const std::optional<std::string> reverse_token = lookup_matrix_cell(matrix, dst_label, src_label);

      if (!forward_token.has_value() && !reverse_token.has_value()) {
        absl::Status parse_status = handle_parse_error(
            absl::InvalidArgumentError(
                absl::StrCat("runtime topology matrix has no cell for GPU pair ", src_label, " <-> ", dst_label)),
            options.strict,
            "NVLINK runtime topology pair");
        if (!parse_status.ok()) {
          return parse_status;
        }
        continue;
      }

      int forward_count = 0;
      if (forward_token.has_value()) {
        auto count_or = parse_nvlink_count_token(*forward_token, src_label, dst_label);
        if (!count_or.ok()) {
          absl::Status parse_status =
              handle_parse_error(count_or.status(), options.strict, "NVLINK runtime topology token");
          if (!parse_status.ok()) {
            return parse_status;
          }
        } else {
          forward_count = count_or.value();
        }
      }

      int reverse_count = 0;
      if (reverse_token.has_value()) {
        auto count_or = parse_nvlink_count_token(*reverse_token, dst_label, src_label);
        if (!count_or.ok()) {
          absl::Status parse_status =
              handle_parse_error(count_or.status(), options.strict, "NVLINK runtime topology token");
          if (!parse_status.ok()) {
            return parse_status;
          }
        } else {
          reverse_count = count_or.value();
        }
      }

      if (forward_count > 0 && reverse_count > 0 && forward_count != reverse_count) {
        absl::Status parse_status = handle_parse_error(
            absl::InvalidArgumentError(
                absl::StrCat(
                    "runtime topology matrix mismatch for ",
                    src_label,
                    " <-> ",
                    dst_label,
                    " (forward=",
                    forward_count,
                    ", reverse=",
                    reverse_count,
                    ")")),
            options.strict,
            "NVLINK runtime topology pair");
        if (!parse_status.ok()) {
          return parse_status;
        }
      }

      const int final_link_count = std::max(forward_count, reverse_count);
      if (final_link_count <= 0) {
        continue;
      }

      const std::string edge_key = make_edge_key(src_gpu.gpu_uuid, dst_gpu.gpu_uuid);
      auto edge_it = edge_records.find(edge_key);
      if (edge_it == edge_records.end()) {
        EdgeAccumulator edge;
        if (src_gpu.gpu_uuid <= dst_gpu.gpu_uuid) {
          edge.src_gpu_uuid = src_gpu.gpu_uuid;
          edge.dst_gpu_uuid = dst_gpu.gpu_uuid;
        } else {
          edge.src_gpu_uuid = dst_gpu.gpu_uuid;
          edge.dst_gpu_uuid = src_gpu.gpu_uuid;
        }
        edge.link_count = final_link_count;
        edge_records.emplace(edge_key, std::move(edge));
      } else {
        edge_it->second.link_count = std::max(edge_it->second.link_count, final_link_count);
      }
    }
  }

  snapshot.edges.reserve(edge_records.size());
  for (const auto& [edge_key, edge] : edge_records) {
    snapshot.edges.push_back(
        NvlinkEdge{edge.src_gpu_uuid, edge.dst_gpu_uuid, edge.link_count, edge.bandwidth_hint_gbps});
  }

  sort_snapshot(&snapshot);
  return snapshot;
}

absl::StatusOr<NvlinkSnapshot> load_nvlink_runtime_probe(NvlinkRuntimeProbeOptions options) {
  std::string gpu_query_output;
  if (!options.gpu_query_output_override.empty()) {
    gpu_query_output = options.gpu_query_output_override;
  } else {
    if (options.gpu_query_command.empty()) {
      return absl::InvalidArgumentError("gpu_query_command must not be empty");
    }
    auto query_or = run_command_capture_stdout(options.gpu_query_command);
    if (!query_or.ok()) {
      return absl::Status(
          query_or.status().code(),
          absl::StrCat("failed to run NVLINK runtime GPU query: ", query_or.status().message()));
    }
    gpu_query_output = std::move(query_or).value();
  }

  std::string topology_matrix_output;
  if (!options.topology_matrix_output_override.empty()) {
    topology_matrix_output = options.topology_matrix_output_override;
  } else {
    if (options.topology_matrix_command.empty()) {
      return absl::InvalidArgumentError("topology_matrix_command must not be empty");
    }
    auto topology_or = run_command_capture_stdout(options.topology_matrix_command);
    if (!topology_or.ok()) {
      return absl::Status(
          topology_or.status().code(),
          absl::StrCat("failed to run NVLINK runtime topology query: ", topology_or.status().message()));
    }
    topology_matrix_output = std::move(topology_or).value();
  }

  return parse_nvlink_runtime_probe_outputs(
      gpu_query_output, topology_matrix_output, NvlinkSnapshotOptions{.strict = options.strict});
}

} // namespace tensorcast::communicator::topology::discovery
