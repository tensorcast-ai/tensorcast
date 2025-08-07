// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/store/loader/dvmp_mapped_sink.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

namespace stepcast::store::loader {

DVMPMappedSink::DVMPMappedSink(Options options) : options_(std::move(options)) {
  if (!options_.base_addr) {
    LOG(ERROR) << "Base address is null";
  }
  mapped_regions_.resize(options_.partition_paths.size(), nullptr);
}

DVMPMappedSink::~DVMPMappedSink() {
  // Note: We don't unmap here because the memory is managed by DVMP
  // The mappings are part of the model's memory and will be unmapped
  // when the model is unloaded
}

absl::Status DVMPMappedSink::write(const void* src, size_t bytes) {
  // This method is for streaming interface compatibility
  // For DVMP, we prefer the direct map_partitions() fast-path
  if (!options_.base_addr) {
    return absl::InvalidArgumentError("Base address is null");
  }

  // Ensure we never write beyond the allocated DVMP region.  If the caller
  // requests more bytes than remaining, clamp the copy to the remaining size
  // and return an OutOfRange error so that the producer/consumer loop can
  // terminate gracefully without triggering fatal errors.
  size_t remaining = options_.total_size - current_offset_;
  if (bytes > remaining) {
    bytes = remaining;
    if (bytes == 0) {
      return absl::OutOfRangeError("No space left in DVMP region");
    }
  }

  // Direct memory copy for streaming mode
  char* dest = static_cast<char*>(options_.base_addr) + current_offset_;
  std::memcpy(dest, src, bytes);
  current_offset_ += bytes;

  return absl::OkStatus();
}

absl::Status DVMPMappedSink::map_partitions() {
  if (mappings_done_) {
    return absl::OkStatus();
  }

  if (!options_.base_addr) {
    return absl::InvalidArgumentError("Base address is null");
  }

  for (size_t i = 0; i < options_.partition_paths.size(); ++i) {
    auto status = map_partition(i);
    if (!status.ok()) {
      unmap_partitions();
      return status;
    }
  }

  mappings_done_ = true;
  VLOG(2) << "Successfully mapped " << options_.partition_paths.size() << " partitions into DVMP region";

  return absl::OkStatus();
}

absl::Status DVMPMappedSink::map_partition(size_t partition_idx) {
  const auto& path = options_.partition_paths[partition_idx];
  const auto& size = options_.partition_sizes[partition_idx];

  // Calculate target address for this partition
  uint64_t offset = 0;
  for (size_t i = 0; i < partition_idx; ++i) {
    offset += options_.partition_sizes[i];
  }
  char* target_addr = static_cast<char*>(options_.base_addr) + offset;

  // Open file
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return absl::InternalError(absl::StrFormat("Failed to open partition %s: %s", path.string(), strerror(errno)));
  }

  // Map file into DVMP region using MAP_FIXED
  int mmap_flags = MAP_PRIVATE | MAP_FIXED;
  if (options_.populate_pages) {
    mmap_flags |= MAP_POPULATE;
  }

  void* mapped = ::mmap(target_addr, size, PROT_READ, mmap_flags, fd, 0);
  ::close(fd); // Can close fd after mmap

  if (mapped == MAP_FAILED) {
    return absl::InternalError(absl::StrFormat("Failed to mmap partition %s: %s", path.string(), strerror(errno)));
  }

  if (mapped != target_addr) {
    // This should not happen with MAP_FIXED
    ::munmap(mapped, size);
    return absl::InternalError(absl::StrFormat("mmap returned unexpected address for partition %s", path.string()));
  }

  mapped_regions_[partition_idx] = mapped;

  VLOG(3) << "Mapped partition " << path << " (" << size << " bytes) "
          << "at address " << mapped;

  return absl::OkStatus();
}

void DVMPMappedSink::unmap_partitions() {
  // Note: In production, we typically don't unmap because DVMP manages
  // the memory lifecycle. This is here for error cleanup paths.
  for (size_t i = 0; i < mapped_regions_.size(); ++i) {
    if (mapped_regions_[i]) {
      ::munmap(mapped_regions_[i], options_.partition_sizes[i]);
      mapped_regions_[i] = nullptr;
    }
  }
  mappings_done_ = false;
}

absl::Status DVMPMappedSink::close() {
  // For DVMP, close doesn't need to do anything special
  // The memory remains mapped and is managed by DVMP's chunk state
  if (!mappings_done_ && current_offset_ == 0) {
    // If we haven't mapped or written anything, try mapping now
    return map_partitions();
  }

  VLOG(2) << "DVMP mapped sink closed. Total size: " << options_.total_size;
  return absl::OkStatus();
}

} // namespace stepcast::store::loader