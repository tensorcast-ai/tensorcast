// Copyright (c) 2025, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#include "aligned_buffer.h"

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace tensorcast::checkpoint {

AlignedBuffer::AlignedBuffer(const std::string& filename)
    : fd_(-1), buf_size_(kBufferSize), buf_pos_(0), file_offset_(0), direct_io_(true) {
  // First attempt using O_DIRECT. Capture the errno in case it fails so we can
  // report it alongside any subsequent fallback failure.
  fd_ = open(filename.c_str(), O_WRONLY | O_CREAT | O_DIRECT | O_TRUNC, 0644);

  int direct_errno = 0; // 0 means the first attempt succeeded.
  if (fd_ < 0) {
    direct_errno = errno; // Save errno from the O_DIRECT attempt.

    // Fallback without O_DIRECT so that environments (e.g.
    // tmpfs or certain FUSE mounts) that do not support direct I/O can still
    // run the code path.
    fd_ = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if (fd_ >= 0) {
      // Fallback succeeded – let the rest of the class know we are using the
      // buffered I/O path.
      direct_io_ = false;

      // Log the first failure so that operators are aware that O_DIRECT is not
      // supported for this path. Use PLOG so the message automatically
      // contains a textual description of `errno`.
      errno = direct_errno; // Ensure PLOG reports the original errno value.
      PLOG(WARNING) << "Opening file " << filename << " with O_DIRECT failed; fallback without O_DIRECT succeeded.";
    } else {
      // Both attempts failed – report the fallback errno via PLOG and include
      // the first attempt's errno code for context.
      PLOG(ERROR) << "Failed to open file " << filename
                  << " even without O_DIRECT. First attempt errno=" << direct_errno;
    }
  }

  buffer_ = aligned_alloc(kAlignment, kBufferSize);
}

AlignedBuffer::~AlignedBuffer() {
  // Flush any remaining data, padding so that O_DIRECT constraints are met.
  if (buffer_ && buf_pos_) {
    flush_buffer(/*pad_to_alignment=*/true);
  }

  if (buffer_) {
    free(buffer_);
  }

  if (fd_ >= 0) {
    // Ensure all data is written to disk
    fsync(fd_);
    close(fd_);
  }
}

size_t AlignedBuffer::write_data(const void* data, size_t size) {
  size_t written = 0;
  while (written < size) {
    // if data size is larger than buffer size and buffer is empty
    // write data directly to file
    if (size - written > buf_size_ && buf_pos_ == 0) {
      const size_t direct_write_size = (size - written) / kAlignment * kAlignment;

      // allocate aligned memory
      void* direct_write_buf = aligned_alloc(kAlignment, direct_write_size);
      if (!direct_write_buf) {
        PLOG(ERROR) << "Failed to allocate aligned memory: kAlignment=" << kAlignment
                    << " direct_write_size=" << direct_write_size;
        exit(1);
      }
      memcpy(direct_write_buf, (char*)data + written, direct_write_size);
      const ssize_t ret = pwrite(fd_, direct_write_buf, direct_write_size, file_offset_);
      PCHECK(ret >= 0) << "Failed to write directly to file";
      CHECK(static_cast<size_t>(ret) == direct_write_size)
          << "Short write when attempting to write " << direct_write_size << " bytes (wrote " << ret << ")";
      written += direct_write_size;
      file_offset_ += direct_write_size;
      free(direct_write_buf);
    }
    const size_t to_write = std::min(size - written, buf_size_ - buf_pos_);
    memcpy(static_cast<char*>(buffer_) + buf_pos_, (char*)data + written, to_write);
    buf_pos_ += to_write;
    written += to_write;

    if (buf_pos_ == buf_size_) {
      flush_buffer(/*pad_to_alignment=*/false);
    }
  }
  return written;
}

size_t AlignedBuffer::write_padding(size_t padding_size) {
  if (padding_size >= 8) {
    LOG(ERROR) << "Padding size should be less than 8 bytes";
    return 0;
  }
  if (padding_size == 0) {
    return 0; // Nothing to do.
  }

  // Zero-initialise the padding region so we do not leak residual bytes from
  // the previous write into the on-disk representation.  This is particularly
  // important when using O_DIRECT where the kernel bypasses the page cache and
  // therefore does not clear the tail.
  std::memset(static_cast<char*>(buffer_) + buf_pos_, 0, padding_size);

  buf_pos_ += padding_size;
  if (buf_pos_ > buf_size_) {
    LOG(ERROR) << "Padding size is too large";
    return 0;
  }
  if (buf_pos_ == buf_size_) {
    // The internal buffer is now full – flush it to disk.  We can pass
    // pad_to_alignment=false here because the buffer size (1 GiB) already
    // satisfies O_DIRECT alignment constraints.
    flush_buffer(/*pad_to_alignment=*/false);
  }
  return padding_size;
}

void AlignedBuffer::flush_buffer(bool pad_to_alignment) {
  if (buf_pos_ == 0) {
    return; // Nothing to do.
  }

  // If we are using O_DIRECT then both offset & length must be multiples of
  // kAlignment.  Optionally pad the tail with zeros to satisfy that.
  if (direct_io_ && pad_to_alignment) {
    const size_t remainder = buf_pos_ % kAlignment;
    if (remainder != 0) {
      const size_t pad = kAlignment - remainder;
      std::memset(static_cast<char*>(buffer_) + buf_pos_, 0, pad);
      buf_pos_ += pad;
    }
  }

  // At this point buf_pos_ is either buf_size_ (1 GiB) or <= buf_size_ but a
  // multiple of kAlignment.
  const ssize_t ret = pwrite(fd_, buffer_, buf_pos_, file_offset_);
  PCHECK(ret >= 0) << "Failed to write buffer to file";
  CHECK(static_cast<size_t>(ret) == buf_pos_)
      << "Short write when flushing buffer: expected " << buf_pos_ << " bytes, wrote " << ret;

  file_offset_ += buf_pos_;
  buf_pos_ = 0;
}

} // namespace tensorcast::checkpoint
