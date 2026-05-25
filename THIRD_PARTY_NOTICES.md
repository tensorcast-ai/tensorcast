# Third-Party Notices

TensorCast-owned code is licensed under MIT unless otherwise noted. Portions derived from ServerlessLLM remain licensed under Apache-2.0. Third-party dependencies are licensed under their respective licenses.

## ServerlessLLM-Derived Code

TensorCast includes files derived from ServerlessLLM:

- `core/checkpoint/aligned_buffer.h`
- `core/checkpoint/aligned_buffer.cc`
- `core/checkpoint/checkpoint.h`
- `core/checkpoint/checkpoint.cc`
- `core/checkpoint/progress_bar.h`
- `core/checkpoint/tensor_writer.h`
- `core/checkpoint/tensor_writer.cc`
- `core/common/memory/cuda_memory.cc`
- `core/common/memory/pinned_buffer_pool.h`
- `core/common/memory/pinned_buffer_pool.cc`
- `core/common/memory/pinned_memory.h`
- `core/common/memory/pinned_memory.cc`

These files retain the upstream ServerlessLLM Apache-2.0 license header:

ServerlessLLM
Copyright (c) ServerlessLLM Team 2024

They have been modified by TensorCast Team, 2025-2026.

## jemalloc Development Header

`third_party/dev_includes/jemalloc/jemalloc.h` is a generated development header
derived from jemalloc 5.3.0 public headers.

Unless otherwise specified, jemalloc source files are distributed under the
following BSD-style license:

Copyright (C) 2002-2022 Jason Evans <jasone@canonware.com>.
All rights reserved.
Copyright (C) 2007-2012 Mozilla Foundation. All rights reserved.
Copyright (C) 2009-2022 Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER(S) BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.

## clang-format-tidy Configuration

`.clang-tidy` is derived from the public `acodcha/clang-format-tidy`
configuration and retains its upstream MIT license notice in the file.

Copyright (c) 2023-2025 Alexandre Coderre-Chabot.

## Major Dependencies

TensorCast depends on third-party projects including PyTorch, CUDA/NVIDIA
components, gRPC, Protocol Buffers, Folly, Abseil, DuckDB, FastAPI,
OpenTelemetry, Catch2, yaml-cpp, Bazel rules, and jemalloc. Those dependencies
are licensed by their respective copyright holders.
