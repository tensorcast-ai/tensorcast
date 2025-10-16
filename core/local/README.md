# Core/Local 目录说明

本目录包含本地开发和测试相关的代码，包括实现、测试和基准测试。

## 目录结构

```
core/local/
├── chunk/        # 数据块相关实现（DataChunk）
├── loader/       # 数据加载器实现
├── test/         # 单元测试
└── benchmark/    # 性能基准测试
```

## 构建选项

### 构建实现库

```bash
# 构建所有实现库
bazel build //core/local/chunk:data_chunk_lib //core/local/loader:disk_chunk_loader

# 单独构建某个库
bazel build //core/local/chunk:data_chunk_lib
bazel build //core/local/loader:disk_chunk_loader
```

### 运行单元测试

```bash
# 运行所有单元测试
bazel test //core/local/test:...

# 运行特定测试
bazel test //core/local/test:data_chunk_test
bazel test //core/local/test:disk_chunk_loader_test
```

### 构建和运行基准测试

```bash
# 构建基准测试二进制文件
bazel build //core/local/benchmark:data_chunk_benchmark

# 运行基准测试
./bazel-bin/core/local/benchmark/data_chunk_benchmark \
    <output_file> <file_size_gib> <chunk_size_mib>
```

### 基准测试性能优化

基准测试使用以下优化选项：
- `-O3`: 最高优化级别
- `-march=native`: 针对本地 CPU 架构优化
- `-mtune=native`: 针对本地 CPU 调优

这些选项仅应用于基准测试，不会影响单元测试或生产代码。

## 使用建议

1. **开发阶段**: 运行单元测试确保功能正确
   ```bash
   bazel test //core/local/test:...
   ```

2. **性能分析**: 在需要性能调优时运行基准测试
   ```bash
   bazel build //core/local/benchmark:data_chunk_benchmark
   ```

3. **CI/CD**: 在 CI 中只运行单元测试（不运行 benchmark，因其运行较慢）
   ```bash
   bazel test //core/local/test:... --test_timeout=300
   ```

## 注意事项

- 基准测试可能需要较长时间运行，通常不会在常规 CI 流程中执行
- benchmark 使用 `-O3` 优化，可能会增加构建时间
- 单元测试和 benchmark 是独立的，可以分别构建和运行
