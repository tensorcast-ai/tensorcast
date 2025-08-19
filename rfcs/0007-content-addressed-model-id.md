## RFC-0007 内容寻址 Model ID（精简版）

### 1. 背景与目标

- 问题：现有 `model_id` 依赖外部命名（路径/来源），同一内容在磁盘/内存/P2P 下难以聚合与稳定寻址。
- 目标：采用“内容寻址”统一标识。以结构指纹 + 数据指纹生成稳定 `model_id`，让磁盘、内存、P2P 多源在同一 ID 下聚合与路由。
- 成功标准：
  - 相同内容在不同来源得到一致 `model_id`。
  - 加载与 P2P 快路径保持毫秒级校验（KEY_POINTS/SEGMENT）。
  - 计算/注册阶段开销可控（GPU 优先、CPU 回退）。

### 2. 方案总览

- **ID 规范（mi2）**：`model_id = "mi2:" + index_multihash + ":" + data_multihash`（Multihash + Multibase32，自描述算法，默认 `sha2-256`）。
- **结构指纹**：对“规范化 Index 字节”（Canonical Index）做 Multihash，得 `index_multihash`。
- **数据指纹**：对“规范化线性数据流”执行树形哈希，取根后再做 Multihash，得 `data_multihash`。
- **统一路由**：Global Store 以 `model_id` 为主键聚合副本；Local Store 根据 `model_id` 选择最优来源（内存 > 本地磁盘 > 远端）。

### 3. 规范（必须遵循）

#### 3.1 Model ID 格式

- `model_id = "mi2:" + index_multihash + ":" + data_multihash`
- Multihash：默认 `sha2-256`；支持后续算法演进（自描述）。
- Multibase：默认 base32。

#### 3.2 Canonical Index（结构指纹输入）

- 编码：优先 Canonical CBOR；过渡期允许严格 Canonical JSON（键有序，字段固定）。
- 外层键：`tensor_name` 严格升序。
- 记录字段顺序（固定）：`offset,size,shape,stride,dtype,storage_offset`。
- 8 字节对齐：与 RFC-0006 一致。
- 生成位置：统一在 C++ 核心实现生成（避免跨语言差异）。

稳定分组与布局（替代“按指针分组”的不稳定排序）：

- 分组键：`(dtype_code, device_id, group_key)`，其中 `group_key = H(sorted(tensor_names_in_group))`，`H` 与 `index_multihash` 同算法。
- 组间升序、组内 `tensor_name` 升序；保持 8B 对齐。

#### 3.3 规范化线性数据流（数据指纹输入）

- 定义：按 Canonical Index 描述的 coalesced 布局，在逻辑区间 `[0, total_size)` 顺序读取的字节序列。
- 读取来源（统一抽象为 `SeekableSource`）：
  - 磁盘：分片按文件名排序顺序读取（`FilePartitionSource`）。
  - P2P：`RemoteKeySource` 流式读取。
  - 内存：Begin 阶段分配的连续显存（coalesced）直接读取（GPU 优先）。

#### 3.4 指纹计算

- `index_multihash = MULTIHASH(canonical_index_bytes)`。
- `data_multihash = MULTIHASH(TREE_HASH(stream_bytes(0..total_size)))`。
- 树形哈希：分块 1–16 MiB，`sha2-256` 叶/根；GPU 并行优先、CPU/PCIe 回退。
- 运行期校验：加载/P2P 使用 KEY_POINTS/SEGMENT；FULL 校验作为可选。

### 4. 系统集成

#### 4.1 Global Store（以 `model_id` 为主键）

- 表：
  - `models(model_id PK, index_multihash, data_multihash, schema_version, encoding, hash_params_json, created_at, ...)`
  - `model_indices(index_multihash PK, schema_version, encoding, size_bytes, index_data BLOB, created_at, ...)`
  - `model_replicas(id PK, model_id FK, source_type ENUM('DISK','MEMORY','P2P'), location|disk_path, device_id, created_at, ...)`
- RPC：`RegisterModelReplica(model_id, ...)`、`GetModelInfoById(model_id)`、`GetModelIndex(index_key)`。

#### 4.2 Local Store / StoreDaemon

- Begin：创建仅内存模型、分配显存、返回 `registration_id + cuda_ipc_handle`。
- Commit：在守护进程/核心侧计算 `index_multihash` 与 `data_multihash`，拼装 `mi2` 并返回 `ModelDescriptor`；同时向 GS 注册副本。
- Prepare（统一入口）：`prepare(DeviceKey, mode, hints)` 支持 `hints.model_id`（内容寻址）与 `hints.disk_path`（显式磁盘）。禁止将 `mi2:` 当作路径。

#### 4.3 Python API

- `register_tensor_dict(state_dict, ...) -> (state_dict, commit_info)`；`commit_info.model_id` 以 `mi2:` 开头。
- 辅助：`generate_model_id_from_state_dict(...)`、`generate_model_id_from_path(...)`（便于审计与迁移）。

### 5. ModelDescriptor（Commit 返回）

- 字段（最小集）：
  - `model_id`（`mi2:...`）、`index_multihash`、`data_multihash`
  - `schema_version`、`encoding`（建议 `cbor`）
  - `total_size`、`hash_params`（如 `chunk_size`、`fanout`）

- 落盘文件：`model_descriptor.json`
  - 保存端（`save_dict`/`save_tensors`）必须在目标目录写入 `model_descriptor.json`，与 `tensor_index.(cbor|json)` 一并输出（推荐 `cbor`）。
  - 该规则适用于标准数据格式（参见“Data File Format Specification”）与 `safetensors` 目录结构。

### 6. 实施计划（阶段）

- A：Canonical Index（CBOR）与稳定分组/布局；定义线性空间。
- B：树形哈希与 Multihash 封装；GPU 优先、CPU 回退。
- C：Commit 返回 `ModelDescriptor`；Python/Proto 对齐。
- D：GS/DB：以 `model_id` 为主键；`model_replicas.model_id` 外键；RPC 切换。
- E：CheckpointStore：`prepare(..., model_id=...)` 基于 GS 路由；运行期轻校验；迁移工具回填 ID。

### 7. 当前状态（代码同步 2025-08-19）

- 已完成（与本仓库当前实现一致）：
  - C++（哈希与管线）：
    - 新增 `core/store/loader/source_hash.{h,cc}`：基于 `SeekableSource` 的统一树形哈希（4MiB 叶 → Merkle 根 → multibase32 multihash）。
    - 新增 `core/store/loader/disk_dir_hash.{h,cc}`：标准分片目录（`tensor.data*`）的数据 multihash，通过 `FilePartitionSource` 调用上述统一管线。
    - `core/common/model_hash.{h,cc}` 现仅保留 GPU Buffer 的数据哈希（`compute_data_multihash_from_gpu(...)`）与 Index multihash（`compute_index_multihash(...)`）；磁盘目录哈希下沉至 loader 层。
  - PyBind：新增统一接口（`scstore/_C`）
    - `save_model_to_disk(tensor_names, tensor_data, meta_state_dict, path, config) -> descriptor`
    - `inspect_or_generate_descriptor(path) -> descriptor`
  - Python：
    - `save_dict(...)` 改为调用 `save_model_to_disk(...)`，一次性写出 `tensor_index.(cbor|json)` + `model_descriptor.json`，并返回 `descriptor`；移除了 Python 侧 multihash/树形哈希与 base32 编码等冗余逻辑。
    - `load_dict(...)` 在加载完成后调用 `inspect_or_generate_descriptor(...)`，若缺失则计算并补写 `model_descriptor.json`；保持 KEY_POINTS/SEGMENT 的后台校验逻辑。
    - 删除了 `load_dict_pure_local(...)`，各单测改为直接调用 `load_dict(...)`。
  - 工具与单测：
    - `scstore/tools/backfill_descriptor.py` 切换为使用 `_C.inspect_or_generate_descriptor`，不再在 Python 侧重算哈希；
    - `core/testing/common.cc` 以及相关 C++ 单测调用统一的 loader 管线（`loader::compute_data_multihash_from_disk_dir`）。
    - 相关示例与测试（`examples/*`、`tests/python/*`）已同步至新接口。

- 待完成（下一步 TODO）：
  - Canonical Index（C++ 权威）：
    - 在 C++ 侧统一生成 Canonical Index（优先 CBOR，短期可保留严格 Canonical JSON），并实现“稳定分组 + 8B 对齐”；完全移除 Python 侧的 index 组装与排序。
  - 统一 `SeekableSource` 与加载内核：
    - 已完成：磁盘分片的数据哈希统一到 `SeekableSource` 管线；
    - 待完成：P2P 远端与内存（GPU/CPU）来源补齐 `SeekableSource` 适配，保存/加载/哈希共享同一流水线（chunk 化、pinned 环形缓冲、多线程）。
  - 标准分片门禁与强校验（DiskLoader）：
    - 已实现门禁：标准分片目录必须包含 `model_descriptor.json` 与 `tensor_index.(cbor|json)`；缺失返回 `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`（`DiskLoader::initialize`）。
  - Commit/Prepare 路径细化：
    - 在 Commit 内部统一 Canonical Index 与 `mi2` 生成（现已部分落地）；AUTO 路由策略细化与 GS 侧去重/聚合完善；严格校验开关（FULL/Merkle 证明）。
  - safetensors 规范化 Index（C++）：
    - 完成 `BuildCanonicalIndexFromSafetensors(...)`（外层键升序、字段顺序固定、`storage_offset=0`、row-major stride），用于缺失描述符与 index 的目录回填。
  - GPU 并行树形哈希：
    - 将 `compute_data_multihash_from_gpu(...)` 升级为 GPU 并行树形归约（多流/多核），提供显著吞吐；保留 CPU 路径作为回退。
  - 文档/示例与 CLI：
    - 统一示例、CLI 与文档到 `save_model_to_disk`/`inspect_or_generate_descriptor` 与 `prepare(..., model_id=..., disk_path=...)` 的新路径；新增回填工具说明与错误码表。

### 8. 关键取舍

- 仅用 `tensor_index_key` 无法区分“同布局不同数据”，不适合作为唯一内容标识。
- 运行期继续采用 KEY_POINTS/SEGMENT，`data_multihash` 仅在保存/注册/Commit 计算。
- 稳定分组排序必须与环境无关；否则将破坏跨来源一致性。

### 9. 风险与缓解

- 历史模型排序不稳定：视为不同 `model_id` 并存；提供批量规范化与回填工具。
- Commit 开销：支持 GPU 优先/CPU 回退与异步回填；仅在需要强一致时阻塞。
- 负 stride/混合 dtype/多设备：由 v2 语义覆盖；稳定分组与 8B 对齐为硬约束。

### 10. 示例

```python
from scstore.torch_util import register_tensor_dict

sd, info = register_tensor_dict(model.state_dict(), "name", device_id=0)
assert info["model_id"].startswith("mi2:")
```

    ```python
# 加载（优先内容寻址）
store.prepare(device_key, mode="AUTO", model_id=info["model_id"])  # GS 路由
```

### 11. 与验证系统的一致性

- `data_multihash` 是唯一标识的组成，仅在保存/注册/Commit 计算；
- 运行期默认 KEY_POINTS/SEGMENT；必要时可启用 FULL 校验或 Merkle 部分证明。

### 12. 兼容性与迁移

- 新老并存期：`disk_path` 可显式触发磁盘加载；`mi2:` 仅作内容寻址 ID，不作路径。
- 标准数据格式目录：加载时必须已存在 `model_descriptor.json` 与 `tensor_index.(cbor|json)`；历史目录需先通过迁移工具生成后再加载。
- `safetensors` 目录：若存在 `model_descriptor.json` 则以其为准并校验一致性；若缺失，允许在完成加载后计算并写回 `model_descriptor.json` 与 Canonical Index（具备写权限时）。


### 13. 磁盘保存与加载（disk_loader）规范

#### 13.1 保存端（save_dict/save_tensors）

- 必须输出：
  - `tensor_index.cbor`（或严格 Canonical JSON：`tensor_index.json`，过渡期）
  - `model_descriptor.json`（含 `model_id`、`index_multihash`、`data_multihash`、`schema_version`、`encoding`、`total_size`、`hash_params`）
- 标准数据格式（参见 `web-docs/.../data-format.md`）：按规范写入分区数据与索引，目录内必须包含上述两个文件。
- `safetensors`：若保存到 `safetensors` 目录，同样在该目录写入 `model_descriptor.json`；如存在多文件负载，索引/描述符依然位于目录根。

#### 13.2 加载端（DiskLoader）

- 标准分片（参见“Data File Format Specification”）
  - 目录要求（目标行为）：必须存在 `model_descriptor.json` 与 `tensor_index.(cbor|json)`；缺失时返回 `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`。
  - 当前实现（过渡期）：若缺失 `model_descriptor.json`，允许在“完成加载后”计算并补写；后续将切换为强制门禁。
  - 加载流程：按文件名排序读取分片，建立线性数据流并完成内存映射。
  - 加载后验证（确保 ID 与数据对齐）：
    - 立即执行轻量校验（KEY_POINTS/SEGMENT），与描述符内信息一致；
    - 当启用严格校验时（当前实现：`LoadingHints::verify == FULL_DIGEST`），计算/或以 Merkle 证明方式验证树形哈希根，确保与 `data_multihash` 完全一致；不一致返回 `DataCorruption(MODEL_ID_MISMATCH)`。
  - 成功后：若 GS 可用，按 `model_id` 注册/更新本地副本信息。

- `safetensors` 目录
  - 若存在 `model_descriptor.json`：将其视为权威 `model_id` 来源；完成加载后按上述策略校验；不一致返回 `DataCorruption(MODEL_ID_MISMATCH)`。
  - 若不存在 `model_descriptor.json`：
    - 完成加载后，基于 Canonical Index 与线性数据流计算 `index_multihash`/`data_multihash`，生成 `model_id`；
    - 将 `model_descriptor.json` 写入目录（无法写入返回 `PermissionDenied(DESCRIPTOR_NOT_WRITABLE)`），并可选持久化 Canonical Index；
    - 若 GS 可用，以新生成的 `model_id` 注册副本。

#### 13.3 统一约束与错误

- 任何情况下，禁止将 `mi2:` 当作磁盘路径解释。
- 错误码建议：
  - `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`：标准数据格式加载缺失描述符
  - `DataCorruption(MODEL_ID_MISMATCH)`：加载后校验与 `model_id` 不一致
  - `PermissionDenied(DESCRIPTOR_NOT_WRITABLE)`：无法生成/落盘描述符


### 14. 代码级实施计划（与当前仓库对齐）

- 保存端（Python）（已完成）
  - 文件：`scstore/torch_util.py`
    - `save_dict(...)`：
      - 使用 `_canonical_index_bytes_from_mapping(...)` 生成规范化 index 字节；仍写 `tensor_index.json`，但生成 bytes 后再落盘以保证稳定序列化。
      - 基于 `tensor_index` 计算逻辑 `total_size = max(offset+size)`。
      - 标准分片：用 `_collect_partition_segments_for_stream(model_dir, actual_size=total_size)` 收集段；调用 `_compute_data_multihash_from_segments(...)` 得 `data_mh`。
      - `index_mh = _to_multibase_multihash_sha256(_sha256(index_bytes))`；`_write_model_descriptor(...)` 写 `model_descriptor.json`。
    - safetensors（如存在该保存路径）：
      - 用 `_collect_safetensors_segments_for_stream(...)` 计算 `data_mh`；
      - 由 `build_indices_from_safetensors(...)` 得映射 → `_canonical_index_bytes_from_mapping(...)` → `index_mh`；
      - 写 `model_descriptor.json` 至目录根。

- 磁盘加载（C++）（已完成）
  - 文件：`core/store/loader/disk_loader.cc`/`disk_loader.h`
    - `DiskLoader::initialize()`：
      - 标准分片：强制存在 `model_descriptor.json` + `tensor_index.(json|cbor)`；缺失返回 `FailedPrecondition(MODEL_DESCRIPTOR_REQUIRED)`；存在则读取并缓存描述符。
      - safetensors：允许缺失 `model_descriptor.json`（后续回填）。
  - 文件：`core/store/checkpoint_store.cc`
    - `CheckpointStore::load_from_disk_internal(...)`（已在等待 `LOADED` 之后构建 `ModelHandle`）：
      - 获取 GPU 基址与大小，调用 `model_hash::compute_data_multihash_from_gpu(...)` 得 `data_mh`。
      - 标准分片：读取 `tensor_index.(json|cbor)` 生成 canonical 字节，`compute_index_multihash(index_bytes, "")` 得 `index_mh`。
      - safetensors：
        - 若存在 `model_descriptor.json`：直接使用其中 `index_multihash`。
        - 若缺失：调用下述新增 API 构建 v2 规范化 index 字节，计算 `index_mh` 并写回 `model_descriptor.json`（不可写报 `PermissionDenied`）。
      - 若有描述符：强校验 `descriptor.data_multihash == data_mh`；不一致报 `DataCorruption(MODEL_ID_MISMATCH)`；可附加 KEY_POINTS/SEGMENT 快速校验。
      - 若 GS 可用：按 `model_id` 注册/更新本地副本信息。

- safetensors 规范化 Index（C++ 新增）（已完成）
  - 文件：`core/store/loader/safetensors_util.{h,cc}`
    - 新增：`absl::StatusOr<std::string> BuildCanonicalIndexFromSafetensors(const std::vector<std::filesystem::path>& files);`
      - 解析 header JSON，构造 v2 规范映射（外层键升序、字段顺序固定、`storage_offset=0`、row-major stride），序列化为严格 canonical JSON 字节。

### 15. 测试计划

- C++：
  - 标准分片目录缺少 `model_descriptor.json` → `DiskLoader::initialize()` 返回 `FailedPrecondition`。
  - 标准分片存在描述符且匹配 → `load_from_disk_internal()` 成功；篡改数据或描述符 → `MODEL_ID_MISMATCH`。
  - safetensors：有/无描述符两路（生成/校验/报错）。
- Python：
  - `save_dict` 生成 canonical `tensor_index.json` + `model_descriptor.json`；断言 `mi2:` 与 `total_size` 为逻辑大小。

### 16. 迁移与交付

- 迁移工具：批量为存量目录生成 `model_descriptor.json`（标准分片与 safetensors）。
- 交付顺序：
  - 批次 1：保存端写描述符 + 标准分片加载门禁；
  - 批次 2：加载后 ID 校验 + safetensors 回填；
  - 批次 3：迁移工具 + GS 注册细化与端到端回归。

