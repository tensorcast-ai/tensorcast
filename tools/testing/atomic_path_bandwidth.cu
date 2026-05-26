// Copyright (c) 2025-2026, TensorCast Team.
//
// Standalone microbenchmark for TensorCast cold-start copy-path triage.
// Build example:
//   /data/cuda/cuda-12.8/bin/nvcc -O3 -std=c++17 \
//     tools/testing/atomic_path_bandwidth.cu -o /tmp/tensorcast/atomic_path_bandwidth

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Args {
  std::string mode = "disk";
  std::string method = "mmap";
  std::string model_dir;
  std::string devices = "0";
  int threads = 1;
  int files = 1;
  int repeats = 1;
  size_t chunk_bytes = 256ULL * 1024 * 1024;
  uint64_t limit_bytes = 0;
  bool pinned = true;
  bool drop_cache = false;
  bool sync_per_chunk = true;
};

struct SafetensorsFile {
  std::filesystem::path path;
  uint64_t file_size = 0;
  uint64_t data_start = 0;
  uint64_t data_size = 0;
};

double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

void die(const std::string& msg) {
  std::cerr << "error: " << msg << "\n";
  std::exit(1);
}

uint64_t parse_size_arg(const std::string& s) {
  if (s.empty())
    return 0;
  char suffix = s.back();
  double value = 0.0;
  std::string number = s;
  if ((suffix < '0' || suffix > '9') && suffix != '.') {
    number = s.substr(0, s.size() - 1);
  } else {
    suffix = 'b';
  }
  value = std::stod(number);
  double mul = 1.0;
  switch (suffix) {
    case 'k':
    case 'K':
      mul = 1024.0;
      break;
    case 'm':
    case 'M':
      mul = 1024.0 * 1024.0;
      break;
    case 'g':
    case 'G':
      mul = 1024.0 * 1024.0 * 1024.0;
      break;
    default:
      mul = 1.0;
      break;
  }
  return static_cast<uint64_t>(value * mul);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc)
        die(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--mode")
      args.mode = need_value("--mode");
    else if (key == "--method")
      args.method = need_value("--method");
    else if (key == "--model-dir")
      args.model_dir = need_value("--model-dir");
    else if (key == "--devices")
      args.devices = need_value("--devices");
    else if (key == "--threads")
      args.threads = std::stoi(need_value("--threads"));
    else if (key == "--files")
      args.files = std::stoi(need_value("--files"));
    else if (key == "--repeats")
      args.repeats = std::stoi(need_value("--repeats"));
    else if (key == "--chunk")
      args.chunk_bytes = parse_size_arg(need_value("--chunk"));
    else if (key == "--limit")
      args.limit_bytes = parse_size_arg(need_value("--limit"));
    else if (key == "--pinned")
      args.pinned = std::stoi(need_value("--pinned")) != 0;
    else if (key == "--drop-cache")
      args.drop_cache = std::stoi(need_value("--drop-cache")) != 0;
    else if (key == "--sync-per-chunk")
      args.sync_per_chunk = std::stoi(need_value("--sync-per-chunk")) != 0;
    else
      die("unknown argument: " + key);
  }
  if (args.threads <= 0)
    die("--threads must be positive");
  if (args.files <= 0)
    die("--files must be positive");
  if (args.repeats <= 0)
    die("--repeats must be positive");
  if (args.chunk_bytes == 0)
    die("--chunk must be non-zero");
  return args;
}

void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    die(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

std::vector<int> parse_devices(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty())
      out.push_back(std::stoi(item));
  }
  if (out.empty())
    out.push_back(0);
  return out;
}

uint64_t file_size_of(const std::filesystem::path& p) {
  struct stat st{};
  if (::stat(p.c_str(), &st) != 0)
    die("stat failed for " + p.string() + ": " + std::strerror(errno));
  return static_cast<uint64_t>(st.st_size);
}

uint64_t read_u64_le(int fd) {
  uint8_t bytes[8];
  ssize_t got = ::pread(fd, bytes, sizeof(bytes), 0);
  if (got != static_cast<ssize_t>(sizeof(bytes)))
    die("failed to read safetensors header length");
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) {
    v = (v << 8) | bytes[i];
  }
  return v;
}

std::vector<SafetensorsFile> list_safetensors(const std::string& model_dir, int max_files) {
  if (model_dir.empty())
    die("--model-dir is required for disk modes");
  std::vector<std::filesystem::path> paths;
  for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
    if (entry.path().extension() == ".safetensors")
      paths.push_back(entry.path());
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty())
    die("no .safetensors files found");
  if (max_files > static_cast<int>(paths.size()))
    max_files = static_cast<int>(paths.size());
  std::vector<SafetensorsFile> out;
  out.reserve(max_files);
  for (int i = 0; i < max_files; ++i) {
    int fd = ::open(paths[i].c_str(), O_RDONLY);
    if (fd < 0)
      die("open failed for " + paths[i].string() + ": " + std::strerror(errno));
    uint64_t fsz = file_size_of(paths[i]);
    uint64_t header_len = read_u64_le(fd);
    ::close(fd);
    uint64_t data_start = 8 + header_len;
    if (data_start > fsz)
      die("invalid safetensors header for " + paths[i].string());
    out.push_back(SafetensorsFile{paths[i], fsz, data_start, fsz - data_start});
  }
  return out;
}

void* alloc_host(size_t bytes, bool pinned) {
  void* ptr = nullptr;
  if (pinned) {
    cuda_check(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc");
  } else {
    if (posix_memalign(&ptr, 4096, bytes) != 0)
      die("posix_memalign failed");
  }
  std::memset(ptr, 0, bytes);
  return ptr;
}

void free_host(void* ptr, bool pinned) {
  if (ptr == nullptr)
    return;
  if (pinned)
    cuda_check(cudaFreeHost(ptr), "cudaFreeHost");
  else
    free(ptr);
}

size_t pread_fully(int fd, uint64_t offset, void* dst, size_t bytes) {
  size_t total = 0;
  char* out = static_cast<char*>(dst);
  while (total < bytes) {
    ssize_t got = ::pread(fd, out + total, bytes - total, static_cast<off_t>(offset + total));
    if (got < 0) {
      if (errno == EINTR)
        continue;
      die(std::string("pread failed: ") + std::strerror(errno));
    }
    if (got == 0)
      break;
    total += static_cast<size_t>(got);
  }
  return total;
}

struct ThreadResult {
  uint64_t bytes = 0;
  double sec = 0.0;
};

ThreadResult run_disk_one(
    const std::vector<SafetensorsFile>& files,
    int thread_id,
    const Args& args,
    std::atomic<int>* ready,
    std::atomic<bool>* go) {
  void* buffer = alloc_host(args.chunk_bytes, args.pinned);
  ready->fetch_add(1, std::memory_order_release);
  while (!go->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  uint64_t done = 0;
  const double start = now_sec();
  for (size_t i = thread_id; i < files.size(); i += args.threads) {
    const auto& f = files[i];
    int fd = ::open(f.path.c_str(), O_RDONLY);
    if (fd < 0)
      die("open failed for " + f.path.string() + ": " + std::strerror(errno));
    if (args.drop_cache) {
      (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
    uint64_t remaining = f.data_size;
    if (args.limit_bytes != 0) {
      const uint64_t per_file_limit = args.limit_bytes;
      remaining = std::min<uint64_t>(remaining, per_file_limit);
    }
    if (args.method == "pread") {
      uint64_t off = 0;
      while (off < remaining) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, remaining - off));
        size_t got = pread_fully(fd, f.data_start + off, buffer, n);
        if (got != n)
          die("short pread");
        off += n;
        done += n;
      }
    } else if (args.method == "mmap") {
      void* mapped = ::mmap(nullptr, static_cast<size_t>(f.file_size), PROT_READ, MAP_PRIVATE, fd, 0);
      if (mapped == MAP_FAILED)
        die("mmap failed for " + f.path.string() + ": " + std::strerror(errno));
      if (args.drop_cache) {
        (void)::madvise(mapped, static_cast<size_t>(f.file_size), MADV_DONTNEED);
        (void)::madvise(mapped, static_cast<size_t>(f.file_size), MADV_SEQUENTIAL);
      }
      const auto* src = static_cast<const uint8_t*>(mapped) + f.data_start;
      uint64_t off = 0;
      while (off < remaining) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, remaining - off));
        std::memcpy(buffer, src + off, n);
        off += n;
        done += n;
      }
      ::munmap(mapped, static_cast<size_t>(f.file_size));
    } else {
      die("unknown disk --method: " + args.method);
    }
    ::close(fd);
  }
  const double sec = now_sec() - start;
  free_host(buffer, args.pinned);
  return ThreadResult{done, sec};
}

void run_disk(const Args& args) {
  auto files = list_safetensors(args.model_dir, args.files);
  std::vector<std::thread> workers;
  std::vector<ThreadResult> results(args.threads);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (int t = 0; t < args.threads; ++t) {
    workers.emplace_back([&, t] { results[t] = run_disk_one(files, t, args, &ready, &go); });
  }
  while (ready.load(std::memory_order_acquire) < args.threads) {
    std::this_thread::yield();
  }
  const double active_start = now_sec();
  go.store(true, std::memory_order_release);
  for (auto& th : workers)
    th.join();
  const double active_wall = now_sec() - active_start;
  uint64_t bytes = 0;
  double max_thread_sec = 0.0;
  for (const auto& r : results) {
    bytes += r.bytes;
    max_thread_sec = std::max(max_thread_sec, r.sec);
  }
  const double active_gib_per_sec =
      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / std::max(active_wall, 1e-9);
  std::cout << "RESULT mode=disk method=" << args.method << " pinned=" << (args.pinned ? 1 : 0)
            << " drop_cache=" << (args.drop_cache ? 1 : 0) << " threads=" << args.threads << " files=" << files.size()
            << " chunk_bytes=" << args.chunk_bytes << " bytes=" << bytes << " active_wall_sec=" << active_wall
            << " max_thread_sec=" << max_thread_sec << " active_gib_per_sec=" << active_gib_per_sec << "\n";
}

ThreadResult run_disk_h2d_one(
    const std::vector<SafetensorsFile>& files,
    const std::vector<int>& devices,
    int thread_id,
    const Args& args,
    std::atomic<int>* ready,
    std::atomic<bool>* go) {
  const int device = devices[thread_id % devices.size()];
  cuda_check(cudaSetDevice(device), "cudaSetDevice");
  void* host = alloc_host(args.chunk_bytes, args.pinned);
  void* dev = nullptr;
  cuda_check(cudaMalloc(&dev, args.chunk_bytes), "cudaMalloc");
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  ready->fetch_add(1, std::memory_order_release);
  while (!go->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  uint64_t done = 0;
  const double start = now_sec();
  for (size_t i = thread_id; i < files.size(); i += args.threads) {
    const auto& f = files[i];
    int fd = ::open(f.path.c_str(), O_RDONLY);
    if (fd < 0)
      die("open failed for " + f.path.string() + ": " + std::strerror(errno));
    if (args.drop_cache) {
      (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
    uint64_t remaining = f.data_size;
    if (args.limit_bytes != 0)
      remaining = std::min<uint64_t>(remaining, args.limit_bytes);
    if (args.method == "mmap") {
      void* mapped = ::mmap(nullptr, static_cast<size_t>(f.file_size), PROT_READ, MAP_PRIVATE, fd, 0);
      if (mapped == MAP_FAILED)
        die("mmap failed for " + f.path.string() + ": " + std::strerror(errno));
      if (args.drop_cache) {
        (void)::madvise(mapped, static_cast<size_t>(f.file_size), MADV_DONTNEED);
        (void)::madvise(mapped, static_cast<size_t>(f.file_size), MADV_SEQUENTIAL);
      }
      const auto* src = static_cast<const uint8_t*>(mapped) + f.data_start;
      uint64_t off = 0;
      while (off < remaining) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, remaining - off));
        std::memcpy(host, src + off, n);
        cuda_check(cudaMemcpyAsync(dev, host, n, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync disk-h2d");
        if (args.sync_per_chunk)
          cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize disk-h2d");
        off += n;
        done += n;
      }
      ::munmap(mapped, static_cast<size_t>(f.file_size));
    } else if (args.method == "pread") {
      uint64_t off = 0;
      while (off < remaining) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, remaining - off));
        size_t got = pread_fully(fd, f.data_start + off, host, n);
        if (got != n)
          die("short pread in disk-h2d");
        cuda_check(cudaMemcpyAsync(dev, host, n, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync disk-h2d");
        if (args.sync_per_chunk)
          cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize disk-h2d");
        off += n;
        done += n;
      }
    } else {
      die("unknown disk-h2d --method: " + args.method);
    }
    ::close(fd);
  }
  if (!args.sync_per_chunk)
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize disk-h2d final");
  const double sec = now_sec() - start;
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  cuda_check(cudaFree(dev), "cudaFree");
  free_host(host, args.pinned);
  return ThreadResult{done, sec};
}

void run_disk_h2d(const Args& args) {
  auto files = list_safetensors(args.model_dir, args.files);
  auto devices = parse_devices(args.devices);
  std::vector<std::thread> workers;
  std::vector<ThreadResult> results(args.threads);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (int t = 0; t < args.threads; ++t) {
    workers.emplace_back([&, t] { results[t] = run_disk_h2d_one(files, devices, t, args, &ready, &go); });
  }
  while (ready.load(std::memory_order_acquire) < args.threads) {
    std::this_thread::yield();
  }
  const double active_start = now_sec();
  go.store(true, std::memory_order_release);
  for (auto& th : workers)
    th.join();
  const double active_wall = now_sec() - active_start;
  uint64_t bytes = 0;
  double max_thread_sec = 0.0;
  for (const auto& r : results) {
    bytes += r.bytes;
    max_thread_sec = std::max(max_thread_sec, r.sec);
  }
  const double active_gib_per_sec =
      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / std::max(active_wall, 1e-9);
  std::cout << "RESULT mode=disk_h2d method=" << args.method << " pinned=" << (args.pinned ? 1 : 0)
            << " drop_cache=" << (args.drop_cache ? 1 : 0) << " sync_per_chunk=" << (args.sync_per_chunk ? 1 : 0)
            << " threads=" << args.threads << " devices=" << args.devices << " files=" << files.size()
            << " chunk_bytes=" << args.chunk_bytes << " bytes=" << bytes << " active_wall_sec=" << active_wall
            << " max_thread_sec=" << max_thread_sec << " active_gib_per_sec=" << active_gib_per_sec << "\n";
}

void run_host_memcpy(const Args& args) {
  uint64_t total_per_thread = args.limit_bytes == 0 ? 64ULL * 1024 * 1024 * 1024 : args.limit_bytes;
  std::vector<std::thread> workers;
  std::vector<ThreadResult> results(args.threads);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (int t = 0; t < args.threads; ++t) {
    workers.emplace_back([&, t] {
      void* src = alloc_host(args.chunk_bytes, false);
      void* dst = alloc_host(args.chunk_bytes, args.pinned);
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      uint64_t done = 0;
      const double start = now_sec();
      while (done < total_per_thread) {
        size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, total_per_thread - done));
        std::memcpy(dst, src, n);
        done += n;
      }
      results[t] = ThreadResult{done, now_sec() - start};
      free_host(dst, args.pinned);
      free_host(src, false);
    });
  }
  while (ready.load(std::memory_order_acquire) < args.threads) {
    std::this_thread::yield();
  }
  const double active_start = now_sec();
  go.store(true, std::memory_order_release);
  for (auto& th : workers)
    th.join();
  const double active_wall = now_sec() - active_start;
  uint64_t done = 0;
  double max_thread_sec = 0.0;
  for (const auto& r : results) {
    done += r.bytes;
    max_thread_sec = std::max(max_thread_sec, r.sec);
  }
  const double active_gib_per_sec =
      static_cast<double>(done) / (1024.0 * 1024.0 * 1024.0) / std::max(active_wall, 1e-9);
  std::cout << "RESULT mode=host_memcpy pinned_dst=" << (args.pinned ? 1 : 0) << " threads=" << args.threads
            << " chunk_bytes=" << args.chunk_bytes << " bytes=" << done << " active_wall_sec=" << active_wall
            << " max_thread_sec=" << max_thread_sec << " active_gib_per_sec=" << active_gib_per_sec << "\n";
}

ThreadResult run_h2d_one(int device, const Args& args, std::atomic<int>* ready, std::atomic<bool>* go) {
  cuda_check(cudaSetDevice(device), "cudaSetDevice");
  void* host = alloc_host(args.chunk_bytes, args.pinned);
  void* dev = nullptr;
  cuda_check(cudaMalloc(&dev, args.chunk_bytes), "cudaMalloc");
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  ready->fetch_add(1, std::memory_order_release);
  while (!go->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  uint64_t total = args.limit_bytes == 0 ? 16ULL * 1024 * 1024 * 1024 : args.limit_bytes;
  uint64_t done = 0;
  const double start = now_sec();
  while (done < total) {
    size_t n = static_cast<size_t>(std::min<uint64_t>(args.chunk_bytes, total - done));
    cuda_check(cudaMemcpyAsync(dev, host, n, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    if (args.sync_per_chunk)
      cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    done += n;
  }
  if (!args.sync_per_chunk)
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize final");
  const double wall = now_sec() - start;
  cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  cuda_check(cudaFree(dev), "cudaFree");
  free_host(host, args.pinned);
  return ThreadResult{done, wall};
}

void run_h2d(const Args& args) {
  auto devices = parse_devices(args.devices);
  std::vector<std::thread> workers;
  std::vector<ThreadResult> results(devices.size());
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (size_t i = 0; i < devices.size(); ++i) {
    workers.emplace_back([&, i] { results[i] = run_h2d_one(devices[i], args, &ready, &go); });
  }
  while (ready.load(std::memory_order_acquire) < static_cast<int>(devices.size())) {
    std::this_thread::yield();
  }
  const double active_start = now_sec();
  go.store(true, std::memory_order_release);
  for (auto& th : workers)
    th.join();
  const double active_wall = now_sec() - active_start;
  uint64_t bytes = 0;
  double max_thread_sec = 0.0;
  for (const auto& r : results) {
    bytes += r.bytes;
    max_thread_sec = std::max(max_thread_sec, r.sec);
  }
  const double active_gib_per_sec =
      static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / std::max(active_wall, 1e-9);
  std::cout << "RESULT mode=h2d pinned=" << (args.pinned ? 1 : 0) << " sync_per_chunk=" << (args.sync_per_chunk ? 1 : 0)
            << " devices=" << args.devices << " chunk_bytes=" << args.chunk_bytes << " bytes=" << bytes
            << " active_wall_sec=" << active_wall << " max_thread_sec=" << max_thread_sec
            << " active_gib_per_sec=" << active_gib_per_sec << "\n";
}

} // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (args.mode == "disk") {
    run_disk(args);
  } else if (args.mode == "disk-h2d") {
    run_disk_h2d(args);
  } else if (args.mode == "host-memcpy") {
    run_host_memcpy(args);
  } else if (args.mode == "h2d") {
    run_h2d(args);
  } else {
    die("unknown --mode: " + args.mode);
  }
  return 0;
}
