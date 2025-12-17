// Copyright (c) 2025, TensorCast Team.

#include "core/common/async_runtime.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/executors/SerialExecutor.h"
#include "folly/executors/thread_factory/NamedThreadFactory.h"
#include "folly/futures/ThreadWheelTimekeeper.h"
#include "folly/io/async/Request.h"
#include "opentelemetry/common/attribute_value.h"
#include "opentelemetry/common/key_value_iterable_view.h"
#include "opentelemetry/context/context.h"
#include "opentelemetry/metrics/provider.h"

namespace tensorcast::common {
namespace {

constexpr absl::Duration kDefaultDrainTimeout = absl::Seconds(30);

struct TaskCounters {
  absl::Mutex mu;
  absl::CondVar cv;
  int64_t inflight ABSL_GUARDED_BY(mu) = 0;
  int64_t cpu_inflight ABSL_GUARDED_BY(mu) = 0;
  int64_t blocking_inflight ABSL_GUARDED_BY(mu) = 0;
  std::atomic<bool> shutting_down{false};

  struct Snapshot {
    int64_t inflight{0};
    int64_t cpu_inflight{0};
    int64_t blocking_inflight{0};
  };

  Snapshot snapshot() {
    absl::MutexLock lock(&mu);
    return Snapshot{
        .inflight = inflight,
        .cpu_inflight = cpu_inflight,
        .blocking_inflight = blocking_inflight,
    };
  }

  void on_enqueued_cpu() {
    absl::MutexLock lock(&mu);
    ++inflight;
    ++cpu_inflight;
  }

  void on_enqueued_blocking() {
    absl::MutexLock lock(&mu);
    ++inflight;
    ++blocking_inflight;
  }

  void on_finished_cpu() {
    absl::MutexLock lock(&mu);
    ABSL_CHECK(inflight > 0);
    ABSL_CHECK(cpu_inflight > 0);
    --inflight;
    --cpu_inflight;
    if (inflight == 0) {
      cv.SignalAll();
    }
  }

  void on_finished_blocking() {
    absl::MutexLock lock(&mu);
    ABSL_CHECK(inflight > 0);
    ABSL_CHECK(blocking_inflight > 0);
    --inflight;
    --blocking_inflight;
    if (inflight == 0) {
      cv.SignalAll();
    }
  }

  absl::Status drain_until(absl::Time deadline) {
    absl::MutexLock lock(&mu);
    while (inflight != 0) {
      const absl::Duration remaining = deadline - absl::Now();
      if (remaining <= absl::ZeroDuration()) {
        return absl::DeadlineExceededError("AsyncRuntime drain deadline exceeded");
      }
      cv.WaitWithTimeout(&mu, remaining);
    }
    return absl::OkStatus();
  }
};

enum class ExecutorKind : uint8_t { kCpu, kBlocking };

struct AsyncRuntimeMetrics {
  explicit AsyncRuntimeMetrics(std::string runtime_name) : runtime_name_(std::move(runtime_name)) {
    cpu_attrs_.emplace("executor", opentelemetry::common::AttributeValue(std::string("cpu")));
    cpu_attrs_.emplace("runtime", opentelemetry::common::AttributeValue(runtime_name_));
    blocking_attrs_.emplace("executor", opentelemetry::common::AttributeValue(std::string("blocking")));
    blocking_attrs_.emplace("runtime", opentelemetry::common::AttributeValue(runtime_name_));
    common_attrs_.emplace("runtime", opentelemetry::common::AttributeValue(runtime_name_));
    try {
      meter_ = opentelemetry::metrics::Provider::GetMeterProvider()->GetMeter("tensorcast.daemon", "1.0.0");
      tasks_total_ = meter_->CreateDoubleCounter("tc_async_runtime_tasks_total");
      queue_latency_ms_ = meter_->CreateDoubleHistogram("tc_async_runtime_queue_latency_ms");
      task_runtime_ms_ = meter_->CreateDoubleHistogram("tc_async_runtime_task_runtime_ms");
      drain_duration_ms_ = meter_->CreateDoubleHistogram("tc_async_runtime_drain_duration_ms");
      drain_timeouts_total_ = meter_->CreateDoubleCounter("tc_async_runtime_drain_timeouts_total");
      inflight_gauge_ = meter_->CreateDoubleObservableGauge("tc_async_runtime_inflight_tasks_gauge");
    } catch (...) {
      // Best-effort; metrics must never affect control flow.
    }
  }

  void register_inflight_gauge(TaskCounters* counters) {
    if (inflight_gauge_) {
      inflight_gauge_->AddCallback(&AsyncRuntimeMetrics::inflight_gauge_callback, counters);
    }
  }

  void on_enqueued(ExecutorKind kind) noexcept {
    if (!tasks_total_) {
      return;
    }
    try {
      const auto& attrs = attrs_for_(kind);
      tasks_total_->Add(1.0, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    } catch (...) {
    }
  }

  void record_queue_latency(ExecutorKind kind, std::chrono::nanoseconds latency) noexcept {
    if (!queue_latency_ms_) {
      return;
    }
    try {
      const double ms = std::chrono::duration<double, std::milli>(latency).count();
      const auto& attrs = attrs_for_(kind);
      queue_latency_ms_->Record(
          ms, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    } catch (...) {
    }
  }

  void record_task_runtime(ExecutorKind kind, std::chrono::nanoseconds runtime) noexcept {
    if (!task_runtime_ms_) {
      return;
    }
    try {
      const double ms = std::chrono::duration<double, std::milli>(runtime).count();
      const auto& attrs = attrs_for_(kind);
      task_runtime_ms_->Record(
          ms, opentelemetry::common::KeyValueIterableView(attrs), opentelemetry::context::Context{});
    } catch (...) {
    }
  }

  void record_drain(absl::Duration duration, bool ok) noexcept {
    if (!drain_duration_ms_) {
      return;
    }
    try {
      const double ms = absl::ToDoubleMilliseconds(duration);
      drain_duration_ms_->Record(
          ms, opentelemetry::common::KeyValueIterableView(common_attrs_), opentelemetry::context::Context{});
      if (!ok && drain_timeouts_total_) {
        drain_timeouts_total_->Add(
            1.0, opentelemetry::common::KeyValueIterableView(common_attrs_), opentelemetry::context::Context{});
      }
    } catch (...) {
    }
  }

  static void inflight_gauge_callback(opentelemetry::metrics::ObserverResult result, void* state) noexcept {
    auto* counters = static_cast<TaskCounters*>(state);
    if (counters == nullptr) {
      return;
    }
    auto obs =
        opentelemetry::nostd::get<opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObserverResultT<double>>>(
            result);
    if (!obs) {
      return;
    }
    TaskCounters::Snapshot snap = counters->snapshot();
    obs->Observe(static_cast<double>(snap.cpu_inflight), {{"executor", opentelemetry::common::AttributeValue("cpu")}});
    obs->Observe(
        static_cast<double>(snap.blocking_inflight), {{"executor", opentelemetry::common::AttributeValue("blocking")}});
    obs->Observe(static_cast<double>(snap.inflight), {{"executor", opentelemetry::common::AttributeValue("total")}});
  }

 private:
  [[nodiscard]] const std::map<std::string, opentelemetry::common::AttributeValue>& attrs_for_(
      ExecutorKind kind) const {
    if (kind == ExecutorKind::kCpu) {
      return cpu_attrs_;
    }
    return blocking_attrs_;
  }

  std::string runtime_name_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Meter> meter_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> tasks_total_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> queue_latency_ms_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> task_runtime_ms_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Histogram<double>> drain_duration_ms_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::Counter<double>> drain_timeouts_total_;
  opentelemetry::nostd::shared_ptr<opentelemetry::metrics::ObservableInstrument> inflight_gauge_;

  std::map<std::string, opentelemetry::common::AttributeValue> cpu_attrs_;
  std::map<std::string, opentelemetry::common::AttributeValue> blocking_attrs_;
  std::map<std::string, opentelemetry::common::AttributeValue> common_attrs_;
};

class TrackingExecutor final : public folly::Executor {
 public:
  static KeepAlive<TrackingExecutor> create(
      KeepAlive<> delegate,
      std::shared_ptr<TaskCounters> counters,
      ExecutorKind kind,
      std::string name,
      std::shared_ptr<AsyncRuntimeMetrics> metrics) {
    return makeKeepAlive(
        new TrackingExecutor(std::move(delegate), std::move(counters), kind, std::move(name), std::move(metrics)));
  }

  void detach_delegate() {
    absl::MutexLock lock(&delegate_mu_);
    delegate_.reset();
  }

  void add(folly::Func func) override {
    if (!func) {
      return;
    }
    KeepAlive<> delegate;
    {
      absl::MutexLock lock(&delegate_mu_);
      delegate = delegate_;
    }
    if (!delegate) {
      return;
    }
    const auto enqueued_at = std::chrono::steady_clock::now();
    auto request_context = folly::RequestContext::saveContext();
    on_enqueued_();
    if (metrics_) {
      metrics_->on_enqueued(kind_);
    }
    delegate->add([func = std::move(func),
                   counters = counters_,
                   kind = kind_,
                   metrics = metrics_,
                   enqueued_at,
                   request_context = std::move(request_context)]() mutable {
      const auto started_at = std::chrono::steady_clock::now();
      folly::RequestContextScopeGuard context_guard(std::move(request_context));
      if (metrics) {
        metrics->record_queue_latency(kind, started_at - enqueued_at);
      }
      absl::Cleanup done = [&]() {
        const auto finished_at = std::chrono::steady_clock::now();
        if (metrics) {
          metrics->record_task_runtime(kind, finished_at - started_at);
        }
        if (kind == ExecutorKind::kCpu) {
          counters->on_finished_cpu();
        } else {
          counters->on_finished_blocking();
        }
      };
      func();
    });
  }

  void addWithPriority(folly::Func func, int8_t priority) override {
    if (!func) {
      return;
    }
    KeepAlive<> delegate;
    {
      absl::MutexLock lock(&delegate_mu_);
      delegate = delegate_;
    }
    if (!delegate) {
      return;
    }
    const auto enqueued_at = std::chrono::steady_clock::now();
    auto request_context = folly::RequestContext::saveContext();
    on_enqueued_();
    if (metrics_) {
      metrics_->on_enqueued(kind_);
    }
    delegate->addWithPriority(
        [func = std::move(func),
         counters = counters_,
         kind = kind_,
         metrics = metrics_,
         enqueued_at,
         request_context = std::move(request_context)]() mutable {
          const auto started_at = std::chrono::steady_clock::now();
          folly::RequestContextScopeGuard context_guard(std::move(request_context));
          if (metrics) {
            metrics->record_queue_latency(kind, started_at - enqueued_at);
          }
          absl::Cleanup done = [&]() {
            const auto finished_at = std::chrono::steady_clock::now();
            if (metrics) {
              metrics->record_task_runtime(kind, finished_at - started_at);
            }
            if (kind == ExecutorKind::kCpu) {
              counters->on_finished_cpu();
            } else {
              counters->on_finished_blocking();
            }
          };
          func();
        },
        priority);
  }

  uint8_t getNumPriorities() const override {
    KeepAlive<> delegate;
    {
      absl::MutexLock lock(&delegate_mu_);
      delegate = delegate_;
    }
    if (!delegate) {
      return 1;
    }
    return delegate->getNumPriorities();
  }

 protected:
  bool keepAliveAcquire() noexcept override {
    const auto prev = keep_alive_counter_.fetch_add(1, std::memory_order_relaxed);
    ABSL_DCHECK(prev > 0);
    return true;
  }

  void keepAliveRelease() noexcept override {
    const auto prev = keep_alive_counter_.fetch_sub(1, std::memory_order_acq_rel);
    ABSL_DCHECK(prev > 0);
    if (prev == 1) {
      delete this;
    }
  }

 private:
  TrackingExecutor(
      KeepAlive<> delegate,
      std::shared_ptr<TaskCounters> counters,
      ExecutorKind kind,
      std::string name,
      std::shared_ptr<AsyncRuntimeMetrics> metrics)
      : delegate_(std::move(delegate)),
        counters_(std::move(counters)),
        kind_(kind),
        name_(std::move(name)),
        metrics_(std::move(metrics)) {}

  ~TrackingExecutor() override {
    ABSL_DCHECK(keep_alive_counter_.load(std::memory_order_relaxed) == 0);
  }

  void on_enqueued_() {
    if (kind_ == ExecutorKind::kCpu) {
      counters_->on_enqueued_cpu();
    } else {
      counters_->on_enqueued_blocking();
    }
  }

  mutable absl::Mutex delegate_mu_;
  KeepAlive<> delegate_ ABSL_GUARDED_BY(delegate_mu_);
  std::shared_ptr<TaskCounters> counters_;
  ExecutorKind kind_;
  std::string name_;
  std::shared_ptr<AsyncRuntimeMetrics> metrics_;
  std::atomic<int64_t> keep_alive_counter_{1};
};

size_t pick_default_threads(size_t requested, size_t fallback) {
  if (requested != 0) {
    return requested;
  }
  if (fallback == 0) {
    return 1;
  }
  return fallback;
}

} // namespace

struct AsyncRuntime::Impl {
  explicit Impl(Options opts)
      : counters(std::make_shared<TaskCounters>()),
        metrics(std::make_shared<AsyncRuntimeMetrics>(opts.thread_name_prefix)),
        cpu_pool(
            std::make_unique<folly::CPUThreadPoolExecutor>(
                pick_default_threads(opts.cpu_threads, std::thread::hardware_concurrency()),
                std::make_shared<folly::NamedThreadFactory>(opts.thread_name_prefix + "-cpu"))),
        blocking_pool(
            std::make_unique<folly::CPUThreadPoolExecutor>(
                pick_default_threads(
                    std::max<size_t>(2, opts.blocking_threads),
                    std::max<size_t>(2, std::thread::hardware_concurrency() / 2)),
                std::make_shared<folly::NamedThreadFactory>(opts.thread_name_prefix + "-blocking"))),
        timekeeper(std::make_unique<folly::ThreadWheelTimekeeper>()) {
    metrics->register_inflight_gauge(counters.get());
    auto cpu_ka = folly::getKeepAliveToken(*cpu_pool);
    auto blocking_ka = folly::getKeepAliveToken(*blocking_pool);
    cpu_executor = TrackingExecutor::create(cpu_ka, counters, ExecutorKind::kCpu, "cpu", metrics);
    blocking_executor = TrackingExecutor::create(blocking_ka, counters, ExecutorKind::kBlocking, "blocking", metrics);
    serial_executor = folly::SerialExecutor::create(cpu_executor.copy());
  }

  std::shared_ptr<TaskCounters> counters;
  std::shared_ptr<AsyncRuntimeMetrics> metrics;
  std::unique_ptr<folly::CPUThreadPoolExecutor> cpu_pool;
  std::unique_ptr<folly::CPUThreadPoolExecutor> blocking_pool;
  folly::Executor::KeepAlive<TrackingExecutor> cpu_executor;
  folly::Executor::KeepAlive<TrackingExecutor> blocking_executor;
  folly::Executor::KeepAlive<folly::SerialExecutor> serial_executor;
  std::unique_ptr<folly::ThreadWheelTimekeeper> timekeeper;
  std::atomic<bool> drained{false};
};

AsyncRuntime::AsyncRuntime() : AsyncRuntime(Options{}) {}

AsyncRuntime::AsyncRuntime(Options opts) : impl_(std::make_unique<Impl>(std::move(opts))) {}

AsyncRuntime::~AsyncRuntime() {
  shutdown();
  const absl::Time deadline = absl::Now() + kDefaultDrainTimeout;
  const absl::Status st = drain(deadline);
  if (!st.ok()) {
    LOG(FATAL) << "AsyncRuntime drain failed during destruction: " << st;
  }
}

folly::Executor::KeepAlive<> AsyncRuntime::cpu_executor() const {
  return impl_->cpu_executor.copy();
}

folly::Executor::KeepAlive<> AsyncRuntime::blocking_executor() const {
  return impl_->blocking_executor.copy();
}

folly::Executor::KeepAlive<> AsyncRuntime::serial_executor() const {
  return impl_->serial_executor.copy();
}

folly::Timekeeper& AsyncRuntime::timekeeper() const {
  return *impl_->timekeeper;
}

void AsyncRuntime::shutdown() {
  impl_->counters->shutting_down.store(true, std::memory_order_release);
}

bool AsyncRuntime::is_shutting_down() const {
  return impl_->counters->shutting_down.load(std::memory_order_acquire);
}

absl::Status AsyncRuntime::drain(absl::Time deadline) {
  if (impl_->drained.load(std::memory_order_acquire)) {
    return absl::OkStatus();
  }

  const absl::Time start_time = absl::Now();
  absl::Status st = impl_->counters->drain_until(deadline);
  if (!st.ok()) {
    impl_->metrics->record_drain(absl::Now() - start_time, /*ok=*/false);
    return st;
  }

  // After counters hit zero, stop/join thread pools to guarantee no work can
  // outlive the runtime object.
  impl_->cpu_executor->detach_delegate();
  impl_->blocking_executor->detach_delegate();
  impl_->serial_executor.reset();
  impl_->cpu_executor.reset();
  impl_->blocking_executor.reset();
  impl_->cpu_pool->stop();
  impl_->blocking_pool->stop();
  impl_->cpu_pool->join();
  impl_->blocking_pool->join();
  impl_->drained.store(true, std::memory_order_release);
  impl_->metrics->record_drain(absl::Now() - start_time, /*ok=*/true);
  return absl::OkStatus();
}

} // namespace tensorcast::common
