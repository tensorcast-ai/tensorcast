# Linux 仅支持的 Launch / Connect 最佳实践（含 CLI 与库入口、进程与日志的极致细节）

本文提供一个可直接落地的工程方案，复刻 Ray 的“init 既可启动本地进程又可连接已有集群”的框架能力，但只关注 Linux 平台与进程管控层面，不涉及业务参数。核心包括：

- 单一实现，既用于 CLI（start/stop/status/logs），也用于库 API（`init()/shutdown()`）。
- 两种模式：Launch/Owner（本地启动并拥有子进程）与 Connect-only（仅连接已有服务）。
- 子进程启动、命运绑定（父死子亡，PR_SET_PDEATHSIG）、信号屏蔽、日志落地与可选 tee 回流、优雅停止（SIGTERM→SIGKILL）。

---

## 目录与元数据布局

- 根目录：`~/.mytool/`
  - `instances/<instance_id>/`
    - `session/`：会话文件、端口缓存等
    - `logs/`：`<role>.out|<role>.err`
    - `pids.json`：记录各角色进程的 PID/命令/日志路径/启动时间
    - `meta.json`：`address/dashboard_url/created_at/user/version`
  - `current_instance`：文本文件，保存最近一次成功启动的 `<instance_id>`（供 `address=auto`）

`instance_id` 若未指定，建议生成 `YYYYMMDD-HHMMSS-<rand4>`。

---

## 模式决策（统一用于 CLI 与库 API）

1. 解析 `address`：
   - 形如 `host:port`：尝试连接，失败即报错（不启动）。
   - `address=None`：依次读取 `MYTOOL_ADDRESS` 环境变量 → `~/.mytool/current_instance` 指向的实例并探活；找到则 connect-only，否则走本地启动。
   - `address="auto"`：与 `None` 相同，但未找到时报错（不启动）。
   - `address="local"`：强制本地启动（即使已有实例）。
2. 返回 `Context`：`{ address, is_owner, session_dir, dashboard_url, close() }`。
   - Launch → `is_owner=True`，持有本地子进程生命周期；
   - Connect-only → `is_owner=False`，仅保存连接信息与本地状态。

---

## Linux 子进程启动与命运绑定：最底层实现

核心思想：使用 `subprocess.Popen` 启动二进制/脚本，通过 `preexec_fn` 设置 PR_SET_PDEATHSIG 实现“父死子亡”，屏蔽/忽略部分信号，独立进程组便于成组发送信号。阻塞模式下用 `PIPE` 并 tee 到控制台；非阻塞模式直接重定向到日志文件。

### 关键代码：prctl(PR_SET_PDEATHSIG) 与 preexec_fn

```python
import ctypes
import os
import signal

libc = ctypes.CDLL("libc.so.6", use_errno=True)
PR_SET_PDEATHSIG = 1  # from linux/prctl.h

def _set_pdeathsig(sig=signal.SIGKILL):
    """Ask kernel to deliver `sig` to child when parent dies (Linux only)."""
    res = libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    if res != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_PDEATHSIG) failed: errno={err}")

def _preexec_fate_sharing(ignore_sigint: bool = True,
                          block_signals=(signal.SIGTTOU, signal.SIGTTIN, signal.SIGTSTP),
                          pdeathsig: int = signal.SIGKILL):
    """Run in child after fork, before exec. Linux-only.

    Keep this lightweight; Python `preexec_fn` runs in a forked copy of the parent.
    """
    # 独立进程组（便于向整组发信号）。也可用 Popen(start_new_session=True) 代替。
    os.setsid()
    # 父死子亡
    _set_pdeathsig(pdeathsig)
    # 子进程忽略 Ctrl-C（SIGINT），由父进程统一管理退出策略
    if ignore_sigint:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    # 可选：屏蔽部分 TTY 相关信号
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, list(block_signals))
    except (AttributeError, OSError):
        pass
```

注意：在多线程父进程中使用 `preexec_fn` 需要谨慎，尽量避免做复杂操作。上述调用集合是业界常用实践（与 Ray 类似）。

### 日志文件打开与 tee

```python
from pathlib import Path

def open_log_binary(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 二进制追加模式。行缓冲仅适用于文本模式，这里依赖子进程自身的行缓冲。
    return open(p, "ab", buffering=0)

def _pump(src, sinks):
    """逐行从 src 读出（bytes），写入多个 sink（文件/终端）。"""
    import sys
    for line in iter(src.readline, b""):
        for s in sinks:
            try:
                if s in (sys.stdout, sys.stderr):
                    # 终端：解码为 utf-8，容错替换
                    s.write(line.decode("utf-8", errors="replace"))
                    s.flush()
                else:
                    s.write(line)
                    s.flush()
            except Exception:
                pass

def tee_process_output(proc, stdout_file, stderr_file, to_console=True):
    """在阻塞模式下，将子进程输出同时写文件与控制台。"""
    import sys, threading
    t1 = threading.Thread(target=_pump, args=(proc.stdout, [stdout_file] + ([sys.stdout] if to_console else [])), daemon=True)
    t2 = threading.Thread(target=_pump, args=(proc.stderr, [stderr_file] + ([sys.stderr] if to_console else [])), daemon=True)
    t1.start(); t2.start()
    return t1, t2
```

### Popen 参数矩阵（精确示例）

Owner + 非阻塞（常用）：

```python
import subprocess

env = {**os.environ, "MYTOOL_INSTANCE": instance_id}
stdout_f = open_log_binary(f"{logs_dir}/daemon.out")
stderr_f = open_log_binary(f"{logs_dir}/daemon.err")

proc = subprocess.Popen(
    args=["/usr/local/bin/my-daemon", "--port", str(port), "--session", session_dir],
    stdin=subprocess.DEVNULL,
    stdout=stdout_f,
    stderr=stderr_f,
    cwd=session_dir,
    env=env,
    preexec_fn=lambda: _preexec_fate_sharing(ignore_sigint=True, pdeathsig=signal.SIGKILL),
    close_fds=True,
    # 也可不设，已在 preexec 内 setsid；二者择一：
    # start_new_session=True,
)
```

Owner + 阻塞（CLI `--block` 或库显式要求）：

```python
stdout_f = open_log_binary(f"{logs_dir}/daemon.out")
stderr_f = open_log_binary(f"{logs_dir}/daemon.err")

proc = subprocess.Popen(
    args=["/usr/local/bin/my-daemon", "--port", str(port), "--session", session_dir],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=session_dir,
    env=env,
    preexec_fn=lambda: _preexec_fate_sharing(ignore_sigint=True, pdeathsig=signal.SIGKILL),
    close_fds=True,
)

# tee 到文件与控制台
_t1, _t2 = tee_process_output(proc, stdout_f, stderr_f, to_console=True)
ret = proc.wait()
```

Connect-only：不执行 `Popen`，只建立到已有服务的连接（例如通过 socket/HTTP/RPC）。

---

## 进程注册与实例元数据

`pids.json` 建议结构：

```json
{
  "instance_id": "20240911-1730-ab12",
  "processes": [
    {
      "role": "daemon",
      "pid": 12345,
      "cmd": ["/usr/local/bin/my-daemon", "--port", "18080"],
      "stdout": "/home/user/.mytool/instances/.../logs/daemon.out",
      "stderr": "/home/user/.mytool/instances/.../logs/daemon.err",
      "start_time": 1726066200.123
    },
    {
      "role": "webui",
      "pid": 12346,
      "cmd": ["/usr/local/bin/my-webui", "--bind", "127.0.0.1:8265"],
      "stdout": ".../logs/webui.out",
      "stderr": ".../logs/webui.err",
      "start_time": 1726066200.456
    }
  ]
}
```

安全写入（`fcntl` 锁）示例：

```python
import fcntl, json, time
from pathlib import Path

def write_json_locked(path: str, data: dict):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def read_json_locked(path: str) -> dict:
    with open(path, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        data = json.load(f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data
```

`meta.json` 写入 `address/dashboard_url/created_at` 等；`~/.mytool/current_instance` 保存 `instance_id` 文本。

---

## 优雅停止与强制清理（stop 与 shutdown）

约定：启动时在 `preexec_fn` 内调用了 `os.setsid()`（或 `start_new_session=True`），子进程成为会话首进程，其进程组 ID = 其 PID。这样可以使用 `os.killpg(pid, SIGTERM)` 一次性向整组发送信号。

```python
import errno

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM  # 存在但无权限

def _kill_gracefully(pgid: int, grace: float = 10.0) -> bool:
    """向进程组发送 SIGTERM 并等待，返回是否如期退出。"""
    import time
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.2)
    return False

def _kill_force(pgid: int):
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

def stop_instance(instance_dir: str, role_order=("worker", "webui", "api", "daemon"), grace: float = 10.0):
    p = os.path.join(instance_dir, "pids.json")
    if not os.path.exists(p):
        return
    data = read_json_locked(p)
    # 按角色顺序优雅终止
    for role in role_order:
        for proc in [x for x in data.get("processes", []) if x["role"] == role]:
            pid = int(proc["pid"])
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if not _kill_gracefully(pgid, grace=grace):
                _kill_force(pgid)
```

命令行 `mytool stop` 与库侧 `shutdown()` 均可调用 `stop_instance()`；Owner 模式在 `atexit` 中也应调用之。

---

## 健康检查与就绪判定（启动后阻塞等待）

可选方案：

1. 端口就绪：
   ```python
   import socket, time

   def wait_port_open(host: str, port: int, timeout: float = 15.0) -> bool:
       deadline = time.time() + timeout
       while time.time() < deadline:
           with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
               s.settimeout(0.5)
               try:
                   if s.connect_ex((host, port)) == 0:
                       return True
               except OSError:
                   pass
           time.sleep(0.2)
       return False
   ```
2. 日志探测：在 `tee` 线程或日志落地文件中查找 `"Ready"` 关键行；
3. Unix socket/文件出现：等待特定套接字或文件生成。

失败即回滚：终止已起的全部子进程并清理会话目录，返回非零。

---

## 库 API：`init()` / `shutdown()` 与 `Context`

```python
# file: api.py
import threading, atexit

class Context:
    def __init__(self, address: str, is_owner: bool, session_dir: str, dashboard_url: str = None):
        self.address = address
        self.is_owner = is_owner
        self.session_dir = session_dir
        self.dashboard_url = dashboard_url
        self._closed = False

    def close(self):
        if self._closed:
            return
        if self.is_owner:
            stop_instance(self.session_dir)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

_ctx_lock = threading.Lock()
_current_ctx: Context | None = None

def current_context() -> Context | None:
    return _current_ctx

def init(address: str | None = None, *, block: bool = False, log_to_caller: bool = True,
         temp_dir: str | None = None, instance_id: str | None = None,
         stdout_file: str | None = None, stderr_file: str | None = None) -> Context:
    """Launch-or-connect 入口，Linux-only 实现。"""
    # 0) 驱动进程自身的 SIGTERM -> 直接退出（便于与子进程命运绑定）
    def _sigterm_handler(signum, frame):
        os._exit(signum)  # 直接退出，交给内核 PDEATHSIG 清理子进程
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # 1) 解析 address：省略若干探测细节，示例化
    resolved = resolve_address(address)  # 需要用户实现：返回 (mode, address)
    mode, addr = resolved.mode, resolved.address

    if mode == "connect":
        ctx = Context(address=addr, is_owner=False, session_dir=create_session(temp_dir))
    else:
        # launch：启动 daemon(+可选 webui)，并写 pids.json / meta.json
        inst = create_instance_dirs(temp_dir, instance_id)
        env = {**os.environ, "MYTOOL_INSTANCE": inst.id}
        # 非阻塞：直接重定向到文件；阻塞：PIPE+tee
        if not block:
            so = open_log_binary(f"{inst.logs}/daemon.out") if not stdout_file else open_log_binary(stdout_file)
            se = open_log_binary(f"{inst.logs}/daemon.err") if not stderr_file else open_log_binary(stderr_file)
            proc = subprocess.Popen(
                ["/usr/local/bin/my-daemon", "--session", inst.session],
                stdin=subprocess.DEVNULL, stdout=so, stderr=se, cwd=inst.session, env=env,
                preexec_fn=lambda: _preexec_fate_sharing(), close_fds=True)
            register_process(inst, role="daemon", proc=proc, so=so, se=se)
            write_current_instance(inst.id)
            ctx = Context(address=discover_address(inst), is_owner=True, session_dir=inst.root)
        else:
            so = open_log_binary(f"{inst.logs}/daemon.out")
            se = open_log_binary(f"{inst.logs}/daemon.err")
            proc = subprocess.Popen(
                ["/usr/local/bin/my-daemon", "--session", inst.session],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=inst.session, env=env, preexec_fn=lambda: _preexec_fate_sharing(), close_fds=True)
            register_process(inst, role="daemon", proc=proc, so=so, se=se)
            tee_process_output(proc, so, se, to_console=log_to_caller)
            ret = proc.wait()
            # 按需处理退出码并清理；示例直接返回上下文
            ctx = Context(address=discover_address(inst), is_owner=True, session_dir=inst.root)

    global _current_ctx
    with _ctx_lock:
        _current_ctx = ctx
    atexit.register(lambda: _current_ctx and _current_ctx.close())
    return ctx

def shutdown():
    ctx = current_context()
    if ctx:
        ctx.close()
```

要点：

- 驱动进程设置 SIGTERM 处理器为直接退出；配合子进程的 PR_SET_PDEATHSIG(SIGKILL)，可形成“父死子亡”。
- Launch 模式：阻塞与非阻塞分别对应 PIPE+tee 与直接文件重定向。
- Connect-only：不启动本地子进程，只建立连接并返回 `Context`。

---

## 与 Ray 行为的对应关系

- Launch（Owner）：
  - `preexec_fn` 设置 PR_SET_PDEATHSIG 与信号策略；
  - `pids.json` 登记所有子进程；
  - 日志落地到 `logs/`，CLI 阻塞模式支持 tee 到终端；
  - `shutdown()/stop` 有序发信号终止，超时后 SIGKILL。
- Connect-only：
  - 不启动本地进程；
  - `init()` 只创建最小会话与连接信息；
  - `shutdown()` 仅断开，不影响远端生命周期。

---

## 额外建议

- 对 `pids.json`/`meta.json` 设定权限 `0o600`，目录 `0o700`，避免多用户串扰。
- 健康检查尽量使用二进制提供的 `/healthz`/端口就绪信号；无则使用日志关键行。
- 针对后台模式提供 `mytool logs -f` 便于 tail 观察。
- 启动失败要回滚：打印关键日志片段、终止已起进程、删除 `current_instance` 指针。

以上为 Linux 平台的完整实现蓝图与代码基石，可直接复用到你自己的 CLI 与库 API（类似 `ray.init` 的双态行为）。
