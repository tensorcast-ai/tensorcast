# TensorCast Release Guide (SOP)

Step-by-step procedure for cutting a tensorcast release to PyPI.

---

## 1. Release Matrix (v0.1.0)

| Axis | Value |
|---|---|
| tensorcast version | `0.1.0` |
| torch | `2.11.0` |
| CUDA | `12.8` (PyTorch index `cu128`) |
| Python | 3.10 / 3.11 / 3.12 |
| Platform | `manylinux_2_28_x86_64` (glibc ≥ 2.28) |
| OS | Linux only (kernel ≥ 5.10) |

---

## 2. Release Flow

Two-stage pipeline:

- **Stage A** — local build, `linux_x86_64` tag. Quick iteration on packaging logic. **Not for PyPI.**
- **Stage B** — manylinux docker build, `manylinux_2_28_x86_64` tag. **Only wheel allowed on PyPI.**

```
Stage A: tools/release.sh build           (local, linux_x86_64)
   │
   ▼  packaging logic verified, smoke tests PASS
Stage B: tools/release.sh build --in-docker --pypi   (docker, manylinux_2_28)
   │
   ▼
   tools/release.sh check               (twine check)
   tools/release.sh publish-test        (TestPyPI dry-run)
   tools/release.sh publish             (production PyPI)
```

### 2.1 Stage A — Local Validation

```bash
tools/release.sh build
# → dist/tensorcast-0.1.0+torch211.cu128-cp310-cp310-linux_x86_64.whl
```

Install into a clean venv and run smoke tests in §4.

To preview the PyPI version string locally:

```bash
tools/release.sh build --pypi
# → dist/tensorcast-0.1.0-cp310-cp310-linux_x86_64.whl
```

### 2.2 Stage B — Manylinux Wheel for PyPI

```bash
# Build the Docker image once
docker build -f docker/release.Dockerfile -t tensorcast-builder:latest .

# Run the build inside the container
docker run --rm \
  -v $(pwd):/io \
  -w /io \
  -e IN_DOCKER=1 \
  tensorcast-builder:latest \
  bash docker/build_in_docker.sh --pypi
# → dist/tensorcast-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

`auditwheel repair` runs automatically inside the container, producing the
`manylinux_2_28_x86_64` tag required by PyPI.

### 2.3 Pre-flight Checks

```bash
tools/release.sh check                                          # twine check
unzip -l dist/tensorcast-0.1.0-*manylinux*.whl | grep -E 'libtorch|libcuda|libcudnn'  # must be empty
du -h dist/tensorcast-0.1.0-*manylinux*.whl                   # < 200 MB
```

### 2.4 TestPyPI Dry-run

```bash
tools/release.sh publish-test
```

Verify in a clean venv:

```bash
python -m venv /tmp/tc && source /tmp/tc/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            tensorcast==0.1.0
python -c "import tensorcast; print(tensorcast.__version__)"
tensorcast-cli daemon start --help
```

If anything fails, fix and bump to `0.1.0.post1` — TestPyPI rejects re-uploading
the same version.

### 2.5 Production PyPI

```bash
tools/release.sh publish
```

Then:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Create the GitHub Release at <https://github.com/tensorcast-ai/tensorcast/releases/new>;
attach the manylinux wheel(s) as a backup mirror.

---

## 3. Versioning

Three version strings are produced by `setup.py`, controlled by env vars:

| Mode | Trigger | Version string | Use |
|---|---|---|---|
| Dev | default | `0.1.0.dev0+<gitsha>.torch211.cu128` | local iterative builds |
| Internal release | `RELEASE=1` | `0.1.0+torch211.cu128` | internal matrix wheels |
| PyPI release | `RELEASE_PYPI=1` (set by `--pypi`) | `0.1.0` | PyPI upload |

`version.txt` is the source of truth. Bump it when starting a new release cycle.

---

## 4. Smoke Tests for a Built Wheel

Run these after Stage A or Stage B to verify the wheel is healthy.

### 4.1 Install and basic checks

```bash
python -m venv /tmp/tc && source /tmp/tc/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install dist/tensorcast-0.1.0-*.whl

# Import + version
python -c "import tensorcast; print(tensorcast.__version__)"

# CLI launcher
tensorcast-cli daemon start --help

# Daemon binary exists and is executable
ls -l $(python -c "from tensorcast.cli_utils.proc import ensure_cpp_daemon_binary; print(ensure_cpp_daemon_binary())")
```

### 4.2 ABI guard

In a separate venv, force an incompatible torch version:

```bash
pip install torch==2.10.0
python -c "import tensorcast" 2>&1 | grep "tensorcast was built against"
# → ImportError with clear mismatch message

# Escape hatch
TENSORCAST_SKIP_TORCH_ABI_CHECK=1 python -c "import tensorcast"
```

### 4.3 Module imports

Verify key modules import cleanly:

```bash
python -c "
import tensorcast
import tensorcast._C
import tensorcast.cli
import tensorcast.api.store
import tensorcast.runtime
print('All imports OK')
"
```

### 4.4 Wheel content audit

```bash
# No torch/CUDA libraries bundled
unzip -l dist/tensorcast-0.1.0-*.whl | grep -E 'libtorch|libcuda|libcudnn' || echo "PASS: no bundled CUDA/torch libs"

# Daemon is present and executable
unzip -l dist/tensorcast-0.1.0-*.whl | grep tensorcast_daemon

# Configs are present
unzip -l dist/tensorcast-0.1.0-*.whl | grep 'examples/config.*\.yaml'
```

---

## 5. Building from Source for a Non-default Torch / CUDA

Users who need a different torch version must build locally:

```bash
git clone https://github.com/tensorcast-ai/tensorcast.git
cd tensorcast
tools/release.sh build --torch-version 2.13.0 --cuda-version cu128
pip install dist/tensorcast-*.whl
```

The ABI fail-fast guard reads the torch version baked into
`tensorcast/_version.py` at build time, so no source edits are needed.

---

## 6. Hotfix / Patch Flow

Patch releases (`0.1.1`, `0.1.2`) reuse the same flow:

1. Bump `version.txt` on `main`.
2. Cherry-pick fixes.
3. Re-run §2.2 / §2.4 / §2.5.
4. Tag `v0.1.1`.

---

## 7. Reference

- `tools/release.sh` — release driver.
- `tools/wheel_post_process.py` — patchelf + strip + chmod + auditwheel.
- `docker/release.Dockerfile` — manylinux build environment.
- `docker/build_in_docker.sh` — config swap and build orchestration inside Docker.
