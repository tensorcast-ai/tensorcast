# TensorCast Release Guide (SOP)

Step-by-step procedure for cutting a tensorcast release to PyPI. v0.1.0 is the
first public release; this document captures the manual flow until we migrate
it to GitHub Actions in v0.2.

---

## 0. Prerequisites (one-time)

### 0.1 PyPI / TestPyPI accounts

1. Register at <https://pypi.org/account/register/> and
   <https://test.pypi.org/account/register/> (they are separate user systems).
2. On each site: **Account Settings → API tokens → Add API token**, scope it
   to the `tensorcast` project once the name is reserved.
3. Save the tokens in `~/.pypirc`:

   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEI...

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = pypi-AgEI...
   ```

4. `chmod 600 ~/.pypirc` so other users on the host cannot read the tokens.

### 0.2 Tooling

- `uv` (Python package manager) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `docker` — required for Stage B (manylinux build).
- The release toolchain (`wheel`, `auditwheel`, `patchelf`, `twine`) lives in
  the `release` dependency group in `pyproject.toml`. `tools/release.sh`
  invokes `uv sync --group release` automatically before any subcommand that
  needs them (`post-process`, `check`, `publish*`), so you only need `uv` on
  PATH — no system `apt install patchelf` step required.

---

## 1. Release matrix (v0.1.0)

| Axis | Value |
|---|---|
| tensorcast version | `0.1.0` |
| torch | `2.11.0` |
| CUDA | `12.8` (PyTorch index `cu128`) |
| Python | 3.10 / 3.11 / 3.12 |
| Platform | `manylinux_2_28_x86_64` (glibc ≥ 2.28) |
| OS | Linux only (kernel ≥ 5.10) |

Other torch versions are not published as wheels; users with a different
torch must build from source — see §5.

---

## 2. Release flow

The pipeline is two-stage:

- **Stage A** — local build, `linux_x86_64` platform tag, **does not go to
  PyPI**. Used to iterate on packaging logic (RPATH, daemon copy, ABI
  guard) without paying the docker round-trip.
- **Stage B** — manylinux_2_28 docker build, `manylinux_2_28_x86_64`
  platform tag, **the only wheel allowed on PyPI**.

```
Stage A: tools/release.sh build           (local Ubuntu, linux_x86_64)
   │
   ▼  packaging logic verified, e2e PASS
Stage B: tools/release.sh build --in-docker --pypi   (docker, manylinux_2_28)
   │
   ▼
   tools/release.sh check               (twine check)
   tools/release.sh publish-test        (TestPyPI dry-run)
   tools/release.sh publish             (production PyPI)
```

### 2.1 Stage A — local validation

```bash
tools/release.sh build
# → dist/tensorcast-0.1.0+torch211.cu128-cp310-cp310-linux_x86_64.whl
```

Install into a clean venv and run the smoke tests in §4. Iterate here until
everything is green.

To preview the PyPI-uploadable version string locally (still `linux_x86_64`,
still not for PyPI):

```bash
tools/release.sh build --pypi
# → dist/tensorcast-0.1.0-cp310-cp310-linux_x86_64.whl
```

### 2.2 Stage B — manylinux wheel for PyPI

```bash
tools/release.sh build --in-docker --pypi
# → dist/tensorcast-0.1.0-cp310-cp310-manylinux_2_28_x86_64.whl
```

The first run pulls the `quay.io/pypa/manylinux_2_28_x86_64` image and
installs Bazel + clang18 inside it (see `docker/release.Dockerfile`); later
runs reuse the cached image. The script reruns itself inside the container
with `IN_DOCKER=1`, which makes `tools/wheel_post_process.py` invoke
`auditwheel repair` automatically.

Repeat for each target Python (3.10 / 3.11 / 3.12). Today this requires
switching the docker image's active Python interpreter; v0.2 will automate
that via a matrix.

### 2.3 Pre-flight checks

```bash
tools/release.sh check                                          # twine check
unzip -l dist/tensorcast-0.1.0-*manylinux*.whl | grep libtorch  # must be empty
unzip -l dist/tensorcast-0.1.0-*manylinux*.whl | grep libcuda   # must be empty
du -h dist/tensorcast-0.1.0-*manylinux*.whl                     # < 200 MB
```

### 2.4 TestPyPI dry-run

```bash
tools/release.sh publish-test
# Then in a clean venv:
python -m venv /tmp/tc && source /tmp/tc/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            tensorcast==0.1.0
python -c "import tensorcast; print(tensorcast.__version__)"
tensorcast-cli daemon start --help
```

If anything fails, fix and **bump to `0.1.0.post1`** — TestPyPI (like PyPI)
rejects re-uploading the same version.

### 2.5 Production PyPI

```bash
tools/release.sh publish
# (prompts you to retype the version from version.txt as confirmation)
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
| Internal release | `RELEASE=1` (set by `release.sh build`) | `0.1.0+torch211.cu128` | internal matrix wheels, build manifests |
| PyPI release | `RELEASE_PYPI=1` (set by `release.sh build --pypi`) | `0.1.0` | the only string PyPI accepts |

`version.txt` is the source of truth for the base version. Bump it when you
start a new release cycle.

---

## 4. Smoke tests for a built wheel

```bash
python -m venv /tmp/tc && source /tmp/tc/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install dist/tensorcast-0.1.0-*.whl

# 1. Import + version
python -c "import tensorcast; print(tensorcast.__version__)"

# 2. CLI launcher
tensorcast-cli daemon start --help

# 3. ABI guard fires when torch is wrong (in another venv)
pip install torch==2.10.0
python -c "import tensorcast" 2>&1 | grep "tensorcast was built against"

# 4. Minimal e2e (needs CUDA driver)
python <<'EOF'
import tensorcast as tc, torch
tc.init(mode="create")
state = {"layer.weight": torch.randn(8, 8, device="cuda")}
tc.register(state, key="demo:model:001")
handle = tc.artifact(key="demo:model:001")
tensors = handle.tensor_dict(device="cuda:0")
print(tensors)
EOF
```

---

## 5. Building from source for a non-default torch / CUDA

Users who must run a different torch version (e.g. coexisting with another
library that pins torch 2.13) must build locally:

```bash
git clone https://github.com/tensorcast-ai/tensorcast.git
cd tensorcast
tools/release.sh build --torch-version 2.13.0 --cuda-version cu128
pip install dist/tensorcast-*.whl
```

`tools/release.sh build` patches `pyproject.toml` in memory for the duration
of the build, runs `tools/update_module_http_archives.py` to align Bazel's
LibTorch / NVIDIA pins, and restores `pyproject.toml` afterwards. The wheel
that drops in `dist/` is `linux_x86_64`-tagged (PyPA does not accept this
for upload, but pip installs it fine locally).

The ABI fail-fast guard reads the torch version baked into
`tensorcast/_version.py` at build time, so no source edits are needed.

---

## 6. Hotfix / patch flow

Patch releases (`0.1.1`, `0.1.2`) reuse the same flow with these tweaks:

1. Bump `version.txt` on `main`.
2. Land any cherry-picked fixes.
3. Re-run §2.2 / §2.4 / §2.5.
4. Tag `v0.1.1`.

Long-lived `releases/v0.1.x` branches are not used until we have enough
release volume to need them (likely v0.2+).

---

## 7. Reference

- `tools/release.sh` — release driver (this guide is its user manual).
- `tools/wheel_post_process.py` — patchelf + strip + chmod + auditwheel.
- `tools/torch_version_manager.py` — pyproject/Bazel/.venv torch consistency.
- `tools/update_module_http_archives.py` — keep `MODULE.bazel` in sync with `uv.lock`.
- `tools/uv-lock-cache/README.md` — per-matrix `uv.lock` cache details.
- `docker/release.Dockerfile` — manylinux_2_28 build environment (Phase
  4-Docker / Stage B).
