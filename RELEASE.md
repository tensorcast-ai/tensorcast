# TensorCast Release Guide

> **One-pager** — the essential information for a fast and painless release.

---

## 1. Quick Release (Build & Publish)

```bash
# CPU build (defaults to Torch 2.6.0)
./tools/release.sh

# Specify Torch & CUDA version (example: Torch 2.6.0 + CUDA 11.8)
./tools/release.sh --torch-version 2.6.0 --cuda-version cu118
```

Optional flags:

| Flag | Description |
|------|-------------|
| `--build-version` | Custom build number (default `0.0.2`) |
| `--skip-uv-sync`  | Skip `uv sync` (useful in CI with a pre-built env) |
| `--cache-uv-lock` | Cache the generated `uv.lock` for instant reuse |

After a successful run you will find the wheel(s) and `build_manifest_<version>.txt` in `dist/`.

---

## 2. Torch Version Management

All Torch versions are handled by `tools/manage_torch_version.py` (invoked transparently by `release.sh`).

```bash
# Check current status (pyproject.toml / .venv / Bazel)
./tools/release.sh status

# Write the Torch version installed in .venv back to pyproject.toml
./tools/release.sh update-pyproject 2.6.0

# Make .venv match pyproject.toml
./tools/release.sh sync-venv
```

The script validates that Torch is consistent across:
1. `pyproject.toml` (build-system & dependencies)
2. The version installed in `.venv` (detected via C++ headers)
3. `MODULE.bazel` references

Any mismatch aborts the build with clear remediation hints.

---

## 3. UV Lock Cache (Optional but Recommended)

`tools/uv-lock-cache/` stores pre-resolved `uv.lock` files for specific Torch/CUDA pairs:

```bash
# Cache current lock file (example: Torch 2.6.0 + CUDA 11.8)
./tools/release.sh --torch-version 2.6.0 --cuda-version cu118 --cache-uv-lock

# Future builds of the same combo reuse the cache automatically
./tools/release.sh --torch-version 2.6.0 --cuda-version cu118
```

Naming scheme:
- CPU:  `uv.lock.torch{version}-default`
- CUDA: `uv.lock.torch{version}-{cuda}` (e.g. `uv.lock.torch260-cu118`)

---

## 4. FAQ

| Issue | Resolution |
|-------|------------|
| **`uv sync` fails** | Check network, clear pip cache, or use `--skip-uv-sync` in CI. |
| **Version mismatch stops build** | Run `./tools/release.sh status`, then fix with `update-pyproject` or `sync-venv`. |
| **Bazel cannot find Torch** | Ensure `MODULE.bazel` points to the Python path inside `.venv`. |

---

## 5. Reference Scripts & Docs

- `tools/release.sh` — unified entry point
- `tools/manage_torch_version.py` — core version logic
- `tools/uv-lock-cache/README.md` — lock-file cache details
- `tools/README_torch_version_management.md` — full design doc

Happy shipping! 🚀