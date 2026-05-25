# Contributing

Thank you for contributing to TensorCast.

## Development

Use the repository setup and test commands in `README.md` and `AGENTS.md`.

Typical local setup:

```bash
uv sync --locked --all-extras --dev
source .venv/bin/activate
pre-commit install
```

Python tests should be run with:

```bash
pytest tests/python/...
```

C++ tests should be run with the matching Bazel target:

```bash
bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake
```

Before opening a pull request, run the focused tests for your change and the
relevant linters:

```bash
ruff check .
ruff format . --check
```

If you modify protocol buffers, regenerate Python stubs and C++ headers:

```bash
bash tools/build_proto_python.sh
```

## Developer Certificate of Origin

TensorCast uses the Developer Certificate of Origin (DCO) for inbound
contributions. By signing off a commit, you certify that you have the right to
submit the contribution under the project licenses.

Sign off every commit with:

```bash
git commit -s
```

This adds a `Signed-off-by` line to the commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Do not submit code that you do not have the right to contribute.
