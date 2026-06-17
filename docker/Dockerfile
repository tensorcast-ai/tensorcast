FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

ARG TENSORCAST_BUILD_COMMIT

ENV TENSORCAST_BUILD_COMMIT=${TENSORCAST_BUILD_COMMIT}
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.10

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --no-install-project --no-dev

COPY .python-version /app/.python-version
COPY setup.py /app/setup.py
COPY version.txt /app/version.txt
COPY versions.py /app/versions.py
COPY dev_dep_versions.yml /app/dev_dep_versions.yml
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md

COPY tensorcast/global_store /app/tensorcast/global_store
COPY tensorcast/*.py /app/tensorcast/
COPY tensorcast/proto/*.py /app/tensorcast/proto/
COPY tensorcast/cli_utils /app/tensorcast/cli_utils

FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

ARG TENSORCAST_BUILD_COMMIT

ENV TENSORCAST_BUILD_COMMIT=${TENSORCAST_BUILD_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    iproute2 \
    iputils-ping \
    net-tools \
    openssh-server \
    unzip \
    vim \
    zip \
    && rm -rf /var/lib/apt/lists/*

RUN GRPC_HEALTH_PROBE_VERSION=v0.4.39 && \
    curl -fsSL "https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64" \
      -o /bin/grpc_health_probe && \
    chmod +x /bin/grpc_health_probe

RUN uv python install 3.10

COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.python-version /app/.python-version
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY --from=builder /app/version.txt /app/version.txt
COPY --from=builder /app/versions.py /app/versions.py
COPY --from=builder /app/dev_dep_versions.yml /app/dev_dep_versions.yml
COPY --from=builder /app/setup.py /app/setup.py
COPY --from=builder /app/README.md /app/README.md
COPY --from=builder /app/tensorcast /app/tensorcast

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

COPY docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["/app/entrypoint.sh", "help"]
