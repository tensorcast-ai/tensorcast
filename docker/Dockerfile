FROM hub.i.basemind.com/stepcast/python:uv-bookworm-slim as builder

ARG HTTPS_PROXY
ARG HTTP_PROXY
ARG NO_PROXY

ENV HTTPS_PROXY=$HTTPS_PROXY
ENV HTTP_PROXY=$HTTP_PROXY
ENV NO_PROXY=$NO_PROXY

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

ENV UV_PYTHON_INSTALL_DIR=/python

ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.10

RUN apt update && apt install -y git

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra global --extra local --extra ui --frozen --no-install-project --no-dev

ADD .python-version /app/.python-version
ADD setup.py /app/setup.py
ADD version.txt /app/version.txt
ADD versions.py /app/versions.py
ADD dev_dep_versions.yml /app/dev_dep_versions.yml
ADD pyproject.toml /app/pyproject.toml
ADD uv.lock /app/uv.lock
ADD README.md /app/README.md

ADD scstore/global_store /app/scstore/global_store
ADD scstore/*.py /app/scstore/
ADD scstore/proto/*.py /app/scstore/proto/
ADD scstore/cli_utils /app/scstore/cli_utils
ADD scstore/store_daemon /app/scstore/store_daemon

ADD .git /app/.git

# Remove the uv pip install from builder stage
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --no-deps -e .

FROM hub.i.basemind.com/stepcast/debian:bookworm-slim as ui_builder

RUN apt-get update && apt-get install -y nodejs npm curl build-essential zip unzip openjdk-17-jdk-headless

ADD scstore/global_store /app/scstore/global_store

# -----------------------------------------------------------------------------
# Build the Global-Store Web-UI (static assets served by FastAPI backend)
# -----------------------------------------------------------------------------
RUN npm install -g pnpm && \
    cd /app/scstore/global_store/webui_frontend && \
    pnpm install --frozen-lockfile --force && \
    pnpm tsc --noEmit && \
    pnpm build


FROM hub.i.basemind.com/stepcast/python:uv-bookworm-slim

RUN apt update && apt install -y git curl zip unzip openjdk-17-jdk-headless openssh-server vim net-tools iputils-ping iproute2

# Install grpc-health-probe for health checks
RUN GRPC_HEALTH_PROBE_VERSION=v0.4.39 && \
    curl -fsSL https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/${GRPC_HEALTH_PROBE_VERSION}/grpc_health_probe-linux-amd64 \
    -o /bin/grpc_health_probe && \
    chmod +x /bin/grpc_health_probe

RUN uv python install 3.10

COPY --from=builder --chown=python:python /python /python

# First copy only the dependencies (.venv) - this layer can be cached when only code changes
# /app/.venv/lib/python3.10/site-packages/nvidia 2.7G
# /app/.venv/lib/python3.10/site-packages/torch 1.5G
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Copy configuration files that rarely change
COPY --from=builder --chown=app:app /app/.python-version /app/.python-version
COPY --from=builder --chown=app:app /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=app:app /app/uv.lock /app/uv.lock

# Copy version and setup files
COPY --from=builder --chown=app:app /app/version.txt /app/version.txt
COPY --from=builder --chown=app:app /app/versions.py /app/versions.py
COPY --from=builder --chown=app:app /app/dev_dep_versions.yml /app/dev_dep_versions.yml
COPY --from=builder --chown=app:app /app/setup.py /app/setup.py
COPY --from=builder --chown=app:app /app/README.md /app/README.md

# Copy application code (changes more frequently)
COPY --from=builder --chown=app:app /app/scstore /app/scstore

# Copy .git last as it changes with every commit
COPY --from=builder --chown=app:app /app/.git /app/.git

WORKDIR /app

# Install the application in editable mode after copying all files
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

ADD scstore/lib/libscstore.so /app/scstore/lib/libscstore.so

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ADD docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ADD scstore/*.so /app/scstore/

COPY --from=ui_builder --chown=python:python /app/scstore/global_store /app/scstore/global_store

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command prints help if no arguments supplied
CMD ["/app/entrypoint.sh", "help"]