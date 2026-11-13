FROM python:3.11-slim-bookworm AS build
ARG UV_VERSION=0.13.0
ARG INSTALL_DISCOVERY=false
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==${UV_VERSION}
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN if [ "$INSTALL_DISCOVERY" = "true" ]; then uv sync --frozen --no-dev --no-editable --extra discovery; else uv sync --frozen --no-dev --no-editable; fi

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app \
    && mkdir /data && chown app:app /data
ENV PATH="/app/.venv/bin:$PATH" \
    APP_ENV=production \
    CONFIG_PATH=/app/configs/pipeline_config.yaml DATA_DIR=/data \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY configs ./configs
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "ticket_classification.controllers.healthcheck", "api"]
CMD ["ticket-api"]
