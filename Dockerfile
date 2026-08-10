# ---------------------------------------------------------------------------
# SiliconStat -- multi-stage image
#
# Stage 1 builds the React dashboard; stage 2 installs the Python package and
# copies the built dashboard in, so the single container serves both the REST
# API and the UI from one origin (no CORS, no second process).
# ---------------------------------------------------------------------------

# --- stage 1: frontend ------------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build


# --- stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SILICONSTAT_DB=/data/siliconstat.sqlite \
    SILICONSTAT_EXAMPLES=/app/examples \
    SILICONSTAT_FRONTEND=/app/frontend/dist

# libgomp is needed by the SciPy / scikit-learn wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip && pip install ".[all]"

COPY examples/ ./examples/
COPY docs/ ./docs/
COPY --from=frontend /build/dist ./frontend/dist

RUN useradd --create-home --uid 10001 siliconstat \
 && mkdir -p /data && chown -R siliconstat:siliconstat /data /app
USER siliconstat

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "siliconstat.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
