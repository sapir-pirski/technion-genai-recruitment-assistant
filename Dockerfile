# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libreoffice-writer fonts-dejavu-core libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv /opt/venv

WORKDIR /app
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

ARG APP_UID=1000
ARG APP_GID=1000
RUN test "$APP_UID" -gt 0 \
    && (getent group "$APP_GID" >/dev/null || groupadd --gid "$APP_GID" app) \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home --shell /usr/sbin/nologin app \
    && mkdir -p data/jobs config artifacts/ingestion \
    && chown -R app data config artifacts

COPY app/ ./app/
COPY streamlit_app/ ./streamlit_app/
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY ops/container_entrypoint.py ./ops/container_entrypoint.py
COPY ops/launch_support.py ./ops/launch_support.py
COPY LICENSE.md ./
COPY tests/fixtures/ ./tests/fixtures/

USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=6 \
    CMD ["python", "-c", "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2).read() == b'ok'"]
ENTRYPOINT ["python", "-m", "ops.container_entrypoint"]
CMD ["python", "-m", "streamlit", "run", "streamlit_app/streamlit_main.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
