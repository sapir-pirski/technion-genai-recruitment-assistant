#!/usr/bin/env bash
# Scheduled runs use the built app image; they do not install packages or launch the UI.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ "${1:-}" == "--hosted" ]]; then
    shift
    if [[ ! -x .venv/bin/python ]]; then
        echo "Create the native environment before running hosted synchronization." >&2
        exit 1
    fi
    exec .venv/bin/python - "$@" <<'PY'
import json
import os
import runpy
import sys
from pathlib import Path
from sqlalchemy.engine import make_url

path = Path("artifacts/hosted-database-connection.json")
try:
    if path.stat().st_mode & 0o077:
        raise ValueError("Credential file must be owner-only")
    url = json.loads(path.read_text())["database_url"]
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql" or not parsed.host or parsed.host in {
        "localhost", "127.0.0.1", "::1", "postgres"
    }:
        raise ValueError("A hosted PostgreSQL connection is required")
    if parsed.query.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("Hosted synchronization requires TLS")
except Exception:
    print("Configure an owner-only artifacts/hosted-database-connection.json with database_url.", file=sys.stderr)
    raise SystemExit(1)
os.environ["DATABASE_URL"] = url
sys.argv = ["app.main", "sync-jobs", "--trigger", "scheduled", "--cleanup", *sys.argv[1:]]
runpy.run_module("app.main", run_name="__main__")
PY
fi
if [[ "${1:-}" == "--native" ]]; then
    shift
    if [[ ! -x .venv/bin/python ]]; then
        echo "Create the native environment before scheduling." >&2
        exit 1
    fi
    exec .venv/bin/python -m app.main sync-jobs --trigger scheduled --cleanup "$@"
fi
# LaunchAgent/systemd may have a smaller PATH than an interactive terminal.
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required; run ./run-full-project.sh first." >&2
    exit 1
fi
if ! docker image inspect recruitment-chatbot:local >/dev/null 2>&1; then
    echo "Build the application image with ./run-full-project.sh before scheduling." >&2
    exit 1
fi
export APP_UID="$(id -u)" APP_GID="$(id -g)"
# The shared PostgreSQL ingestion lease prevents overlapping native/container runs.
exec docker compose run --rm --no-deps --pull never -T app python -m app.main sync-jobs --trigger scheduled --cleanup "$@"
