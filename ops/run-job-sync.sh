#!/usr/bin/env bash
# Scheduled runs use the built app image; they do not install packages or launch the UI.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
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
