#!/usr/bin/env bash
# Run the prepared project environment; setup and PDF ingestion are explicit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: ./run-full-project.sh [Streamlit options]

Run Streamlit in the foreground on port 8502. Stop with Ctrl+C.
Examples:
  ./run-full-project.sh
  ./run-full-project.sh --server.port 8503
  ./run-full-project.sh --server.headless true

First create .venv, install requirements.txt, configure .env, and explicitly run:
  .venv/bin/python -m app.main ingest
HELP
    exit 0
fi

if [[ ! -x .venv/bin/python ]]; then
    cat >&2 <<'HELP'
Create and prepare the project environment first:
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
Then configure .env and run PDF ingestion. See README.md.
HELP
    exit 1
fi

exec .venv/bin/python -m streamlit run streamlit_app/streamlit_main.py --server.port 8502 "$@"
