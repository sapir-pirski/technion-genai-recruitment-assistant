#!/usr/bin/env bash
# Scheduled runs reuse the project venv and hosted database configuration.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ ! -x .venv/bin/python ]]; then
    echo "Run ./run-full-project.sh to prepare the project venv first." >&2
    exit 1
fi
exec .venv/bin/python -m app.main sync-jobs --trigger scheduled --cleanup "$@"
