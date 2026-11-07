#!/usr/bin/env bash
# Repeatable Docker/native setup and lifecycle commands.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
project_root="$PWD"
native=no
action=start
open_browser=yes
port=""
explicit_port=no
streamlit_args=()
phase="argument validation"

fail() { echo "$1" >&2; exit 1; }
usage() {
    cat <<'HELP'
Usage: ./run-full-project.sh [--native] [--port NUMBER] [--no-browser]
       ./run-full-project.sh [--native] --status|--logs|--stop|--restart

Default: cached Docker build, runtime/dependency checks, database readiness,
conditional PDF ingestion, healthy app startup, then open the browser.
Native: reuse/install Python 3.11+ (prefer 3.13), repair .venv, install only
missing/changed requirements, and install LibreOffice when supported.
--port is explicit; otherwise reuse this project's port or choose a free one.
--stop preserves data. Native stop affects only this launcher's managed process.
PYTHON_BIN selects a native interpreter. No credentials are printed or overwritten.
HELP
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --native) native=yes; shift ;;
        --no-browser) open_browser=no; shift ;;
        --status|--logs|--stop|--restart)
            [[ "$action" == start ]] || fail "Choose one lifecycle action."
            action="${1#--}"; shift ;;
        --port|--server.port)
            [[ $# -ge 2 ]] || fail "Supply a port between 1 and 65535."
            [[ "$2" =~ ^[0-9]+$ ]] && [[ "$2" -ge 1 && "$2" -le 65535 ]] || fail "Use a port between 1 and 65535."
            port="$2"; explicit_port=yes; shift 2 ;;
        --server.port=*)
            port="${1#*=}"
            [[ "$port" =~ ^[0-9]+$ ]] && [[ "$port" -ge 1 && "$port" -le 65535 ]] || fail "Use a port between 1 and 65535."
            explicit_port=yes; shift ;;
        *) [[ "$native" == yes ]] || fail "Unknown option; use --help."
            streamlit_args+=("$1"); shift ;;
    esac
done
umask 077
runtime="$project_root/artifacts/launcher"
mkdir -p "$runtime"
launcher_log="$runtime/launcher.log"
touch "$launcher_log"
chmod 600 "$launcher_log"
trap 'echo "$phase failed. See artifacts/launcher/launcher.log; no existing database or credentials were reset." >&2' ERR

stage() {
    phase="$1"; shift
    echo "$phase..."
    "$@" >>"$launcher_log" 2>&1
}
checksum() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$@" | awk '{print $1}'
    else fail "A SHA-256 utility (sha256sum or shasum) is required."
    fi
}
build_fingerprint() {
    {
        printf '%s\n' "$APP_UID:$APP_GID"
        while IFS= read -r input; do
            case "$input" in */__pycache__/*|*.pyc|*.pyo|*.pyd|*/.DS_Store|*/.env*|*/secrets.toml) continue ;; esac
            printf '%s\n' "$input"
            if [[ -f "$input" ]]; then checksum "$input"; else echo missing; fi
        done < <({
            printf '%s\n' Dockerfile .dockerignore requirements.txt LICENSE.md .streamlit/config.toml ops/container_entrypoint.py ops/launch_support.py
            printf '%s\n' tests/fixtures/quality_cases.json tests/fixtures/workflow_cases.json
            find app streamlit_app -type f 2>/dev/null || true
        } | LC_ALL=C sort)
    } | checksum
}
ensure_image() {
    local fingerprint image_id saved_fingerprint saved_image
    fingerprint="$(build_fingerprint)"
    image_id="$(docker image inspect --format '{{.Id}}' recruitment-chatbot:local 2>>"$launcher_log" || true)"
    saved_fingerprint=""; saved_image=""
    if [[ -f "$runtime/docker-build.state" ]]; then
        saved_fingerprint="$(sed -n '1p' "$runtime/docker-build.state")"
        saved_image="$(sed -n '2p' "$runtime/docker-build.state")"
    fi
    if [[ -n "$image_id" && "$fingerprint" == "$saved_fingerprint" && "$image_id" == "$saved_image" ]]; then
        echo "Application image is current; reusing it."
        return
    fi
    stage "Building changed application inputs" docker compose build app
    image_id="$(docker image inspect --format '{{.Id}}' recruitment-chatbot:local 2>>"$launcher_log")"
    printf '%s\n%s\n' "$fingerprint" "$image_id" > "$runtime/docker-build.state.tmp"
    mv "$runtime/docker-build.state.tmp" "$runtime/docker-build.state"
}
busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
choose_port() {
    if [[ -z "$port" ]]; then
        port=8502
        while busy "$port"; do
            port=$((port + 1))
            [[ "$port" -le 8602 ]] || fail "No free port found; choose --port explicitly."
        done
    elif busy "$port"; then
        fail "Port $port is occupied by another process. Choose --port NUMBER."
    fi
}
open_page() {
    local url="http://127.0.0.1:$port"
    echo "App ready: $url"
    [[ "$open_browser" == yes ]] || return 0
    case "$(uname -s)" in
        Darwin) open "$url" >>"$launcher_log" 2>&1 || echo "Open $url in your browser." ;;
        Linux)
            if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v xdg-open >/dev/null 2>&1; then
                xdg-open "$url" >>"$launcher_log" 2>&1 || echo "Open $url in your browser."
            elif command -v powershell.exe >/dev/null 2>&1; then
                powershell.exe -NoProfile -Command "Start-Process '$url'" >>"$launcher_log" 2>&1 || echo "Open $url in your browser."
            else echo "No desktop browser available; use the URL above."
            fi ;;
        *) echo "Open $url in your browser." ;;
    esac
}
prepare_env() {
    if [[ ! -e .env ]]; then
        cp .env.example .env
        chmod 600 .env
        fail "Created .env. Enter OPENAI_API_KEY and the hosted DATABASE_URL, then rerun."
    fi
}
docker_ready() {
    command -v docker >/dev/null 2>&1 || fail "Install Docker with Compose, or use --native with hosted PostgreSQL."
    docker compose version >>"$launcher_log" 2>&1 || fail "Docker Compose is required."
    if ! docker info >/dev/null 2>&1; then
        if [[ "$action" == status || "$action" == stop || "$action" == logs ]]; then
            echo "Docker is unavailable; check daemon readiness and access permissions. No containers were changed."
            exit 0
        fi
        echo "Starting Docker..."
        case "$(uname -s)" in
            Darwin) open -ga Docker >>"$launcher_log" 2>&1 || fail "Start Docker Desktop and rerun." ;;
            Linux) systemctl --user start docker >>"$launcher_log" 2>&1 || true ;;
        esac
        local deadline=$((SECONDS + 90))
        until docker info >/dev/null 2>&1; do
            [[ "$SECONDS" -lt "$deadline" ]] || fail "Docker did not become ready. Start its daemon and rerun."
            sleep 1
        done
    fi
}
compatible_python() { "$1" -c 'import sys; sys.exit(sys.version_info < (3,11))' >/dev/null 2>&1; }
find_python() {
    python_bin=""
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        compatible_python "$PYTHON_BIN" || fail "PYTHON_BIN must be a working Python 3.11+ interpreter."
        python_bin="$PYTHON_BIN"; return
    fi
    for candidate in .venv/bin/python python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && compatible_python "$candidate"; then
            python_bin="$(command -v "$candidate")"; return
        fi
    done
}
install_python() {
    local uv_bin=""
    if command -v uv >/dev/null 2>&1; then uv_bin="$(command -v uv)"
    elif [[ -x "$runtime/tools/uv" ]]; then uv_bin="$runtime/tools/uv"
    else
        command -v curl >/dev/null 2>&1 || fail "curl is needed to download the Python installer."
        stage "Downloading the official uv installer" curl --fail --location --proto '=https' --tlsv1.2 --retry 2 https://astral.sh/uv/install.sh -o "$runtime/uv-installer.sh"
        stage "Installing project-local uv" env UV_UNMANAGED_INSTALL="$runtime/tools" sh "$runtime/uv-installer.sh"
        uv_bin="$runtime/tools/uv"
    fi
    stage "Installing project-managed Python 3.13" env UV_PYTHON_INSTALL_DIR="$runtime/python" "$uv_bin" python install --no-bin 3.13
    python_bin="$(UV_PYTHON_INSTALL_DIR="$runtime/python" "$uv_bin" python find --managed-python --no-project --no-python-downloads 3.13 2>>"$launcher_log")"
    compatible_python "$python_bin" || fail "Python installation did not produce a usable runtime."
}
converter_ready() {
    .venv/bin/python -c 'from app.profiles import document_converter; raise SystemExit(0 if document_converter() else 1)' >/dev/null 2>&1
}
install_converter() {
    case "$(uname -s)" in
        Darwin)
            command -v brew >/dev/null 2>&1 || fail "Homebrew is needed to install native LibreOffice; install it or use Docker mode."
            stage "Installing LibreOffice" brew install --cask libreoffice ;;
        Linux)
            local elevate=()
            if [[ "$(id -u)" != 0 ]]; then
                command -v sudo >/dev/null 2>&1 || fail "Installing LibreOffice needs sudo or an administrator; Docker mode includes it."
                sudo -v || fail "Administrator access is needed to install LibreOffice."
                elevate=(sudo)
            fi
            if command -v apt-get >/dev/null 2>&1; then
                stage "Updating package metadata" ${elevate[@]+"${elevate[@]}"} apt-get update
                stage "Installing LibreOffice" ${elevate[@]+"${elevate[@]}"} apt-get install -y libreoffice-writer
            elif command -v dnf >/dev/null 2>&1; then
                stage "Installing LibreOffice" ${elevate[@]+"${elevate[@]}"} dnf install -y libreoffice-writer
            else fail "Install LibreOffice with your platform's package manager, or use Docker mode."
            fi ;;
        *) fail "Automatic native setup supports macOS/Linux/WSL; use Docker on this platform." ;;
    esac
    converter_ready || fail "LibreOffice is still unavailable; check the launcher log."
}

if [[ "$native" == no ]]; then
    docker_ready
    [[ -e .env ]] || {
        if [[ "$action" == start || "$action" == restart ]]; then prepare_env
        else fail "No .env exists yet; run the launcher without lifecycle flags first."
        fi
    }
    case "$action" in
        status) docker compose ps; exit 0 ;;
        logs) docker compose logs --tail 100 app; exit 0 ;;
        stop) stage "Stopping the app container" docker compose stop app; echo "Stopped; the hosted database is unchanged."; exit 0 ;;
    esac
    export APP_UID="$(id -u)" APP_GID="$(id -g)"
    existing_port="$(docker compose port app 8501 2>>"$launcher_log" | awk -F: 'NR==1 {print $NF}' || true)"
    if [[ -n "$existing_port" && "$explicit_port" == no ]]; then port="$existing_port"
    elif [[ -z "$existing_port" || "$port" != "$existing_port" ]]; then choose_port
    fi
    [[ "$port" =~ ^[0-9]+$ ]] || fail "Could not determine the app port."
    export APP_HTTP_PORT="$port"
    ensure_image
    phase="container runtime, requirements and credentials"
    echo "Checking Python, requirements, credentials and document support..."
    docker compose run --rm --no-deps -T app --check-config >>"$launcher_log" 2>&1
    stage "Checking PDFs and published indexes" docker compose run --rm --no-deps -T app python -m app.main bootstrap --if-needed
    if [[ "$action" == restart ]]; then
        stage "Restarting the application" docker compose up -d --no-deps --force-recreate --wait --wait-timeout 90 app
    else
        stage "Starting or reusing the application" docker compose up -d --no-deps --wait --wait-timeout 90 app
    fi
    open_page
    echo "Use --status, --logs, --stop or --restart for this project."
    exit 0
fi

find_python
if [[ "$action" == status || "$action" == stop || "$action" == logs ]]; then
    if [[ -z "$python_bin" ]]; then
        echo "No compatible native Python is available; no managed app can be inspected."
        exit 0
    fi
    "$python_bin" -m ops.launch_support "$action"
    exit 0
fi
prepare_env
[[ -n "$python_bin" ]] || install_python
phase="native virtual environment"
native_changed=no
if ! [[ -x .venv/bin/python && -f .venv/pyvenv.cfg ]] || ! .venv/bin/python -c 'import pathlib,sys; raise SystemExit(sys.version_info<(3,11) or pathlib.Path(sys.prefix).resolve()!=pathlib.Path(".venv").resolve())' >/dev/null 2>&1; then
    python_bin="$("$python_bin" -c 'import os,sys; print(os.path.realpath(getattr(sys,"_base_executable",sys.executable)))')"
    case "$python_bin" in "$project_root"/.venv/*) install_python ;; esac
    if [[ -e .venv || -L .venv ]]; then
        backup="$runtime/venv-backup-$(date +%Y%m%d%H%M%S)-$$"
        mv .venv "$backup"
        echo "Preserved the incomplete environment under artifacts/launcher/."
    fi
    stage "Creating the native virtual environment" "$python_bin" -m venv .venv
    native_changed=yes
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    stage "Repairing pip" .venv/bin/python -m ensurepip --upgrade
fi
echo "Native runtime: $(.venv/bin/python --version)"
if .venv/bin/python -m ops.launch_support dependencies; then
    echo "Requirements are already installed and consistent."
else
    stage "Installing missing or changed requirements" .venv/bin/python -m pip install -r requirements.txt
    native_changed=yes
    .venv/bin/python -m ops.launch_support dependencies || fail "Dependencies remain inconsistent; inspect the launcher log."
fi
converter_ready || install_converter
phase="native configuration"
.venv/bin/python -m ops.launch_support config >>"$launcher_log" 2>&1
existing_port="$(.venv/bin/python -m ops.launch_support port)"
if [[ -n "$existing_port" ]]; then
    if [[ "$explicit_port" == yes && "$port" != "$existing_port" && "$action" != restart ]]; then
        fail "This project already has a managed native app; use --native --restart --port NUMBER."
    fi
    [[ "$explicit_port" == yes ]] || port="$existing_port"
    if [[ "$action" == restart || "$native_changed" == yes ]]; then
        .venv/bin/python -m ops.launch_support stop
        existing_port=""
    fi
fi
if [[ -z "$existing_port" || "$action" == restart ]]; then choose_port; fi
stage "Checking PDFs and published indexes" .venv/bin/python -m app.main bootstrap --if-needed
phase="native application readiness"
port="$(.venv/bin/python -m ops.launch_support start --port "$port" ${streamlit_args[@]+"${streamlit_args[@]}"})"
open_page
echo "Use --native --status, --logs, --stop or --restart. App output: artifacts/launcher/native.log"
