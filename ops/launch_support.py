"""Launcher checks and managed native processes. Importable without app dependencies."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "artifacts/launcher"
STATE = RUNTIME / "native.json"
SERVER = ROOT / "streamlit_app/streamlit_main.py"


def dependencies_ready():
    commands = [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-index",
            "--quiet",
            "-r",
            str(ROOT / "requirements.txt"),
        ],
        [sys.executable, "-m", "pip", "check"],
    ]
    return all(
        subprocess.run(command, capture_output=True, timeout=90).returncode == 0
        for command in commands
    )


def configuration_target():
    from dotenv import dotenv_values
    from sqlalchemy.engine import make_url

    from app.config import Settings

    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    missing = [
        key
        for key in ("OPENAI_API_KEY", "DATABASE_URL", "POSTGRES_PASSWORD")
        if not values.get(key) or str(values[key]).startswith("replace-with-")
    ]
    if missing:
        raise RuntimeError("Fill in " + ", ".join(missing) + " in .env.")
    try:
        settings = Settings()
        settings.require_api_key()
        url = make_url(settings.database_url)
        local = (
            url.host in {"localhost", "127.0.0.1"}
            and url.port == 55432
            and url.database == "recruitment"
        )
        if local and (url.username != "recruitment" or url.password != values["POSTGRES_PASSWORD"]):
            raise RuntimeError(
                "Local DATABASE_URL and POSTGRES_PASSWORD must use the same credentials."
            )
        return "local" if local else "hosted"
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Check DATABASE_URL and application configuration in .env.") from None


def process_identity(pid):
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "args="],
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    marker = f"-m streamlit run {SERVER}"
    if result.returncode != 0 or marker not in value:
        return ""
    # macOS framework Python re-execs with a different executable path. Its
    # process start time and Streamlit arguments remain stable across that exec.
    started = " ".join(value.split()[:5])
    arguments = value[value.index(marker) :]
    return hashlib.sha256(f"{started}\0{arguments}".encode()).hexdigest()


def native_state():
    try:
        state = json.loads(STATE.read_text())
        identity = process_identity(int(state["pid"]))
        return state if identity and identity == state["identity"] else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
            return response.read() == b"ok"
    except Exception:
        return False


def wait_ready(port, *, pid=None, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pid and not process_identity(pid):
            raise RuntimeError("The native app exited. Check --native --logs.")
        if healthy(port):
            return
        time.sleep(0.5)
    raise RuntimeError("The app did not become ready. Check --logs.")


def stop_native(*, quiet=False):
    state = native_state()
    if state is None:
        if not quiet:
            print("No managed native app is running.")
        return
    os.killpg(state["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 20
    while native_state() and time.monotonic() < deadline:
        time.sleep(0.2)
    if native_state():
        os.killpg(state["pid"], signal.SIGKILL)
    STATE.unlink(missing_ok=True)
    if not quiet:
        print("Native app stopped; database and files retained.")


def start_native(port, extra):
    digest = hashlib.sha256(sys.version.encode())
    digest.update(json.dumps(extra).encode())
    environment_file = ROOT / ".env"
    if environment_file.exists():
        info = environment_file.stat()
        digest.update(f"{info.st_mtime_ns}:{info.st_size}".encode())
    for directory in (ROOT / "app", ROOT / "streamlit_app", ROOT / ".streamlit"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".txt", ".js", ".css", ".toml"}:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    existing = native_state()
    if existing and existing.get("fingerprint") == fingerprint:
        wait_ready(existing["port"], pid=existing["pid"])
        return existing["port"]
    if existing:
        stop_native(quiet=True)
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise RuntimeError("The selected port is in use; choose another --port.") from None
    RUNTIME.mkdir(parents=True, exist_ok=True)
    log_path = RUNTIME / "native.log"
    descriptor = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a") as log:
        process = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "streamlit",
                "run",
                str(SERVER),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
                *extra,
            ],
            cwd=ROOT,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    identity = process_identity(process.pid)
    state = {"pid": process.pid, "identity": identity, "port": port, "fingerprint": fingerprint}
    STATE.write_text(json.dumps(state))
    STATE.chmod(0o600)
    wait_ready(port, pid=process.pid)
    return port


def managed_action(action, port, extra):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "native.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return start_native(port, extra) if action == "start" else stop_native()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["dependencies", "config", "port", "status", "start", "stop", "logs", "wait"],
    )
    parser.add_argument("--port", type=int, default=8502)
    args, extra = parser.parse_known_args()
    try:
        if args.action == "dependencies":
            raise SystemExit(0 if dependencies_ready() else 1)
        if args.action == "config":
            print(configuration_target())
        elif args.action == "port":
            state = native_state()
            print(state["port"] if state else "")
        elif args.action == "status":
            state = native_state()
            print(
                f"Native app: http://127.0.0.1:{state['port']} ({'ready' if healthy(state['port']) else 'starting'})"
                if state
                else "No managed native app is running."
            )
        elif args.action == "stop":
            managed_action("stop", args.port, extra)
        elif args.action == "start":
            print(managed_action("start", args.port, extra))
        elif args.action == "wait":
            wait_ready(args.port)
        elif args.action == "logs":
            path = RUNTIME / "native.log"
            print(
                "".join(path.read_text(errors="replace").splitlines(keepends=True)[-100:])
                if path.exists()
                else "No native app log yet.",
                end="",
            )
    except Exception as error:
        # Only our static RuntimeError messages may be displayed; provider exceptions stay private.
        print(
            str(error)
            if isinstance(error, RuntimeError)
            else f"Launcher check failed ({type(error).__name__}). See artifacts/launcher/.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
