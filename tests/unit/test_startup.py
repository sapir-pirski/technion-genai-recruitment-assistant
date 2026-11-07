import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import URL, make_url

from ops import launch_support as launcher
from ops.container_entrypoint import container_database_url

ROOT = Path(__file__).resolve().parents[2]


def test_hosted_worker_requires_private_remote_credentials_and_keeps_them_out_of_schedule(tmp_path):
    worker = tmp_path / "ops/run-job-sync.sh"
    worker.parent.mkdir()
    worker.write_bytes((ROOT / "ops/run-job-sync.sh").read_bytes())
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/bash\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    app = tmp_path / "app"
    app.mkdir()
    (app / "__init__.py").touch()
    (app / "main.py").write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path('received.json').write_text(json.dumps({'url':os.environ['DATABASE_URL'],"
        "'arguments':sys.argv[1:]}))\n"
    )
    credentials = tmp_path / "artifacts/hosted-database-connection.json"
    credentials.parent.mkdir()
    remote = "postgresql://worker:fictitious-secret@example.test/jobs?sslmode=require"
    command = ["bash", str(worker), "--hosted", "--offline"]
    missing = subprocess.run(command, capture_output=True, text=True)
    assert missing.returncode == 1 and "owner-only" in missing.stderr
    credentials.write_text(json.dumps({"database_url": remote}))
    credentials.chmod(0o600)
    ready = subprocess.run(command, capture_output=True, text=True)
    assert ready.returncode == 0, ready.stderr
    received = json.loads((tmp_path / "received.json").read_text())
    assert received == {
        "url": remote,
        "arguments": ["sync-jobs", "--trigger", "scheduled", "--cleanup", "--offline"],
    }
    for unsafe_url, mode in [
        (remote, 0o644),
        ("postgresql://worker:fake@localhost/jobs", 0o600),
        (remote.replace("sslmode=require", "sslmode=disable"), 0o600),
    ]:
        credentials.write_text(json.dumps({"database_url": unsafe_url}))
        credentials.chmod(mode)
        failed = subprocess.run(command, capture_output=True, text=True)
        assert failed.returncode == 1 and unsafe_url not in failed.stderr
    output = tmp_path / "cloud.plist"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops/install_schedule.py"),
            "--hosted",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    config = plistlib.loads(output.read_bytes())
    assert config["Label"] == "com.recruitment.job-sync-cloud"
    assert config["ProgramArguments"][-1] == "--hosted"
    assert config["StartCalendarInterval"] == {"Hour": 3, "Minute": 37}
    assert "EnvironmentVariables" not in config and remote not in str(config)


def test_container_url_preserves_credentials_and_hosted_options():
    original = URL.create(
        "postgresql+psycopg",
        username="recruitment",
        password="fictitious:@/password",
        host="127.0.0.1",
        port=55432,
        database="recruitment",
    )
    translated = make_url(container_database_url(original.render_as_string(hide_password=False)))
    assert translated.host == "postgres" and translated.port == 5432
    assert translated.password == original.password and translated.database == original.database
    remote = "postgresql+psycopg://person:fictitious@example.test:5432/jobs?sslmode=require"
    assert container_database_url(remote) == remote
    with pytest.raises(ValueError, match="container-reachable"):
        container_database_url("postgresql://person:fictitious@localhost:5433/other")


def fake_docker(tmp_path):
    directory = tmp_path / "bin"
    directory.mkdir()
    executable = directory / "docker"
    executable.write_text("""#!/bin/bash
echo "$*" >> "$DOCKER_TEST_CALLS"
if [[ "$*" == *"--check-config"* ]]; then
    echo "${DOCKER_TEST_TARGET:-local}"
elif [[ "$*" == "compose port app 8501" ]]; then
    [[ -z "${DOCKER_TEST_PORT:-}" ]] || echo "127.0.0.1:$DOCKER_TEST_PORT"
elif [[ "$*" == "compose build app" ]]; then
    touch "$DOCKER_TEST_IMAGE"
elif [[ "$*" == "image inspect "* ]]; then
    [[ ! -f "$DOCKER_TEST_IMAGE" ]] || echo 'sha256:fictitious-image'
fi
exit 0
""")
    executable.chmod(0o755)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=fictitious\nDATABASE_URL=fictitious\nPOSTGRES_PASSWORD=fictitious\n"
    )
    launcher = tmp_path / "run-full-project.sh"
    launcher.write_bytes((ROOT / launcher.name).read_bytes())
    environment = {
        **os.environ,
        "PATH": str(directory) + os.pathsep + os.defpath,
        "DOCKER_TEST_CALLS": str(tmp_path / "calls"),
        "DOCKER_TEST_IMAGE": str(tmp_path / "image-ready"),
    }
    return launcher, environment


def test_launcher_builds_bootstraps_then_starts_only_the_required_services(tmp_path):
    target = "local"
    launcher, environment = fake_docker(tmp_path)
    environment["DOCKER_TEST_TARGET"] = target
    completed = subprocess.run(
        ["bash", str(launcher), "--port", "8517", "--no-browser"],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0 and "127.0.0.1:8517" in completed.stdout
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls.index("compose build app") < next(
        i for i, v in enumerate(calls) if v.endswith("app.main bootstrap --if-needed")
    )
    assert ("compose up -d --wait --wait-timeout 60 postgres" in calls) is (target == "local")
    assert calls[-1] == "compose up -d --no-deps --wait --wait-timeout 90 app"
    assert not any("down" in value or " -v" in value for value in calls)


def test_launcher_copies_template_without_overwriting_existing_credentials(tmp_path):
    launcher, environment = fake_docker(tmp_path)
    template = tmp_path / ".env.example"
    template.write_text("OPENAI_API_KEY=\nDATABASE_URL=\nPOSTGRES_PASSWORD=\n")
    original = (tmp_path / ".env").read_bytes()
    subprocess.run(
        ["bash", str(launcher), "--no-browser"], env=environment, check=True, capture_output=True
    )
    assert (tmp_path / ".env").read_bytes() == original
    (tmp_path / ".env").unlink()
    (tmp_path / "calls").unlink()
    completed = subprocess.run(["bash", str(launcher)], env=environment, capture_output=True)
    assert completed.returncode == 1
    assert (tmp_path / ".env").read_bytes() == template.read_bytes()
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600
    assert "compose build app" not in (tmp_path / "calls").read_text()


def test_browser_opens_after_readiness_and_healthy_project_port_is_reused(tmp_path):
    launcher, environment = fake_docker(tmp_path)
    environment["DOCKER_TEST_PORT"] = "8518"
    for name, body in {
        "uname": "echo Darwin",
        "open": 'echo "browser:$*" >> "$DOCKER_TEST_CALLS"',
    }.items():
        path = tmp_path / "bin" / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher)], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0 and "127.0.0.1:8518" in result.stdout
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls[-1] == "browser:http://127.0.0.1:8518"
    assert calls[-2].endswith("--wait-timeout 90 app")


RUNTIME_SCRIPT = r"""#!/bin/bash
echo "python:$*" >> "$NATIVE_TEST_CALLS"
case "$1" in
 --version) echo 'Python 3.13.0'; exit 0 ;;
 -c)
   case "$2" in
    *"_base_executable"*) echo "$NATIVE_TEST_ROOT/artifacts/launcher/python/python" ;;
    *"document_converter"*) test -f "$NATIVE_TEST_CONVERTER"; exit $? ;;
   esac
   exit 0 ;;
esac
if [[ "$1 $2" == '-m venv' ]]; then
 mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; touch "$3/pyvenv.cfg"
 rm -f "$NATIVE_TEST_DEPS"; exit 0
elif [[ "$1 $2 $3" == '-m pip install' ]]; then
 touch "$NATIVE_TEST_DEPS"; exit 0
elif [[ "$1 $2" == '-m pip' ]]; then exit 0
elif [[ "$2" == ops.launch_support ]]; then
 case "$3" in
  dependencies) test -f "$NATIVE_TEST_DEPS"; exit $? ;;
  config) echo hosted ;;
  port) [[ ! -f "$NATIVE_TEST_PORT" ]] || cat "$NATIVE_TEST_PORT" ;;
  start) echo "$5" > "$NATIVE_TEST_PORT"; echo "$5" ;;
  stop) rm -f "$NATIVE_TEST_PORT" ;;
  status) echo 'No managed native app is running.' ;;
 esac
fi
exit 0
"""


def executable(path, content):
    path.write_text(content)
    path.chmod(0o755)


def test_missing_python_dependencies_and_converter_are_installed_once_and_repairable(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    launcher = tmp_path / "run-full-project.sh"
    launcher.write_bytes((ROOT / launcher.name).read_bytes())
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=fictitious\nDATABASE_URL=fictitious\nPOSTGRES_PASSWORD=fictitious\n"
    )
    runtime = tmp_path / "runtime-template"
    executable(runtime, RUNTIME_SCRIPT)
    for name in ("python3", "python3.11", "python3.12", "python3.13"):
        executable(binary / name, "#!/bin/bash\nexit 1\n")
    executable(binary / "uname", "#!/bin/bash\necho Darwin\n")
    executable(
        binary / "brew",
        '#!/bin/bash\necho "brew:$*" >> "$NATIVE_TEST_CALLS"\ntouch "$NATIVE_TEST_CONVERTER"\n',
    )
    uv = tmp_path / "fake-uv"
    executable(
        uv,
        """#!/bin/bash
echo "uv:$*" >> "$NATIVE_TEST_CALLS"
if [[ "$1 $2" == 'python install' ]]; then
 mkdir -p "$UV_PYTHON_INSTALL_DIR"; cp "$NATIVE_TEST_RUNTIME" "$UV_PYTHON_INSTALL_DIR/python"
else echo "$UV_PYTHON_INSTALL_DIR/python"
fi
""",
    )
    installer = tmp_path / "installer-template"
    installer.write_text(
        'mkdir -p "$UV_UNMANAGED_INSTALL"\ncp "$NATIVE_TEST_UV" "$UV_UNMANAGED_INSTALL/uv"\n'
    )
    executable(
        binary / "curl",
        """#!/bin/bash
echo "curl:$*" >> "$NATIVE_TEST_CALLS"
while [[ $# -gt 0 ]]; do
 if [[ "$1" == -o ]]; then cp "$NATIVE_TEST_INSTALLER" "$2"; exit 0; fi
 shift
done
exit 1
""",
    )
    environment = {
        **os.environ,
        "PATH": str(binary) + os.pathsep + os.defpath,
        "NATIVE_TEST_CALLS": str(tmp_path / "calls"),
        "NATIVE_TEST_ROOT": str(tmp_path),
        "NATIVE_TEST_DEPS": str(tmp_path / "deps-ready"),
        "NATIVE_TEST_CONVERTER": str(tmp_path / "converter-ready"),
        "NATIVE_TEST_PORT": str(tmp_path / "port"),
        "NATIVE_TEST_RUNTIME": str(runtime),
        "NATIVE_TEST_UV": str(uv),
        "NATIVE_TEST_INSTALLER": str(installer),
    }
    environment.pop("PYTHON_BIN", None)
    command = ["bash", str(launcher), "--native", "--port", "8527", "--no-browser"]
    first = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "127.0.0.1:8527" in first.stdout
    second = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    calls = (tmp_path / "calls").read_text()
    assert calls.count("uv:python install") == 1
    assert calls.count("python:-m pip install -r requirements.txt") == 1
    assert calls.count("brew:install --cask libreoffice") == 1
    assert "curl:--fail --location --proto =https" in calls
    (tmp_path / ".venv/pyvenv.cfg").unlink()
    repaired = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert repaired.returncode == 0, repaired.stderr
    assert list((tmp_path / "artifacts/launcher").glob("venv-backup-*"))
    assert (tmp_path / ".venv/pyvenv.cfg").is_file()


def test_stale_process_record_never_signals_an_unrelated_process(tmp_path, monkeypatch):
    state = tmp_path / "native.json"
    state.write_text(json.dumps({"pid": 123, "identity": "old", "port": 8502}))
    monkeypatch.setattr(launcher, "STATE", state)
    monkeypatch.setattr(launcher, "process_identity", lambda pid: "another-process")
    kill = Mock(side_effect=AssertionError("Must not signal a reused PID"))
    monkeypatch.setattr(launcher.os, "killpg", kill)
    launcher.stop_native()
    kill.assert_not_called()
