import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sqlalchemy.engine import URL

from ops import launch_support as launcher

ROOT = Path(__file__).resolve().parents[2]


def test_native_schedule_keeps_cloud_credentials_out_of_plist(tmp_path):
    output = tmp_path / "job-sync.plist"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "ops/install_schedule.py"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    config = plistlib.loads(output.read_bytes())
    assert config["Label"] == "com.recruitment.job-sync"
    assert config["ProgramArguments"] == ["/bin/bash", str(ROOT / "ops/run-job-sync.sh")]
    assert config["StartCalendarInterval"] == {"Hour": 3, "Minute": 17}
    assert "EnvironmentVariables" not in config


def test_hosted_database_url_requires_tls_and_preserves_credentials():
    original = URL.create(
        "postgresql+psycopg",
        username="recruitment",
        password="fictitious:@/password",
        host="database.example.test",
        port=5432,
        database="recruitment",
        query={"sslmode": "require", "channel_binding": "require"},
    )
    encoded = original.render_as_string(hide_password=False)
    assert launcher.hosted_database_url(encoded) == encoded
    for host in ("localhost", "127.0.0.1", "::1", "postgres"):
        with pytest.raises(ValueError, match="hosted PostgreSQL"):
            launcher.hosted_database_url(
                original.set(host=host).render_as_string(hide_password=False)
            )
    with pytest.raises(ValueError, match="requires TLS"):
        launcher.hosted_database_url(
            original.set(query={"sslmode": "disable"}).render_as_string(hide_password=False)
        )


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
    (tmp_path / ".env").write_text("OPENAI_API_KEY=fictitious\nDATABASE_URL=fictitious\n")
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
    original_env = (tmp_path / ".env").read_bytes()
    executable(binary / "open", '#!/bin/bash\necho "browser:$*" >> "$NATIVE_TEST_CALLS"\n')
    command = ["bash", str(launcher), "--port", "8527"]
    first = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "127.0.0.1:8527" in first.stdout
    first_calls = (tmp_path / "calls").read_text().splitlines()
    assert first_calls[-1] == "browser:http://127.0.0.1:8527"
    assert first_calls[-2].startswith("python:-m ops.launch_support start")
    assert (tmp_path / ".env").read_bytes() == original_env
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
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\nDATABASE_URL=\n")
    (tmp_path / ".env").unlink()
    absent = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert absent.returncode == 1
    assert (tmp_path / ".env").read_bytes() == (tmp_path / ".env.example").read_bytes()
    assert (tmp_path / ".env").stat().st_mode & 0o777 == 0o600


def test_stale_process_record_never_signals_an_unrelated_process(tmp_path, monkeypatch):
    state = tmp_path / "native.json"
    state.write_text(json.dumps({"pid": 123, "identity": "old", "port": 8502}))
    monkeypatch.setattr(launcher, "STATE", state)
    monkeypatch.setattr(launcher, "process_identity", lambda pid: "another-process")
    kill = Mock(side_effect=AssertionError("Must not signal a reused PID"))
    monkeypatch.setattr(launcher.os, "killpg", kill)
    launcher.stop_native()
    kill.assert_not_called()


def test_dependency_readiness_checks_cpu_wheels_markers_and_extras_without_downloads(
    tmp_path, monkeypatch
):
    from packaging import markers

    (tmp_path / "requirements.txt").write_text(
        'torch @ https://download.pytorch.org/whl/cpu/torch-2.15.0%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl ; sys_platform == "linux"\n'
        'missing-package==1.0 ; sys_platform == "win32"\n'
        "psycopg[binary]==3.3.6\n"
    )
    environment = {**markers.default_environment(), "sys_platform": "linux"}
    monkeypatch.setattr(markers, "default_environment", lambda: environment)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    versions = {"torch": "2.15.0+cpu", "psycopg": "3.3.6", "psycopg-binary": "3.3.6"}

    def installed(name):
        if name not in versions:
            raise launcher.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(launcher.metadata, "version", installed)
    monkeypatch.setattr(
        launcher.metadata,
        "requires",
        lambda name: ['psycopg-binary==3.3.6 ; extra == "binary"', 'unused-pool ; extra == "pool"'],
    )
    run = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(launcher.subprocess, "run", run)
    assert launcher.dependencies_ready()
    assert run.call_args.args[0] == [sys.executable, "-m", "pip", "check"]
    versions["torch"] = "2.15.0"
    assert not launcher.dependencies_ready()
    versions["torch"] = "2.15.0+cpu"
    versions.pop("psycopg-binary")
    assert not launcher.dependencies_ready()
    assert run.call_count == 1


def test_model_preflight_and_runtime_identity_follow_local_activation(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    identity = {"provider": "openai", "revision": None}
    monkeypatch.setattr(launcher, "exit_model_identity", lambda: identity)
    baseline = launcher.runtime_fingerprint([])
    assert launcher.runtime_fingerprint([]) == baseline
    identity.update(provider="huggingface", revision="first-trained-artifact")
    trained = launcher.runtime_fingerprint([])
    assert trained != baseline and launcher.runtime_fingerprint([]) == trained
    identity["revision"] = "replacement-trained-artifact"
    assert launcher.runtime_fingerprint([]) != trained

    settings = SimpleNamespace(
        exit_provider="huggingface",
        hf_exit_local_path=tmp_path,
        hf_exit_revision="verified-revision",
    )
    monkeypatch.setattr("app.config.Settings", lambda: settings)
    verify = Mock()
    monkeypatch.setattr("app.agents.backends.huggingface.verify_local_artifact", verify)
    assert "verified" in launcher.verify_exit_model()
    verify.assert_called_once_with(tmp_path, "verified-revision")
    verify.side_effect = FileNotFoundError("private-path-and-credential")
    with pytest.raises(RuntimeError, match="Restore its verified files") as error:
        launcher.verify_exit_model()
    assert "private-path-and-credential" not in str(error.value)
    settings.exit_provider = "openai"
    verify.reset_mock()
    assert "OpenAI" in launcher.verify_exit_model()
    verify.assert_not_called()
    monkeypatch.setattr(
        "app.config.Settings", Mock(side_effect=ValueError("private-activation-details"))
    )
    with pytest.raises(RuntimeError, match="Restore its verified files") as error:
        launcher.verify_exit_model()
    assert "private-activation-details" not in str(error.value)
