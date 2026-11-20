import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.engine import URL

from app.config import Settings

ROOT = Path(__file__).resolve().parents[2]


def test_hosted_database_settings_require_tls_and_preserve_credentials():
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
    assert Settings(_env_file=None, database_url=encoded).database_url == encoded
    for sslmode in ("disable", "prefer", "allow"):
        with pytest.raises(ValueError, match="require TLS") as error:
            Settings(
                _env_file=None,
                database_url=original.set(query={"sslmode": sslmode}).render_as_string(
                    hide_password=False
                ),
            )
        assert "fictitious" not in str(error.value)
    local = original.set(host="127.0.0.1", query={}).render_as_string(hide_password=False)
    assert Settings(_env_file=None, database_url=local).database_url == local


def launcher_copy(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    script = project / "run-full-project.sh"
    script.write_bytes((ROOT / script.name).read_bytes())
    return project, script


def test_help_runs_before_environment_setup_without_creating_files(tmp_path):
    project, script = launcher_copy(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--help"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0 and "Ctrl+C" in result.stdout
    assert "--server.port" in result.stdout
    assert list(project.iterdir()) == [script]


def test_missing_environment_has_actionable_instructions_and_preserves_credentials(tmp_path):
    project, script = launcher_copy(tmp_path)
    env_file = project / ".env"
    original = "OPENAI_API_KEY=fictitious\nDATABASE_URL=fictitious\n"
    env_file.write_text(original)
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1 and "pip install -r requirements.txt" in result.stderr
    assert env_file.read_text() == original
    assert not (project / ".venv").exists() and not (project / "artifacts").exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_foreground_launcher_uses_project_python_and_forwards_arguments(tmp_path, exit_code):
    project, script = launcher_copy(tmp_path)
    binary = project / ".venv/bin/python"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
Path(os.environ["STARTUP_TEST_LOG"]).write_text(json.dumps({"cwd": os.getcwd(), "args": sys.argv[1:], "pid": os.getpid()}))
sys.exit(int(os.environ["STARTUP_TEST_EXIT"]))
"""
    )
    binary.chmod(0o755)
    log = tmp_path / "calls.json"
    environment = {**os.environ, "STARTUP_TEST_LOG": str(log), "STARTUP_TEST_EXIT": str(exit_code)}
    with subprocess.Popen(
        ["bash", str(script), "--server.port", "8527", "--server.headless", "true"],
        cwd=tmp_path,
        env=environment,
    ) as process:
        assert process.wait(timeout=10) == exit_code
        called = json.loads(log.read_text())
        assert called["pid"] == process.pid
    assert called["cwd"] == str(project)
    assert called["args"] == [
        "-m",
        "streamlit",
        "run",
        "streamlit_app/streamlit_main.py",
        "--server.port",
        "8502",
        "--server.port",
        "8527",
        "--server.headless",
        "true",
    ]
    assert not (project / ".env").exists() and not (project / "artifacts").exists()
