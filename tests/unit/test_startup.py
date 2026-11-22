import hashlib
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from langchain_openai import ChatOpenAI
from sqlalchemy.engine import URL

from app.agents.backends.common import OPERATION_PROMPTS, prompt_spec
from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from ops import launch_support as launcher

ROOT = Path(__file__).resolve().parents[2]


def test_workspace_recovery_is_bounded_read_only_and_does_not_retry_setup_errors(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example:fake@localhost/unit-test")
    import psycopg
    from psycopg.errors import AdminShutdown, InvalidPassword, UndefinedTable
    from psycopg_pool import PoolTimeout
    from sqlalchemy.exc import OperationalError

    from app.ingestion.models import DEFAULT_JOB_ID
    from streamlit_app.startup import StartupFailure, load_workspace, transient_database_error

    sleeps = Mock()
    monkeypatch.setattr("streamlit_app.startup.time.sleep", sleeps)
    service = Mock()
    service.state.return_value = {"job_id": DEFAULT_JOB_ID}
    service.conversations.messages.return_value = [{"role": "user", "content": "Hello"}]
    job = {"job_id": DEFAULT_JOB_ID, "company_name": "Example", "title": "Python Developer"}
    service.registry.list_jobs.return_value = [job]
    service.registry.get_job.return_value = job
    downloads = Mock(return_value=[])
    on_retry = Mock()
    factory = Mock(side_effect=[AdminShutdown("synthetic disconnect"), service, service])
    service.state.side_effect = [
        AdminShutdown("synthetic mid-read disconnect"),
        {"job_id": DEFAULT_JOB_ID},
    ]
    result = load_workspace(
        Settings(_env_file=None), factory, "same-conversation", None, downloads, on_retry
    )
    assert result.job == job and result.messages == service.conversations.messages.return_value
    assert factory.call_count == 3 and on_retry.call_count == 2
    assert [call.args[0] for call in sleeps.call_args_list] == [0.5, 1.0]
    service.turn.assert_not_called()
    assert all(call.args == ("same-conversation",) for call in service.state.call_args_list)

    # Document, logo, and catalog reads belong to the same recovery boundary.
    factory.side_effect = None
    factory.return_value = service
    service.state.side_effect = None
    downloads.side_effect = [AdminShutdown("synthetic document read disconnect"), []]
    result = load_workspace(
        Settings(_env_file=None), factory, "same-conversation", None, downloads, on_retry
    )
    assert result.downloads == []

    for error in (AdminShutdown(), PoolTimeout(), psycopg.OperationalError("connection refused")):
        assert transient_database_error(error)
        assert transient_database_error(OperationalError(None, None, error))
    for error in (
        InvalidPassword(),
        UndefinedTable(),
        ValueError("invalid configuration"),
        psycopg.OperationalError("password authentication failed"),
        psycopg.OperationalError("certificate verify failed"),
    ):
        assert not transient_database_error(error)
        factory.reset_mock()
        factory.side_effect = error
        with pytest.raises(StartupFailure) as failure:
            load_workspace(
                Settings(_env_file=None), factory, "same-conversation", None, downloads, on_retry
            )
        assert not failure.value.retryable and failure.value.stage == "service"
        assert factory.call_count == 1
    factory.reset_mock()
    factory.side_effect = AdminShutdown("synthetic persistent outage")
    with pytest.raises(StartupFailure) as failure:
        load_workspace(
            Settings(_env_file=None), factory, "same-conversation", None, downloads, on_retry
        )
    assert failure.value.retryable and factory.call_count == 3


def test_failed_service_initialization_releases_partial_resources(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://example:fake@localhost/unit-test")
    from psycopg.errors import AdminShutdown

    import app.service as module

    registry, repository, retriever, pool = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(module, "JobRegistry", Mock(return_value=registry))
    monkeypatch.setattr(module, "ConnectionPool", Mock(return_value=pool))
    # Keep the real health-check callback available on the mocked constructor.
    pool.wait.side_effect = AdminShutdown("synthetic construction failure")
    with pytest.raises(AdminShutdown):
        module.ChatService(
            Settings(_env_file=None), backend=Mock(), repository=repository, retriever=retriever
        )
    pool.close.assert_called_once()
    registry.close.assert_called_once()
    repository.engine.dispose.assert_called_once()
    retriever.close.assert_called_once()

    # Failure during registry construction must also dispose its own engine,
    # since no registry object has been assigned to ChatService yet.
    import app.ingestion.registry as registry_module

    engine = Mock()
    monkeypatch.setattr(registry_module, "make_engine", Mock(return_value=engine))
    monkeypatch.setattr(registry_module, "initialize_database", Mock(side_effect=AdminShutdown()))
    with pytest.raises(AdminShutdown):
        registry_module.JobRegistry(Settings(_env_file=None))
    engine.dispose.assert_called_once()


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
    second = subprocess.run(
        ["bash", str(launcher), "--port=08527", "--no-browser"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert "127.0.0.1:8527" in second.stdout
    calls = (tmp_path / "calls").read_text()
    assert calls.count("uv:python install") == 1
    assert calls.count("python:-m pip install -r requirements.txt") == 1
    assert calls.count("brew:install --cask libreoffice") == 1
    assert calls.count("browser:") == 1
    assert "curl:--fail --location --proto =https" in calls
    for invalid_port in ("0", "65536", "999999999999999999999", "08x", "-1", "1+1", ""):
        invalid = subprocess.run(
            ["bash", str(launcher), "--port", invalid_port],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert invalid.returncode == 1
        assert "between 1 and 65535" in invalid.stderr
    assert (tmp_path / "calls").read_text() == calls
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


def test_model_preflight_and_runtime_identity_follow_selected_artifact(tmp_path, monkeypatch):
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
        hf_exit_repo_id="",
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

    settings.hf_exit_repo_id = "example/exit-model"
    artifact = Mock(return_value=tmp_path / "pinned-runtime")
    monkeypatch.setattr("app.agents.backends.huggingface.artifact_directory", artifact)
    verify.reset_mock(side_effect=True)
    assert "pinned Hugging Face artifact" in launcher.verify_exit_model()
    artifact.assert_called_once_with(settings)
    verify.assert_called_once_with(tmp_path / "pinned-runtime", "verified-revision")

    settings.exit_provider = "openai"
    verify.reset_mock()
    assert "OpenAI" in launcher.verify_exit_model()
    verify.assert_not_called()
    assert artifact.call_count == 1
    monkeypatch.setattr(
        "app.config.Settings", Mock(side_effect=ValueError("private-activation-details"))
    )
    with pytest.raises(RuntimeError, match="Restore its verified files") as error:
        launcher.verify_exit_model()
    assert "private-activation-details" not in str(error.value)


@pytest.mark.parametrize(
    "operation,output",
    [
        ("main", None),
        (
            "profile_extraction",
            {"name": None, "experience": [], "skills": ["Python"], "education": [], "projects": []},
        ),
    ],
)
def test_operation_selects_model_and_sends_compatible_bounded_parameters(
    now, backend, operation, output
):
    settings = Settings(
        _env_file=None,
        openai_api_key="test-only-key",
        database_url="postgresql://test:test@localhost/test",
    )
    seen = []
    content = json.dumps(output or backend.plan.return_value.model_dump())

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-policy",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch(
            "app.agents.backends.openai.ChatOpenAI",
            side_effect=lambda **options: ChatOpenAI(http_client=client, **options),
        ):
            adapter = OpenAIBackend(settings)
        state = {"messages": [], "reference_time": now.isoformat(), "timezone": "UTC"}
        calls = {
            "main": lambda: adapter.plan(state),
            "profile_extraction": lambda: adapter.extract_profile("Python"),
        }
        calls[operation]()
        if operation == "main":
            seen.clear()
            # Chat revisions use the existing non-reasoning budget model and system prompt.
            content = json.dumps(
                {
                    "facts": {
                        "name": None,
                        "experience": [],
                        "skills": [],
                        "education": [],
                        "projects": [],
                    },
                    "superseded_fact_ids": [],
                    "confirmed_fact_ids": [],
                }
            )
            adapter.extract_chat_profile({"latest_message": "I use Python."})
            assert seen[0]["model"] == settings.model_for("chat_profile") == settings.chat_model
            assert seen[0]["messages"][0]["role"] == "system"
            assert seen[0]["max_completion_tokens"] == settings.max_output_tokens
            assert "reasoning_effort" not in seen[0]
            seen.clear()
            content = json.dumps({"decision": "info_not_needed", "question": None})
            adapter.info_decision(state)
            assert seen[0]["reasoning_effort"] == "minimal"
            assert seen[0]["max_completion_tokens"] == 1024
            seen.clear()
            content = json.dumps(backend.plan.return_value.model_dump())
            adapter.plan(state)
    assert len(seen) == 1
    request = seen[0]
    policy = settings.model_policy()[operation]
    assert request["model"] == policy["model"]
    assert request["max_completion_tokens"] == policy["max_completion_tokens"]
    assert "max_tokens" not in request
    spec = prompt_spec(OPERATION_PROMPTS[operation], policy["model"])
    assert request["messages"][0] == {"role": spec["instruction_role"], "content": spec["content"]}
    assert request["messages"][1]["role"] == "user"
    assert (
        adapter.prompt_manifest[operation]["prompt_sha256"]
        == hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest()
    )
    if policy["model"] == settings.helper_model:
        assert request["reasoning_effort"] == policy["reasoning_effort"]
        assert "temperature" not in request
    else:
        assert request["temperature"] == 0
        assert "reasoning_effort" not in request
