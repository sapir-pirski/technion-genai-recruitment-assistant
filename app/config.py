import hashlib
import json
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
LOCAL_EXIT = ROOT / "artifacts/hf-exit/runtime"


def local_exit_revision():
    manifest = LOCAL_EXIT / "exit_classifier.json"
    if not manifest.is_file():
        return ""
    return json.loads(manifest.read_text())["files"]["model.onnx"]


def default_exit_provider():
    activation = ROOT / "artifacts/hf-exit/active.json"
    if not activation.is_file():
        return "openai"
    if json.loads(activation.read_text()).get("artifact_sha256") != local_exit_revision():
        raise ValueError("The local Exit artifact changed; verify and activate it again.")
    return "huggingface"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Credentials belong in .env or the deployment's secret settings.
    openai_api_key: SecretStr = SecretStr("")
    hf_token: SecretStr = SecretStr("")
    database_url: str = Field(default="", repr=False)

    # Configure application behavior here instead of duplicating defaults in .env.
    chat_model: str = "gpt-4o-mini"
    helper_model: str = "gpt-5-nano"
    judge_model: str = "gpt-4o-mini"
    exit_provider: Literal["openai", "huggingface"] = Field(default_factory=default_exit_provider)
    hf_exit_model: str = "answerdotai/ModernBERT-base (fine-tuned)"
    hf_exit_revision: str = Field(default_factory=local_exit_revision)
    hf_exit_local_path: Path = LOCAL_EXIT
    embedding_model: str = "text-embedding-3-small"
    job_sources_path: Path = ROOT / "config/job_sources.json"
    job_manifest_path: Path = ROOT / "config/job_corpus_manifest.json"
    job_data_path: Path = ROOT / "data"
    ingestion_report_path: Path = ROOT / "artifacts/ingestion"
    conversation_export_path: Path = ROOT / "artifacts/conversations.json"
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    dataset_version: str = "assignment-v1"
    max_advisor_consultations: int = Field(default=3, ge=1, le=8)
    corpus_limit: int = Field(default=10, ge=1, le=10000)
    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=120, ge=0)
    pipeline_version: str = "jobs-v1"
    embedding_dimensions: int | None = Field(default=None, ge=1)
    business_timezone: str = "UTC"
    request_timeout: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_output_tokens: int = Field(default=1500, ge=256)
    model_temperature: float | None = Field(default=0.0, ge=0, le=2)

    def model_for(self, operation: str) -> str:
        if operation == "judge":
            return self.judge_model
        if operation == "exit":
            if self.exit_provider == "huggingface":
                return self.hf_exit_model
            return self.helper_model
        if operation in {"profile_extraction", "followups", "info_decision"}:
            return self.helper_model
        if operation in {"main", "info", "scheduling", "chat_profile"}:
            return self.chat_model
        raise ValueError("Unknown model operation")

    def model_policy(self) -> dict:
        policy = {}
        for operation in (
            "main",
            "info",
            "scheduling",
            "info_decision",
            "exit",
            "profile_extraction",
            "chat_profile",
            "followups",
            "judge",
        ):
            model = self.model_for(operation)
            if operation == "exit" and self.exit_provider == "huggingface":
                policy[operation] = {
                    "provider": "huggingface",
                    "model": model,
                    "revision": self.hf_exit_revision,
                    "task": "text-classification",
                    "reasoning_effort": None,
                    "temperature": None,
                    "max_completion_tokens": None,
                }
                continue
            # These defaults target GPT-5 nano and GPT-4o mini. Changing families
            # requires checking the target model's supported API parameters.
            nano = model in {"gpt-5-nano", "gpt-5-nano-2025-08-07"}
            cap = {"exit": 512, "info_decision": 1024, "followups": 768}.get(
                operation, self.max_output_tokens
            )
            policy[operation] = {
                "model": model,
                "reasoning_effort": (
                    "minimal" if operation in {"followups", "info_decision"} else "low"
                )
                if nano
                else None,
                "temperature": None if nano else self.model_temperature,
                "max_completion_tokens": min(cap, self.max_output_tokens),
            }
        return policy

    @property
    def job_pdf_path(self) -> Path:
        return self.job_data_path / "jobs"

    @field_validator("database_url")
    @classmethod
    def postgres_only(cls, value):
        if not value or make_url(value).get_backend_name() != "postgresql":
            raise ValueError(
                "Set DATABASE_URL to the PostgreSQL database used by the whole application"
            )
        return (
            make_url(value)
            .set(drivername="postgresql+psycopg")
            .render_as_string(hide_password=False)
        )

    @property
    def postgres_dsn(self) -> str:
        return (
            make_url(self.database_url)
            .set(drivername="postgresql")
            .render_as_string(hide_password=False)
        )

    @field_validator("model_temperature", "embedding_dimensions", mode="before")
    @classmethod
    def optional_temperature(cls, value):
        return None if value in ("", "none", "null") else value

    @field_validator("business_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @property
    def collection_name(self) -> str:
        return "python_job_openai"

    def require_api_key(self) -> str:
        key = self.openai_api_key.get_secret_value().strip()
        if not key:
            raise ValueError(
                "Set OPENAI_API_KEY in .env or Streamlit secrets before running agents or ingestion."
            )
        return key

    @property
    def embedding_identity(self) -> str:
        return self.embedding_model

    @property
    def index_collection(self) -> str:
        identity = f"{self.embedding_identity}:{self.embedding_dimensions}:{self.pipeline_version}:{self.chunk_size}:{self.chunk_overlap}"
        suffix = hashlib.sha256(identity.encode()).hexdigest()[:12]
        return f"{self.collection_name}_{suffix}"
