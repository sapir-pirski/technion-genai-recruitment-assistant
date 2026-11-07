"""Hosted database validation and server readiness; never prints secrets."""

import os
import sys
import time

from sqlalchemy.engine import make_url


def container_database_url(value):
    url = make_url(value)
    if (
        url.get_backend_name() != "postgresql"
        or not url.host
        or url.host in {"localhost", "127.0.0.1", "::1", "postgres"}
    ):
        raise ValueError("Use the hosted PostgreSQL database")
    if url.query.get("sslmode") not in {"require", "verify-ca", "verify-full"}:
        raise ValueError("The hosted database requires TLS")
    return url.render_as_string(hide_password=False)


def main():
    command = sys.argv[1:]
    try:
        if value := os.environ.get("DATABASE_URL"):
            os.environ["DATABASE_URL"] = container_database_url(value)
        if command == ["--check-config"]:
            from app.config import Settings
            from app.profiles import document_converter
            from ops.launch_support import dependencies_ready

            missing = [
                key
                for key in ("OPENAI_API_KEY", "DATABASE_URL")
                if not os.environ.get(key) or os.environ[key].startswith("replace-with-")
            ]
            if missing:
                print("Fill in " + ", ".join(missing) + " in .env.", file=sys.stderr)
                raise SystemExit(1)
            settings = Settings()
            settings.require_api_key()
            if sys.version_info < (3, 11) or not dependencies_ready():
                raise RuntimeError(
                    "Rebuild the application image to repair its Python dependencies"
                )
            if not document_converter():
                raise RuntimeError("Document converter is missing")
            container_database_url(settings.database_url)
            print("hosted")
            return
        if command[:3] == ["python", "-m", "streamlit"]:
            from app.config import Settings
            from app.storage.database import initialize_database, make_engine

            settings = Settings()
            settings.require_api_key()
            url = make_url(settings.database_url).update_query_dict({"connect_timeout": "3"})
            engine = make_engine(url.render_as_string(hide_password=False))
            deadline = time.monotonic() + 60
            try:
                while True:
                    try:
                        initialize_database(engine)
                        break
                    except Exception:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Database did not become ready") from None
                        time.sleep(1)
            finally:
                engine.dispose()
        if not command:
            raise ValueError("Supply a server or application command")
        os.execvp(command[0], command)
    except Exception as error:
        print(
            f"Container startup failed ({type(error).__name__}). Check credentials, database availability and container configuration.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
