"""Install/remove the current user's macOS daily job; no sudo is needed."""

import argparse
import plistlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.recruitment.job-sync"


def main():
    parser = argparse.ArgumentParser(
        description="Daily local/hosted job sync (requires Asia/Jerusalem)"
    )
    parser.add_argument("--remove", action="store_true")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--native", action="store_true", help="Schedule the host venv instead of Docker"
    )
    target.add_argument(
        "--hosted", action="store_true", help="Sync the hosted database daily at 03:37"
    )
    parser.add_argument(
        "--output", type=Path, help="Write a reviewable plist without installing it"
    )
    args = parser.parse_args()
    import os
    import sys

    label = LABEL + ("-cloud" if args.hosted else "")
    minute = 37 if args.hosted else 17
    log_prefix = "cloud" if args.hosted else "scheduled"
    domain = f"gui/{os.getuid()}"
    destination = args.output or Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if args.remove:
        if args.output:
            parser.error("--remove cannot be combined with --output")
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True
        )
        destination.unlink(missing_ok=True)
        print("Daily job sync removed.")
        return
    if not args.output:
        if sys.platform != "darwin":
            parser.error("Use the documented systemd timer on Linux")
        zone = Path("/etc/localtime").resolve().as_posix()
        if not zone.endswith("Asia/Jerusalem"):
            parser.error(
                "The host timezone must be Asia/Jerusalem; use the timezone-aware systemd example otherwise"
            )
        if (args.native or args.hosted) and not (ROOT / ".venv/bin/python").exists():
            parser.error("Create the native project venv first")
        if args.hosted:
            credentials = ROOT / "artifacts/hosted-database-connection.json"
            if not credentials.is_file() or credentials.stat().st_mode & 0o077:
                parser.error("Configure the owner-only hosted database credential file first")
        if not (args.native or args.hosted) and not shutil.which("docker"):
            parser.error("Install Docker and build the app with ./run-full-project.sh first")
    logs = ROOT / "artifacts/ingestion"
    logs.mkdir(parents=True, exist_ok=True)
    config = {
        "Label": label,
        "ProgramArguments": ["/bin/bash", str(ROOT / "ops/run-job-sync.sh")]
        + (["--hosted"] if args.hosted else ["--native"] if args.native else []),
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": {"Hour": 3, "Minute": minute},
        "StandardOutPath": str(logs / f"{log_prefix}.stdout.log"),
        "StandardErrorPath": str(logs / f"{log_prefix}.stderr.log"),
        "ProcessType": "Background",
        "RunAtLoad": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(plistlib.dumps(config))
    if not args.output:
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{label}"], check=False, capture_output=True
        )
        subprocess.run(["launchctl", "bootstrap", domain, str(destination)], check=True)
        subprocess.run(["launchctl", "enable", f"{domain}/{label}"], check=True)
    print(
        f"{'Prepared' if args.output else 'Installed'} daily 03:{minute:02d} Asia/Jerusalem sync: {destination}"
    )


if __name__ == "__main__":
    main()
