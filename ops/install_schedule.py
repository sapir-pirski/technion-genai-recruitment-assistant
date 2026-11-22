"""Install/remove the current user's macOS daily job; no sudo is needed."""

import argparse
import plistlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = "com.recruitment.job-sync"


def main():
    parser = argparse.ArgumentParser(
        description="Daily job sync at 03:17 (requires Asia/Jerusalem)"
    )
    parser.add_argument("--remove", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Write a reviewable plist without installing it"
    )
    args = parser.parse_args()
    import os
    import sys

    label = LABEL
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
        if not (ROOT / ".venv/bin/python").exists():
            parser.error("Create the native project venv first")
    logs = ROOT / "artifacts/ingestion"
    logs.mkdir(parents=True, exist_ok=True)
    config = {
        "Label": label,
        "ProgramArguments": ["/bin/bash", str(ROOT / "ops/run-job-sync.sh")],
        "WorkingDirectory": str(ROOT),
        "StartCalendarInterval": {"Hour": 3, "Minute": 17},
        "StandardOutPath": str(logs / "scheduled.stdout.log"),
        "StandardErrorPath": str(logs / "scheduled.stderr.log"),
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
        f"{'Prepared' if args.output else 'Installed'} daily 03:17 Asia/Jerusalem sync: {destination}"
    )


if __name__ == "__main__":
    main()
