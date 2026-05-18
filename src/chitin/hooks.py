# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "chitin" / "config.toml"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return tomllib.loads(CONFIG_PATH.read_text())


def get_post_process_command(cli_override: str | None = None) -> str | None:
    if cli_override is not None:
        return cli_override
    cfg = load_config()
    return cfg.get("hooks", {}).get("post_process")


def run_post_process(
    command_template: str, input_path: Path, quiet: bool = False
) -> str | None:
    cmd = command_template.replace("{input}", str(input_path))

    if not quiet:
        print(f"chitin: running post-process hook: {cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    stderr_lines = []
    for line in proc.stderr:
        decoded = line.decode("utf-8", errors="replace")
        stderr_lines.append(decoded)
        if not quiet:
            sys.stderr.write(decoded)

    stdout = proc.stdout.read().decode("utf-8", errors="replace").strip()
    proc.wait()

    if proc.returncode != 0:
        if not quiet:
            print(f"chitin: post-process hook exited with code {proc.returncode}")
        return None

    if stdout and not quiet:
        print(f"chitin: hook result: {stdout}")

    return stdout or None
