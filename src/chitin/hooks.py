from __future__ import annotations

import shlex
import subprocess
import sys
import threading
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
    cmd_str = command_template.replace("{input}", shlex.quote(str(input_path)))
    cmd = shlex.split(cmd_str)

    if not quiet:
        print(f"chitin: running post-process hook: {cmd_str}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def _stream_stderr():
        for line in proc.stderr:
            decoded = line.decode("utf-8", errors="replace")
            if not quiet:
                sys.stderr.write(decoded)

    stderr_thread = threading.Thread(target=_stream_stderr, daemon=True)
    stderr_thread.start()

    stdout = proc.stdout.read().decode("utf-8", errors="replace").strip()
    stderr_thread.join()
    proc.wait()

    if proc.returncode != 0:
        if not quiet:
            print(f"chitin: post-process hook exited with code {proc.returncode}")
        return None

    if stdout and not quiet:
        print(f"chitin: hook result: {stdout}")

    return stdout or None
