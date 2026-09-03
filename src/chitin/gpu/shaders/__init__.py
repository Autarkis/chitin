"""WGSL shader sources for GPU compute primitives."""

from pathlib import Path

_SHADER_DIR = Path(__file__).parent


def load_shader(name: str) -> str:
    return (_SHADER_DIR / f"{name}.wgsl").read_text()
