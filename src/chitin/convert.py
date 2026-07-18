from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

BLENDER_SCRIPT = """\
import bpy, sys
argv = sys.argv[sys.argv.index("--") + 1:]
src, dst = argv[0], argv[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=src)
bpy.ops.export_scene.gltf(filepath=dst, export_format="GLB")
"""

BLENDER_INSTALL_HINT = (
    "blender not found. FBX import requires Blender on PATH.\n"
    "  macOS:   brew install --cask blender\n"
    "  Ubuntu:  sudo snap install blender --classic\n"
    "  Windows: winget install BlenderFoundation.Blender"
)


class BlenderNotFoundError(RuntimeError):
    """Raised when Blender is required (FBX import) but not on PATH."""


def find_blender() -> str | None:
    blender = shutil.which("blender")
    if blender:
        return blender
    candidates = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None


def convert_fbx_to_glb(input_path: str | Path, output_path: str | Path) -> None:
    """Convert an FBX file to GLB via headless Blender.

    Raises BlenderNotFoundError if Blender is not on PATH, or RuntimeError if
    the conversion subprocess fails.
    """
    blender = find_blender()
    if not blender:
        raise BlenderNotFoundError(BLENDER_INSTALL_HINT)

    script_path = Path(output_path).parent / ".chitin_fbx2glb.py"
    script_path.write_text(BLENDER_SCRIPT)

    try:
        result = subprocess.run(
            [
                blender,
                "-b",
                "--python",
                str(script_path),
                "--",
                str(input_path),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"blender FBX->GLB conversion failed:\n{result.stderr}")
    finally:
        script_path.unlink(missing_ok=True)
