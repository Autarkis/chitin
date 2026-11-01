"""The package gate must accept npm's hoisted workspace dependencies."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "npm_gate", Path(__file__).resolve().parents[1] / "scripts" / "npm_gate.py"
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_gate_accepts_hoisted_workspace_install(tmp_path):
    package = tmp_path / "repo" / "integrations" / "wasm-lite"
    package.mkdir(parents=True)
    (tmp_path / "repo" / "node_modules").mkdir()
    assert not (package / "node_modules").exists()
    assert _module.has_node_modules(package)


def test_gate_accepts_package_local_install(tmp_path):
    package = tmp_path / "repo" / "integrations" / "web"
    (package / "node_modules").mkdir(parents=True)
    assert _module.has_node_modules(package)


def test_sibling_install_does_not_satisfy_package(tmp_path):
    package = tmp_path / "repo" / "integrations" / "wasm-lite"
    package.mkdir(parents=True)
    (package.parent / "web" / "node_modules").mkdir(parents=True)
    assert not _module.has_node_modules(package)
