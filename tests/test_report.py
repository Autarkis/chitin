import hashlib

from fastapi.testclient import TestClient

from chitin import __version__
from chitin_service.app import app, set_store
from chitin_service.store import Store

import pytest
import trimesh


@pytest.fixture
def client(tmp_path):
    store = Store(tmp_path)
    set_store(store)
    yield TestClient(app)
    set_store(None)


@pytest.fixture
def box_glb(tmp_path):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    path = tmp_path / "box.glb"
    mesh.export(str(path), file_type="glb")
    return path


def test_report_fields(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    job_id = resp.json()["id"]

    resp = client.get(f"/v1/jobs/{job_id}/artifacts/report.json")
    assert resp.status_code == 200
    report = resp.json()

    assert report["status"] == "complete"
    assert report["input_kind"] == "glb"
    assert report["collider_kind"] in ("static", "rigged")
    assert isinstance(report["pipeline"], list)
    assert len(report["pipeline"]) >= 1
    assert report["hull_count"] >= 1
    assert report["source_vertices"] > 0
    assert report["processed_vertices"] > 0
    assert isinstance(report["warnings"], list)
    assert isinstance(report["detected"], dict)
    canonical = report["compilation_report"]
    assert canonical["report_version"] == 1
    assert canonical["verdict"]["status"] == "pass"
    assert canonical["runtime"]["kind"] == "python_native"
    assert canonical["reproducibility"]["scope"] == "same_runtime_toolchain"

    # compiler_version pins the base version plus dependency versions
    # (coacd/open3d/trimesh) so a dependency upgrade invalidates caches.
    assert report["compiler_version"].startswith(__version__)
    assert "coacd" in report["compiler_version"]
    assert "json" in report["artifacts"]

    artifact_resp = client.get(
        f"/v1/jobs/{job_id}/artifacts/{report['artifacts']['json']}"
    )
    assert artifact_resp.status_code == 200
    artifact_bytes = artifact_resp.content

    assert canonical["output"]["byte_length"] == len(artifact_bytes)
    sha256 = canonical["reproducibility"]["artifact_sha256"]
    assert sha256 is not None
    assert len(sha256) == 64
    assert sha256 == hashlib.sha256(artifact_bytes).hexdigest()


def test_report_pipeline_has_parse_and_decompose(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    job_id = resp.json()["id"]

    resp = client.get(f"/v1/jobs/{job_id}/artifacts/report.json")
    pipeline = resp.json()["pipeline"]
    assert "parse" in pipeline
    assert "decompose" in pipeline
