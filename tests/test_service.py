import pytest
import trimesh
from fastapi.testclient import TestClient

from chitin_service.app import app, set_store
from chitin_service.models import JobConfig
from chitin_service.store import Store


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


def test_submit_and_complete(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "complete"

    events = [e["status"] for e in data["events"]]
    assert "uploaded" in events
    assert "preflighted" in events
    assert "running" in events
    assert "complete" in events


def test_get_job(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    job_id = resp.json()["id"]

    resp = client.get(f"/v1/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "complete"


def test_get_events(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    job_id = resp.json()["id"]

    resp = client.get(f"/v1/jobs/{job_id}/events")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert len(events) >= 4


def test_list_and_download_artifacts(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "json"},
        )
    job_id = resp.json()["id"]

    resp = client.get(f"/v1/jobs/{job_id}/artifacts")
    assert resp.status_code == 200
    names = resp.json()["artifacts"]
    assert "colliders.json" in names
    assert "report.json" in names

    resp = client.get(f"/v1/jobs/{job_id}/artifacts/report.json")
    assert resp.status_code == 200
    report = resp.json()
    assert "hull_count" in report
    assert "compiler_version" in report
    assert isinstance(report["warnings"], list)


def test_cache_hit(client, box_glb):
    with open(box_glb, "rb") as f:
        data = f.read()

    resp1 = client.post(
        "/v1/jobs",
        files={"file": ("box.glb", data, "model/gltf-binary")},
        params={"outputs": "json"},
    )
    assert resp1.status_code == 201
    first_id = resp1.json()["id"]

    resp2 = client.post(
        "/v1/jobs",
        files={"file": ("box.glb", data, "model/gltf-binary")},
        params={"outputs": "json"},
    )
    assert resp2.status_code == 201
    assert resp2.json()["cached_from"] == first_id


def test_cache_key_distinguishes_input_kind():
    # Adapter dispatch is extension-based, so identical bytes + config submitted
    # as .glb vs .obj must produce different cache keys (they route through
    # different adapters). Regression for the cross-adapter cache-collision bug.
    cfg = JobConfig()
    assert Store.hash_config(cfg, ["json"], ".glb") != Store.hash_config(
        cfg, ["json"], ".obj"
    )


def test_compiler_version_pins_dependency_versions():
    # A CoACD upgrade must invalidate persisted caches, so its version is part
    # of the compiler-version cache component.
    assert "coacd" in Store.compiler_version()


def test_404_unknown_job(client):
    resp = client.get("/v1/jobs/nonexistent")
    assert resp.status_code == 404


def test_empty_file_rejected(client):
    resp = client.post(
        "/v1/jobs",
        files={"file": ("empty.glb", b"", "model/gltf-binary")},
    )
    assert resp.status_code == 400


def test_bad_format_rejected(client, box_glb):
    with open(box_glb, "rb") as f:
        resp = client.post(
            "/v1/jobs",
            files={"file": ("box.glb", f, "model/gltf-binary")},
            params={"outputs": "mp4"},
        )
    assert resp.status_code == 400


def test_submit_cli_omits_poisson_depth_by_default(monkeypatch, tmp_path):
    # The submit CLI must not force depth 8 (documented-unstable); by default it
    # sends no poisson_depth so the server auto-selects 4-7 per cell.
    import argparse

    from chitin_service import cli

    captured = {}

    class _Resp:
        status_code = 201
        text = ""

        def json(self):
            return {"id": "j1", "status": "complete"}

    def _fake_post(url, files=None, params=None, timeout=None):
        captured["params"] = params
        return _Resp()

    monkeypatch.setattr("httpx.post", _fake_post)
    f = tmp_path / "m.glb"
    f.write_bytes(b"x")
    args = argparse.Namespace(
        file=str(f),
        server="http://x",
        outputs="phys",
        concavity=0.05,
        opacity_threshold=0.5,
        poisson_depth=None,
        min_hull_vertices=4,
        max_hulls=256,
        opacity_is_logit=False,
        coacd_preprocess_mode="auto",
        coacd_preprocess_resolution=50,
        max_decompose_vertices=200_000,
    )

    cli._cmd_submit(args)
    assert "poisson_depth" not in captured["params"]

    args.poisson_depth = 8
    cli._cmd_submit(args)
    assert captured["params"]["poisson_depth"] == 8
