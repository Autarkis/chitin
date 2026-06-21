"""Cross-runtime .phys conformance — Python reader half.

Verifies the production reader (`read_phys`) and validator (`validate_phys`) against
the frozen golden corpus + ``manifest.json`` authored by ``build_fixtures.py``. The
web reader checks the *same* manifest (`integrations/web/test/conformance.test.ts`),
so the two runtimes are pinned to one set of expectations. Numpy-only — runs in CI's
light path (no open3d/coacd).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chitin.phys import read_phys, validate_phys

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))

VALID = sorted(k for k, v in MANIFEST.items() if v["valid"])
INVALID = sorted(k for k, v in MANIFEST.items() if not v["valid"])


def test_corpus_present():
    assert VALID and INVALID
    for name in MANIFEST:
        assert (FIXTURES / name).is_file(), f"missing fixture {name}"


@pytest.mark.parametrize("name", VALID)
def test_valid_fixture_reads_to_manifest(name):
    spec = MANIFEST[name]
    pf = read_phys(FIXTURES / name)

    assert pf.version == spec["version"]
    assert pf.flags == spec["flags"]
    assert pf.has_bones == spec["hasBones"]
    assert pf.has_bind_poses == spec["hasBindPoses"]
    assert pf.has_lod == spec["hasLod"]
    assert len(pf.hulls) == spec["hullCount"]
    assert pf.total_vertices == spec["totalVertices"]
    assert pf.total_indices == spec["totalIndices"]
    assert pf.total_triangles == spec["totalTriangles"]

    assert len(pf.hulls) == len(spec["hulls"])
    for hull, hspec in zip(pf.hulls, spec["hulls"]):
        assert len(hull.vertices) == hspec["vertexCount"]
        assert len(hull.indices) == hspec["indexCount"]
        assert hull.bone_index == hspec["boneIndex"]
        np.testing.assert_allclose(hull.aabb_min, hspec["aabbMin"], atol=1e-4)
        np.testing.assert_allclose(hull.aabb_max, hspec["aabbMax"], atol=1e-4)
        # indices stay within the hull's vertex range
        if len(hull.indices):
            assert int(hull.indices.max()) < len(hull.vertices)

    assert [b.name for b in pf.bones] == [b["name"] for b in spec["bones"]]
    assert len(pf.lod_tiers) == len(spec["lodTiers"])
    for tier, tspec in zip(pf.lod_tiers, spec["lodTiers"]):
        assert tier.hull_count == tspec["hullCount"]
        assert tier.concavity == pytest.approx(tspec["concavity"], abs=1e-4)


@pytest.mark.parametrize("name", VALID)
def test_valid_fixture_has_no_validation_errors(name):
    errors = [i for i in validate_phys(FIXTURES / name) if i.severity == "error"]
    assert errors == [], f"{name}: unexpected errors {[str(e) for e in errors]}"


@pytest.mark.parametrize("name", INVALID)
def test_invalid_fixture_is_rejected(name):
    spec = MANIFEST[name]
    needle = spec["errorContains"].lower()

    # read_phys raises with a message naming the defect
    with pytest.raises(ValueError) as exc:
        read_phys(FIXTURES / name)
    assert needle in str(exc.value).lower(), f"{name}: {exc.value!r} lacks {needle!r}"

    # validate_phys reports at least one error (and names the defect somewhere)
    errors = [i for i in validate_phys(FIXTURES / name) if i.severity == "error"]
    assert errors, f"{name}: validate_phys reported no errors"
    assert any(needle in str(e).lower() for e in errors), (
        f"{name}: no error mentions {needle!r}: {[str(e) for e in errors]}"
    )
