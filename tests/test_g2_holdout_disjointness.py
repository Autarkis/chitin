"""G2 holdout disjointness tests.

Proves that G2 holdout fixtures are disjoint from all spent corpora
and excluded from routine GPU/clipping test discovery.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chitin.trace_fixtures import G2_HOLDOUT_FIXTURES, HOLDOUT_FIXTURES

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
TESTS_DIR = Path(__file__).resolve().parent
THIS_FILE = Path(__file__).name

CORPUS_FILES = [
    DOCS_DIR / "holdout-corpus-0.2.0.json",
    DOCS_DIR / "holdout-corpus-0.3.0.json",
]
RESULTS_FILES = [
    DOCS_DIR / "holdout-results-0.2.0.json",
    DOCS_DIR / "holdout-results-0.3.0.json",
]

# Frozen canonical digests: SHA256(vertices.tobytes() + faces.tobytes()).
# Any change to a G2 generator function must not silently change these.
FROZEN_G2_DIGESTS = {
    "g2_oblique_dipyramid": "4f8102ef027c4d1457bd00895f443625bd16f77c8e4ff8e41db7a96720631b18",
    "g2_razor_star_prism": "67132b46c6e476b7fe8b01ba92c7ab8eee1576c16a38108a4525615f9069f000",
    "g2_separated_octahedra_trio": "ad62f3c42f097c27ac757a2bd27e6601be701ca49c9a520fb4c8d0033ef92287",
    "g2_high_valence_bipyramid": "0eaaf66593b8caf7c89f724db2513f8f2b15fbe01f5cafaced92f5d9b2c5d78f",
    "g2_multiscale_lattice_box": "7f95ac764700d64ff9e55ea0bea7c261cd9b2949e8905a9d01d1fe6933091382",
}


def _fixture_digest(vertices, faces) -> str:
    """Same method as the manifest: SHA256(vertices.tobytes() + faces.tobytes())."""
    return hashlib.sha256(vertices.tobytes() + faces.tobytes()).hexdigest()


def _corpus_names(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["name"] for entry in data["fixtures"]}


def _corpus_digests(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {entry["source_digest"] for entry in data["fixtures"]}


# --- 1. Name disjointness -------------------------------------------------


def test_g2_names_disjoint_from_holdout_fixtures():
    overlap = set(G2_HOLDOUT_FIXTURES) & set(HOLDOUT_FIXTURES)
    assert not overlap, f"G2 names collide with HOLDOUT_FIXTURES: {overlap}"


@pytest.mark.parametrize("corpus_path", CORPUS_FILES, ids=lambda p: p.name)
def test_g2_names_disjoint_from_corpus(corpus_path):
    corpus_names = _corpus_names(corpus_path)
    overlap = set(G2_HOLDOUT_FIXTURES) & corpus_names
    assert not overlap, f"G2 names collide with {corpus_path.name}: {overlap}"


def test_g2_names_all_prefixed():
    for name in G2_HOLDOUT_FIXTURES:
        assert name.startswith("g2_"), f"{name!r} does not carry the g2_ prefix"


# --- 2. Digest disjointness ------------------------------------------------


@pytest.mark.parametrize("corpus_path", CORPUS_FILES, ids=lambda p: p.name)
def test_g2_digests_disjoint_from_corpus(corpus_path):
    corpus_digests = _corpus_digests(corpus_path)
    for name, factory in G2_HOLDOUT_FIXTURES.items():
        vertices, faces = factory()
        digest = _fixture_digest(vertices, faces)
        assert digest not in corpus_digests, (
            f"{name} digest {digest} appears in {corpus_path.name}"
        )


@pytest.mark.parametrize("results_path", RESULTS_FILES, ids=lambda p: p.name)
def test_g2_digests_absent_from_results(results_path):
    raw_text = results_path.read_text(encoding="utf-8")
    for name, factory in G2_HOLDOUT_FIXTURES.items():
        vertices, faces = factory()
        digest = _fixture_digest(vertices, faces)
        assert digest not in raw_text, (
            f"{name} digest {digest} appears in {results_path.name}"
        )


# --- 3. G2 exclusion from routine test discovery ---------------------------


def _test_files():
    return sorted(p for p in TESTS_DIR.rglob("*.py") if p.name != THIS_FILE)


def test_only_this_file_references_g2_fixtures():
    g2_names = list(G2_HOLDOUT_FIXTURES)
    offenders = []
    for path in _test_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "G2_HOLDOUT_FIXTURES" in text or any(name in text for name in g2_names):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, f"G2 fixtures referenced outside this file: {offenders}"


def test_gpu_and_clipping_tests_do_not_reference_g2():
    g2_names = list(G2_HOLDOUT_FIXTURES)
    offenders = []
    for path in _test_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        imports_gpu_or_clip = (
            "chitin.gpu" in text
            or "from chitin import gpu" in text
            or "chitin.clip" in text
            or "chitin.coacd_trace" in text
        )
        if not imports_gpu_or_clip:
            continue
        if "G2_HOLDOUT_FIXTURES" in text or any(name in text for name in g2_names):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, f"GPU/clipping tests reference G2 fixtures: {offenders}"


# --- 4. Frozen digest verification -----------------------------------------


@pytest.mark.parametrize("name", list(FROZEN_G2_DIGESTS))
def test_frozen_digest_matches_generator(name):
    vertices, faces = G2_HOLDOUT_FIXTURES[name]()
    digest = _fixture_digest(vertices, faces)
    assert digest == FROZEN_G2_DIGESTS[name], (
        f"{name} digest drifted: expected {FROZEN_G2_DIGESTS[name]}, got {digest}"
    )


def test_frozen_digest_table_covers_all_g2_fixtures():
    assert set(FROZEN_G2_DIGESTS) == set(G2_HOLDOUT_FIXTURES)
