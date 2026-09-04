"""Adversarial differential testing for Chitin plane predicates.

This module is test tooling, not a runtime predicate implementation.  It treats
the finite IEEE-754 inputs as exact rational numbers, searches close to f32 and
grid decision boundaries, and records small reproducers when Policy 0.2.0
disagrees with that exact-input oracle.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from chitin.f32_policy import POLICY_0_2_0, QuantizationPolicy
from chitin.f32_predicates import classify_plane_f32


@dataclass(frozen=True)
class PlaneCase:
    """One vertex cloud and plane presented to a classifier."""

    vertices: np.ndarray
    plane_point: np.ndarray
    plane_normal: np.ndarray
    label: str = ""

    def validate(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if len(self.vertices) == 0:
            raise ValueError("vertices must not be empty")
        if self.plane_point.shape != (3,) or self.plane_normal.shape != (3,):
            raise ValueError("plane_point and plane_normal must have shape (3,)")
        if not (
            np.all(np.isfinite(self.vertices))
            and np.all(np.isfinite(self.plane_point))
            and np.all(np.isfinite(self.plane_normal))
        ):
            raise ValueError("exact-input oracle requires finite inputs")
        if not np.any(self.plane_normal):
            raise ValueError("plane_normal must be nonzero")


@dataclass(frozen=True)
class DifferentialResult:
    """Exact, f64, and policy classifications for one case."""

    exact_signs: np.ndarray
    f64_signs: np.ndarray
    f32_signs: np.ndarray
    fast_path_count: int
    ambiguity_path_count: int

    @property
    def f32_exact_mismatch(self) -> np.ndarray:
        return self.f32_signs != self.exact_signs

    @property
    def f64_exact_mismatch(self) -> np.ndarray:
        return self.f64_signs != self.exact_signs

    @property
    def num_f32_exact_mismatch(self) -> int:
        return int(np.count_nonzero(self.f32_exact_mismatch))

    @property
    def num_f64_exact_mismatch(self) -> int:
        return int(np.count_nonzero(self.f64_exact_mismatch))


@dataclass(frozen=True)
class AdversarialFinding:
    """A differential or metamorphic failure retained by the search."""

    case: PlaneCase
    result: DifferentialResult
    failure_types: tuple[str, ...]
    coverage_bucket: tuple[str, ...]
    score: float


def _fraction(value: np.floating | float) -> Fraction:
    numerator, denominator = float(value).as_integer_ratio()
    return Fraction(numerator, denominator)


def classify_plane_exact(case: PlaneCase) -> np.ndarray:
    """Return exact signs of the supplied finite binary floating-point inputs.

    This does not claim the source geometry is mathematically exact.  It answers
    the narrower and useful testing question: what is the exact sign of the
    numbers that actually entered the predicate?
    """

    case.validate()
    point = [_fraction(x) for x in case.plane_point]
    normal = [_fraction(x) for x in case.plane_normal]
    signs = np.empty(len(case.vertices), dtype=np.int8)
    for row_index, vertex in enumerate(case.vertices):
        dot = sum(
            (_fraction(vertex[axis]) - point[axis]) * normal[axis] for axis in range(3)
        )
        signs[row_index] = 1 if dot > 0 else -1 if dot < 0 else 0
    return signs


def evaluate_case(
    case: PlaneCase, policy: QuantizationPolicy = POLICY_0_2_0
) -> DifferentialResult:
    """Classify one case with exact-input, f64, and policy predicates."""

    case.validate()
    vertices = case.vertices.astype(np.float64, copy=False)
    point = case.plane_point.astype(np.float64, copy=False)
    normal = case.plane_normal.astype(np.float64, copy=False)
    exact = classify_plane_exact(case)
    f64 = np.sign(np.dot(vertices - point, normal)).astype(np.int8)
    candidate = classify_plane_f32(vertices, point, normal, policy)
    return DifferentialResult(
        exact_signs=exact,
        f64_signs=f64,
        f32_signs=candidate.signs,
        fast_path_count=candidate.fast_path_count,
        ambiguity_path_count=candidate.ambiguity_path_count,
    )


def generate_boundary_case(rng: np.random.Generator, index: int) -> PlaneCase:
    """Generate a finite case with one vertex close to a decision boundary.

    Geometry starts in f32, matching common mesh inputs.  The plane is then
    placed on the vertex or moved by f64 ULPs, f32 ULPs, or fractions of a
    normalized grid cell.  Axis and oblique normals probe casting, quantization,
    and dot-product cancellation boundaries without making every generated case
    a guaranteed failure.
    """

    vertex_count = int(rng.integers(4, 17))
    exponent = int(rng.integers(-12, 13))
    scale = np.float32(2.0**exponent)
    center = rng.integers(-8, 9, size=3).astype(np.float32) * scale
    vertices = center + rng.normal(size=(vertex_count, 3)).astype(np.float32) * scale
    vertices = vertices.astype(np.float64)

    target_index = int(rng.integers(0, vertex_count))
    target = vertices[target_index]
    if index % 2 == 0:
        axis = int(rng.integers(0, 3))
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0
    else:
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        axis = int(np.argmax(np.abs(normal)))

    point = target.copy()
    direction = np.inf if rng.integers(0, 2) else -np.inf
    steps = int(rng.integers(1, 5))
    boundary_kind = ("f64_ulp", "f32_ulp", "grid_cell", "on_plane")[index % 4]
    if boundary_kind == "f64_ulp":
        for _ in range(steps):
            point[axis] = np.nextafter(point[axis], direction)
    elif boundary_kind == "f32_ulp":
        value = np.float32(point[axis])
        target_direction = np.float32(direction)
        for _ in range(steps):
            value = np.nextafter(value, target_direction, dtype=np.float32)
        point[axis] = float(value)
    elif boundary_kind == "grid_cell":
        extent = max(float(np.max(np.abs(vertices - vertices.mean(axis=0)))), 1e-30)
        grid_cell = 2.0 * extent / POLICY_0_2_0.grid_scale
        fraction = (0.25, 0.5, 1.0, 2.0)[steps - 1]
        point[axis] += (-1.0 if direction < 0 else 1.0) * grid_cell * fraction

    return PlaneCase(
        vertices=vertices,
        plane_point=point,
        plane_normal=normal,
        label=f"seeded-{boundary_kind}-{index:06d}",
    )


def permute_vertices(case: PlaneCase, order: np.ndarray) -> PlaneCase:
    """Return a vertex-order metamorph of ``case``."""

    if sorted(int(x) for x in order) != list(range(len(case.vertices))):
        raise ValueError("order must be a permutation of all vertex indices")
    return replace(case, vertices=case.vertices[order], label=f"{case.label}:permute")


def power_of_two_scale(case: PlaneCase, exponent: int) -> PlaneCase:
    """Scale positions by an exact power of two; plane normal is unchanged."""

    factor = float(2.0**exponent)
    scaled = replace(
        case,
        vertices=case.vertices * factor,
        plane_point=case.plane_point * factor,
        label=f"{case.label}:scale2^{exponent}",
    )
    scaled.validate()
    return scaled


def translate(case: PlaneCase, offset: np.ndarray) -> PlaneCase:
    """Translate vertices and plane point by the same finite offset."""

    offset = np.asarray(offset, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("offset must be a finite shape-(3,) vector")
    translated = replace(
        case,
        vertices=case.vertices + offset,
        plane_point=case.plane_point + offset,
        label=f"{case.label}:translate",
    )
    translated.validate()
    return translated


def _metamorphic_failures(
    case: PlaneCase,
    baseline: DifferentialResult,
    rng: np.random.Generator,
    policy: QuantizationPolicy,
) -> list[str]:
    failures: list[str] = []

    order = rng.permutation(len(case.vertices))
    permuted = evaluate_case(permute_vertices(case, order), policy)
    inverse = np.argsort(order)
    if not np.array_equal(permuted.exact_signs[inverse], baseline.exact_signs):
        raise AssertionError("vertex permutation changed exact classifications")
    if not np.array_equal(permuted.f32_signs[inverse], baseline.f32_signs):
        failures.append("vertex_permutation")

    exponent = int(rng.integers(-6, 7))
    scaled = evaluate_case(power_of_two_scale(case, exponent), policy)
    if np.array_equal(scaled.exact_signs, baseline.exact_signs) and not np.array_equal(
        scaled.f32_signs, baseline.f32_signs
    ):
        failures.append("power_of_two_scale")

    extent = max(float(np.max(np.abs(case.vertices))), 1.0)
    offset = rng.integers(-4, 5, size=3).astype(np.float64) * extent
    translated = evaluate_case(translate(case, offset), policy)
    if np.array_equal(
        translated.exact_signs, baseline.exact_signs
    ) and not np.array_equal(translated.f32_signs, baseline.f32_signs):
        failures.append("translation")

    return failures


def _finding_score(
    case: PlaneCase, result: DifferentialResult, failures: list[str]
) -> float:
    signed = np.dot(case.vertices - case.plane_point, case.plane_normal)
    scale = max(float(np.max(np.ptp(case.vertices, axis=0))), np.finfo(float).tiny)
    proximity = float(np.min(np.abs(signed)) / scale)
    proximity_score = min(50.0, -np.log10(max(proximity, np.finfo(float).tiny)))
    return (
        1000.0 * result.num_f32_exact_mismatch
        + 100.0 * len(failures)
        + proximity_score
        + result.ambiguity_path_count / max(1, len(case.vertices))
    )


def _coverage_bucket(
    case: PlaneCase, result: DifferentialResult, failures: list[str]
) -> tuple[str, ...]:
    """Describe a numerical niche for MAP-Elites-style result retention."""

    nonzero_normal_axes = int(np.count_nonzero(case.plane_normal))
    normal_family = "axis" if nonzero_normal_axes == 1 else "oblique"
    extent = max(float(np.max(np.ptp(case.vertices, axis=0))), np.finfo(float).tiny)
    scale_bin = int(np.floor(np.log2(extent) / 4.0))
    if result.ambiguity_path_count == 0:
        ambiguity_family = "fast"
    elif result.ambiguity_path_count == len(case.vertices):
        ambiguity_family = "ambiguity-all"
    else:
        ambiguity_family = "ambiguity-mixed"
    boundary_family = next(
        (
            kind
            for kind in ("f64_ulp", "f32_ulp", "grid_cell", "on_plane")
            if kind in case.label
        ),
        "external",
    )
    return (
        boundary_family,
        normal_family,
        f"scale-bin-{scale_bin}",
        ambiguity_family,
        *sorted(set(failures)),
    )


def search_adversaries(
    *,
    seed: int,
    cases: int,
    policy: QuantizationPolicy = POLICY_0_2_0,
    generator: Callable[[np.random.Generator, int], PlaneCase] = generate_boundary_case,
) -> list[AdversarialFinding]:
    """Run a deterministic boundary search with MAP-Elites-style retention.

    Only the highest-scoring finding in each numerical coverage niche is kept.
    This prevents a long campaign from returning thousands of equivalent ULP
    failures while preserving different boundary, normal, scale, path, and
    failure families.
    """

    if cases < 0:
        raise ValueError("cases must be nonnegative")
    rng = np.random.default_rng(seed)
    archive: dict[tuple[str, ...], AdversarialFinding] = {}
    for index in range(cases):
        case = generator(rng, index)
        result = evaluate_case(case, policy)
        failures: list[str] = []
        if result.num_f32_exact_mismatch:
            failures.append("f32_exact")
        if result.num_f64_exact_mismatch:
            failures.append("f64_exact")
        failures.extend(_metamorphic_failures(case, result, rng, policy))
        if failures:
            bucket = _coverage_bucket(case, result, failures)
            finding = AdversarialFinding(
                case=case,
                result=result,
                failure_types=tuple(sorted(set(failures))),
                coverage_bucket=bucket,
                score=_finding_score(case, result, failures),
            )
            incumbent = archive.get(bucket)
            if incumbent is None or finding.score > incumbent.score:
                archive[bucket] = finding
    findings = list(archive.values())
    findings.sort(key=lambda finding: (-finding.score, finding.case.label))
    return findings


def shrink_f32_exact_failure(
    case: PlaneCase, policy: QuantizationPolicy = POLICY_0_2_0
) -> PlaneCase:
    """Greedily remove vertices while preserving an f32/exact disagreement."""

    if evaluate_case(case, policy).num_f32_exact_mismatch == 0:
        raise ValueError("case does not contain an f32/exact disagreement")
    current = case
    changed = True
    while changed and len(current.vertices) > 1:
        changed = False
        for index in range(len(current.vertices)):
            candidate_vertices = np.delete(current.vertices, index, axis=0)
            if len(candidate_vertices) == 0:
                continue
            candidate = replace(
                current,
                vertices=candidate_vertices,
                label=f"{case.label}:shrunk",
            )
            if evaluate_case(candidate, policy).num_f32_exact_mismatch:
                current = candidate
                changed = True
                break
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_findings(
    findings: Iterable[AdversarialFinding],
    output_dir: Path,
    *,
    seed: int,
    policy: QuantizationPolicy = POLICY_0_2_0,
) -> Path:
    """Write immutable NPZ reproducers and a digest-bearing manifest."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    entries = []
    for index, finding in enumerate(findings):
        shrunk = (
            shrink_f32_exact_failure(finding.case, policy)
            if finding.result.num_f32_exact_mismatch
            else finding.case
        )
        filename = f"finding_{index:06d}.npz"
        path = output_dir / filename
        np.savez(
            path,
            vertices=shrunk.vertices,
            plane_point=shrunk.plane_point,
            plane_normal=shrunk.plane_normal,
        )
        entries.append(
            {
                "file": filename,
                "sha256": _sha256(path),
                "source_label": finding.case.label,
                "failure_types": list(finding.failure_types),
                "coverage_bucket": list(finding.coverage_bucket),
                "score": finding.score,
                "vertices_before": len(finding.case.vertices),
                "vertices_after": len(shrunk.vertices),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "generator": "boundary-directed-v1",
        "seed": seed,
        "policy_version": policy.version,
        "findings": entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
