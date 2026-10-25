"""Deterministic boundary-preserving triangulation for host cap assembly."""

from __future__ import annotations

import math

import numpy as np


def triangulate_polygon(loop: np.ndarray, points: np.ndarray) -> list[list[int]]:
    """Maximize the minimum triangle area over valid polygon triangulations.

    Dynamic programming avoids leaving collinear boundary vertices in a final
    ear. Invalid diagonals are excluded before selection; no fan fallback or
    deletion of boundary vertices is permitted. Coordinates remain unchanged.
    """
    n = len(loop)
    if n < 3 or len(set(map(int, loop))) != n:
        raise ValueError("cap requires a simple loop of at least three vertices")
    p = [(float(x), float(y)) for x, y in points]

    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on(a, b, c):
        return (
            orient(a, b, c) == 0
            and min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
        )

    def intersects(a, b, c, d):
        ab_c, ab_d = orient(a, b, c), orient(a, b, d)
        cd_a, cd_b = orient(c, d, a), orient(c, d, b)
        return (
            (ab_c * ab_d < 0 and cd_a * cd_b < 0)
            or on(a, b, c)
            or on(a, b, d)
            or on(c, d, a)
            or on(c, d, b)
        )

    def inside(q):
        result = False
        for i in range(n):
            a, b = p[i], p[(i + 1) % n]
            if (a[1] > q[1]) != (b[1] > q[1]):
                x = a[0] + (q[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if q[0] < x:
                    result = not result
        return result

    for i in range(n):
        if p[i] == p[(i + 1) % n]:
            raise ValueError("collapsed cap boundary edge")
        for j in range(i + 1, n):
            if len({i, (i + 1) % n, j, (j + 1) % n}) == 4 and intersects(
                p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]
            ):
                raise ValueError("self-intersecting cap boundary")

    area = math.fsum(orient(p[0], p[i], p[i + 1]) for i in range(1, n - 1))
    if area == 0:
        raise ValueError("zero-area cap boundary")
    sign = 1 if area > 0 else -1
    valid = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            boundary = j == i + 1 or (i == 0 and j == n - 1)
            ok = boundary
            if not boundary:
                ok = inside(((p[i][0] + p[j][0]) / 2, (p[i][1] + p[j][1]) / 2))
                for k in range(n):
                    k1 = (k + 1) % n
                    if k not in (i, j) and on(p[i], p[j], p[k]):
                        ok = False
                    if len({i, j, k, k1}) == 4 and intersects(p[i], p[j], p[k], p[k1]):
                        ok = False
            valid[i, j] = valid[j, i] = ok

    score = np.full((n, n), -1.0)
    split = np.full((n, n), -1, dtype=int)
    for i in range(n - 1):
        score[i, i + 1] = math.inf
    for width in range(2, n):
        for i in range(n - width):
            j = i + width
            if not valid[i, j]:
                continue
            for k in range(i + 1, j):
                cross = sign * orient(p[i], p[k], p[j])
                if cross <= 0 or score[i, k] < 0 or score[k, j] < 0:
                    continue
                candidate = min(cross, score[i, k], score[k, j])
                if candidate > score[i, j]:
                    score[i, j], split[i, j] = candidate, k
    if split[0, n - 1] < 0:
        raise ValueError("no boundary-preserving cap triangulation")
    result = []
    pending = [(0, n - 1)]
    while pending:
        i, j = pending.pop()
        if j <= i + 1:
            continue
        k = int(split[i, j])
        result.append([int(loop[i]), int(loop[k]), int(loop[j])])
        pending.extend([(k, j), (i, k)])
    return result
