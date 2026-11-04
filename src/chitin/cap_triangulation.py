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

    pts = np.array(p, dtype=np.float64)
    px, py = pts[:, 0], pts[:, 1]
    nxt = (np.arange(n) + 1) % n
    if (pts == pts[nxt]).all(axis=1).any():
        raise ValueError("collapsed cap boundary edge")
    cx, cy = px[nxt], py[nxt]  # edge k runs (px[k], py[k]) -> (cx[k], cy[k])

    def v_orient(ax, ay, bx, by, qx, qy):
        return (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)

    def v_on(ax, ay, bx, by, qx, qy):
        return (
            (v_orient(ax, ay, bx, by, qx, qy) == 0)
            & (np.minimum(ax, bx) <= qx)
            & (qx <= np.maximum(ax, bx))
            & (np.minimum(ay, by) <= qy)
            & (qy <= np.maximum(ay, by))
        )

    def v_intersects(ax, ay, bx, by, qx, qy, rx, ry):
        ab_q = v_orient(ax, ay, bx, by, qx, qy)
        ab_r = v_orient(ax, ay, bx, by, rx, ry)
        qr_a = v_orient(qx, qy, rx, ry, ax, ay)
        qr_b = v_orient(qx, qy, rx, ry, bx, by)
        return (
            ((ab_q * ab_r < 0) & (qr_a * qr_b < 0))
            | v_on(ax, ay, bx, by, qx, qy)
            | v_on(ax, ay, bx, by, rx, ry)
            | v_on(qx, qy, rx, ry, ax, ay)
            | v_on(qx, qy, rx, ry, bx, by)
        )

    ks = np.arange(n)
    with np.errstate(all="ignore"):
        for i in range(n):
            i1 = (i + 1) % n
            js = np.arange(i + 1, n)
            distinct = (js != i1) & (((js + 1) % n) != i)
            js = js[distinct]
            if len(js) == 0:
                continue
            hit = v_intersects(
                px[i],
                py[i],
                cx[i],
                cy[i],
                px[js],
                py[js],
                cx[js],
                cy[js],
            )
            if hit.any():
                raise ValueError("self-intersecting cap boundary")

    area = math.fsum(orient(p[0], p[i], p[i + 1]) for i in range(1, n - 1))
    if area == 0:
        raise ValueError("zero-area cap boundary")
    sign = 1 if area > 0 else -1
    valid = np.eye(n, dtype=bool)
    with np.errstate(all="ignore"):
        for i in range(n):
            js = np.arange(i + 1, n)
            boundary = (js == i + 1) | ((i == 0) & (js == n - 1))
            valid[i, js[boundary]] = valid[js[boundary], i] = True
            js = js[~boundary]
            if len(js) == 0:
                continue
            # point-in-polygon of each diagonal midpoint (even-odd rule)
            qx = ((px[i] + px[js]) / 2)[:, None]
            qy = ((py[i] + py[js]) / 2)[:, None]
            straddle = (py[None, :] > qy) != (cy[None, :] > qy)
            xs = (
                px[None, :]
                + (qy - py[None, :]) * (cx - px)[None, :] / (cy - py)[None, :]
            )
            crossing = straddle & (qx < xs)
            ok = (crossing.sum(axis=1) % 2) == 1
            ax, ay = px[i], py[i]
            bx, by = px[js][:, None], py[js][:, None]
            kx, ky = px[None, :], py[None, :]
            k1x, k1y = cx[None, :], cy[None, :]
            not_ij = (ks[None, :] != i) & (ks[None, :] != js[:, None])
            bad_on = not_ij & v_on(ax, ay, bx, by, kx, ky)
            distinct4 = (
                not_ij
                & (ks[None, :] != i)
                & (nxt[None, :] != i)
                & (nxt[None, :] != js[:, None])
            )
            bad_x = distinct4 & v_intersects(ax, ay, bx, by, kx, ky, k1x, k1y)
            ok &= ~(bad_on | bad_x).any(axis=1)
            valid[i, js] = valid[js, i] = ok

    score = np.full((n, n), -1.0)
    split = np.full((n, n), -1, dtype=int)
    for i in range(n - 1):
        score[i, i + 1] = math.inf
    with np.errstate(all="ignore"):
        for width in range(2, n):
            rows = np.arange(n - width)
            cols = rows + width
            kk = rows[:, None] + np.arange(1, width)[None, :]
            ii = rows[:, None]
            jj = cols[:, None]
            cross = sign * v_orient(px[ii], py[ii], px[kk], py[kk], px[jj], py[jj])
            s_ik = score[ii, kk]
            s_kj = score[kk, jj]
            ok = (cross > 0) & (s_ik >= 0) & (s_kj >= 0)
            cand = np.where(ok, np.minimum(np.minimum(cross, s_ik), s_kj), -np.inf)
            best = cand.argmax(axis=1)
            best_val = cand[np.arange(len(rows)), best]
            upd = valid[rows, cols] & (best_val > -1.0)
            score[rows[upd], cols[upd]] = best_val[upd]
            split[rows[upd], cols[upd]] = rows[upd] + 1 + best[upd]
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
