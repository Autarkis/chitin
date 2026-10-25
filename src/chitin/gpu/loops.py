"""Boundary loop extraction from welded clipping edges.

Extracts closed loops from the parity-deduped boundary edges produced by
the weld stage.  Each loop is an ordered ring of vertex indices (source
vertices below vertex_count, intersection vertices at or above it).

CPU-side: boundary edge counts are mesh-scale (hundreds, not millions),
and the adjacency walk is inherently serial.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chitin.gpu.weld import WeldResult


@dataclass
class LoopExtractionResult:
    loops: list[np.ndarray]  # each (Li,) uint32 -- ordered ring of vertex indices
    valid: bool  # True iff all boundary edges form closed valence-2 loops


def extract_boundary_loops(weld_result: WeldResult) -> LoopExtractionResult:
    """Extract closed boundary loops from welded boundary edges.

    The boundary edges in weld_result.boundary_edges are already welded
    (intersection indices remapped) and parity-deduped (interior on-plane
    edges cancelled).  This function builds an adjacency graph, validates
    that every vertex has valence exactly 2 (necessary for closed loops),
    and walks the graph to extract one or more closed loops.

    Determinism: when a vertex has two unused neighbors, the walk always
    picks the neighbor with the smaller vertex index first.  This makes
    loop extraction independent of boundary-edge input order.
    """
    boundary = weld_result.boundary_edges
    if len(boundary) == 0:
        return LoopExtractionResult(loops=[], valid=True)

    adjacency: dict[int, list[int]] = {}
    for edge in boundary:
        a, b = int(edge[0]), int(edge[1])
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    for nbrs in adjacency.values():
        if len(nbrs) != 2:
            return LoopExtractionResult(loops=[], valid=False)
        nbrs.sort()

    used_edges: set[frozenset[int]] = set()
    loops: list[np.ndarray] = []

    # Sorted for the same determinism reason as the adjacency lists above.
    sorted_edges = sorted(
        (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1]))) for e in boundary
    )

    for a, b in sorted_edges:
        key = frozenset((a, b))
        if key in used_edges:
            continue
        loop = [a, b]
        used_edges.add(key)
        current = b
        closed = False
        while True:
            neighbors = sorted(
                n
                for n in adjacency[current]
                if frozenset((current, n)) not in used_edges
            )
            if not neighbors:
                break
            nxt = neighbors[0]
            used_edges.add(frozenset((current, nxt)))
            if nxt == loop[0]:
                closed = True
                break
            loop.append(nxt)
            current = nxt
        if not closed:
            return LoopExtractionResult(loops=[], valid=False)
        loops.append(np.array(loop, dtype=np.uint32))

    return LoopExtractionResult(loops=loops, valid=True)
