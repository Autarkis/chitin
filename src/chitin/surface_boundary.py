"""Recover cap boundaries from retained triangle incidence."""

from collections import defaultdict

import numpy as np


def surface_loops(faces: np.ndarray) -> list[np.ndarray]:
    """Walk boundary edges through connected incident-face fans.

    Two surface sheets can touch at a vertex without sharing an edge. Their
    boundary walks remain separate because pairing uses face connectivity,
    rather than arbitrary neighbor order at the shared vertex.
    """
    incidence = defaultdict(list)
    for fi, tri in enumerate(faces):
        if len(set(map(int, tri))) < 3:
            raise ValueError("repeated vertex in surface face")
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            incidence[min(a, b), max(a, b)].append(fi)
    if any(len(items) > 2 for items in incidence.values()):
        raise ValueError("nonmanifold surface edge")
    boundary = sorted(edge for edge, items in incidence.items() if len(items) == 1)
    if not boundary:
        return []
    at_vertex = defaultdict(list)
    for edge, items in incidence.items():
        for v in edge:
            at_vertex[v].append((edge, items))
    paired = {}
    for v, edges in at_vertex.items():
        parents = {}

        def find(x, parents=parents):
            parents.setdefault(x, x)
            while parents[x] != x:
                x = parents[x]
            return x

        for _, items in edges:
            if len(items) == 2:
                parents[find(items[0])] = find(items[1])
        groups = defaultdict(list)
        for edge, items in edges:
            if len(items) == 1:
                groups[find(items[0])].append(edge)
        for group in groups.values():
            if len(group) != 2:
                raise ValueError("open or branching surface boundary fan")
            a, b = group
            paired[v, a] = b
            paired[v, b] = a
    used = set()
    loops = []
    for first in boundary:
        if first in used:
            continue
        start, vertex = first
        edge = first
        loop = [start]
        while edge not in used:
            used.add(edge)
            if vertex == start and paired[vertex, edge] == first:
                break
            loop.append(vertex)
            next_edge = paired[vertex, edge]
            vertex = next_edge[1] if next_edge[0] == vertex else next_edge[0]
            edge = next_edge
        else:
            raise ValueError("boundary walk joined an existing loop")
        # A face fan can trace a figure-eight boundary at an on-plane
        # saddle vertex. Split at repeated occurrences along that walk;
        # connectivity still comes from incident faces, not neighbor order.
        path = []
        positions = {}
        for vertex in loop + [loop[0]]:
            if vertex in positions:
                begin = positions[vertex]
                cycle = path[begin:]
                if len(cycle) < 3:
                    raise ValueError("collapsed surface boundary cycle")
                loops.append(np.asarray(cycle, dtype=np.int64))
                for item in cycle:
                    del positions[item]
                path = path[:begin]
            positions[vertex] = len(path)
            path.append(vertex)
    return loops
