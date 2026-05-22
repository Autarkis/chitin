from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chitin.result import ExtractionResult


def export_json(result: ExtractionResult, path: str | Path) -> None:
    hull_dicts = []
    for h in result.hulls:
        d = {
            "vertices": h.vertices.tolist(),
            "indices": h.indices.tolist(),
        }
        if h.bone_name is not None:
            d["bone_name"] = h.bone_name
        if h.bone_index is not None:
            d["bone_index"] = h.bone_index
        hull_dicts.append(d)

    meta = {
        "hull_count": len(result.hulls),
        "source_vertex_count": result.source_vertex_count,
        "mesh_vertex_count": result.mesh_vertex_count,
    }
    if result.bones:
        meta["rigged"] = True
        meta["bones"] = [
            {
                "name": b.name,
                "index": b.index,
                "bind_transform": b.bind_transform.tolist(),
            }
            for b in result.bones
        ]

    data = {"hulls": hull_dicts, "meta": meta}
    Path(path).write_text(json.dumps(data))
