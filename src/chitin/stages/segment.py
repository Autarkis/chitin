# Existing-check: scripts/, ~/.claude/scripts/, devops_tools/ - no match
from __future__ import annotations

import numpy as np


def segment_by_bone(
    vertices: np.ndarray,
    faces: np.ndarray,
    joint_indices: np.ndarray,
    joint_weights: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    dominant_bone = joint_indices[
        np.arange(len(joint_indices)), joint_weights.argmax(axis=1)
    ]

    face_vertex_bones = dominant_bone[faces]
    face_bone = np.zeros(len(faces), dtype=np.int32)
    for i, fb in enumerate(face_vertex_bones):
        bones, counts = np.unique(fb, return_counts=True)
        face_bone[i] = bones[counts.argmax()]

    segments: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bone_idx in np.unique(face_bone):
        bone_faces = faces[face_bone == bone_idx]
        used_verts = np.unique(bone_faces)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[used_verts] = np.arange(len(used_verts))
        segments[int(bone_idx)] = (vertices[used_verts], remap[bone_faces])

    return segments
