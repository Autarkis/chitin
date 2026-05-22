from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BuildPlan:
    input_kind: str
    collider_kind: str = "unknown"
    pipeline: list[str] = field(default_factory=list)
    source_vertices: int = 0
    processed_vertices: int = 0
    decimated: bool = False
    detected: dict = field(default_factory=dict)

    def step(self, name: str) -> None:
        self.pipeline.append(name)

    def to_dict(self) -> dict:
        return {
            "input_kind": self.input_kind,
            "collider_kind": self.collider_kind,
            "pipeline": self.pipeline,
            "source_vertices": self.source_vertices,
            "processed_vertices": self.processed_vertices,
            "decimated": self.decimated,
            "detected": self.detected,
        }
