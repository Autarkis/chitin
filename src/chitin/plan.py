from __future__ import annotations

from dataclasses import dataclass, field


# Diagnostics an acceptance policy gates on. A build that subdivides its input
# -- per bone, per octree cell, per seam-repair group -- decomposes each piece
# under a throwaway plan so the pieces' pipeline steps don't flood the asset's,
# and octree cells run in a separate process where a shared plan isn't reachable
# at all. These counters have to be carried back either way: a strict profile
# reads a missing counter as zero, so an unmerged CoACD timeout is not a
# rejected build, it is an accepted one with a bounding box in it.
CHILD_COUNTERS = ("coacd_timeouts", "fallback_hulls")


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

    def child_counters(self) -> dict[str, int]:
        """This plan's gate-relevant counters, ready to merge into a parent.

        Picklable, so it survives the process boundary an octree cell crosses.
        """
        return {
            k: int(self.detected[k]) for k in CHILD_COUNTERS if self.detected.get(k)
        }

    def merge_counters(self, counters: dict[str, int]) -> None:
        """Add a child's counters (from :meth:`child_counters`) into this plan."""
        for key, value in counters.items():
            self.detected[key] = int(self.detected.get(key, 0)) + int(value)

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
