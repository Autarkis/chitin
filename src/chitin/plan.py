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

# Gate-relevant facts that are true-or-false rather than countable, carried back
# the same way. They merge by AND: one piece decomposed without CoACD pinned to
# a single thread is enough to make the whole asset unreproducible.
CHILD_FLAGS = ("coacd_deterministic",)


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

    def child_signals(self) -> dict:
        """This plan's gate-relevant counters and flags, ready to merge upward.

        Picklable, so it survives the process boundary an octree cell crosses.
        """
        signals: dict = {
            k: int(self.detected[k]) for k in CHILD_COUNTERS if self.detected.get(k)
        }
        for k in CHILD_FLAGS:
            if k in self.detected:
                signals[k] = bool(self.detected[k])
        return signals

    def merge_signals(self, signals: dict) -> None:
        """Fold a child's signals (from :meth:`child_signals`) into this plan.

        Counters add; flags AND, so a single unreproducible piece is not hidden
        by the pieces around it.
        """
        for key, value in signals.items():
            if key in CHILD_FLAGS:
                current = self.detected.get(key, True)
                self.detected[key] = bool(current) and bool(value)
            else:
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
