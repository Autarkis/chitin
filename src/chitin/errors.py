from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CompilationError(Exception):
    """A compilation failure that prevents producing a valid collider."""

    code: str
    evidence: dict = field(default_factory=dict)
    message: str = ""

    def __str__(self) -> str:
        return f"CompilationError({self.code}): {self.message}"
