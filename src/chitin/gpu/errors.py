"""Error types for the GPU compute backend."""

from __future__ import annotations


class DeviceLostError(RuntimeError):
    """Raised when an operation is attempted after the GPU device has been lost."""


class CapacityError(RuntimeError):
    """Raised when a requested allocation exceeds a device limit."""


class ShaderCompilationError(RuntimeError):
    """Raised when a WGSL shader module fails to compile."""
