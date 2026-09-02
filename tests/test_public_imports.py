"""Every name in chitin.__all__ must resolve at import time."""

import chitin


def test_all_exports_resolve():
    for name in chitin.__all__:
        assert hasattr(chitin, name), f"{name} in __all__ but not importable"
