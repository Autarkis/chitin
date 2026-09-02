"""Verify CompilationError semantics."""

from chitin.errors import CompilationError


def test_compilation_error_is_exception():
    err = CompilationError(
        code="COACD_TIMEOUT",
        evidence={"timeout_seconds": 0.001},
        message="timed out",
    )
    assert isinstance(err, Exception)
    assert err.code == "COACD_TIMEOUT"
    assert err.evidence["timeout_seconds"] == 0.001
    assert "timed out" in str(err)


def test_compilation_error_default_evidence():
    err = CompilationError(code="TEST", message="test")
    assert err.evidence == {}
