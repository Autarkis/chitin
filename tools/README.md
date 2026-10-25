# tools/

This directory holds build tooling for producing a trace-instrumented
build of CoACD, used to capture the corpus in `tests/fixtures/traces/`.

## Traced DLL

`src/chitin/coacd_trace.py` and `src/chitin/coacd_trace_replay.py` load a
shared library (`.dll` / `.so` / `.dylib`) built from a patched CoACD that
emits internal geometric-predicate decision traces (see
`tests/fixtures/traces/README.md` for what the traces contain and how the
gate uses them).

## Not committed to git

Build output directories (e.g. `tools/coacd-traced/`, `tools/coacd-build/`)
and any compiled shared library under `tools/` (`*.dll`, `*.so`, `*.dylib`)
are **not committed** — a full traced build is ~60K+ files and ~1.15 GiB.
These paths are listed in `.gitignore`.

## Rebuilding

The traced DLL is built from:

1. Upstream CoACD (https://github.com/SarahWeiii/CoACD)
2. A minimal instrumentation patch adding trace hooks to the predicate
   decision points CoACD's decomposition passes through.

Only the instrumentation patch is stored in this repo (not the full CoACD
source tree or any build output): `tools/coacd-v2-instrumentation.patch`.
It applies on top of upstream CoACD and adds the trace hooks at the predicate
decision points. The patch as captured also carries two bits of noise that are
not part of the instrumentation: a `-dirty` submodule pointer for `3rd/cdt`
and a formatter pass over `python/package/__init__.py`; both can be dropped
when re-applying.

Still to record here: the exact upstream CoACD commit/tag the patch applies
against and the build steps that produce the traced shared library. Tracked
under chitin#108.
