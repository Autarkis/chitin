# Contributing to Chitin

Thanks for your interest in Chitin! It's an MIT-licensed physics asset compiler,
and contributions of all kinds are welcome — bug reports, fixes, docs, new engine
readers, and format tooling.

## Repository layout

Chitin is one repo spanning several languages. Know which part you're touching:

| Path | Language | What it is |
|------|----------|------------|
| `src/chitin/` | Python | The compiler: mesh/scan → convex hulls → `.phys` |
| `src/chitin_service/` | Python | Local build service (FastAPI) |
| `integrations/web/` | TypeScript | `@autarkis/chitin-web` — `.phys` reader + Rapier/Three bindings |
| `integrations/wasm/` + `wasm-lite/` | C++/WASM + TS | Browser CoACD compiler (`@autarkis/chitin-lite`) |
| `integrations/unity/` | C# | Unity importer |
| `integrations/unreal/` | C++ | Unreal importer |
| `tests/conformance/` | Python + TS | Cross-runtime `.phys` golden corpus |

## Python development

Requires **Python 3.12** (open3d has no 3.13 wheel yet).

```bash
python -m pip install -e ".[dev]"   # installs open3d, coacd, trimesh, ruff, pytest, ...
pytest                              # full suite
ruff check .                        # lint (the enforced linter)
ruff format .                       # format
```

> **Important:** install the `[dev]` (or `[splat]`) extra so **open3d** is present.
> Without it, the entire point-cloud / Poisson / spatial / CoACD test path **skips
> silently** — the suite still reports green while leaving that code untested. If
> your change touches reconstruction, decomposition, or the spatial path, make sure
> those tests actually run locally (the full suite takes several minutes; the
> spatial thin-shell tests are the slow ones).

We lint and format with **ruff** only. `mypy`/`pyright` are not enforced (the code
uses runtime-guarded `| None` narrowing), so don't add type-checking gates.

## Web / TypeScript development

```bash
cd integrations/web        # or integrations/wasm-lite
npm ci
npm test                   # vitest
npm run build              # tsc
```

## The `.phys` format is versioned

`.phys` is a stable binary contract with readers in Python, TypeScript, C#, and
C++. If you change the format:

1. Update the writer **and every reader** (`src/chitin/phys.py`,
   `integrations/web/src/phys-parser.ts`, Unity, Unreal).
2. Regenerate the cross-runtime corpus and keep both copies in sync:
   ```bash
   PYTHONPATH=src python tests/conformance/build_fixtures.py
   cp tests/conformance/fixtures/*.phys tests/conformance/manifest.json \
      integrations/web/test/conformance/
   ```
3. Run the conformance tests on both sides (`pytest tests/conformance/`,
   `cd integrations/web && npm test`).

## Building the WASM CoACD module (browser compiler only)

Only needed if you work on `@autarkis/chitin-lite`'s decomposition. See
[`integrations/wasm/README.md`](integrations/wasm/README.md): it needs Emscripten
and the CoACD source, then `./build.sh`.

## Pull requests

- Keep the change focused; match the style of the surrounding code.
- Add or update tests. `ruff check`, `ruff format --check`, and the test suites
  must pass (CI runs them on Python + both web packages).
- If you change the `.phys` format, include conformance fixtures and update all
  readers (see above).
- Commit messages: a short imperative subject line (`scope: do the thing`); a body
  only when there's a non-obvious *why*.

By contributing, you agree your work is licensed under the repository's MIT License.
