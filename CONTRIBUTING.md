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
pre-commit install                  # install the git hook (do this once)
pytest                              # full suite
ruff check .                        # lint (the enforced linter)
ruff format .                       # format
```

`pre-commit install` wires `ruff check --fix`, `ruff format`, and a few
whitespace/YAML hygiene hooks into `git commit`, so the two cheap CI gates fail
locally in a second instead of on a pushed branch. If a hook rewrites a file, the
commit is aborted — re-`git add` the fixed file and commit again. To check the
whole tree at once: `pre-commit run --all-files`.

The ruff version is pinned **exactly** in the `dev` extra and pinned again as the
`rev` in `.pre-commit-config.yaml`; the two must stay equal. ruff's formatter
output is not stable across minor releases, so if they drift, the hook and CI
disagree and you get a commit that passes locally and fails `ruff format --check`
in CI. `pre-commit autoupdate` bumps only the config — update `pyproject.toml` to
match in the same commit.

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

The pre-commit hook runs both of those (via `scripts/npm_gate.py`) for whichever
package a commit touches — a change under `integrations/web/` runs only the web
suite, and a Python-only commit runs neither, so you don't pay for Node unless
you edited TypeScript. It needs `npm ci` to have been run in that package; if
`node_modules` is missing the hook tells you which directory to run it in rather
than skipping the check.

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
- PRs are **squash-merged**. The PR title becomes the merge commit subject, so
  use the same `scope: subject` format — the CI conventions check enforces it.
- All PRs require the `gate` CI job to pass before merge.
- External contributions require one approving review from a
  [code owner](.github/CODEOWNERS) before merge.

By contributing, you agree your work is licensed under the repository's MIT License.
