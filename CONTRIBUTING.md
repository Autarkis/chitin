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
uv sync --locked --extra dev       # installs the exact CI dependency set
uv run pre-commit install           # install the git hook (do this once)
uv run pytest                       # full suite
uv run ruff check .                 # lint (the enforced linter)
uv run ruff format .                # format
```

CI pins `uv` and restores its content-addressed package cache using `uv.lock`.
When dependencies change, update and commit the lockfile with `uv lock`; CI uses
`uv sync --locked --extra dev`, so an out-of-date lockfile fails instead of
silently resolving a different environment. A pip editable install remains
supported for development, but it does not reproduce CI's exact resolution.

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

## CI checks

The required `gate` job summarizes the jobs in `.github/workflows/ci.yml` and is
the stable branch-protection target. Pull requests run the expensive Python and
native WASM paths only when their inputs change; pushes to `main` and manual
runs always exercise both paths. Changes to the CI workflow itself exercise
everything.

Python validation runs on Linux, macOS, and Windows in two parallel shards:

```bash
pytest --ignore=tests/test_point_cloud.py  # core shard
pytest tests/test_point_cloud.py           # Open3D/Poisson shard
```

Both shards sync the same locked `dev` environment so a missing optional native
dependency cannot turn tests into skips. The `uv` package cache avoids resolving,
redownloading, and repeatedly unpacking packages through pip on every disposable
runner. CI reports the slowest tests with `--durations=25`; keep real native-path
coverage, but do not repeat an identical extraction merely to assert another part
of its result. Shared expensive test results must be copied before each consumer
so tests cannot leak mutations.

The browser gate builds and functionally tests the native CoACD and Poisson WASM
modules once. An exact cache key covers the Emscripten version, native dependency
reference, build script, and binding sources; a cache hit also skips Emscripten
setup. The build job packs and validates the release candidate, builds the
browser harness and Collider Lab once, and uploads those ready-to-test outputs
as one artifact. Separate `chrome`, `firefox`, and `safari` Playwright jobs only
install their test dependencies and browser before consuming it. Those project
names map to Playwright's `chromium`, `firefox`, and `webkit` browser installers
respectively. Each browser job keeps one Playwright worker because CoACD is CPU-
and memory-intensive; parallelism is provided by the job matrix instead.

When changing path filters, keep them conservative: a false positive costs a CI
run, while a false negative can merge untested code. Generated Python contract
inputs and generators belong to the Python path even when the changed file is
JSON or JavaScript.

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
  Dependabot and GitHub Advanced Security PRs are exempt because their titles
  are generated by GitHub.
- All PRs require the `gate` CI job to pass before merge.
- External contributions require one approving review from a
  [code owner](.github/CODEOWNERS) before merge.

By contributing, you agree your work is licensed under the repository's MIT License.
