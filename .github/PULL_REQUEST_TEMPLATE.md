<!-- Thanks for contributing to Chitin! -->

## What & why

<!-- What does this change, and why? Link any related issue (e.g. Closes #123). -->

## Component(s)

<!-- Check all that apply -->
- [ ] Python compiler / service
- [ ] `.phys` format / spec
- [ ] Python `.phys` reader / validator
- [ ] `@autarkis/chitin-web` (TypeScript)
- [ ] `@autarkis/chitin-lite` (WASM)
- [ ] Unity / Unreal
- [ ] Docs

## Checklist

- [ ] Tests added/updated and passing (`pytest`; `npm test` for web changes)
- [ ] `ruff check .` and `ruff format --check .` pass
- [ ] If reconstruction/spatial code changed: ran the full suite with `open3d`
      installed (`pip install -e ".[dev]"`) so the point-cloud path isn't skipped
- [ ] If the `.phys` format changed: updated **all** readers (Python, TS, Unity,
      Unreal) and regenerated the conformance corpus
- [ ] Docs updated if behavior/CLI/API changed
