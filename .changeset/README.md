# Changesets

This folder is managed by [@changesets/cli](https://github.com/changesets/changesets).

To add a changeset, run `npx changeset` from the repo root and follow the prompts.
Each changeset declares which packages are affected and the semver bump level
(patch / minor / major), plus a short changelog fragment.

On merge to `main`, the release workflow consumes these fragments to version-bump
`package.json`, assemble per-package `CHANGELOG.md` files, and publish to npm.
