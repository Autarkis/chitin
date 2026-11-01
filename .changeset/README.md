# Changesets

This folder is managed by [@changesets/cli](https://github.com/changesets/changesets).

To add a changeset, run `npx changeset` from the repo root and follow the prompts.
Each changeset declares which packages are affected and the semver bump level
(patch / minor / major), plus a short changelog fragment.

On push to `main`, the release workflow creates/updates a version PR; after
versioning it can publish lite/web. WASM is excluded by the current publish
script and has a separate tag workflow. Versions remain independent.

A successful release-PR update does not imply npm publication. Verify the
exact package artifacts and published versions before promoting a release.
