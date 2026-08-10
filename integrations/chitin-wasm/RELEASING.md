# Releasing `@autarkis/chitin-wasm`

npm trusted publishing can only be configured after the package exists. Bootstrap
one prerelease with token authentication, then configure OIDC for later releases
via `release-wasm.yml` on a `wasm-v*` tag.

## One-time bootstrap (before `wasm-v0.2.0`)

Build the artifacts (needs Emscripten) and stage them into the package. The
`prepack` guard refuses to pack without them.

```bash
( cd integrations/wasm && bash build.sh )
cp integrations/wasm/dist/{coacd,poisson}.{mjs,wasm} integrations/chitin-wasm/
```

Publish a prerelease under a non-`latest` dist-tag with your npm token, then
revert the temporary version bump and remove the staged artifacts.

```bash
cd integrations/chitin-wasm
npm version 0.2.0-bootstrap.0 --no-git-tag-version
npm publish --access public --tag bootstrap
git checkout package.json
rm -f coacd.mjs coacd.wasm
```

On npmjs.com, open the package → Settings → Trusted Publisher and add repository
`Autarkis/chitin`, workflow `release-wasm.yml`, environment `npm-publish`.

In GitHub → Settings → Environments, create the `npm-publish` environment and add
protection rules. The `environment:` key in the workflow does not make it
protected on its own.

## Every release after the bootstrap

Bump `version` in `package.json`, commit, then tag `wasm-v<version>` and push it.
`release-wasm.yml` checks the tag against `package.json`, builds, runs the wrapper
gate, attaches the assets to the release, and publishes to npm via OIDC.

The `0.2.0-bootstrap.0` prerelease can stay (it is not `latest`) or be hidden with
`npm deprecate`.
