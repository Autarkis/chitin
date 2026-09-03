# Traced CoACD Build Contract

Pinned build producing the instrumented `lib_coacd.dll` used by the f32 predicate gate.

## Upstream

| Field | Value |
|-------|-------|
| Repository | `mhamber/CoACD` |
| Tag | `v1.0.14` |
| Commit | `1401ce2` |

## Instrumentation patches

Applied in order on a clean checkout of the upstream commit above.

| Order | Patch | SHA-256 |
|-------|-------|---------|
| 1 | `0001-trace-concatenated-stream-trace-hook-with-ZIP64-flus.patch` | `c99e7e55d57787746e446f6cfdff2fc75a0906a8d066ec9d00979a538f195492` |
| 2 | `0002-trace-v2-instrumentation-hooks-in-clip-cost-mcts-pro.patch` | `7b25666c79d62b6c287f9f0d8ab83302c27814d90cfc7f82ca1afaf8f57b00bc` |

Corresponding branch in the local clone: `chitin/traced-v2` (HEAD `85f97fc`).

## Toolchain

| Component | Version |
|-----------|---------|
| CMake | 4.0.3 |
| MSVC | 14.51.36247 (Visual Studio 17 2022) |
| Generator | Visual Studio 17 2022 |
| Platform | x64, Release |
| zlib | vendored (linked for crc32 in trace hook) |

## Build flags

```
cmake -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config Release
```

## Deployed artifact

| Field | Value |
|-------|-------|
| Path | `<conda>/Lib/site-packages/coacd/lib_coacd.dll` |
| SHA-256 | `dd295d37ad6579545f1017c7125bfe8daab65b52a9ff1853a104b8a2851853d3` |

## Trace format

| Field | Value |
|-------|-------|
| Stream version | v3 (concatenated arrays + int64 offset tables) |
| Container | `.npz` (ZIP64) |
| Sidecar | `meta.json`, `clips.json`, `planes.json`, `mcts.json`, `costs.json` |
| Oracle data | Per-clip `oracle_sides` (int8 vertex sign from C++ `Side()`) |

## Reproducibility

```bash
tools/build-traced-coacd.sh
```

Clones upstream at pinned commit, applies patches, builds, and verifies DLL hash.
