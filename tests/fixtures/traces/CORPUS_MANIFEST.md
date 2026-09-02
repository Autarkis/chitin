# Trace Corpus Manifest

Stream v3 format. Captured with traced CoACD v1.0.14 (see `tools/BUILD_CONTRACT.md`).

## CI tier (tracked in git)

Replayed in every CI run. Small enough for bare git.

| Fixture | Size | Clips | Parts | arrays.npz SHA-256 |
|---------|------|-------|-------|-------------------|
| box | 3,458 | 0 | 1 | `f2778d3f5ddb58e309bf903667899940b9cbed192b9102791194d889697f125c` |
| icosphere | 125,186 | 0 | 1 | `5ff57d55f916ed43e9c54c359f7bb2bde9545248e426625d26bfc853025e0e87` |
| thin_panel | 3,458 | 0 | 1 | `874302dd2e001fb74d1235ba9302100464a78f246e4d48f831d013a9fddf57c5` |
| l_shape | 3,983,614 | 1,054 | 2 | `48a45262c932e01d278544522f15d45d25b59a9f3bc920f5ddcbbdd0e8f68420` |
| thin_u_channel | 7,303,282 | 2,108 | 3 | `1b1223a253fedc2a86e6c000d6a5d873bf586aab0bbdf04f5175bcae9bb36e40` |
| cross_bracket | 11,659,946 | 2,108 | 3 | `fa2e275d0da0bc8c539b0b772e0795d3818f883af9a91c3b2a810544f174593e` |
| staircase | 21,661,216 | 5,226 | 5 | `9f9be026aecacccb40891d0c60b1874b70fbcdbc9c8d972c4d9300fb4b0760c5` |

Total: ~45 MB, 11,550 clips.

## External tier (not tracked)

Too large for git. Stored as release artifacts or CAS.

| Fixture | Size | Clips | Parts | arrays.npz SHA-256 |
|---------|------|-------|-------|-------------------|
| t_shape | 1,679,745,996 | 1,054 | 2 | `293790274a89a0c7549f6d86394017a2620fa95ccb71dbe7e52a26c85d10b202` |
| curved_pipe_quarter | 2,563,929,826 | 3,162 | 4 | `dce6de15b4b3560df0cb799803e84beaead93b1eafb8f55dc288a8e49c41ef14` |
| h_shape | 13,178,564,920 | 20,954 | 16 | `b42e20807a3cf4fc2b6d8048dfc434e83f13f137c4658951de5471a290fa6972` |

Total: ~17.4 GB, 25,170 clips.

## Holdout protocol

External-tier fixtures are holdout candidates. They must not influence regression
floor tuning. The holdout evaluation runs once, records an immutable result, and
issues the final PASS/FAIL verdict before #108/#101 close.

## Integrity check

CI verifies corpus digests via `tests/conftest.py::verify_corpus_integrity`.
Missing or corrupted CI-tier fixtures fail the gate when `CHITIN_GATE_FINAL=1`.
