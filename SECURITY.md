# Security Policy

## Supported versions

Chitin is pre-1.0 and iterating quickly. Security fixes land on `main` and in the
latest release. Please test against the most recent version before reporting.

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Chitin parses untrusted, externally-produced 3D assets (meshes, point clouds,
gaussian splats) and binary `.phys` sidecars, so parser and decoder robustness is
a real concern. If you find a way to make the compiler or any `.phys` reader
crash, hang, read out of bounds, exhaust memory, or otherwise misbehave on crafted
input, report it privately:

- Use GitHub's **[Report a vulnerability](https://github.com/Autarkis/chitin/security/advisories/new)**
  (Security → Advisories), or
- Email **autarkis@gmail.com** with steps to reproduce and, if possible, a minimal
  offending asset or `.phys` file.

Please give us a reasonable window to fix and release before public disclosure.

## Scope

In scope: the Python compiler and service, the `.phys` readers/validators (Python,
TypeScript, C#, C++), and the WASM compiler. The `.phys` validator is expected to
**reject** malformed input (bad magic, unknown versions/flags, out-of-range
offsets, non-contiguous ranges, non-finite AABBs/bind transforms) rather than
crash or produce garbage — a case where it does not is a valid report.

Out of scope: vulnerabilities in third-party dependencies (report those upstream)
and in assets you author yourself.
