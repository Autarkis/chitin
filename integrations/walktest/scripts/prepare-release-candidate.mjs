import {
  copyFileSync,
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const here = dirname(fileURLToPath(import.meta.url));
const walktest = resolve(here, "..");
const repo = resolve(walktest, "../..");
const npmCli = process.env.npm_execpath;
if (!npmCli) {
  throw new Error("npm_execpath is unavailable; run this script through npm run");
}

const wasmLite = join(repo, "integrations", "wasm-lite");
const web = join(repo, "integrations", "web");
const wasmPackage = join(repo, "integrations", "chitin-wasm");
const wasmDist = join(repo, "integrations", "wasm", "dist");
const candidate = join(walktest, ".release-candidate");
const tarballs = join(candidate, "tarballs");
const publicPackages = join(walktest, "harness", "dist", "packages");

function run(command, args, cwd) {
  const result = spawnSync(command, args, {
    cwd,
    encoding: "utf8",
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} ${args.join(" ")} failed with exit code ${result.status}`);
  }
}

function runNpm(args, cwd) {
  run(process.execPath, [npmCli, ...args], cwd);
}

function requireFile(path, hint) {
  try {
    return readFileSync(path);
  } catch {
    throw new Error(`missing ${path}; ${hint}`);
  }
}

function ensureBuildDependencies(packageDir) {
  const tsc = join(
    packageDir,
    "node_modules",
    ".bin",
    process.platform === "win32" ? "tsc.cmd" : "tsc",
  );
  if (!existsSync(tsc)) runNpm(["ci"], packageDir);
}

// Build the exact JS that will enter the tarballs. The native WASM artifacts
// are produced by integrations/wasm/build.sh and are deliberately not rebuilt here.
ensureBuildDependencies(wasmLite);
ensureBuildDependencies(web);
runNpm(["run", "build"], wasmLite);
runNpm(["run", "build"], web);

// The release package intentionally keeps generated binaries out of git. Stage
// them exactly as the release workflow does, then let its prepack check inspect
// them while creating the local tarball.
const nativeArtifacts = ["coacd.mjs", "coacd.wasm", "poisson.mjs", "poisson.wasm"];
for (const artifact of nativeArtifacts) {
  const source = join(wasmDist, artifact);
  requireFile(source, "run integrations/wasm/build.sh first");
  copyFileSync(source, join(wasmPackage, artifact));
}

rmSync(candidate, { recursive: true, force: true });
mkdirSync(tarballs, { recursive: true });
for (const packageDir of [wasmLite, web, wasmPackage]) {
  runNpm(["pack", "--pack-destination", tarballs], packageDir);
}

const packed = readdirSync(tarballs)
  .filter((name) => name.endsWith(".tgz"))
  .map((name) => join(tarballs, name));
if (packed.length !== 3) {
  throw new Error(`expected 3 packed Chitin packages, found ${packed.length}`);
}

// Each tarball must actually contain the build outputs the consumer imports at
// runtime; a silent packaging gap (missing "files" entry, a build that didn't
// run) would otherwise only surface as a runtime import failure downstream.
const requiredTarballFiles = {
  "autarkis-chitin-wasm": [
    "package/coacd.mjs",
    "package/coacd.wasm",
    "package/poisson.mjs",
    "package/poisson.wasm",
  ],
  "autarkis-chitin-lite": [
    "package/dist/index.js",
    "package/dist/index.d.ts",
    "package/dist/worker.js",
  ],
  "autarkis-chitin-web": [
    "package/dist/index.js",
    "package/dist/index.d.ts",
    "package/dist/rapier.js",
    "package/dist/rapier.d.ts",
  ],
};

for (const tarball of packed) {
  const tarballName = basename(tarball);
  const prefix = Object.keys(requiredTarballFiles).find((name) =>
    tarballName.startsWith(name),
  );
  if (!prefix) {
    throw new Error(`tarball ${tarballName} does not match any known Chitin package prefix`);
  }
  const listing = spawnSync("tar", ["tzf", tarball], { encoding: "utf8" });
  if (listing.error) throw listing.error;
  if (listing.status !== 0) {
    throw new Error(`tar tzf ${tarballName} failed with exit code ${listing.status}`);
  }
  const entries = new Set(
    listing.stdout
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean),
  );
  for (const requiredFile of requiredTarballFiles[prefix]) {
    if (!entries.has(requiredFile)) {
      throw new Error(`tarball ${tarballName} missing: ${requiredFile}`);
    }
  }
}

// walktest is the isolated consumer application. --no-save and
// --package-lock=false ensure preparing a candidate never rewrites its manifest
// or lockfile; imports below resolve only from the installed tarball contents.
runNpm(
  ["install", "--no-save", "--package-lock=false", "--ignore-scripts", ...packed],
  walktest,
);

const installedScope = join(walktest, "node_modules", "@autarkis");

// Cross-package exports validation: each installed package's `exports` map
// must resolve to files that actually exist in the tarball contents, so a
// stale or misconfigured `exports` field is caught here rather than as a
// downstream import failure in the consumer app.
function collectExportTargets(value) {
  if (typeof value === "string") return [value];
  if (Array.isArray(value)) return value.flatMap(collectExportTargets);
  if (value && typeof value === "object") {
    return Object.values(value).flatMap(collectExportTargets);
  }
  return [];
}

const exportsCheckedAt = new Date().toISOString();
for (const name of ["chitin-lite", "chitin-web", "chitin-wasm"]) {
  const packageDir = join(installedScope, name);
  const manifest = JSON.parse(readFileSync(join(packageDir, "package.json"), "utf8"));
  if (!manifest.exports) continue;
  for (const [exportKey, exportValue] of Object.entries(manifest.exports)) {
    for (const target of collectExportTargets(exportValue)) {
      const resolvedPath = join(packageDir, target);
      if (!existsSync(resolvedPath)) {
        throw new Error(
          `package @autarkis/${name} export "${exportKey}" points to missing file: ${target}`,
        );
      }
    }
  }
}
const exportsVerified = true;

rmSync(publicPackages, { recursive: true, force: true });
const installedLite = join(installedScope, "chitin-lite");
const installedWasm = join(installedScope, "chitin-wasm");

// A module Worker resolves its relative imports at runtime, so serve the whole
// packaged dist directory instead of copying a source-tree worker entry point.
cpSync(join(installedLite, "dist"), join(publicPackages, "chitin-lite"), {
  recursive: true,
});
mkdirSync(join(publicPackages, "coacd"), { recursive: true });
copyFileSync(join(installedWasm, "coacd.mjs"), join(publicPackages, "coacd", "coacd.mjs"));
copyFileSync(join(installedWasm, "coacd.wasm"), join(publicPackages, "coacd", "coacd.wasm"));

const packageNames = [
  "chitin-lite",
  "chitin-web",
  "chitin-wasm",
];
const versions = Object.fromEntries(
  packageNames.map((name) => {
    const manifest = JSON.parse(
      readFileSync(join(installedScope, name, "package.json"), "utf8"),
    );
    return [manifest.name, manifest.version];
  }),
);
writeFileSync(
  join(publicPackages, "candidate.json"),
  `${JSON.stringify(
    {
      packages: versions,
      compatibility: { exportsVerified, checkedAt: exportsCheckedAt },
    },
    null,
    2,
  )}\n`,
);

console.log(`release candidate staged from ${packed.length} tarballs`);
for (const [name, version] of Object.entries(versions)) {
  console.log(`  ${name}@${version}`);
}
