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
import { dirname, join, resolve } from "node:path";
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
const coacdPackage = join(repo, "integrations", "coacd-wasm");
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

// Build the exact JS that will enter the tarballs. The CoACD binary itself is
// produced by integrations/wasm/build.sh and is deliberately not rebuilt here.
ensureBuildDependencies(wasmLite);
ensureBuildDependencies(web);
runNpm(["run", "build"], wasmLite);
runNpm(["run", "build"], web);

const coacdMjs = join(wasmDist, "coacd.mjs");
const coacdWasm = join(wasmDist, "coacd.wasm");
requireFile(coacdMjs, "run integrations/wasm/build.sh first");
requireFile(coacdWasm, "run integrations/wasm/build.sh first");

// The release package intentionally keeps generated binaries out of git. Stage
// them exactly as the release workflow does, then let its prepack check inspect
// them while creating the local tarball.
copyFileSync(coacdMjs, join(coacdPackage, "coacd.mjs"));
copyFileSync(coacdWasm, join(coacdPackage, "coacd.wasm"));

rmSync(candidate, { recursive: true, force: true });
mkdirSync(tarballs, { recursive: true });
for (const packageDir of [wasmLite, web, coacdPackage]) {
  runNpm(["pack", "--pack-destination", tarballs], packageDir);
}

const packed = readdirSync(tarballs)
  .filter((name) => name.endsWith(".tgz"))
  .map((name) => join(tarballs, name));
if (packed.length !== 3) {
  throw new Error(`expected 3 packed Chitin packages, found ${packed.length}`);
}

// walktest is the isolated consumer application. --no-save and
// --package-lock=false ensure preparing a candidate never rewrites its manifest
// or lockfile; imports below resolve only from the installed tarball contents.
runNpm(
  ["install", "--no-save", "--package-lock=false", "--ignore-scripts", ...packed],
  walktest,
);

rmSync(publicPackages, { recursive: true, force: true });
const installedScope = join(walktest, "node_modules", "@autarkis");
const installedLite = join(installedScope, "chitin-lite");
const installedWasm = join(installedScope, "chitin-coacd-wasm");

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
  "chitin-coacd-wasm",
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
  `${JSON.stringify({ packages: versions }, null, 2)}\n`,
);

console.log(`release candidate staged from ${packed.length} tarballs`);
for (const [name, version] of Object.entries(versions)) {
  console.log(`  ${name}@${version}`);
}
