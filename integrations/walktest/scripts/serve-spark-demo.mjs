import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const walktest = resolve(here, "..");
const serveEntry = resolve(walktest, "node_modules", "serve", "build", "main.js");
const demoUrl = "http://127.0.0.1:3219/spark-demo";
const expectedTitle = "<title>Chitin × Spark — Hull Shadows on Splats</title>";

if (process.argv.includes("--help")) {
  console.log("serve-spark-demo.mjs");
  console.log(`Serves the built Spark demo at ${demoUrl} and opens it in the default browser.`);
  process.exit(0);
}

async function demoIsReady() {
  try {
    const response = await fetch(demoUrl, { signal: AbortSignal.timeout(2_000) });
    return response.ok && (await response.text()).includes(expectedTitle);
  } catch {
    return false;
  }
}

function openDemo() {
  let command;
  let args;
  if (process.platform === "win32") {
    command = "powershell.exe";
    args = ["-NoProfile", "-Command", `Start-Process '${demoUrl}'`];
  } else if (process.platform === "darwin") {
    command = "open";
    args = [demoUrl];
  } else {
    command = "xdg-open";
    args = [demoUrl];
  }

  const opener = spawn(command, args, {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  opener.on("error", () => {
    console.warn(`Open ${demoUrl} in a browser.`);
  });
  opener.unref();
}

if (await demoIsReady()) {
  console.log("Chitin Spark demo is already running. Opening it now...");
  openDemo();
  process.exit(0);
}

if (!existsSync(serveEntry)) {
  throw new Error("serve is not installed; run npm ci in integrations/walktest");
}

const server = spawn(
  process.execPath,
  [
    serveEntry,
    "harness",
    "--listen",
    "tcp://127.0.0.1:3219",
    "--no-port-switching",
    "--no-clipboard",
  ],
  { cwd: walktest, stdio: "inherit" },
);

let serverClosed = false;
let serverExitCode = 0;
server.once("exit", (code) => {
  serverClosed = true;
  serverExitCode = code ?? 1;
});

for (let attempt = 0; attempt < 100 && !serverClosed; attempt += 1) {
  if (await demoIsReady()) {
    console.log(`Opening ${demoUrl}`);
    openDemo();
    break;
  }
  await new Promise((resolveDelay) => setTimeout(resolveDelay, 100));
}

if (!serverClosed && !(await demoIsReady())) {
  server.kill();
  throw new Error(`Spark demo did not become ready at ${demoUrl}`);
}

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => server.kill(signal));
}

if (!serverClosed) {
  serverExitCode = await new Promise((resolveExit) => {
    server.once("exit", (code) => resolveExit(code ?? 1));
  });
}
process.exitCode = serverExitCode;
