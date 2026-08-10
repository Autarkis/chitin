import RAPIER from "@dimforge/rapier3d-compat";
import {
  ChitinError,
  ChitinCompiler,
  validateCompilationReport,
  type CompilationStage,
} from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import { addToWorld } from "@autarkis/chitin-web/rapier";

import { buildMinimalGltf, packGlb } from "../../../scripts/glb-pack.mjs";

interface CandidateManifest {
  packages: Record<string, string>;
}

interface ReleaseCandidateResult {
  packages: Record<string, string>;
  hashes: [string, string];
  deterministic: boolean;
  cancelledWith: string;
  recovered: boolean;
  inputPreserved: boolean;
  stages: CompilationStage[];
  physVersion: number;
  physBytes: number;
  hullCount: number;
  rapierColliderCount: number;
  fallingBodyY: number;
  reportVersion: number;
  verdictStatus: string;
  reportHash: string | null;
  reportDeterministic: boolean | null;
  reportProblems: string[];
}

interface ReleaseCandidateAPI {
  ready: boolean;
  run(): Promise<ReleaseCandidateResult>;
}

declare global {
  interface Window {
    __chitinReleaseCandidate: ReleaseCandidateAPI;
  }
}

const status = document.getElementById("status")!;

// Concave L-prism from the native/WASM wrapper gate. A healthy decomposition
// produces multiple hulls, so this catches a packaged pipeline that merely
// returns one bounding box or an empty result.
function lPrism(): { vertices: Float64Array; faces: Int32Array } {
  return {
    vertices: new Float64Array([
      -0.48, -0.58, -0.25, 0.72, -0.58, -0.25, 0.72, 0.02, -0.25,
      0.12, 0.02, -0.25, 0.12, 0.82, -0.25, -0.48, 0.82, -0.25,
      -0.48, -0.58, 0.25, 0.72, -0.58, 0.25, 0.72, 0.02, 0.25,
      0.12, 0.02, 0.25, 0.12, 0.82, 0.25, -0.48, 0.82, 0.25,
    ]),
    faces: new Int32Array([
      2, 1, 0, 5, 4, 3, 3, 2, 0, 0, 5, 3, 6, 7, 8, 9, 10, 11,
      6, 8, 9, 9, 11, 6, 7, 6, 1, 1, 6, 0, 8, 7, 2, 2, 7, 1,
      9, 8, 3, 3, 8, 2, 10, 9, 4, 4, 9, 3, 6, 11, 0, 0, 11, 5,
      11, 10, 5, 5, 10, 4,
    ]),
  };
}

function lPrismGlb(): ArrayBuffer {
  const mesh = lPrism();
  const positions = Float32Array.from(mesh.vertices);
  const indices = Uint16Array.from(mesh.faces);
  const { document, binary } = buildMinimalGltf(positions, indices, {
    generator: "chitin release-candidate gate",
    bounds: false,
  });
  return packGlb(document, binary);
}

async function sha256(buffer: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function run(): Promise<ReleaseCandidateResult> {
  const candidateResponse = await fetch("./dist/packages/candidate.json");
  if (!candidateResponse.ok) {
    throw new Error(`candidate manifest failed to load (${candidateResponse.status})`);
  }
  const candidate = (await candidateResponse.json()) as CandidateManifest;

  const stages: CompilationStage[] = [];
  const compiler = new ChitinCompiler({
    wasm: {
      js: new URL("./dist/packages/coacd/coacd.mjs", location.href).href,
      wasm: new URL("./dist/packages/coacd/coacd.wasm", location.href).href,
      version: candidate.packages["@autarkis/chitin-wasm"],
    },
    workerUrl: new URL("./dist/packages/chitin-lite/worker.js", location.href),
  });

  let cancelledWith = "";
  try {
    const controller = new AbortController();
    const pending = compiler.compileGlb(lPrismGlb(), {
      signal: controller.signal,
      onProgress: ({ stage }) => {
        if (stage === "decomposing") controller.abort();
      },
    });
    await pending;
    throw new Error("cancelled decomposition unexpectedly resolved");
  } catch (error) {
    if (!(error instanceof ChitinError)) throw error;
    cancelledWith = error.code;
  }

  try {
    const firstInput = lPrismGlb();
    const firstBytes = firstInput.byteLength;
    const first = await compiler.compileGlb(firstInput, {
      onProgress: ({ stage }) => stages.push(stage),
    });
    const inputPreserved = firstInput.byteLength === firstBytes;
    const second = await compiler.compileGlb(lPrismGlb());
    const hashes = [await sha256(first.phys), await sha256(second.phys)] as [
      string,
      string,
    ];

    const phys = parsePhys(first.phys);
    const compilationReport = first.report;
    await RAPIER.init();
    const world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    const fixedBody = addToWorld(RAPIER, world, phys);

    const fallingBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic().setTranslation(-0.2, 2, 0),
    );
    world.createCollider(RAPIER.ColliderDesc.ball(0.1), fallingBody);
    for (let i = 0; i < 240; i++) world.step();
    const fallingBodyY = fallingBody.translation().y;

    const result: ReleaseCandidateResult = {
      packages: candidate.packages,
      hashes,
      deterministic: hashes[0] === hashes[1],
      cancelledWith,
      recovered: first.hulls.length >= 2,
      inputPreserved,
      stages,
      physVersion: phys.version,
      physBytes: first.phys.byteLength,
      hullCount: phys.hulls.length,
      rapierColliderCount: fixedBody.numColliders(),
      fallingBodyY,
      reportVersion: compilationReport.report_version,
      verdictStatus: compilationReport.verdict.status,
      reportHash: compilationReport.reproducibility.artifact_sha256,
      reportDeterministic: compilationReport.reproducibility.deterministic,
      reportProblems: validateCompilationReport(compilationReport),
    };
    status.textContent = JSON.stringify(result, null, 2);
    return result;
  } finally {
    compiler.terminate();
  }
}

window.__chitinReleaseCandidate = { ready: true, run };
status.textContent = "release candidate ready";
