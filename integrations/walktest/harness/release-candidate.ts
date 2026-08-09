import RAPIER from "@dimforge/rapier3d-compat";
import { SparkRenderer, SpzWriter } from "@sparkjsdev/spark";
import * as THREE from "three";
import {
  ChitinError,
  ChitinCompiler,
  validateCompilationReport,
  type CompilationStage,
} from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import { addToWorld } from "@autarkis/chitin-web/rapier";
import { loadSparkAsset } from "@autarkis/chitin-web/spark";
import { physTransformFromObject3D } from "@autarkis/chitin-web/three";

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
  sparkSplatCount: number;
  sparkRenderedPixels: number;
  sparkShadowMeshCount: number;
  sparkShadowChangedPixels: number;
  sparkBoundsCenterError: number;
  sparkBodyX: number;
  reportVersion: number;
  verdictStatus: string;
  reportHash: string | null;
  reportDeterministic: boolean | null;
  reportProblems: string[];
  exportsResolved: string[];
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
      -0.48, -0.58, -0.25, 0.72, -0.58, -0.25, 0.72, 0.02, -0.25, 0.12, 0.02,
      -0.25, 0.12, 0.82, -0.25, -0.48, 0.82, -0.25, -0.48, -0.58, 0.25, 0.72,
      -0.58, 0.25, 0.72, 0.02, 0.25, 0.12, 0.02, 0.25, 0.12, 0.82, 0.25, -0.48,
      0.82, 0.25,
    ]),
    faces: new Int32Array([
      2, 1, 0, 5, 4, 3, 3, 2, 0, 0, 5, 3, 6, 7, 8, 9, 10, 11, 6, 8, 9, 9, 11, 6,
      7, 6, 1, 1, 6, 0, 8, 7, 2, 2, 7, 1, 9, 8, 3, 3, 8, 2, 10, 9, 4, 4, 9, 3,
      6, 11, 0, 0, 11, 5, 11, 10, 5, 5, 10, 4,
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

async function lPrismSpz(): Promise<Uint8Array> {
  const vertices = lPrism().vertices;
  const count = vertices.length / 3;
  const writer = new SpzWriter({ numSplats: count, shDegree: 0 });
  for (let i = 0; i < count; i++) {
    writer.setCenter(
      i,
      vertices[i * 3],
      vertices[i * 3 + 1],
      vertices[i * 3 + 2],
    );
    writer.setAlpha(i, 1);
    writer.setRgb(i, 0.2 + (i % 3) * 0.25, 0.55, 0.85);
    writer.setScale(i, 0.16, 0.16, 0.16);
    writer.setQuat(i, 0, 0, 0, 1);
  }
  return writer.finalize();
}

interface SparkProbeResult {
  splatCount: number;
  renderedPixels: number;
  shadowMeshCount: number;
  shadowChangedPixels: number;
  boundsCenterError: number;
  bodyX: number;
}

async function probeSparkSidecar(
  phys: ReturnType<typeof parsePhys>,
  world: RAPIER.World,
): Promise<SparkProbeResult> {
  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setSize(160, 160, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  const target = new THREE.WebGLRenderTarget(160, 160, {
    depthBuffer: true,
  });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x202020);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 30);
  camera.position.set(3.2, 2.4, 4.2);
  camera.lookAt(0, 0, 0);

  const sparkRenderer = new SparkRenderer({ renderer });
  scene.add(sparkRenderer);
  const asset = await loadSparkAsset({
    visual: {
      fileBytes: await lPrismSpz(),
      fileName: "l-prism.spz",
    },
    phys,
    shadows: { mode: "merged-hulls" },
  });
  asset.root.position.x = 0.25;
  scene.add(asset.root);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(7, 7),
    new THREE.MeshStandardMaterial({ color: 0xf0f0f0, roughness: 1 }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.72;
  floor.receiveShadow = true;
  scene.add(floor);

  const light = new THREE.DirectionalLight(0xffffff, 2.5);
  light.position.set(-3, 5, 4);
  light.castShadow = true;
  light.shadow.mapSize.set(512, 512);
  light.shadow.camera.left = -4;
  light.shadow.camera.right = 4;
  light.shadow.camera.top = 4;
  light.shadow.camera.bottom = -4;
  light.shadow.camera.near = 0.1;
  light.shadow.camera.far = 15;
  scene.add(light, light.target, new THREE.AmbientLight(0xffffff, 0.2));

  const readPixels = (): Uint8Array => {
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const pixels = new Uint8Array(160 * 160 * 4);
    renderer.readRenderTargetPixels(target, 0, 0, 160, 160, pixels);
    return pixels;
  };

  // Prove the decoded Spark splats participate in the actual browser render.
  floor.visible = false;
  if (asset.shadowProxy) asset.shadowProxy.visible = false;
  scene.remove(asset.root);
  await sparkRenderer.update({ scene, camera });
  const emptyPixels = readPixels();
  scene.add(asset.root);
  await sparkRenderer.update({ scene, camera });
  const splatPixels = readPixels();
  let renderedPixels = 0;
  for (let i = 0; i < splatPixels.length; i += 4) {
    const difference =
      Math.abs(splatPixels[i] - emptyPixels[i]) +
      Math.abs(splatPixels[i + 1] - emptyPixels[i + 1]) +
      Math.abs(splatPixels[i + 2] - emptyPixels[i + 2]);
    if (difference > 12) renderedPixels++;
  }

  // Compare otherwise identical frames with the hull caster disabled/enabled.
  floor.visible = true;
  if (asset.shadowProxy) asset.shadowProxy.visible = true;
  asset.visual.visible = false;
  asset.shadowProxy?.traverse((object) => {
    if (object instanceof THREE.Mesh) object.castShadow = false;
  });
  const unshadowed = readPixels();
  asset.shadowProxy?.traverse((object) => {
    if (object instanceof THREE.Mesh) object.castShadow = true;
  });
  light.shadow.needsUpdate = true;
  const shadowed = readPixels();
  let shadowChangedPixels = 0;
  for (let i = 0; i < shadowed.length; i += 4) {
    const difference =
      Math.abs(shadowed[i] - unshadowed[i]) +
      Math.abs(shadowed[i + 1] - unshadowed[i + 1]) +
      Math.abs(shadowed[i + 2] - unshadowed[i + 2]);
    if (difference > 12) shadowChangedPixels++;
  }

  const visualCenter = asset.bounds.visual.getCenter(new THREE.Vector3());
  const physicsCenter = new THREE.Vector3();
  if (asset.bounds.physics) {
    physicsCenter
      .fromArray(asset.bounds.physics.min)
      .add(new THREE.Vector3().fromArray(asset.bounds.physics.max))
      .multiplyScalar(0.5);
  }
  const body = addToWorld(RAPIER, world, asset.phys, undefined, {
    transform: physTransformFromObject3D(asset.root),
  });
  const result: SparkProbeResult = {
    splatCount:
      asset.visual.splats?.getNumSplats() ??
      asset.visual.packedSplats?.getNumSplats() ??
      0,
    renderedPixels,
    shadowMeshCount:
      asset.shadowProxy?.children.filter((child) => child instanceof THREE.Mesh)
        .length ?? 0,
    shadowChangedPixels,
    boundsCenterError: visualCenter.distanceTo(physicsCenter),
    bodyX: body.translation().x,
  };

  world.removeRigidBody(body);
  asset.dispose();
  sparkRenderer.dispose();
  floor.geometry.dispose();
  (floor.material as THREE.Material).dispose();
  target.dispose();
  renderer.dispose();
  return result;
}

async function run(): Promise<ReleaseCandidateResult> {
  const candidateResponse = await fetch("./dist/packages/candidate.json");
  if (!candidateResponse.ok) {
    throw new Error(
      `candidate manifest failed to load (${candidateResponse.status})`,
    );
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
    const spark = await probeSparkSidecar(phys, world);

    const exportsResolved: string[] = [];
    if (typeof ChitinCompiler === "function") exportsResolved.push("@autarkis/chitin-lite");
    if (typeof parsePhys === "function") exportsResolved.push("@autarkis/chitin-web");
    if (typeof addToWorld === "function") exportsResolved.push("@autarkis/chitin-web/rapier");

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
      sparkSplatCount: spark.splatCount,
      sparkRenderedPixels: spark.renderedPixels,
      sparkShadowMeshCount: spark.shadowMeshCount,
      sparkShadowChangedPixels: spark.shadowChangedPixels,
      sparkBoundsCenterError: spark.boundsCenterError,
      sparkBodyX: spark.bodyX,
      reportVersion: compilationReport.report_version,
      verdictStatus: compilationReport.verdict.status,
      reportHash: compilationReport.reproducibility.artifact_sha256,
      reportDeterministic: compilationReport.reproducibility.deterministic,
      reportProblems: validateCompilationReport(compilationReport),
      exportsResolved,
    };
    status.textContent = JSON.stringify(result, null, 2);
    return result;
  } finally {
    compiler.terminate();
  }
}

window.__chitinReleaseCandidate = { ready: true, run };
status.textContent = "release candidate ready";
