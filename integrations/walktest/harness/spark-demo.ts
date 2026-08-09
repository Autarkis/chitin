import { ChitinCompiler } from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import { attachSparkDirectionalShadow } from "@autarkis/chitin-web/spark";
import {
  createHullGeometries,
  createShadowProxy,
} from "@autarkis/chitin-web/three";
import { SparkRenderer, SplatMesh, SpzWriter } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { ConvexGeometry } from "three/addons/geometries/ConvexGeometry.js";

import { buildMinimalGltf, packGlb } from "../../../scripts/glb-pack.mjs";

const viewport = document.getElementById("viewport")!;
const status = document.getElementById("status")!;
const strengthInput = document.getElementById("strength") as HTMLInputElement;
const strengthValue = document.getElementById("strength-value")!;
const heightInput = document.getElementById("height") as HTMLInputElement;
const heightValue = document.getElementById("height-value")!;
const animateInput = document.getElementById("animate") as HTMLInputElement;
const hullsInput = document.getElementById("hulls") as HTMLInputElement;
const receiverHullsInput = document.getElementById(
  "receiver-hulls",
) as HTMLInputElement;
const receiverSplatsInput = document.getElementById(
  "receiver-splats",
) as HTMLInputElement;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080b12);
scene.fog = new THREE.FogExp2(0x080b12, 0.035);

const camera = new THREE.PerspectiveCamera(
  42,
  window.innerWidth / window.innerHeight,
  0.05,
  80,
);
camera.position.set(4.8, 3.2, -6.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.45, 0);
controls.enableDamping = true;
controls.minDistance = 3;
controls.maxDistance = 14;
controls.maxPolarAngle = Math.PI * 0.48;

const sparkRenderer = new SparkRenderer({ renderer });
scene.add(sparkRenderer);

const light = new THREE.DirectionalLight(0xfff4df, 4.5);
light.position.set(-3.8, 7, 4.6);
light.target.position.set(0, 0, 0);
light.castShadow = true;
light.shadow.mapSize.set(1024, 1024);
light.shadow.camera.left = -5;
light.shadow.camera.right = 5;
light.shadow.camera.top = 5;
light.shadow.camera.bottom = -5;
light.shadow.camera.near = 0.1;
light.shadow.camera.far = 18;
light.shadow.bias = -0.00015;
scene.add(light, light.target);
scene.add(new THREE.HemisphereLight(0x9ab9ff, 0x111827, 1.4));

const grid = new THREE.GridHelper(12, 24, 0x284057, 0x172535);
grid.position.y = -0.055;
grid.material.transparent = true;
grid.material.opacity = 0.45;
scene.add(grid);

const casterRoot = new THREE.Group();
const hullEdges = new THREE.Group();
casterRoot.add(hullEdges);
scene.add(casterRoot);

const receiverHullPreview = new THREE.Group();
receiverHullPreview.name = "Gaussian-derived receiver hull preview";
scene.add(receiverHullPreview);

let receiverStrength = Number(strengthInput.value);
const casterBaseX = -0.55;

const ROBOT_HEAD_URL = "https://sparkjs.dev/assets/splats/robot-head.spz";

type GaussianSample = {
  center: THREE.Vector3;
  scales: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

type GaussianCluster = {
  seen: number;
  samples: GaussianSample[];
};

function buildReceiverHullPreview(visual: SplatMesh): number {
  const bounds = new THREE.Box3();
  visual.forEachSplat((_index, center, _scales, _quaternion, opacity) => {
    if (opacity >= 0.2) bounds.expandByPoint(center);
  });
  if (bounds.isEmpty()) return 0;

  const divisions = new THREE.Vector3(6, 8, 6);
  const size = bounds.getSize(new THREE.Vector3());
  const diagonal = size.length();
  const clusters = new Map<number, GaussianCluster>();
  const maxSamplesPerCluster = 36;

  visual.forEachSplat((index, center, scales, quaternion, opacity) => {
    if (opacity < 0.2) return;
    const x = Math.min(
      divisions.x - 1,
      Math.floor(
        ((center.x - bounds.min.x) / Math.max(size.x, 1e-6)) * divisions.x,
      ),
    );
    const y = Math.min(
      divisions.y - 1,
      Math.floor(
        ((center.y - bounds.min.y) / Math.max(size.y, 1e-6)) * divisions.y,
      ),
    );
    const z = Math.min(
      divisions.z - 1,
      Math.floor(
        ((center.z - bounds.min.z) / Math.max(size.z, 1e-6)) * divisions.z,
      ),
    );
    const key = x + y * divisions.x + z * divisions.x * divisions.y;
    const cluster = clusters.get(key) ?? { seen: 0, samples: [] };
    cluster.seen++;

    const sample = {
      center: center.clone(),
      scales: scales.clone(),
      quaternion: quaternion.clone(),
    };
    if (cluster.samples.length < maxSamplesPerCluster) {
      cluster.samples.push(sample);
    } else {
      const replacement = ((index * 2654435761) >>> 0) % cluster.seen;
      if (replacement < maxSamplesPerCluster) {
        cluster.samples[replacement] = sample;
      }
    }
    clusters.set(key, cluster);
  });

  const fillMaterial = new THREE.MeshBasicMaterial({
    color: 0xffd166,
    depthWrite: false,
    opacity: 0.055,
    side: THREE.DoubleSide,
    transparent: true,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0xffd166,
    opacity: 0.72,
    transparent: true,
  });
  const minSupport = diagonal * 0.0015;
  const maxSupport = diagonal * 0.045;
  const axis = new THREE.Vector3();
  let hullCount = 0;

  for (const cluster of clusters.values()) {
    if (cluster.samples.length < 4) continue;
    const points: THREE.Vector3[] = [];
    for (const sample of cluster.samples) {
      for (let component = 0; component < 3; component++) {
        axis.set(
          component === 0 ? 1 : 0,
          component === 1 ? 1 : 0,
          component === 2 ? 1 : 0,
        );
        axis.applyQuaternion(sample.quaternion);
        const radius = THREE.MathUtils.clamp(
          sample.scales.getComponent(component) * 1.75,
          minSupport,
          maxSupport,
        );
        axis.multiplyScalar(radius);
        points.push(sample.center.clone().add(axis));
        points.push(sample.center.clone().sub(axis));
      }
    }
    try {
      const geometry = new ConvexGeometry(points);
      receiverHullPreview.add(new THREE.Mesh(geometry, fillMaterial));
      receiverHullPreview.add(
        new THREE.LineSegments(
          new THREE.EdgesGeometry(geometry, 20),
          edgeMaterial,
        ),
      );
      hullCount++;
    } catch {
      // Nearly planar edge clusters do not define a stable 3D convex hull.
    }
  }

  receiverHullPreview.position.copy(visual.position);
  receiverHullPreview.quaternion.copy(visual.quaternion);
  receiverHullPreview.scale.copy(visual.scale);
  receiverHullPreview.visible = receiverHullsInput.checked;
  return hullCount;
}

async function buildReceiverSplats(): Promise<SplatMesh> {
  status.textContent = "Loading Spark’s robot-head example capture…";
  try {
    const visual = new SplatMesh({
      url: ROBOT_HEAD_URL,
      enableLod: false,
    });
    await visual.initialized;
    visual.rotation.x = Math.PI;
    visual.scale.setScalar(1.05);
    visual.position.set(0.15, 0.62, -0.15);
    scene.add(visual);
    const count =
      visual.splats?.getNumSplats() ??
      visual.packedSplats?.getNumSplats() ??
      0;
    document.getElementById("splat-count")!.textContent =
      count.toLocaleString();
    document.getElementById("receiver-name")!.textContent =
      "Spark robot head";
    return visual;
  } catch (error) {
    console.warn(
      "Spark example unavailable; using the local procedural fallback",
      error,
    );
    status.textContent =
      "Spark example unavailable — building the local splat fallback…";
    return buildProceduralReceiverSplats();
  }
}

async function buildProceduralReceiverSplats(): Promise<SplatMesh> {
  const width = 49;
  const spacing = 0.105;
  const count = width * width;
  const writer = new SpzWriter({ numSplats: count, shDegree: 0 });
  let index = 0;
  for (let z = 0; z < width; z++) {
    for (let x = 0; x < width; x++) {
      const grain = ((x * 37 + z * 61 + x * z * 3) % 101) / 100;
      const crossGrain = ((x * 71 + z * 29 + x * z * 7) % 97) / 96;
      const px =
        (x - (width - 1) / 2) * spacing +
        (grain - 0.5) * spacing * 0.24;
      const pz =
        (z - (width - 1) / 2) * spacing +
        (crossGrain - 0.5) * spacing * 0.24;
      const py =
        Math.sin(px * 1.65) * Math.cos(pz * 1.35) * 0.025 +
        (grain - 0.5) * 0.008;
      const radial = Math.hypot(px, pz) / 4;
      const footprint = spacing * (0.27 + grain * 0.09);
      writer.setCenter(index, px, py, pz);
      writer.setAlpha(index, Math.max(0.78, 0.96 - radial * 0.1));
      writer.setRgb(
        index,
        0.38 + grain * 0.18 + (x / width) * 0.06,
        0.55 + crossGrain * 0.16,
        0.8 + grain * 0.12,
      );
      writer.setScale(index, footprint, 0.012, footprint * 0.9);
      writer.setQuat(index, 0, 0, 0, 1);
      index++;
    }
  }
  const visual = new SplatMesh({
    fileBytes: await writer.finalize(),
    fileName: "chitin-shadow-receiver.spz",
    enableLod: false,
  });
  await visual.initialized;
  scene.add(visual);
  document.getElementById("splat-count")!.textContent = count.toLocaleString();
  document.getElementById("receiver-name")!.textContent = "local fallback";
  return visual;
}

function lPrismGlb(): ArrayBuffer {
  const positions = new Float32Array([
    -0.7, -0.42, -0.3, 0.75, -0.42, -0.3, 0.75, 0.18, -0.3, 0.12, 0.18, -0.3,
    0.12, 0.92, -0.3, -0.7, 0.92, -0.3, -0.7, -0.42, 0.3, 0.75, -0.42, 0.3,
    0.75, 0.18, 0.3, 0.12, 0.18, 0.3, 0.12, 0.92, 0.3, -0.7, 0.92, 0.3,
  ]);
  const indices = new Uint16Array([
    2, 1, 0, 5, 4, 3, 3, 2, 0, 0, 5, 3, 6, 7, 8, 9, 10, 11, 6, 8, 9, 9, 11, 6,
    7, 6, 1, 1, 6, 0, 8, 7, 2, 2, 7, 1, 9, 8, 3, 3, 8, 2, 10, 9, 4, 4, 9, 3, 6,
    11, 0, 0, 11, 5, 11, 10, 5, 5, 10, 4,
  ]);
  const { document, binary } = buildMinimalGltf(positions, indices, {
    generator: "chitin Spark shadow demo",
    bounds: false,
  });
  return packGlb(document, binary);
}

async function buildCaster() {
  const compiler = new ChitinCompiler({
    wasm: {
      js: new URL("./dist/packages/coacd/coacd.mjs", location.href).href,
      wasm: new URL("./dist/packages/coacd/coacd.wasm", location.href).href,
      version: "0.2.0",
    },
    workerUrl: new URL("./dist/packages/chitin-lite/worker.js", location.href),
  });
  try {
    status.textContent =
      "Compiling the caster into a full-detail .phys sidecar…";
    const result = await compiler.compileGlb(lPrismGlb());
    const phys = parsePhys(result.phys);
    document.getElementById("hull-count")!.textContent = String(
      phys.hulls.length,
    );

    const proxy = createShadowProxy(phys, { mode: "merged-hulls" });
    casterRoot.add(proxy);

    const material = new THREE.MeshStandardMaterial({
      color: 0xff6f58,
      roughness: 0.55,
      metalness: 0.05,
      emissive: 0x24100c,
    });
    for (const geometry of createHullGeometries(phys)) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = false;
      casterRoot.add(mesh);
      const edge = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry, 24),
        new THREE.LineBasicMaterial({
          color: 0x72e1bf,
          transparent: true,
          opacity: 0.8,
        }),
      );
      hullEdges.add(edge);
    }
    casterRoot.scale.setScalar(0.72);
    casterRoot.position.set(casterBaseX, Number(heightInput.value), 0.7);
  } finally {
    compiler.terminate();
  }
}

async function init() {
  const receiverVisual = await buildReceiverSplats();
  status.textContent =
    "Deriving a decomposed hull preview from the Gaussian support…";
  const receiverHullCount = buildReceiverHullPreview(receiverVisual);
  document.getElementById("receiver-hull-count")!.textContent =
    receiverHullCount.toLocaleString();
  const receiver = attachSparkDirectionalShadow({
    visual: receiverVisual,
    renderer,
    light,
    strength: receiverStrength,
    bias: -0.0002,
  });
  await buildCaster();

  // Three allocates directional shadow depth textures lazily. Render once to
  // populate it, then regenerate Spark's splat buffer with the real texture so
  // the first frame labelled Live is already settled.
  renderer.render(scene, camera);
  receiver.update();
  await sparkRenderer.update({ scene, camera });
  renderer.render(scene, camera);
  status.textContent =
    "Live — amber hulls are derived from the SPZ; mint hulls are the Chitin .phys caster.";

  strengthInput.addEventListener("input", () => {
    receiverStrength = Number(strengthInput.value);
    receiver.strength = receiverStrength;
    strengthValue.textContent = `${Math.round(receiverStrength * 100)}%`;
  });
  heightInput.addEventListener("input", () => {
    casterRoot.position.y = Number(heightInput.value);
    heightValue.textContent = Number(heightInput.value).toFixed(2);
  });
  hullsInput.addEventListener("change", () => {
    hullEdges.visible = hullsInput.checked;
  });
  receiverHullsInput.addEventListener("change", () => {
    receiverHullPreview.visible = receiverHullsInput.checked;
    viewport.dataset.receiverHullsVisible = String(receiverHullsInput.checked);
  });
  receiverSplatsInput.addEventListener("change", () => {
    receiverVisual.visible = receiverSplatsInput.checked;
    viewport.dataset.receiverSplatsVisible = String(receiverSplatsInput.checked);
  });
  viewport.dataset.receiverHullsVisible = String(receiverHullsInput.checked);
  viewport.dataset.receiverSplatsVisible = String(receiverSplatsInput.checked);

  const startedAt = performance.now();
  renderer.setAnimationLoop(() => {
    const time = (performance.now() - startedAt) / 1000;
    if (animateInput.checked) {
      casterRoot.position.x = casterBaseX + Math.sin(time * 0.7) * 0.65;
      casterRoot.rotation.y = Math.sin(time * 0.43) * 0.32;
    }
    controls.update();
    renderer.render(scene, camera);
  });
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init().catch((error) => {
  console.error(error);
  status.textContent = `Demo failed: ${error instanceof Error ? error.message : String(error)}`;
});
