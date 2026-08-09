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

import { buildMinimalGltf, packGlb } from "../../../scripts/glb-pack.mjs";

const viewport = document.getElementById("viewport")!;
const status = document.getElementById("status")!;
const strengthInput = document.getElementById("strength") as HTMLInputElement;
const strengthValue = document.getElementById("strength-value")!;
const heightInput = document.getElementById("height") as HTMLInputElement;
const heightValue = document.getElementById("height-value")!;
const animateInput = document.getElementById("animate") as HTMLInputElement;
const hullsInput = document.getElementById("hulls") as HTMLInputElement;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080b12);
scene.fog = new THREE.FogExp2(0x080b12, 0.035);

const camera = new THREE.PerspectiveCamera(
  42,
  window.innerWidth / window.innerHeight,
  0.05,
  80,
);
camera.position.set(5.2, 3.5, 6.1);

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

let receiverStrength = Number(strengthInput.value);

async function buildReceiverSplats(): Promise<SplatMesh> {
  const width = 57;
  const spacing = 0.09;
  const count = width * width;
  const writer = new SpzWriter({ numSplats: count, shDegree: 0 });
  let index = 0;
  for (let z = 0; z < width; z++) {
    for (let x = 0; x < width; x++) {
      const px = (x - (width - 1) / 2) * spacing;
      const pz = (z - (width - 1) / 2) * spacing;
      const radial = Math.hypot(px, pz) / 4;
      writer.setCenter(index, px, 0, pz);
      writer.setAlpha(index, Math.max(0.72, 0.97 - radial * 0.12));
      writer.setRgb(
        index,
        0.48 + (x / width) * 0.16,
        0.61 + (z / width) * 0.1,
        0.84,
      );
      writer.setScale(index, spacing * 0.72, 0.018, spacing * 0.72);
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
    casterRoot.position.y = Number(heightInput.value);
  } finally {
    compiler.terminate();
  }
}

async function init() {
  const receiverVisual = await buildReceiverSplats();
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
    "Live — drag to orbit. The visible mesh does not cast; its Chitin hull proxy does.";

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

  const startedAt = performance.now();
  renderer.setAnimationLoop(() => {
    const time = (performance.now() - startedAt) / 1000;
    if (animateInput.checked) {
      casterRoot.position.x = Math.sin(time * 0.7) * 1.2;
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
