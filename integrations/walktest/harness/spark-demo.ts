import {
  ChitinCompiler,
  type GaussianFieldInput,
} from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import {
  attachSparkDirectionalShadow,
  bindSparkToPhysSurface,
} from "@autarkis/chitin-web/spark";
import {
  createHullGeometries,
  createShadowProxy,
} from "@autarkis/chitin-web/three";
import { SparkRenderer, SplatMesh, SpzWriter } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";

import { buildMinimalGltf, packGlb } from "../../../scripts/glb-pack.mjs";

const viewport = document.getElementById("viewport")!;
const status = document.getElementById("status")!;
const strengthInput = document.getElementById("strength") as HTMLInputElement;
const strengthValue = document.getElementById("strength-value")!;
const aoStrengthInput = document.getElementById(
  "ao-strength",
) as HTMLInputElement;
const aoStrengthValue = document.getElementById("ao-strength-value")!;
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
const environmentInput = document.getElementById(
  "environment",
) as HTMLSelectElement;
const lightingInput = document.getElementById("lighting") as HTMLInputElement;
const selfShadowsInput = document.getElementById(
  "self-shadows",
) as HTMLInputElement;
const ambientOcclusionInput = document.getElementById(
  "ambient-occlusion",
) as HTMLInputElement;

const LIVE_STATUS =
  "Live — amber and mint geometry are parsed Chitin .phys hulls; Spark supplies appearance only.";
const LIGHTING_OFF_STATUS =
  "Live — real-time lighting is off; Spark capture colors remain baked.";
const ENVIRONMENTS: Record<string, string | null> = {
  studio: null,
  venice:
    "https://threejs.org/examples/textures/equirectangular/venice_sunset_1k.hdr",
  overpass:
    "https://threejs.org/examples/textures/equirectangular/pedestrian_overpass_1k.hdr",
};

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

const roomEnvironment = new RoomEnvironment();
const environmentGenerator = new THREE.PMREMGenerator(renderer);
const studioEnvironment = environmentGenerator.fromScene(
  roomEnvironment,
  0.04,
).texture;
roomEnvironment.dispose();
environmentGenerator.dispose();
scene.environment = studioEnvironment;
scene.environmentIntensity = 0.55;
scene.backgroundIntensity = 0.42;
scene.backgroundBlurriness = 0.16;
viewport.dataset.environment = "studio";
viewport.dataset.lighting = String(lightingInput.checked);
let activeHdri: THREE.DataTexture | null = null;
let environmentRequest = 0;

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
const hemisphereLight = new THREE.HemisphereLight(
  0x9ab9ff,
  0x111827,
  1.4,
);
scene.add(hemisphereLight);

function createHullDepthLight(position: THREE.Vector3): THREE.DirectionalLight {
  const depthLight = new THREE.DirectionalLight(0xffffff, 0.000001);
  depthLight.position.copy(position);
  depthLight.target.position.set(0.15, 0.62, -0.15);
  depthLight.castShadow = true;
  depthLight.shadow.mapSize.set(1024, 1024);
  depthLight.shadow.camera.left = -3.2;
  depthLight.shadow.camera.right = 3.2;
  depthLight.shadow.camera.top = 3.2;
  depthLight.shadow.camera.bottom = -3.2;
  depthLight.shadow.camera.near = 0.1;
  depthLight.shadow.camera.far = 18;
  depthLight.shadow.bias = -0.00015;
  depthLight.shadow.camera.layers.set(1);
  scene.add(depthLight, depthLight.target);
  return depthLight;
}

light.shadow.camera.layers.set(0);
const selfShadowLight = createHullDepthLight(light.position.clone());
const aoLights = [
  new THREE.Vector3(4.5, 4.5, 3.5),
  new THREE.Vector3(-4.5, 4.5, -3.5),
].map(createHullDepthLight);

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
receiverHullPreview.name = "Robot Chitin .phys hull preview";
scene.add(receiverHullPreview);

const receiverSelfShadowProxy = new THREE.Group();
receiverSelfShadowProxy.name = "Robot Chitin .phys lighting proxy";
scene.add(receiverSelfShadowProxy);

let receiverStrength = Number(strengthInput.value);
let ambientOcclusionStrength = Number(aoStrengthInput.value);
const casterBaseX = -0.55;

const ROBOT_HEAD_URL = "https://sparkjs.dev/assets/splats/robot-head.spz";

async function applyEnvironment(name: string): Promise<void> {
  const url = ENVIRONMENTS[name];
  if (url === undefined) return;
  const request = ++environmentRequest;
  environmentInput.disabled = true;

  try {
    if (url === null) {
      activeHdri?.dispose();
      activeHdri = null;
      scene.environment = studioEnvironment;
      scene.background = new THREE.Color(0x080b12);
      scene.environmentIntensity = lightingInput.checked ? 0.55 : 0;
      scene.backgroundIntensity = 0.42;
      viewport.dataset.environment = name;
      return;
    }

    status.textContent = `Loading ${environmentInput.selectedOptions[0]?.textContent ?? "HDRI"}…`;
    const texture = await new RGBELoader().loadAsync(url);
    if (request !== environmentRequest) {
      texture.dispose();
      return;
    }
    texture.mapping = THREE.EquirectangularReflectionMapping;
    activeHdri?.dispose();
    activeHdri = texture;
    scene.environment = texture;
    scene.background = texture;
    scene.environmentIntensity = lightingInput.checked ? 0.7 : 0;
    scene.backgroundIntensity = 0.36;
    viewport.dataset.environment = name;
  } catch (error) {
    console.warn("Unable to load the selected HDR environment", error);
    environmentInput.value = "studio";
    activeHdri?.dispose();
    activeHdri = null;
    scene.environment = studioEnvironment;
    scene.background = new THREE.Color(0x080b12);
    scene.environmentIntensity = lightingInput.checked ? 0.55 : 0;
    viewport.dataset.environment = "studio";
  } finally {
    if (request === environmentRequest) {
      environmentInput.disabled = false;
      status.textContent = LIVE_STATUS;
    }
  }
}

function extractGaussianField(visual: SplatMesh): GaussianFieldInput {
  const count =
    visual.splats?.getNumSplats() ??
    visual.packedSplats?.getNumSplats() ??
    visual.numSplats;
  const centers = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const quaternions = new Float32Array(count * 4);
  const opacities = new Float32Array(count);
  visual.forEachSplat((index, center, scale, quaternion, opacity) => {
    center.toArray(centers, index * 3);
    scale.toArray(scales, index * 3);
    quaternion.toArray(quaternions, index * 4);
    opacities[index] = opacity;
  });
  return { centers, scales, quaternions, opacities };
}

function copyVisualTransform(target: THREE.Object3D, visual: SplatMesh): void {
  target.position.copy(visual.position);
  target.quaternion.copy(visual.quaternion);
  target.scale.copy(visual.scale);
}

function buildRobotPhysGeometry(
  phys: ReturnType<typeof parsePhys>,
  visual: SplatMesh,
): void {
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
  for (const geometry of createHullGeometries(phys)) {
    receiverHullPreview.add(new THREE.Mesh(geometry, fillMaterial));
    receiverHullPreview.add(
      new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 20), edgeMaterial),
    );
  }
  copyVisualTransform(receiverHullPreview, visual);
  receiverHullPreview.visible = receiverHullsInput.checked;
  const shadowProxy = createShadowProxy(phys, { mode: "merged-hulls" });
  shadowProxy.traverse((object) => object.layers.set(1));
  receiverSelfShadowProxy.add(shadowProxy);
  receiverSelfShadowProxy.traverse((object) => object.layers.set(1));
  copyVisualTransform(receiverSelfShadowProxy, visual);
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

async function buildCaster(compiler: ChitinCompiler) {
  status.textContent = "Compiling the caster into a Chitin .phys sidecar…";
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
}

async function init() {
  const receiverVisual = await buildReceiverSplats();
  const compiler = new ChitinCompiler({
    wasm: {
      js: new URL("./dist/packages/coacd/coacd.mjs", location.href).href,
      wasm: new URL("./dist/packages/coacd/coacd.wasm", location.href).href,
      version: "0.2.0",
    },
    workerUrl: new URL("./dist/packages/chitin-lite/worker.js", location.href),
  });
  let robotPhys!: ReturnType<typeof parsePhys>;
  try {
    status.textContent =
      "Reconstructing the robot support and compiling it through Chitin…";
    const robotResult = await compiler.compileGaussianField(
      extractGaussianField(receiverVisual),
      {
        reconstruction: {
          resolution: 72,
          supportMultiplier: 1.75,
          minOpacity: 0.2,
          shellThickness: 1.15,
        },
        decompose: {
          threshold: 0.025,
          maxConvexHull: 96,
          maxChVertex: 64,
        },
      },
    );
    robotPhys = parsePhys(robotResult.phys);
    document.getElementById("receiver-hull-count")!.textContent =
      robotPhys.hulls.length.toLocaleString();
    buildRobotPhysGeometry(robotPhys, receiverVisual);
    await buildCaster(compiler);
  } finally {
    compiler.terminate();
  }

  status.textContent = "Binding Spark appearance to the robot .phys surface…";
  const surface = bindSparkToPhysSurface(receiverVisual, robotPhys);
  const receiver = attachSparkDirectionalShadow({
    visual: receiverVisual,
    renderer,
    light,
    strength: receiverStrength,
    bias: -0.0002,
    normalBias: 0.025,
    surface,
  });
  const selfShadowReceiver = attachSparkDirectionalShadow({
    visual: receiverVisual,
    renderer,
    light: selfShadowLight,
    strength: selfShadowsInput.checked ? receiverStrength * 0.65 : 0,
    bias: -0.0001,
    normalBias: 0.018,
    surface,
  });
  const aoReceivers = aoLights.map((aoLight) =>
    attachSparkDirectionalShadow({
      visual: receiverVisual,
      renderer,
      light: aoLight,
      strength:
        ambientOcclusionInput.checked
          ? ambientOcclusionStrength / aoLights.length
          : 0,
      bias: -0.0001,
      normalBias: 0.014,
      surface,
    }),
  );

  const applyLightingState = (): void => {
    const enabled = lightingInput.checked;
    receiver.strength = enabled ? receiverStrength : 0;
    selfShadowReceiver.strength =
      enabled && selfShadowsInput.checked ? receiverStrength * 0.65 : 0;
    const aoStrength =
      enabled && ambientOcclusionInput.checked
        ? ambientOcclusionStrength / aoReceivers.length
        : 0;
    for (const aoReceiver of aoReceivers) aoReceiver.strength = aoStrength;
    receiverSelfShadowProxy.visible =
      enabled &&
      (selfShadowsInput.checked || ambientOcclusionInput.checked);
    selfShadowLight.visible = enabled && selfShadowsInput.checked;
    for (const aoLight of aoLights) {
      aoLight.visible = enabled && ambientOcclusionInput.checked;
    }
  };
  applyLightingState();

  // Three allocates directional shadow depth textures lazily. Render once to
  // populate it, then regenerate Spark's splat buffer with the real texture so
  // the first frame labelled Live is already settled.
  renderer.render(scene, camera);
  receiver.update();
  selfShadowReceiver.update();
  for (const aoReceiver of aoReceivers) aoReceiver.update();
  await sparkRenderer.update({ scene, camera });
  renderer.render(scene, camera);
  status.textContent = LIVE_STATUS;

  strengthInput.addEventListener("input", () => {
    receiverStrength = Number(strengthInput.value);
    applyLightingState();
    strengthValue.textContent = `${Math.round(receiverStrength * 100)}%`;
  });
  aoStrengthInput.addEventListener("input", () => {
    ambientOcclusionStrength = Number(aoStrengthInput.value);
    applyLightingState();
    aoStrengthValue.textContent = `${Math.round(ambientOcclusionStrength * 100)}%`;
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
  lightingInput.addEventListener("change", () => {
    const enabled = lightingInput.checked;
    light.visible = enabled;
    hemisphereLight.visible = enabled;
    scene.environmentIntensity = enabled
      ? environmentInput.value === "studio"
        ? 0.55
        : 0.7
      : 0;
    applyLightingState();
    strengthInput.disabled = !enabled;
    aoStrengthInput.disabled = !enabled || !ambientOcclusionInput.checked;
    viewport.dataset.lighting = String(enabled);
    status.textContent = enabled ? LIVE_STATUS : LIGHTING_OFF_STATUS;
  });
  selfShadowsInput.addEventListener("change", () => {
    applyLightingState();
    viewport.dataset.selfShadows = String(selfShadowsInput.checked);
  });
  ambientOcclusionInput.addEventListener("change", () => {
    applyLightingState();
    aoStrengthInput.disabled =
      !lightingInput.checked || !ambientOcclusionInput.checked;
    viewport.dataset.ambientOcclusion = String(ambientOcclusionInput.checked);
  });
  viewport.dataset.selfShadows = String(selfShadowsInput.checked);
  viewport.dataset.ambientOcclusion = String(ambientOcclusionInput.checked);
  environmentInput.addEventListener("change", () => {
    void applyEnvironment(environmentInput.value);
  });

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
