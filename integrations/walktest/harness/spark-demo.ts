import { parsePhys } from "@autarkis/chitin-web";
import {
  createDeferredHullLightingPass,
  createHullGeometries,
  createShadowProxy,
} from "@autarkis/chitin-web/three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";

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
const sceneTab = document.getElementById("scene-tab") as HTMLButtonElement;
const hullSetupTab = document.getElementById(
  "hull-setup-tab",
) as HTMLButtonElement;
const scenePanel = document.getElementById("scene-panel")!;
const hullSetupPanel = document.getElementById("hull-setup-panel")!;
const physFileButton = document.getElementById(
  "phys-file-button",
) as HTMLButtonElement;
const physFileInput = document.getElementById(
  "phys-file-input",
) as HTMLInputElement;
const applyHullsButton = document.getElementById(
  "apply-hulls",
) as HTMLButtonElement;
const resetHullsButton = document.getElementById(
  "reset-hulls",
) as HTMLButtonElement;
const bindingStatus = document.getElementById("binding-status")!;
const physFileName = document.getElementById("phys-file-name")!;
const physFileDetail = document.getElementById("phys-file-detail")!;
const setupHullCount = document.getElementById("setup-hull-count")!;
const physVersion = document.getElementById("phys-version")!;
const physSize = document.getElementById("phys-size")!;
const hullPositionInputs = ["x", "y", "z"].map(
  (axis) => document.getElementById(`hull-pos-${axis}`) as HTMLInputElement,
);
const hullRotationInputs = ["x", "y", "z"].map(
  (axis) => document.getElementById(`hull-rot-${axis}`) as HTMLInputElement,
);
const hullScaleInput = document.getElementById(
  "hull-scale",
) as HTMLInputElement;

type PhysArtifact = ReturnType<typeof parsePhys>;

interface LoadedPhys {
  phys: PhysArtifact;
  byteLength: number;
  label: string;
}

const LIVE_STATUS =
  "Live — Spark appearance is composited with deferred lighting from stored Chitin .phys hulls.";
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
light.position.set(-3.8, 7, -4.6);
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

function createHullDepthLight(
  position: THREE.Vector3,
  target = new THREE.Vector3(0.15, 0.62, -0.15),
): THREE.DirectionalLight {
  const depthLight = new THREE.DirectionalLight(0xffffff, 0.000001);
  depthLight.layers.set(1);
  depthLight.position.copy(position);
  depthLight.target.position.copy(target);
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

const selfShadowLight = createHullDepthLight(
  light.position.clone(),
  light.target.position.clone(),
);
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
const receiverSelfShadowProxy = new THREE.Group();
receiverSelfShadowProxy.name = "Robot Chitin .phys lighting proxy";
const receiverBindingRoot = new THREE.Group();
receiverBindingRoot.name = "Robot Chitin appearance-to-physics binding";
receiverBindingRoot.add(receiverHullPreview, receiverSelfShadowProxy);
scene.add(receiverBindingRoot);

let stagedReceiverPhys: LoadedPhys | null = null;
let appliedReceiverPhys: LoadedPhys | null = null;

let receiverStrength = Number(strengthInput.value);
let ambientOcclusionStrength = Number(aoStrengthInput.value);
const casterBaseX = -0.55;

const ROBOT_HEAD_URL = "https://sparkjs.dev/assets/splats/robot-head.spz";
const ROBOT_PHYS_URL = "./assets/robot-head.phys";
const CASTER_PHYS_URL = "./assets/l-prism.phys";

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

function copyVisualTransform(target: THREE.Object3D, visual: SplatMesh): void {
  target.position.copy(visual.position);
  target.quaternion.copy(visual.quaternion);
  target.scale.copy(visual.scale);
}

function setControlTab(tab: "scene" | "setup"): void {
  const showScene = tab === "scene";
  sceneTab.ariaSelected = String(showScene);
  hullSetupTab.ariaSelected = String(!showScene);
  sceneTab.tabIndex = showScene ? 0 : -1;
  hullSetupTab.tabIndex = showScene ? -1 : 0;
  scenePanel.hidden = !showScene;
  hullSetupPanel.hidden = showScene;
  viewport.dataset.controlTab = tab;
}

function setBindingStatus(
  state: "loading" | "staged" | "applied" | "error",
  message: string,
): void {
  bindingStatus.dataset.state = state;
  bindingStatus.textContent = message;
  viewport.dataset.hullBindingStatus = state;
}

function formatBytes(byteLength: number): string {
  if (byteLength < 1024) return `${byteLength} B`;
  return `${(byteLength / 1024).toFixed(1)} KB`;
}

function updatePhysSummary(artifact: LoadedPhys): void {
  physFileName.textContent = artifact.label;
  physFileDetail.textContent =
    `${artifact.phys.hulls.length.toLocaleString()} hulls · ` +
    `Chitin format v${artifact.phys.version}`;
  setupHullCount.textContent = artifact.phys.hulls.length.toLocaleString();
  physVersion.textContent = `v${artifact.phys.version}`;
  physSize.textContent = formatBytes(artifact.byteLength);
  document.getElementById("receiver-hull-count")!.textContent =
    artifact.phys.hulls.length.toLocaleString();
}

function clearRenderedGroup(group: THREE.Group): void {
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  group.traverse((object) => {
    if (
      !(object instanceof THREE.Mesh || object instanceof THREE.LineSegments)
    ) {
      return;
    }
    geometries.add(object.geometry);
    const objectMaterials = Array.isArray(object.material)
      ? object.material
      : [object.material];
    for (const material of objectMaterials) materials.add(material);
  });
  group.clear();
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
}

function rebuildReceiverPreview(phys: PhysArtifact): void {
  clearRenderedGroup(receiverHullPreview);
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
  receiverHullPreview.visible = receiverHullsInput.checked;
}

function rebuildReceiverLightingProxy(phys: PhysArtifact): void {
  clearRenderedGroup(receiverSelfShadowProxy);
  const shadowProxy = createShadowProxy(phys, { mode: "merged-hulls" });
  shadowProxy.traverse((object) => object.layers.set(1));
  receiverSelfShadowProxy.add(shadowProxy);
  receiverSelfShadowProxy.traverse((object) => object.layers.set(1));
}

function readAlignmentInputs(): {
  position: [number, number, number];
  rotation: [number, number, number];
  scale: number;
} | null {
  const position = hullPositionInputs.map((input) => Number(input.value)) as [
    number,
    number,
    number,
  ];
  const rotation = hullRotationInputs.map((input) => Number(input.value)) as [
    number,
    number,
    number,
  ];
  const scale = Number(hullScaleInput.value);
  if (
    !position.every(Number.isFinite) ||
    !rotation.every(Number.isFinite) ||
    !Number.isFinite(scale) ||
    scale <= 0
  ) {
    return null;
  }
  return { position, rotation, scale };
}

function stageAlignment(): void {
  const alignment = readAlignmentInputs();
  if (alignment === null) {
    applyHullsButton.disabled = true;
    setBindingStatus(
      "error",
      "Enter finite coordinates and a scale above zero.",
    );
    return;
  }
  receiverHullPreview.position.fromArray(alignment.position);
  receiverHullPreview.rotation.set(
    THREE.MathUtils.degToRad(alignment.rotation[0]),
    THREE.MathUtils.degToRad(alignment.rotation[1]),
    THREE.MathUtils.degToRad(alignment.rotation[2]),
  );
  receiverHullPreview.scale.setScalar(alignment.scale);
  applyHullsButton.disabled = stagedReceiverPhys === null;
  setBindingStatus(
    "staged",
    `${stagedReceiverPhys?.label ?? ".phys"} staged — apply to update scene lighting.`,
  );
}

function resetAlignmentInputs(): void {
  for (const input of [...hullPositionInputs, ...hullRotationInputs]) {
    input.value = "0";
  }
  hullScaleInput.value = "1";
  stageAlignment();
}

function stageReceiverPhys(artifact: LoadedPhys): void {
  stagedReceiverPhys = artifact;
  rebuildReceiverPreview(artifact.phys);
  updatePhysSummary(artifact);
  stageAlignment();
}

function applyReceiverBinding(): void {
  if (stagedReceiverPhys === null) return;
  const alignment = readAlignmentInputs();
  if (alignment === null) return;
  rebuildReceiverLightingProxy(stagedReceiverPhys.phys);
  receiverSelfShadowProxy.position.copy(receiverHullPreview.position);
  receiverSelfShadowProxy.quaternion.copy(receiverHullPreview.quaternion);
  receiverSelfShadowProxy.scale.copy(receiverHullPreview.scale);
  appliedReceiverPhys = stagedReceiverPhys;
  applyHullsButton.disabled = true;
  viewport.dataset.hullBindingAsset = appliedReceiverPhys.label;
  viewport.dataset.hullBindingTransform = JSON.stringify(alignment);
  setBindingStatus(
    "applied",
    `${appliedReceiverPhys.label} applied to deferred lighting.`,
  );
}

function initializeReceiverBinding(
  artifact: LoadedPhys,
  visual: SplatMesh,
): void {
  copyVisualTransform(receiverBindingRoot, visual);
  stagedReceiverPhys = artifact;
  appliedReceiverPhys = artifact;
  rebuildReceiverPreview(artifact.phys);
  rebuildReceiverLightingProxy(artifact.phys);
  updatePhysSummary(artifact);
  viewport.dataset.hullBindingAsset = artifact.label;
  viewport.dataset.hullBindingTransform = JSON.stringify({
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    scale: 1,
  });
  setBindingStatus(
    "applied",
    `${artifact.label} loaded and applied to deferred lighting.`,
  );
}

function bindHullSetupControls(): void {
  sceneTab.addEventListener("click", () => setControlTab("scene"));
  hullSetupTab.addEventListener("click", () => setControlTab("setup"));
  for (const tab of [sceneTab, hullSetupTab]) {
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const nextTab = tab === sceneTab ? hullSetupTab : sceneTab;
      const next = nextTab === sceneTab ? "scene" : "setup";
      setControlTab(next);
      nextTab.focus();
    });
  }
  for (const input of [
    ...hullPositionInputs,
    ...hullRotationInputs,
    hullScaleInput,
  ]) {
    input.addEventListener("input", stageAlignment);
  }
  resetHullsButton.addEventListener("click", resetAlignmentInputs);
  applyHullsButton.addEventListener("click", applyReceiverBinding);
  physFileButton.addEventListener("click", () => physFileInput.click());
  physFileInput.addEventListener("change", () => {
    const file = physFileInput.files?.[0];
    if (file === undefined) return;
    void (async () => {
      physFileButton.disabled = true;
      setBindingStatus("loading", `Inspecting ${file.name}…`);
      try {
        if (!file.name.toLowerCase().endsWith(".phys")) {
          throw new Error("Choose a .phys sidecar generated by Chitin.");
        }
        const buffer = await file.arrayBuffer();
        const phys = parsePhys(buffer);
        if (phys.hulls.length === 0) {
          throw new Error("The selected sidecar contains no collider hulls.");
        }
        stageReceiverPhys({
          phys,
          byteLength: buffer.byteLength,
          label: file.name,
        });
      } catch (error) {
        applyHullsButton.disabled = true;
        setBindingStatus(
          "error",
          error instanceof Error ? error.message : String(error),
        );
      } finally {
        physFileButton.disabled = false;
        physFileInput.value = "";
      }
    })();
  });
  setControlTab("scene");
}

async function buildReceiverSplats(): Promise<SplatMesh> {
  status.textContent = "Loading Spark’s robot-head example capture…";
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
  document.getElementById("splat-count")!.textContent = count.toLocaleString();
  document.getElementById("receiver-name")!.textContent = "Spark robot head";
  return visual;
}

async function loadPhys(
  url: string,
  label: string,
): Promise<LoadedPhys> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Unable to load ${label} (${response.status} ${response.statusText})`,
    );
  }
  const buffer = await response.arrayBuffer();
  return { phys: parsePhys(buffer), byteLength: buffer.byteLength, label };
}

function buildCaster(phys: PhysArtifact) {
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
  casterRoot.position.set(casterBaseX, Number(heightInput.value), -0.9);
}

async function init() {
  status.textContent = "Loading paired Spark and Chitin assets…";
  const [receiverVisual, robotPhys, casterPhys] = await Promise.all([
    buildReceiverSplats(),
    loadPhys(ROBOT_PHYS_URL, "robot-head.phys"),
    loadPhys(CASTER_PHYS_URL, "l-prism.phys"),
  ]);
  initializeReceiverBinding(robotPhys, receiverVisual);

  buildCaster(casterPhys.phys);

  status.textContent = "Creating the hull G-buffer and deferred composite…";
  const deferredLighting = createDeferredHullLightingPass({
    renderer,
    scene,
    camera,
    hullLayer: 1,
    lights: [
      {
        light,
        strength: receiverStrength,
        bias: -0.0002,
        normalBias: 0.025,
      },
      {
        light: selfShadowLight,
        strength: selfShadowsInput.checked ? receiverStrength * 0.28 : 0,
        bias: -0.0001,
        normalBias: 0.018,
      },
      ...aoLights.map((aoLight) => ({
        light: aoLight,
        strength:
          ambientOcclusionInput.checked
            ? ambientOcclusionStrength / aoLights.length
            : 0,
        bias: -0.0001,
        normalBias: 0.014,
      })),
    ],
  });
  viewport.dataset.lightingMode = "deferred-hulls";
  viewport.dataset.selfShadowSource = "phys-hulls";
  const [receiver, selfShadowReceiver, ...aoReceivers] =
    deferredLighting.lights;

  const applyLightingState = (): void => {
    const enabled = lightingInput.checked;
    receiver.strength = enabled ? receiverStrength : 0;
    selfShadowReceiver.strength =
      enabled && selfShadowsInput.checked ? receiverStrength * 0.28 : 0;
    const aoStrength =
      enabled && ambientOcclusionInput.checked
        ? ambientOcclusionStrength / aoReceivers.length
        : 0;
    for (const aoReceiver of aoReceivers) aoReceiver.strength = aoStrength;
    selfShadowLight.visible = enabled && selfShadowsInput.checked;
    for (const aoLight of aoLights) {
      aoLight.visible = enabled && ambientOcclusionInput.checked;
    }
  };
  applyLightingState();

  // Three allocates directional shadow depth textures lazily. Render once to
  // populate them, then settle Spark before presenting the deferred composite.
  deferredLighting.render();
  await sparkRenderer.update({ scene, camera });
  deferredLighting.render();
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
    deferredLighting.render();
  });
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

bindHullSetupControls();
init().catch((error) => {
  console.error(error);
  status.textContent = `Demo failed: ${error instanceof Error ? error.message : String(error)}`;
});
