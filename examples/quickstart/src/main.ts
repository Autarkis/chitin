import {
  ChitinCompiler,
  ChitinError,
  type CompilationProgress,
  type CompileGlbResult,
} from "@autarkis/chitin-lite";
import ChitinWorker from "@autarkis/chitin-lite/worker?worker";
import coacdModuleUrl from "@autarkis/chitin-coacd-wasm?url";
import coacdWasmUrl from "@autarkis/chitin-coacd-wasm/coacd.wasm?url";
import { parsePhys } from "@autarkis/chitin-web";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import "./styles.css";

const qualityBenchmarkEnabled = new URLSearchParams(window.location.search)
  .get("qualityBenchmark") === "1";

const $ = <T extends HTMLElement>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`missing UI element: ${selector}`);
  return element;
};

const canvas = $("#viewport") as HTMLCanvasElement;
const viewportPanel = $(".viewport-panel");
const runtimeStatus = $("#runtime-status");
const emptyState = $("#empty-state");
const fileInput = $("#file-input") as HTMLInputElement;
const fileButton = $("#file-button") as HTMLButtonElement;
const sampleGrid = $("#sample-grid");
const replaceButton = $("#replace-button") as HTMLButtonElement;
const fileSummary = $("#file-summary");
const fileName = $("#file-name");
const fileSize = $("#file-size");
const threshold = $("#threshold") as HTMLInputElement;
const thresholdValue = $("#threshold-value") as HTMLOutputElement;
const thresholdStatus = $("#threshold-status");
const progressSection = $("#progress-section");
const progressBar = $("#progress-bar");
const progressCopy = $("#progress-copy");
const cancelButton = $("#cancel-button") as HTMLButtonElement;
const errorCard = $("#error-card");
const errorCode = $("#error-code");
const errorMessage = $("#error-message");
const errorSuggestion = $("#error-suggestion");
const retryButton = $("#retry-button") as HTMLButtonElement;
const resultSection = $("#result-section");
const resultSummary = $("#result-summary");
const resultTime = $("#result-time");
const sourceTriangles = $("#source-triangles");
const colliderTriangles = $("#collider-triangles");
const hullCount = $("#hull-count");
const triangleRatio = $("#triangle-ratio");
const downloadButton = $("#download-button") as HTMLButtonElement;
const reportButton = $("#report-button") as HTMLButtonElement;
const reportOutput = $("#report-output") as HTMLPreElement;
const showSource = $("#show-source") as HTMLInputElement;
const showColliders = $("#show-colliders") as HTMLInputElement;
const explodeColliders = $("#explode-colliders") as HTMLInputElement;
const explodeDistance = $("#explode-distance") as HTMLInputElement;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0e110f, 0.035);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 10_000);
camera.position.set(3.8, 2.8, 4.8);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.target.set(0, 0, 0);

scene.add(new THREE.HemisphereLight(0xe9f2df, 0x222820, 1.7));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(4, 7, 5);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0xd7ff43, 1.1);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

const grid = new THREE.GridHelper(20, 28, 0x313831, 0x202520);
(grid.material as THREE.Material).transparent = true;
(grid.material as THREE.Material).opacity = 0.55;
scene.add(grid);

let sourceRoot = new THREE.Group();
let colliderRoot = new THREE.Group();
scene.add(sourceRoot, colliderRoot);

const compiler = new ChitinCompiler({
  wasm: {
    js: coacdModuleUrl,
    wasm: coacdWasmUrl,
    version: "0.2.0",
  },
  workerFactory: () => new ChitinWorker(),
  maxWorkers: 2,
});

let selectedFile: File | null = null;
let activeController: AbortController | null = null;
let activeCompile: Promise<void> | null = null;
let requestNumber = 0;
let latestResult: CompileGlbResult | null = null;
let downloadUrl: string | null = null;
let thresholdTimer: number | null = null;
let appliedThreshold: number | null = null;
let appliedFile: File | null = null;
let lastErrorCode: string | null = null;
type ColliderPart = {
  group: THREE.Group;
  fill: THREE.MeshStandardMaterial;
  wire: THREE.LineBasicMaterial;
  delay: number;
  basePosition: THREE.Vector3;
  explodeDirection: THREE.Vector3;
};
let colliderParts: ColliderPart[] = [];
let colliderExplosionRadius = 1;
let colliderExplosionCurrent = 0;
let colliderExplosionTarget = 0;
let colliderReveal: {
  startedAt: number | null;
  parts: ColliderPart[];
} | null = null;
let colliderRevealCount = 0;

const stageOrder: CompilationProgress["stage"][] = [
  "reading-input",
  "parsing-input",
  "validating-input",
  "loading-wasm",
  "decomposing",
  "writing-phys",
  "done",
];

const stageCopy: Record<CompilationProgress["stage"], string> = {
  "reading-input": "Reading GLB bytes",
  "parsing-input": "Resolving the active scene",
  "validating-input": "Validating triangle geometry",
  "loading-wasm": "Loading the compiler runtime",
  "decomposing": "Finding convex parts",
  "verifying": "Verifying collider output",
  "writing-phys": "Writing the .phys sidecar",
  done: "Compilation complete",
};

function setRuntime(state: "ready" | "busy" | "error", copy: string): void {
  runtimeStatus.className = `runtime-status ${state}`;
  runtimeStatus.innerHTML = `<i></i> ${copy}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.max(1, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}

function asMaterialArray(material: THREE.Material | THREE.Material[]): THREE.Material[] {
  return Array.isArray(material) ? material : [material];
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh || object instanceof THREE.LineSegments)) return;
    object.geometry.dispose();
    for (const material of asMaterialArray(object.material)) material.dispose();
  });
}

function replaceGroup(current: THREE.Group, next: THREE.Group): THREE.Group {
  scene.remove(current);
  disposeObject(current);
  scene.add(next);
  return next;
}

function fitCamera(preferredRoot?: THREE.Object3D, distanceMultiplier = 2.7): void {
  const visible = new THREE.Group();
  if (preferredRoot) visible.add(preferredRoot.clone());
  else if (sourceRoot.children.length > 0) visible.add(sourceRoot.clone());
  else if (colliderRoot.children.length > 0) visible.add(colliderRoot.clone());
  const box = new THREE.Box3().setFromObject(visible);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.length() * 0.5, 0.2);
  controls.target.copy(center);
  camera.near = Math.max(radius / 1000, 0.001);
  camera.far = Math.max(radius * 100, 100);
  camera.position.copy(center).add(
    new THREE.Vector3(1.25, 0.9, 1.55).normalize().multiplyScalar(radius * distanceMultiplier),
  );
  camera.updateProjectionMatrix();
  controls.update();
  grid.position.y = box.min.y;
  grid.scale.setScalar(Math.max(radius / 4, 0.15));
}

async function showSourcePreview(file: File): Promise<void> {
  const loader = new GLTFLoader();
  const gltf = await loader.parseAsync(await file.arrayBuffer(), "");
  const next = new THREE.Group();
  next.name = "source_geometry";
  next.add(gltf.scene);
  const previewMeshes: THREE.Mesh[] = [];
  next.traverse((object) => {
    if (object instanceof THREE.Mesh) previewMeshes.push(object);
  });
  for (const object of previewMeshes) {
    const usedMaterialArray = Array.isArray(object.material);
    const sourceMaterials = asMaterialArray(object.material);
    for (const material of sourceMaterials) material.dispose();
    const previewMaterials = sourceMaterials.map(() => new THREE.MeshBasicMaterial({
      color: 0xdce6df,
      transparent: true,
      opacity: showColliders.checked ? 0.18 : 0.84,
      depthWrite: !showColliders.checked,
      side: THREE.DoubleSide,
    }));
    object.material = usedMaterialArray ? previewMaterials : previewMaterials[0];
    object.userData.sourceFill = true;
    object.renderOrder = 1;
    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(object.geometry, 24),
      new THREE.LineBasicMaterial({
        color: 0xf2f6f0,
        transparent: true,
        opacity: showColliders.checked ? 0.22 : 0.82,
      }),
    );
    outline.userData.sourceOutline = true;
    outline.renderOrder = 2;
    object.add(outline);
  }
  sourceRoot = replaceGroup(sourceRoot, next);
  updatePreviewLayers();
  fitCamera();
}

function showColliderPreview(result: CompileGlbResult): void {
  const phys = parsePhys(result.phys);
  const next = new THREE.Group();
  next.name = "compiled_colliders";
  const palette = [0xd7ff43, 0x66dcff, 0xffad5c, 0xbb8cff, 0xff7188, 0x58e2aa];
  const revealParts: ColliderPart[] = [];
  const stagger = phys.hulls.length > 1 ? Math.min(42, 420 / (phys.hulls.length - 1)) : 0;
  phys.hulls.forEach((hull, index) => {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(hull.vertices, 3));
    geometry.setIndex(new THREE.BufferAttribute(hull.indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    const center = geometry.boundingBox?.getCenter(new THREE.Vector3()) ?? new THREE.Vector3();
    geometry.translate(-center.x, -center.y, -center.z);
    const color = palette[index % palette.length];
    const fill = new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: 0,
      roughness: 0.74,
      metalness: 0.02,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const wire = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0 });
    const part = new THREE.Group();
    part.position.copy(center);
    part.scale.setScalar(0.78);
    part.add(
      new THREE.Mesh(geometry, fill),
      new THREE.LineSegments(new THREE.WireframeGeometry(geometry), wire),
    );
    next.add(part);
    revealParts.push({
      group: part,
      fill,
      wire,
      delay: index * stagger,
      basePosition: center.clone(),
      explodeDirection: new THREE.Vector3(),
    });
  });
  const bounds = new THREE.Box3().setFromObject(next);
  const objectCenter = bounds.getCenter(new THREE.Vector3());
  colliderExplosionRadius = Math.max(bounds.getSize(new THREE.Vector3()).length() * 0.5, 0.2);
  revealParts.forEach((part, index) => {
    part.explodeDirection.copy(part.basePosition).sub(objectCenter);
    if (part.explodeDirection.lengthSq() < 1e-8) {
      const y = 1 - 2 * ((index + 0.5) / Math.max(1, revealParts.length));
      const radial = Math.sqrt(Math.max(0, 1 - y * y));
      const angle = index * Math.PI * (3 - Math.sqrt(5));
      part.explodeDirection.set(Math.cos(angle) * radial, y, Math.sin(angle) * radial);
    } else {
      part.explodeDirection.normalize();
    }
  });
  colliderParts = revealParts;
  explodeColliders.disabled = false;
  explodeDistance.disabled = !explodeColliders.checked;
  colliderExplosionTarget = explodeColliders.checked ? Number(explodeDistance.value) : 0;
  colliderReveal = { startedAt: null, parts: revealParts };
  colliderRevealCount++;
  viewportPanel.dataset.colliderReveal = "pending";
  colliderRoot = replaceGroup(colliderRoot, next);
  colliderRoot.traverse((object) => { object.renderOrder = 2; });
  updatePreviewLayers();
  fitCamera();
}

function positionColliderParts(amount: number): void {
  for (const part of colliderParts) {
    part.group.position.copy(part.basePosition).addScaledVector(
      part.explodeDirection,
      amount * colliderExplosionRadius,
    );
  }
}

function updateColliderExplosion(): void {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const delta = colliderExplosionTarget - colliderExplosionCurrent;
  colliderExplosionCurrent = reducedMotion || Math.abs(delta) < 0.001
    ? colliderExplosionTarget
    : colliderExplosionCurrent + delta * 0.14;
  positionColliderParts(colliderExplosionCurrent);
  viewportPanel.dataset.exploded = String(colliderExplosionCurrent > 0.001);
}

function updateExplosionControls(): void {
  explodeDistance.disabled = explodeColliders.disabled || !explodeColliders.checked;
  colliderExplosionTarget = explodeColliders.checked ? Number(explodeDistance.value) : 0;
  if (colliderParts.length === 0) return;
  const previousAmount = colliderExplosionCurrent;
  positionColliderParts(colliderExplosionTarget);
  fitCamera(
    explodeColliders.checked ? colliderRoot : (sourceRoot.children.length > 0 ? sourceRoot : colliderRoot),
    explodeColliders.checked ? 3.5 : 2.7,
  );
  positionColliderParts(previousAmount);
}

function finishColliderReveal(): void {
  if (!colliderReveal) return;
  for (const part of colliderReveal.parts) {
    part.group.scale.setScalar(1);
    part.fill.opacity = 0.48;
    part.wire.opacity = 0.92;
  }
  colliderReveal = null;
  viewportPanel.dataset.colliderReveal = "complete";
}

function updateColliderReveal(time: number): void {
  if (!colliderReveal || !colliderRoot.visible) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    finishColliderReveal();
    return;
  }
  colliderReveal.startedAt ??= time;
  viewportPanel.dataset.colliderReveal = "running";
  const duration = 520;
  let complete = true;
  for (const part of colliderReveal.parts) {
    const linear = Math.min(1, Math.max(0, (time - colliderReveal.startedAt - part.delay) / duration));
    const remaining = linear - 1;
    const eased = 1 + 2.2 * remaining ** 3 + 1.2 * remaining ** 2;
    part.group.scale.setScalar(0.78 + 0.22 * eased);
    part.fill.opacity = 0.48 * linear;
    part.wire.opacity = 0.92 * linear;
    if (linear < 1) complete = false;
  }
  if (complete) finishColliderReveal();
}

function updatePreviewLayers(): void {
  sourceRoot.visible = showSource.checked;
  colliderRoot.visible = showColliders.checked;
  if (colliderRoot.visible && colliderReveal?.startedAt === null) {
    viewportPanel.dataset.colliderReveal = "pending";
  }
  const sourceOpacity = showColliders.checked ? 0.18 : 0.84;
  const outlineOpacity = showColliders.checked ? 0.22 : 0.82;
  sourceRoot.traverse((object) => {
    if (object instanceof THREE.LineSegments && object.userData.sourceOutline) {
      const materials = asMaterialArray(object.material);
      for (const material of materials) material.opacity = outlineOpacity;
      return;
    }
    if (!(object instanceof THREE.Mesh) || !object.userData.sourceFill) return;
    const materials = asMaterialArray(object.material);
    for (const material of materials) {
      material.transparent = true;
      material.opacity = sourceOpacity;
      material.depthWrite = !showColliders.checked;
      material.side = THREE.DoubleSide;
    }
  });
}

function updateProgress(progress: CompilationProgress): void {
  const index = Math.max(0, stageOrder.indexOf(progress.stage));
  let percentage = Math.round(((index + 1) / stageOrder.length) * 100);
  if (progress.stage === "decomposing" && progress.total && progress.completed !== undefined) {
    percentage = Math.round(20 + 70 * (progress.completed / progress.total));
  }
  progressBar.style.width = `${percentage}%`;
  progressBar.parentElement?.setAttribute("aria-valuenow", String(percentage));
  const eta = progress.eta_ms !== undefined && progress.eta_ms >= 1000
    ? ` · about ${formatDuration(progress.eta_ms)} left`
    : "";
  progressCopy.textContent = `${progress.message ?? stageCopy[progress.stage]}${eta}`;
}

function showFile(file: File): void {
  fileButton.hidden = true;
  fileSummary.hidden = false;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  emptyState.classList.add("hidden");
}

function resetOutput(): void {
  errorCard.hidden = true;
  if (lastErrorCode === "NON_MANIFOLD") showColliders.checked = true;
  lastErrorCode = null;
  retryButton.textContent = "Try again";
  resultSection.hidden = true;
  reportOutput.hidden = true;
  reportButton.setAttribute("aria-expanded", "false");
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
  latestResult = null;
  colliderParts = [];
  colliderExplosionCurrent = 0;
  colliderExplosionTarget = 0;
  explodeColliders.checked = false;
  explodeColliders.disabled = true;
  explodeDistance.disabled = true;
  colliderReveal = null;
  viewportPanel.dataset.colliderReveal = "idle";
  const emptyColliders = new THREE.Group();
  emptyColliders.name = "compiled_colliders";
  colliderRoot = replaceGroup(colliderRoot, emptyColliders);
  updatePreviewLayers();
}

function showError(error: unknown): void {
  const info =
    error instanceof ChitinError
      ? error.toInfo()
      : {
          code: "COMPILE_ERROR",
          message: error instanceof Error ? error.message : String(error),
          suggestion: "Check the GLB and try again.",
        };
  lastErrorCode = info.code;
  const needsRepair = info.code === "NON_MANIFOLD";
  errorCode.textContent = needsRepair ? "NON_MANIFOLD · GEOMETRY REPAIR NEEDED" : info.code;
  errorMessage.textContent = info.message;
  errorSuggestion.textContent = info.suggestion ?? "Check the GLB and try again.";
  retryButton.textContent = needsRepair ? "Choose another GLB" : "Try again";
  errorCard.hidden = false;
  if (needsRepair) {
    showSource.checked = true;
    showColliders.checked = false;
    updatePreviewLayers();
    thresholdStatus.className = "setting-status pending";
    thresholdStatus.textContent = "Needs full-compiler repair · no collider produced";
    setRuntime("error", "Source ready · collider unavailable");
  } else {
    setRuntime("error", "Compile failed");
  }
}

function hasReportWarning(result: CompileGlbResult, code: string): boolean {
  return result.report.warnings.some((warning) => warning.code === code);
}

function effectiveReportNumber(result: CompileGlbResult, key: string): number | null {
  const value = result.report.config.effective?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function hullBudgetCopy(result: CompileGlbResult): string {
  const budget = effectiveReportNumber(result, "max_hulls");
  const ceiling = effectiveReportNumber(result, "max_hulls_ceiling");
  if (budget === null || ceiling === null || budget === -1) return "";
  return budget === ceiling ? ` · hull budget ${budget}` : ` · hull budget ${budget}/${ceiling}`;
}

function showResult(result: CompileGlbResult, detail: number, file: File): void {
  latestResult = result;
  appliedThreshold = detail;
  appliedFile = file;
  resultSection.hidden = false;
  const hollowShellGuard = hasReportWarning(result, "INTERACTIVE_HOLLOW_SHELL_GUARD");
  const importanceGuard = hasReportWarning(result, "INTERACTIVE_IMPORTANCE_GUARD");
  const adaptiveHullDetail = hasReportWarning(result, "INTERACTIVE_HULL_VERTICES_ADAPTED");
  const budgetCopy = hullBudgetCopy(result);
  resultSummary.textContent = hollowShellGuard
    ? `Detail ${detail.toFixed(2)} requested${budgetCopy} · hollow-shell guard + adaptive hull detail · checks not evaluated`
    : importanceGuard
      ? `Detail ${detail.toFixed(2)} requested${budgetCopy} · scale-aware body detail + adaptive hull budget · checks not evaluated`
      : adaptiveHullDetail
      ? `Detail ${detail.toFixed(2)} applied${budgetCopy} · scene-size + shape-aware hull budget · checks not evaluated`
      : `Detail ${detail.toFixed(2)} applied${budgetCopy} · interactive profile · checks not evaluated`;
  const reused = result.reuse.component_results;
  const reuseCopy = reused > 0
    ? ` · ${reused}/${result.reuse.total_components} parts reused`
    : "";
  resultTime.textContent = `${Math.round(result.report.timings_ms.total ?? 0)} ms · ${formatBytes(result.phys.byteLength)}${reuseCopy}`;
  const sourceTriangleCount = result.source.triangle_count;
  const colliderTriangleCount = result.report.output.triangle_count;
  sourceTriangles.textContent = sourceTriangleCount.toLocaleString();
  colliderTriangles.textContent = colliderTriangleCount.toLocaleString();
  hullCount.textContent = result.hulls.length.toLocaleString();
  triangleRatio.textContent = sourceTriangleCount === 0
    ? "—"
    : `${Math.round((colliderTriangleCount / sourceTriangleCount) * 100)}%`;
  reportOutput.textContent = JSON.stringify(result.report, null, 2);
  updateThresholdStatus();
  setRuntime("ready", "Artifact ready");
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function compileSelected(file: File): Promise<void> {
  const compileThreshold = Number(threshold.value);
  const ownRequest = ++requestNumber;
  activeController?.abort();
  if (activeCompile) await activeCompile.catch(() => {});
  if (ownRequest !== requestNumber) return;

  const preservesPreviousArtifact = selectedFile === file && latestResult !== null;
  selectedFile = file;
  showFile(file);
  if (preservesPreviousArtifact) {
    errorCard.hidden = true;
    lastErrorCode = null;
    resultSummary.textContent = `Showing detail ${appliedThreshold?.toFixed(2) ?? "—"} · compiling ${compileThreshold.toFixed(2)}`;
  } else {
    resetOutput();
  }
  progressSection.hidden = false;
  progressBar.style.width = "4%";
  progressCopy.textContent = "Preparing compilation";
  cancelButton.disabled = false;
  setRuntime("busy", "Compiling locally");
  thresholdStatus.className = "setting-status pending";
  thresholdStatus.textContent = `Compiling detail ${compileThreshold.toFixed(2)}…`;
  const controller = new AbortController();
  activeController = controller;

  const work = (async () => {
    let preview: Promise<void> | null = null;
    try {
      if (!preservesPreviousArtifact || sourceRoot.children.length === 0) {
        preview = showSourcePreview(file).catch((error) => {
          console.warn("Source preview unavailable", error);
        });
      }
      const result = await compiler.compileGlb(file, {
        profile: "interactive",
        signal: controller.signal,
        decompose: { threshold: compileThreshold },
        checkManifold: true,
        quality: qualityBenchmarkEnabled
          ? { surfaceSamples: 2048, volumeSamples: 8192 }
          : false,
        onProgress: updateProgress,
      });
      await preview;
      if (ownRequest !== requestNumber) return;
      showColliderPreview(result);
      showResult(result, compileThreshold, file);
    } catch (error) {
      await preview;
      if (ownRequest !== requestNumber) return;
      if (error instanceof ChitinError && error.code === "CANCELLED") {
        setRuntime("ready", latestResult ? "Previous artifact ready" : "Compilation cancelled");
        progressCopy.textContent = latestResult
          ? "Cancelled — previous collider remains applied"
          : "Cancelled — adjust settings or try again";
        updateThresholdStatus();
      } else {
        showError(error);
      }
    } finally {
      if (ownRequest === requestNumber) {
        activeController = null;
        activeCompile = null;
        cancelButton.disabled = true;
        progressSection.hidden = true;
      }
    }
  })();
  activeCompile = work;
  await work;
}

function updateThresholdStatus(): void {
  const detail = Number(threshold.value);
  thresholdValue.value = detail.toFixed(2);
  if (!selectedFile) {
    thresholdStatus.className = "setting-status";
    thresholdStatus.textContent = "Choose geometry to apply this setting.";
    return;
  }
  if (lastErrorCode === "NON_MANIFOLD") {
    thresholdStatus.className = "setting-status pending";
    thresholdStatus.textContent = "Needs full-compiler repair · no collider produced";
    return;
  }
  if (appliedFile === selectedFile && appliedThreshold === detail && latestResult) {
    thresholdStatus.className = "setting-status applied";
    const hullCopy = `${latestResult.hulls.length} ${latestResult.hulls.length === 1 ? "hull" : "hulls"}`;
    const budgetCopy = hullBudgetCopy(latestResult);
    thresholdStatus.textContent = hasReportWarning(latestResult, "INTERACTIVE_HOLLOW_SHELL_GUARD")
      ? `Requested ${detail.toFixed(2)} · ${hullCopy}${budgetCopy} · hollow-shell guard active`
      : hasReportWarning(latestResult, "INTERACTIVE_IMPORTANCE_GUARD")
        ? `Requested ${detail.toFixed(2)} · ${hullCopy}${budgetCopy} · scale-aware body detail active`
      : `Applied ${detail.toFixed(2)} · ${hullCopy}${budgetCopy}`;
    return;
  }
  thresholdStatus.className = "setting-status pending";
  thresholdStatus.textContent = `Detail ${detail.toFixed(2)} pending…`;
}

function applyThreshold(): void {
  if (thresholdTimer !== null) {
    window.clearTimeout(thresholdTimer);
    thresholdTimer = null;
  }
  if (!selectedFile) {
    updateThresholdStatus();
    return;
  }
  if (lastErrorCode === "NON_MANIFOLD") {
    updateThresholdStatus();
    return;
  }
  const detail = Number(threshold.value);
  if (appliedFile === selectedFile && appliedThreshold === detail && latestResult) {
    updateThresholdStatus();
    return;
  }
  void compileSelected(selectedFile);
}

function scheduleThresholdApply(): void {
  updateThresholdStatus();
  if (!selectedFile) return;
  activeController?.abort();
  if (thresholdTimer !== null) window.clearTimeout(thresholdTimer);
  thresholdTimer = window.setTimeout(() => {
    thresholdTimer = null;
    applyThreshold();
  }, 350);
}

function chooseFiles(files: FileList | File[]): void {
  const file = files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".glb")) {
    showError(
      new ChitinError("INVALID_GLB", "Choose a binary glTF file with a .glb extension.", {
        stage: "reading-input",
        suggestion: "Export the model as a self-contained GLB 2.0 file.",
      }),
    );
    return;
  }
  void compileSelected(file);
}

function download(buffer: ArrayBuffer, name: string, type: string): void {
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = URL.createObjectURL(new Blob([buffer], { type }));
  const anchor = document.createElement("a");
  anchor.href = downloadUrl;
  anchor.download = name;
  anchor.click();
}

const verifiedSamples = {
  wicker: () => fetchSample("./assets/clearcoat-wicker.glb", "clearcoat-wicker.glb"),
  dish: () => fetchSample("./assets/iridescent-dish-with-olives.glb", "iridescent-dish-with-olives.glb"),
  fish: () => fetchSample("./assets/barramundi-fish.glb", "barramundi-fish.glb"),
} satisfies Record<string, () => Promise<File>>;

async function fetchSample(url: string, name: string): Promise<File> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (cause) {
    throw new ChitinError("LOAD_ERROR", `Could not load the ${name} sample.`, {
      stage: "reading-input",
      suggestion: "Reload the demo and try the sample again.",
      retryable: true,
      cause,
    });
  }
  if (!response.ok) {
    throw new ChitinError("LOAD_ERROR", `Could not load the ${name} sample (HTTP ${response.status}).`, {
      stage: "reading-input",
      suggestion: "Reload the demo and try the sample again.",
      retryable: response.status >= 500,
      context: { http_status: response.status, url },
    });
  }
  return new File([await response.arrayBuffer()], name, { type: "model/gltf-binary" });
}

async function loadVerifiedSample(name: keyof typeof verifiedSamples): Promise<void> {
  try {
    const file = await verifiedSamples[name]();
    await compileSelected(file);
  } catch (error) {
    showError(error);
  }
}

fileButton.addEventListener("click", () => fileInput.click());
replaceButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files) chooseFiles(fileInput.files);
  fileInput.value = "";
});
sampleGrid.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-sample]");
  if (!button) return;
  const sample = button.dataset.sample;
  if (!sample || !(sample in verifiedSamples)) return;
  void loadVerifiedSample(sample as keyof typeof verifiedSamples);
});
retryButton.addEventListener("click", () => {
  if (lastErrorCode === "NON_MANIFOLD") fileInput.click();
  else if (selectedFile) void compileSelected(selectedFile);
});
cancelButton.addEventListener("click", () => activeController?.abort());
threshold.addEventListener("input", () => {
  scheduleThresholdApply();
});
threshold.addEventListener("change", applyThreshold);
showSource.addEventListener("change", updatePreviewLayers);
showColliders.addEventListener("change", updatePreviewLayers);
explodeColliders.addEventListener("change", updateExplosionControls);
explodeDistance.addEventListener("input", updateExplosionControls);
downloadButton.addEventListener("click", () => {
  if (!latestResult || !selectedFile) return;
  const name = selectedFile.name.replace(/\.glb$/i, "") + ".phys";
  download(latestResult.phys, name, "application/octet-stream");
});
reportButton.addEventListener("click", () => {
  const opening = reportOutput.hidden;
  reportOutput.hidden = !opening;
  reportButton.setAttribute("aria-expanded", String(opening));
  reportButton.textContent = opening ? "Hide report" : "View report";
});

let dragDepth = 0;
window.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth++;
  viewportPanel.classList.add("dragging");
});
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("dragleave", (event) => {
  event.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) viewportPanel.classList.remove("dragging");
});
window.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  viewportPanel.classList.remove("dragging");
  if (event.dataTransfer?.files) chooseFiles(event.dataTransfer.files);
});

const resizeObserver = new ResizeObserver(([entry]) => {
  const width = Math.max(1, entry.contentRect.width);
  const height = Math.max(1, entry.contentRect.height);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
});
resizeObserver.observe(viewportPanel);

function animate(): void {
  controls.update();
  updateColliderReveal(performance.now());
  updateColliderExplosion();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
setRuntime("ready", "Runtime ready");

declare global {
  interface Window {
    __chitinDemo: {
      ready: boolean;
      state(): {
        busy: boolean;
        hulls: number;
        verdict: string | null;
        reportVersion: number | null;
        appliedThreshold: number | null;
        sourceVisible: boolean;
        colliderVisible: boolean;
        sourceMeshes: number;
        sourceFilledMeshes: number;
        colliderRevealActive: boolean;
        colliderRevealCount: number;
        exploded: boolean;
        explosionAmount: number;
        reusedComponents: number;
      };
    };
  }
}

window.__chitinDemo = {
  ready: true,
  state: () => {
    let sourceMeshes = 0;
    let sourceFilledMeshes = 0;
    sourceRoot.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      sourceMeshes++;
      if (!Array.isArray(object.material) || object.geometry.groups.length > 0) sourceFilledMeshes++;
    });
    return {
      busy: activeCompile !== null,
      hulls: latestResult?.hulls.length ?? 0,
      verdict: latestResult?.report.verdict.status ?? null,
      reportVersion: latestResult?.report.report_version ?? null,
      appliedThreshold,
      sourceVisible: sourceRoot.visible,
      colliderVisible: colliderRoot.visible,
      sourceMeshes,
      sourceFilledMeshes,
      colliderRevealActive: colliderReveal !== null,
      colliderRevealCount,
      exploded: colliderExplosionCurrent > 0.001,
      explosionAmount: colliderExplosionCurrent,
      reusedComponents: latestResult?.reuse.component_results ?? 0,
    };
  },
};
