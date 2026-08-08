import {
  ChitinCompiler,
  ChitinError,
  type CompilationProgress,
  type CompileGlbResult,
} from "@autarkis/chitin-lite";
import ChitinWorker from "@autarkis/chitin-lite/worker?worker";
import coacdModuleUrl from "@autarkis/chitin-coacd-wasm?url";
import coacdWasmUrl from "@autarkis/chitin-coacd-wasm/coacd.wasm?url";

import type { ChitinDemoApi } from "./demo-api";
import { PreviewController } from "./preview-controller";
import {
  appliedThresholdCopy,
  hasQualityDiagnostics,
  metricPercentCopy,
  resultSummaryCopy,
} from "./result-presentation";

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
const fitPresets = $("#fit-presets");
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
const qualityButton = $("#quality-button") as HTMLButtonElement;
const qualityMetrics = $("#quality-metrics");
const measurementNote = $("#measurement-note");
const surfaceCoverage = $("#surface-coverage");
const volumePrecision = $("#volume-precision");
const falseFill = $("#false-fill");
const deepFalseFill = $("#deep-false-fill");
const reportPanel = $("#report-panel");
const reportOutput = $("#report-output") as HTMLPreElement;
const reportStatus = $("#report-status");
const reportProfile = $("#report-profile");
const reportVerdict = $("#report-verdict");
const reportWarnings = $("#report-warnings");
const reportChecks = $("#report-checks");
const showSource = $("#show-source") as HTMLInputElement;
const showColliders = $("#show-colliders") as HTMLInputElement;
const explodeColliders = $("#explode-colliders") as HTMLInputElement;
const explodeDistance = $("#explode-distance") as HTMLInputElement;

const previewController = new PreviewController({
  canvas,
  viewportPanel,
  showSource,
  showColliders,
  explodeColliders,
  explodeDistance,
});

const compiler = new ChitinCompiler({
  wasm: {
    js: coacdModuleUrl,
    wasm: coacdWasmUrl,
    version: "0.2.0",
  },
  workerFactory: () => new ChitinWorker(),
  maxWorkers: 2,
});

type CompileIntent = "artifact" | "quality";

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
let lastFailedIntent: CompileIntent = "artifact";

const stageProgress: Record<CompilationProgress["stage"], number> = {
  "reading-input": 8,
  "parsing-input": 16,
  "validating-input": 24,
  "loading-wasm": 30,
  decomposing: 30,
  verifying: 90,
  "writing-phys": 96,
  done: 100,
};

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

function updateProgress(progress: CompilationProgress): void {
  let percentage = stageProgress[progress.stage];
  if (progress.stage === "decomposing" && progress.total && progress.completed !== undefined) {
    percentage = Math.round(30 + 55 * (progress.completed / progress.total));
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

function displayStatus(value: string | null): string {
  if (!value) return "None";
  return value
    .split("_")
    .map((part, index) => index === 0
      ? part.charAt(0).toUpperCase() + part.slice(1)
      : part)
    .join(" ");
}

function renderReportItems(
  target: HTMLElement,
  items: Array<{ code: string; message: string; tone: string }>,
  emptyCopy: string,
): void {
  target.replaceChildren();
  if (items.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty";
    empty.textContent = emptyCopy;
    target.append(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("li");
    row.className = item.tone;
    const code = document.createElement("strong");
    code.textContent = item.code;
    const message = document.createElement("span");
    message.textContent = item.message;
    row.append(code, message);
    target.append(row);
  }
}

function renderReport(result: CompileGlbResult): void {
  const { report } = result;
  reportStatus.textContent = displayStatus(report.status);
  reportProfile.textContent = displayStatus(report.profile);
  reportVerdict.textContent = displayStatus(report.verdict.status);
  renderReportItems(
    reportWarnings,
    report.warnings.map((warning) => ({
      code: warning.code,
      message: warning.message,
      tone: warning.severity,
    })),
    "No compiler warnings.",
  );
  renderReportItems(
    reportChecks,
    report.verdict.checks.map((check) => ({
      code: check.code,
      message: check.message,
      tone: check.status,
    })),
    report.verdict.status === "not_evaluated"
      ? "Profile acceptance checks were not run."
      : "No profile checks were reported.",
  );
  reportOutput.textContent = JSON.stringify(report, null, 2);
}

function resetQualityPresentation(): void {
  qualityMetrics.hidden = true;
  surfaceCoverage.textContent = "—";
  volumePrecision.textContent = "—";
  falseFill.textContent = "—";
  deepFalseFill.textContent = "—";
  measurementNote.textContent =
    "Complexity is measured. Run local sampling to check geometric fit and coverage.";
  qualityButton.disabled = false;
  qualityButton.textContent = "Run quality diagnostics";
}

function renderQualityPresentation(result: CompileGlbResult): void {
  if (!hasQualityDiagnostics(result)) {
    resetQualityPresentation();
    return;
  }
  qualityMetrics.hidden = false;
  surfaceCoverage.textContent = metricPercentCopy(result, "source_surface_coverage");
  volumePrecision.textContent = metricPercentCopy(result, "collider_volume_precision");
  falseFill.textContent = metricPercentCopy(result, "false_fill_fraction");
  deepFalseFill.textContent = metricPercentCopy(result, "deep_false_fill_fraction");
  measurementNote.textContent =
    "Sampled locally against this artifact. Profile acceptance checks remain not evaluated.";
  qualityButton.disabled = false;
  qualityButton.textContent = "Rerun quality diagnostics";
}

function updatePresetState(): void {
  const detail = Number(threshold.value);
  for (const button of fitPresets.querySelectorAll<HTMLButtonElement>("button[data-detail]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.detail) === detail));
  }
}

function resetOutput(): void {
  errorCard.hidden = true;
  if (lastErrorCode === "NON_MANIFOLD") showColliders.checked = true;
  lastErrorCode = null;
  retryButton.textContent = "Try again";
  resultSection.hidden = true;
  reportPanel.hidden = true;
  reportButton.setAttribute("aria-expanded", "false");
  reportButton.textContent = "View report";
  resetQualityPresentation();
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
  latestResult = null;
  appliedThreshold = null;
  appliedFile = null;
  previewController.clearColliders();
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
  lastFailedIntent = "artifact";
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
    previewController.updateLayers();
    thresholdStatus.className = "setting-status pending";
    thresholdStatus.textContent = "Needs full-compiler repair · no collider produced";
    setRuntime("error", "Source ready · collider unavailable");
  } else {
    setRuntime("error", "Compile failed");
  }
}

function showQualityError(error: unknown): void {
  const info = error instanceof ChitinError
    ? error.toInfo()
    : {
        code: "QUALITY_ERROR",
        message: error instanceof Error ? error.message : String(error),
        suggestion: "Retry diagnostics or keep the compiled artifact without sampled metrics.",
      };
  lastFailedIntent = "quality";
  lastErrorCode = info.code;
  errorCode.textContent = "QUALITY_DIAGNOSTICS_FAILED";
  errorMessage.textContent = info.message;
  errorSuggestion.textContent = info.suggestion ?? "Retry quality diagnostics.";
  retryButton.textContent = "Retry diagnostics";
  errorCard.hidden = false;
  measurementNote.textContent =
    "The artifact remains available, but local quality sampling did not complete.";
  qualityMetrics.hidden = true;
  qualityButton.disabled = false;
  qualityButton.textContent = "Retry quality diagnostics";
  setRuntime("error", "Artifact ready · diagnostics failed");
}

function showResult(result: CompileGlbResult, detail: number, file: File): void {
  latestResult = result;
  appliedThreshold = detail;
  appliedFile = file;
  resultSection.hidden = false;
  resultSummary.textContent = resultSummaryCopy(result, detail);
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
  renderQualityPresentation(result);
  renderReport(result);
  updateThresholdStatus();
  setRuntime("ready", "Artifact ready");
  resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function compileSelected(
  file: File,
  intent: CompileIntent = "artifact",
): Promise<void> {
  const compileThreshold = Number(threshold.value);
  const qualityRequested = intent === "quality" || qualityBenchmarkEnabled;
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
    resultSummary.textContent = intent === "quality"
      ? `Detail ${compileThreshold.toFixed(2)} applied · running quality diagnostics`
      : `Showing detail ${appliedThreshold?.toFixed(2) ?? "—"} · compiling ${compileThreshold.toFixed(2)}`;
  } else {
    resetOutput();
  }
  progressSection.hidden = false;
  progressBar.style.width = "4%";
  progressCopy.textContent = intent === "quality"
    ? "Preparing quality diagnostics"
    : "Preparing compilation";
  cancelButton.disabled = false;
  setRuntime("busy", intent === "quality" ? "Measuring collider fit" : "Compiling locally");
  thresholdStatus.className = "setting-status pending";
  thresholdStatus.textContent = intent === "quality"
    ? `Applied ${compileThreshold.toFixed(2)} · measuring geometric fit…`
    : `Compiling detail ${compileThreshold.toFixed(2)}…`;
  if (intent === "quality") {
    qualityMetrics.hidden = true;
    measurementNote.textContent =
      "Sampling source coverage and collider fill locally. The artifact remains available.";
    qualityButton.disabled = true;
    qualityButton.textContent = "Running quality diagnostics…";
  }
  const controller = new AbortController();
  activeController = controller;

  const work = (async () => {
    let previewLoad: Promise<void> | null = null;
    try {
      if (!preservesPreviousArtifact || !previewController.hasSource()) {
        previewLoad = previewController.showSourcePreview(file).catch((error) => {
          console.warn("Source preview unavailable", error);
        });
      }
      if (intent === "quality") {
        await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      }
      const result = await compiler.compileGlb(file, {
        profile: "interactive",
        signal: controller.signal,
        decompose: { threshold: compileThreshold },
        checkManifold: true,
        quality: qualityRequested
          ? { surfaceSamples: 2048, volumeSamples: 8192 }
          : false,
        onProgress: updateProgress,
      });
      await previewLoad;
      if (ownRequest !== requestNumber) return;
      if (intent === "artifact") previewController.showColliderPreview(result);
      showResult(result, compileThreshold, file);
    } catch (error) {
      await previewLoad;
      if (ownRequest !== requestNumber) return;
      if (error instanceof ChitinError && error.code === "CANCELLED") {
        setRuntime("ready", latestResult ? "Previous artifact ready" : "Compilation cancelled");
        progressCopy.textContent = latestResult
          ? "Cancelled — previous collider remains applied"
          : "Cancelled — adjust settings or try again";
        if (latestResult) renderQualityPresentation(latestResult);
        updateThresholdStatus();
      } else if (intent === "quality" && latestResult) {
        lastFailedIntent = intent;
        showQualityError(error);
      } else {
        lastFailedIntent = intent;
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
  updatePresetState();
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
    thresholdStatus.textContent = appliedThresholdCopy(latestResult, detail);
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
fitPresets.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-detail]");
  const detail = button?.dataset.detail;
  if (!detail) return;
  threshold.value = detail;
  updateThresholdStatus();
  applyThreshold();
});
retryButton.addEventListener("click", () => {
  if (lastErrorCode === "NON_MANIFOLD") fileInput.click();
  else if (selectedFile) void compileSelected(selectedFile, lastFailedIntent);
});
cancelButton.addEventListener("click", () => activeController?.abort());
threshold.addEventListener("input", () => {
  scheduleThresholdApply();
});
threshold.addEventListener("change", applyThreshold);
showSource.addEventListener("change", () => previewController.updateLayers());
showColliders.addEventListener("change", () => previewController.updateLayers());
explodeColliders.addEventListener("change", () => previewController.updateExplosionControls());
explodeDistance.addEventListener("input", () => previewController.updateExplosionControls());
downloadButton.addEventListener("click", () => {
  if (!latestResult || !selectedFile) return;
  const name = selectedFile.name.replace(/\.glb$/i, "") + ".phys";
  download(latestResult.phys, name, "application/octet-stream");
});
qualityButton.addEventListener("click", () => {
  if (!selectedFile || !latestResult || appliedThreshold === null) return;
  if (thresholdTimer !== null) {
    window.clearTimeout(thresholdTimer);
    thresholdTimer = null;
  }
  threshold.value = appliedThreshold.toFixed(2);
  updateThresholdStatus();
  void compileSelected(selectedFile, "quality");
});
reportButton.addEventListener("click", () => {
  const opening = reportPanel.hidden;
  reportPanel.hidden = !opening;
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

window.addEventListener("pagehide", (event) => {
  if (event.persisted) return;
  activeController?.abort();
  compiler.terminate();
  previewController.dispose();
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
});

setRuntime("ready", "Runtime ready");

declare global {
  interface Window {
    __chitinDemo: ChitinDemoApi;
  }
}

window.__chitinDemo = {
  ready: true,
  state: () => ({
    ...previewController.state(),
    busy: activeCompile !== null,
    hulls: latestResult?.hulls.length ?? 0,
    verdict: latestResult?.report.verdict.status ?? null,
    reportVersion: latestResult?.report.report_version ?? null,
    appliedThreshold,
    qualityMeasured: latestResult ? hasQualityDiagnostics(latestResult) : false,
    reusedComponents: latestResult?.reuse.component_results ?? 0,
  }),
};
