import {
  ChitinCompiler,
  ChitinError,
  type BrowserProfileName,
  type CompilationProgress,
  type CompileGlbResult,
} from "@autarkis/chitin-lite";
import ChitinWorker from "@autarkis/chitin-lite/worker?worker";
import coacdModuleUrl from "@autarkis/chitin-wasm?url";
import coacdWasmUrl from "@autarkis/chitin-wasm/coacd.wasm?url";

import type { ChitinDemoApi } from "./demo-api";
import { NullPreviewController, PreviewController, type PreviewApi } from "./preview-controller";
import { SimulationController } from "./simulation-controller";
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
const progressTime = $("#progress-time");
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
const outputSize = $("#output-size");
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
const simulateButton = $("#simulate-button") as HTMLButtonElement;
const simulateStatus = $("#simulate-status") as HTMLParagraphElement;
const showSimulation = $("#show-simulation") as HTMLInputElement;
const profileSelector = $("#profile-selector");
const hullControls = $("#hull-controls");
const hullList = $("#hull-list");
const toggleHulls = $("#toggle-hulls") as HTMLButtonElement;
const codeSnippet = $("#code-snippet");
const copySnippet = $("#copy-snippet") as HTMLButtonElement;
const copyStatus = $("#copy-status");

let previewController: PreviewApi;
let previewAvailable = true;
try {
  previewController = new PreviewController({
    canvas,
    viewportPanel,
    showSource,
    showColliders,
    explodeColliders,
    explodeDistance,
    onTick: (time) => simulationController.tick(time),
  });
} catch {
  previewController = new NullPreviewController();
  canvas.hidden = true;
  previewAvailable = false;
}

const simulationController = new SimulationController();

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

interface ErrorDisplay {
  code: string;
  message: string;
  suggestion: string;
}

interface AppliedArtifact {
  result: CompileGlbResult;
  threshold: number;
}

type DemoPhase =
  | { kind: "idle" }
  | { kind: "compiling"; controller: AbortController; previous: AppliedArtifact | null }
  | ({ kind: "ready" } & AppliedArtifact)
  | { kind: "failed"; error: ErrorDisplay }
  | ({ kind: "quality-failed"; error: ErrorDisplay } & AppliedArtifact);

let phase: DemoPhase = { kind: "idle" };
let selectedFile: File | null = null;
let activeCompile: Promise<void> | null = null;
let requestNumber = 0;
let downloadUrl: string | null = null;
let thresholdTimer: number | null = null;
let simulationRequest = 0;
let selectedProfile: BrowserProfileName = "interactive";
let progressClock: number | null = null;

function phaseArtifact(): AppliedArtifact | null {
  if (phase.kind === "ready" || phase.kind === "quality-failed") return phase;
  if (phase.kind === "compiling") return phase.previous;
  return null;
}

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

function snippet(): string {
  return `import { ChitinCompiler } from "@autarkis/chitin-lite";\nimport ChitinWorker from "@autarkis/chitin-lite/worker?worker";\nimport coacdModuleUrl from "@autarkis/chitin-wasm?url";\nimport coacdWasmUrl from "@autarkis/chitin-wasm/coacd.wasm?url";\n\nconst compiler = new ChitinCompiler({\n  wasm: { js: coacdModuleUrl, wasm: coacdWasmUrl },\n  workerFactory: () => new ChitinWorker(),\n});\n\nconst result = await compiler.compileGlb(file, {\n  profile: "${selectedProfile}",\n  decompose: { threshold: ${Number(threshold.value).toFixed(2)} },\n});`;
}

function updateSnippet(): void {
  codeSnippet.textContent = snippet();
  copyStatus.textContent = "";
}

function renderHullControls(count: number): void {
  hullList.replaceChildren();
  hullControls.hidden = count === 0;
  for (let index = 0; index < count; index++) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.dataset.hull = String(index);
    label.append(input, ` Hull ${index + 1}`);
    hullList.append(label);
  }
  toggleHulls.textContent = "Hide all";
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
  simulationRequest++;
  errorCard.hidden = true;
  if (phase.kind === "failed" && phase.error.code === "NON_MANIFOLD") showColliders.checked = true;
  retryButton.textContent = "Try again";
  resultSection.hidden = true;
  reportPanel.hidden = true;
  reportButton.setAttribute("aria-expanded", "false");
  reportButton.textContent = "View report";
  resetQualityPresentation();
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = null;
  previewController.clearColliders();
  renderHullControls(0);
  simulationController.stop();
  previewController.setSimulationActive(false);
  simulateButton.textContent = "Test in Rapier";
  simulateButton.disabled = !previewAvailable;
  simulateStatus.hidden = true;
  showSimulation.checked = false;
  showSimulation.disabled = true;
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
  phase = {
    kind: "failed",
    error: {
      code: info.code,
      message: info.message,
      suggestion: info.suggestion ?? "Check the GLB and try again.",
    },
  };
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

function showQualityError(error: unknown, artifact: AppliedArtifact): void {
  const info = error instanceof ChitinError
    ? error.toInfo()
    : {
        code: "QUALITY_ERROR",
        message: error instanceof Error ? error.message : String(error),
        suggestion: "Retry diagnostics or keep the compiled artifact without sampled metrics.",
      };
  phase = {
    kind: "quality-failed",
    ...artifact,
    error: {
      code: info.code,
      message: info.message,
      suggestion: info.suggestion ?? "Retry quality diagnostics.",
    },
  };
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

function showResult(result: CompileGlbResult, detail: number): void {
  phase = { kind: "ready", result, threshold: detail };
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
  outputSize.textContent = formatBytes(result.phys.byteLength);
  renderHullControls(result.hulls.length);
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
  const qualityRequested = intent === "quality" || qualityBenchmarkEnabled || selectedProfile !== "interactive";
  const ownRequest = ++requestNumber;
  if (phase.kind === "compiling") phase.controller.abort();
  if (activeCompile) await activeCompile.catch(() => {});
  if (ownRequest !== requestNumber) return;

  const currentArtifact = phaseArtifact();
  const preservesPreviousArtifact = selectedFile === file && currentArtifact !== null;
  selectedFile = file;
  showFile(file);
  if (preservesPreviousArtifact) {
    errorCard.hidden = true;
    resultSummary.textContent = intent === "quality"
      ? `Detail ${compileThreshold.toFixed(2)} applied · running quality diagnostics`
      : `Showing detail ${currentArtifact?.threshold.toFixed(2) ?? "—"} · compiling ${compileThreshold.toFixed(2)}`;
  } else {
    resetOutput();
  }

  const previous = preservesPreviousArtifact ? currentArtifact : null;
  const controller = new AbortController();
  phase = { kind: "compiling", controller, previous };

  progressSection.hidden = false;
  const progressStarted = performance.now();
  if (progressClock !== null) window.clearInterval(progressClock);
  progressTime.textContent = "0 ms";
  progressClock = window.setInterval(() => {
    progressTime.textContent = `${Math.round(performance.now() - progressStarted)} ms`;
  }, 100);
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
        profile: selectedProfile,
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
      showResult(result, compileThreshold);
    } catch (error) {
      await previewLoad;
      if (ownRequest !== requestNumber) return;
      if (error instanceof ChitinError && error.code === "CANCELLED") {
        const prev = phase.kind === "compiling" ? phase.previous : null;
        if (prev) {
          phase = { kind: "ready", result: prev.result, threshold: prev.threshold };
          renderQualityPresentation(prev.result);
        } else {
          phase = { kind: "idle" };
        }
        setRuntime("ready", prev ? "Previous artifact ready" : "Compilation cancelled");
        progressCopy.textContent = prev
          ? "Cancelled — previous collider remains applied"
          : "Cancelled — adjust settings or try again";
        updateThresholdStatus();
      } else {
        const artifact = phaseArtifact();
        if (intent === "quality" && artifact) showQualityError(error, artifact);
        else showError(error);
      }
    } finally {
      if (ownRequest === requestNumber) {
        if (progressClock !== null) window.clearInterval(progressClock);
        progressClock = null;
        progressTime.textContent = `${Math.round(performance.now() - progressStarted)} ms`;
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
  updateSnippet();
  if (!selectedFile) {
    thresholdStatus.className = "setting-status";
    thresholdStatus.textContent = "Choose geometry to apply this setting.";
    return;
  }
  if (phase.kind === "failed" && phase.error.code === "NON_MANIFOLD") {
    thresholdStatus.className = "setting-status pending";
    thresholdStatus.textContent = "Needs full-compiler repair · no collider produced";
    return;
  }
  if ((phase.kind === "ready" || phase.kind === "quality-failed") && phase.threshold === detail) {
    thresholdStatus.className = "setting-status applied";
    thresholdStatus.textContent = appliedThresholdCopy(phase.result, detail);
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
  if (phase.kind === "failed" && phase.error.code === "NON_MANIFOLD") {
    updateThresholdStatus();
    return;
  }
  const detail = Number(threshold.value);
  if ((phase.kind === "ready" || phase.kind === "quality-failed") && phase.threshold === detail) {
    updateThresholdStatus();
    return;
  }
  void compileSelected(selectedFile);
}

function scheduleThresholdApply(): void {
  updateThresholdStatus();
  if (!selectedFile) return;
  if (phase.kind === "compiling") phase.controller.abort();
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
  if (phase.kind === "failed" && phase.error.code === "NON_MANIFOLD") fileInput.click();
  else if (selectedFile) {
    const retryIntent: CompileIntent = phase.kind === "quality-failed" ? "quality" : "artifact";
    void compileSelected(selectedFile, retryIntent);
  }
});
cancelButton.addEventListener("click", () => {
  if (phase.kind === "compiling") phase.controller.abort();
});
threshold.addEventListener("input", () => {
  scheduleThresholdApply();
});
threshold.addEventListener("change", applyThreshold);
showSource.addEventListener("change", () => previewController.updateLayers());
showColliders.addEventListener("change", () => previewController.updateLayers());
explodeColliders.addEventListener("change", () => previewController.updateExplosionControls());
explodeDistance.addEventListener("input", () => previewController.updateExplosionControls());
downloadButton.addEventListener("click", () => {
  const artifact = phaseArtifact();
  if (!artifact || !selectedFile) return;
  const name = selectedFile.name.replace(/\.glb$/i, "") + ".phys";
  download(artifact.result.phys, name, "application/octet-stream");
});
qualityButton.addEventListener("click", () => {
  const artifact = phaseArtifact();
  if (!selectedFile || !artifact) return;
  if (thresholdTimer !== null) {
    window.clearTimeout(thresholdTimer);
    thresholdTimer = null;
  }
  threshold.value = artifact.threshold.toFixed(2);
  updateThresholdStatus();
  void compileSelected(selectedFile, "quality");
});
reportButton.addEventListener("click", () => {
  const opening = reportPanel.hidden;
  reportPanel.hidden = !opening;
  reportButton.setAttribute("aria-expanded", String(opening));
  reportButton.textContent = opening ? "Hide report" : "View report";
});
simulateButton.addEventListener("click", async () => {
  const artifact = phaseArtifact();
  const scene = previewController.getScene();
  if (!artifact || !scene) return;
  const ownRequest = ++simulationRequest;
  simulateButton.disabled = true;
  simulateButton.textContent = "Initializing Rapier…";
  simulateStatus.hidden = false;
  simulateStatus.textContent = "Loading physics engine";
  try {
    await simulationController.start(artifact.result.phys, scene);
    if (ownRequest !== simulationRequest || phaseArtifact() !== artifact) {
      simulationController.stop();
      return;
    }
    previewController.setSimulationActive(true);
    simulateButton.textContent = "Restart simulation";
    simulateButton.disabled = false;
    simulateStatus.textContent = "Sphere dropped — watch the viewport";
    showSimulation.checked = true;
    showSimulation.disabled = false;
  } catch (error) {
    if (ownRequest !== simulationRequest) return;
    simulateButton.textContent = "Test in Rapier";
    simulateButton.disabled = false;
    simulateStatus.textContent = `Simulation failed: ${error instanceof Error ? error.message : String(error)}`;
  }
});
showSimulation.addEventListener("change", () => {
  const group = previewController.getScene()?.getObjectByName("simulation_objects");
  if (group) group.visible = showSimulation.checked;
});
profileSelector.addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-profile]");
  if (!button) return;
  selectedProfile = button.dataset.profile as BrowserProfileName;
  for (const btn of profileSelector.querySelectorAll<HTMLButtonElement>("button[data-profile]")) {
    btn.setAttribute("aria-pressed", String(btn === button));
  }
  updateSnippet();
  if (selectedFile) void compileSelected(selectedFile);
});
hullList.addEventListener("change", (event) => {
  const input = (event.target as HTMLElement).closest<HTMLInputElement>("input[data-hull]");
  if (!input) return;
  previewController.setHullVisible(Number(input.dataset.hull), input.checked);
  toggleHulls.textContent = [...hullList.querySelectorAll<HTMLInputElement>("input")].every((item) => item.checked)
    ? "Hide all" : "Show all";
});
toggleHulls.addEventListener("click", () => {
  const inputs = [...hullList.querySelectorAll<HTMLInputElement>("input")];
  const visible = !inputs.every((input) => input.checked);
  inputs.forEach((input, index) => {
    input.checked = visible;
    previewController.setHullVisible(index, visible);
  });
  toggleHulls.textContent = visible ? "Hide all" : "Show all";
});
copySnippet.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(snippet());
    copyStatus.textContent = "Copied to clipboard";
  } catch {
    copyStatus.textContent = "Clipboard unavailable — select the snippet manually";
  }
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
  if (phase.kind === "compiling") phase.controller.abort();
  compiler.terminate();
  previewController.dispose();
  simulationController.dispose();
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
  previewAvailable,
  state: () => {
    const artifact = phaseArtifact();
    const result = artifact?.result;
    return {
      ...previewController.state(),
      busy: phase.kind === "compiling",
      hulls: result?.hulls.length ?? 0,
      verdict: result?.report.verdict.status ?? null,
      reportVersion: result?.report.report_version ?? null,
      appliedThreshold: artifact?.threshold ?? null,
      qualityMeasured: result ? hasQualityDiagnostics(result) : false,
      reusedComponents: result?.reuse.component_results ?? 0,
      simulationActive: simulationController.isRunning(),
      simulationHeight: simulationController.height(),
      profile: selectedProfile,
    };
  },
};
