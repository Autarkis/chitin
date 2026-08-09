import { ChitinError } from "./errors.js";
import type { CompilationStage } from "./report.js";

export type GlbInput = ArrayBuffer | ArrayBufferView | Blob | URL | string;

function cancelled(stage: CompilationStage, message: string): ChitinError {
  return new ChitinError("CANCELLED", message, {
    stage,
    suggestion: "Start a new compilation when ready.",
    retryable: true,
  });
}

export function throwIfAborted(
  signal: AbortSignal | undefined,
  stage: CompilationStage,
): void {
  if (signal?.aborted) throw cancelled(stage, "compilation aborted by caller");
}

export function mergeSignals(
  caller: AbortSignal | undefined,
  lifecycle: AbortSignal,
): { signal: AbortSignal; cleanup: () => void } {
  if (!caller) return { signal: lifecycle, cleanup: () => {} };
  const controller = new AbortController();
  const abort = () => controller.abort();
  if (caller.aborted || lifecycle.aborted) abort();
  else {
    caller.addEventListener("abort", abort, { once: true });
    lifecycle.addEventListener("abort", abort, { once: true });
  }
  return {
    signal: controller.signal,
    cleanup: () => {
      caller.removeEventListener("abort", abort);
      lifecycle.removeEventListener("abort", abort);
    },
  };
}

function copyView(input: ArrayBufferView): ArrayBuffer {
  return new Uint8Array(input.buffer, input.byteOffset, input.byteLength).slice().buffer;
}

export async function readInput(
  input: GlbInput,
  signal?: AbortSignal,
): Promise<{ buffer: ArrayBuffer; source: string | null }> {
  throwIfAborted(signal, "reading-input");
  if (input instanceof ArrayBuffer) return { buffer: input.slice(0), source: null };
  if (ArrayBuffer.isView(input)) return { buffer: copyView(input), source: null };
  if (typeof Blob !== "undefined" && input instanceof Blob) {
    try {
      const buffer = await input.arrayBuffer();
      throwIfAborted(signal, "reading-input");
      return {
        buffer,
        source: typeof File !== "undefined" && input instanceof File ? input.name : null,
      };
    } catch (cause) {
      if (signal?.aborted) throw cancelled("reading-input", "input read aborted by caller");
      throw new ChitinError("LOAD_ERROR", "could not read GLB blob", {
        stage: "reading-input",
        retryable: true,
        cause,
      });
    }
  }

  const url = input instanceof URL ? input.href : String(input);
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (cause) {
    if (signal?.aborted) throw cancelled("reading-input", "GLB fetch aborted by caller");
    throw new ChitinError("LOAD_ERROR", `could not fetch GLB from ${url}`, {
      stage: "reading-input",
      suggestion: "Check the URL, network connection, and server CORS response.",
      retryable: true,
      context: { url },
      cause,
    });
  }
  if (!response.ok) {
    throw new ChitinError("LOAD_ERROR", `GLB request failed with HTTP ${response.status}`, {
      stage: "reading-input",
      suggestion: "Check that the URL exists and is accessible to this browser.",
      retryable: response.status >= 500,
      context: { url, http_status: response.status },
    });
  }
  try {
    const buffer = await response.arrayBuffer();
    throwIfAborted(signal, "reading-input");
    return { buffer, source: url };
  } catch (cause) {
    if (signal?.aborted) throw cancelled("reading-input", "GLB fetch aborted by caller");
    throw new ChitinError("LOAD_ERROR", `could not read the GLB response from ${url}`, {
      stage: "reading-input",
      retryable: true,
      context: { url },
      cause,
    });
  }
}
