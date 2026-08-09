import { SplatMesh, type SplatMeshOptions } from "@sparkjsdev/spark";
import {
  Box3,
  BufferGeometry,
  Group,
  Material,
  Mesh,
  type Object3D,
} from "three";

import { materializePhysHulls, type PhysBounds } from "./geometry.js";
import { parsePhys, type PhysFile } from "./phys-parser.js";
import { createShadowProxy, type ShadowProxyOptions } from "./three.js";

export type SparkVisualSource = string | URL | SplatMesh | SplatMeshOptions;
export type PhysSource =
  | string
  | URL
  | ArrayBuffer
  | ArrayBufferView
  | PhysFile;

export interface LoadSparkAssetOptions {
  visual: SparkVisualSource;
  phys: PhysSource;
  shadows?: false | ShadowProxyOptions;
  signal?: AbortSignal;
  /** Override fetch for authenticated stores, tests, or non-browser runtimes. */
  fetch?: typeof globalThis.fetch;
  /** Dispose a caller-supplied SplatMesh with the asset. Defaults to false. */
  disposeSuppliedVisual?: boolean;
}

export interface SparkAssetBounds {
  visual: Box3;
  physics: PhysBounds | null;
}

export interface ChitinSparkAsset {
  root: Group;
  visual: SplatMesh;
  phys: PhysFile;
  shadowProxy: Object3D | null;
  bounds: SparkAssetBounds;
  dispose(): void;
}

/**
 * Pair a Spark visual with its engine-neutral `.phys` sidecar. Physics remains
 * opt-in through `@autarkis/chitin-web/rapier`; this adapter never creates a
 * body or imports Rapier.
 */
export async function loadSparkAsset(
  options: LoadSparkAssetOptions,
): Promise<ChitinSparkAsset> {
  throwIfAborted(options.signal);
  const suppliedVisual = options.visual instanceof SplatMesh;
  let visual: SplatMesh;
  if (suppliedVisual) {
    visual = options.visual as SplatMesh;
  } else {
    visual = new SplatMesh(
      normalizeVisualOptions(
        options.visual as Exclude<SparkVisualSource, SplatMesh>,
      ),
    );
  }
  const ownsVisual = !suppliedVisual || options.disposeSuppliedVisual === true;
  const originalParent = visual.parent;
  const originalPosition = visual.position.clone();
  const originalRotation = visual.quaternion.clone();
  const originalScale = visual.scale.clone();
  let shadowProxy: Object3D | null = null;

  try {
    const [, phys] = await Promise.all([
      withAbort(visual.initialized, options.signal),
      loadPhys(options.phys, options.fetch, options.signal),
    ]);
    throwIfAborted(options.signal);

    shadowProxy =
      options.shadows === undefined || options.shadows === false
        ? null
        : createShadowProxy(phys, options.shadows);

    const bounds: SparkAssetBounds = {
      visual: visual.getBoundingBox(),
      physics: materializePhysHulls(phys, {
        pose: "bind",
      }).bounds,
    };

    // The group is the canonical asset transform. Adopt an existing visual's
    // local TRS so the splat, shadow proxy, and separately-created collider all
    // share it instead of silently diverging.
    const root = new Group();
    root.name = "chitin_spark_asset";
    root.position.copy(originalPosition);
    root.quaternion.copy(originalRotation);
    root.scale.copy(originalScale);
    visual.position.set(0, 0, 0);
    visual.quaternion.identity();
    visual.scale.set(1, 1, 1);
    root.add(visual);
    if (shadowProxy !== null) root.add(shadowProxy);

    let disposed = false;
    return {
      root,
      visual,
      phys,
      shadowProxy,
      bounds,
      dispose() {
        if (disposed) return;
        disposed = true;
        root.remove(visual);
        if (shadowProxy !== null) {
          root.remove(shadowProxy);
          disposeThreeObject(shadowProxy);
        }
        if (ownsVisual) {
          visual.dispose();
        } else {
          visual.position.copy(originalPosition);
          visual.quaternion.copy(originalRotation);
          visual.scale.copy(originalScale);
          if (originalParent !== null) originalParent.add(visual);
        }
      },
    };
  } catch (error) {
    if (shadowProxy !== null) disposeThreeObject(shadowProxy);
    if (ownsVisual) visual.dispose();
    throw error;
  }
}

function normalizeVisualOptions(
  source: Exclude<SparkVisualSource, SplatMesh>,
): SplatMeshOptions {
  if (typeof source === "string" || source instanceof URL) {
    return { url: source.toString() };
  }
  return source;
}

async function loadPhys(
  source: PhysSource,
  fetchOverride?: typeof globalThis.fetch,
  signal?: AbortSignal,
): Promise<PhysFile> {
  if (isPhysFile(source)) return source;
  if (source instanceof ArrayBuffer) return parsePhys(source);
  if (ArrayBuffer.isView(source)) {
    const copy = new Uint8Array(source.byteLength);
    copy.set(
      new Uint8Array(source.buffer, source.byteOffset, source.byteLength),
    );
    return parsePhys(copy.buffer);
  }

  const fetcher = fetchOverride ?? globalThis.fetch;
  if (fetcher === undefined) {
    throw new Error("loadSparkAsset: fetch is unavailable for the .phys URL");
  }
  const response = await fetcher(source.toString(), { signal });
  if (!response.ok) {
    throw new Error(
      `loadSparkAsset: failed to fetch .phys (${response.status} ${response.statusText})`,
    );
  }
  return parsePhys(await response.arrayBuffer());
}

function isPhysFile(source: PhysSource): source is PhysFile {
  return (
    typeof source === "object" &&
    source !== null &&
    "hulls" in source &&
    "lodTiers" in source &&
    "version" in source
  );
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw abortError(signal.reason);
}

function withAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (signal === undefined) return promise;
  throwIfAborted(signal);
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortError(signal.reason));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

function abortError(reason: unknown): Error {
  if (reason instanceof Error) return reason;
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  return error;
}

function disposeThreeObject(root: Object3D): void {
  const geometries = new Set<BufferGeometry>();
  const materials = new Set<Material>();
  root.traverse((object) => {
    if (!(object instanceof Mesh)) return;
    geometries.add(object.geometry);
    const meshMaterials = Array.isArray(object.material)
      ? object.material
      : [object.material];
    for (const material of meshMaterials) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material.dispose();
}
