import {
  SplatMesh,
  dyno,
  type GsplatModifier,
  type SplatMeshOptions,
} from "@sparkjsdev/spark";
import {
  Box3,
  BufferGeometry,
  DataTexture,
  DepthTexture,
  DirectionalLight,
  FloatType,
  Group,
  LessEqualCompare,
  Material,
  Matrix4,
  Mesh,
  NearestFilter,
  PCFShadowMap,
  RGBAFormat,
  Texture,
  Triangle,
  UnsignedIntType,
  Vector3,
  WebGLRenderer,
  type Object3D,
} from "three";

import { materializePhysHulls, type PhysBounds } from "./geometry.js";
import { parsePhys, type PhysFile } from "./phys-parser.js";
import { createShadowProxy, type ShadowProxyOptions } from "./three.js";

export type SparkVisualSource = string | URL | SplatMesh | SplatMeshOptions;
export type PhysSource =
  string | URL | ArrayBuffer | ArrayBufferView | PhysFile;

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

export interface AttachSparkDirectionalShadowOptions {
  visual: SplatMesh;
  renderer: WebGLRenderer;
  light: DirectionalLight;
  /** Fraction of source RGB removed at full occlusion. Defaults to 0.65. */
  strength?: number;
  /** Shadow-depth bias. Null follows `light.shadow.bias`. Defaults to null. */
  bias?: number | null;
  /** Offset shadow lookup along the light-facing Gaussian normal. Defaults to 0. */
  normalBias?: number;
  /** Chitin `.phys` surface points/normals bound to the visual's splat order. */
  surface?: SparkPhysSurfaceBinding;
  /** Re-evaluate the receiver every frame for moving lights/casters. */
  dynamic?: boolean;
}

export interface SparkPhysSurfaceBinding {
  /** Local-space closest `.phys` surface point for each splat. */
  points: Float32Array;
  /** Local-space outward `.phys` surface normal for each splat. */
  normals: Float32Array;
}

export interface SparkDirectionalShadowReceiver {
  readonly visual: SplatMesh;
  readonly light: DirectionalLight;
  strength: number;
  bias: number | null;
  normalBias: number;
  dynamic: boolean;
  /** Refresh uniforms and re-evaluate splat visibility on the next Spark pass. */
  update(): void;
  dispose(): void;
}

type SparkOnFrame = NonNullable<SplatMesh["onFrame"]>;

interface SparkFrameHookState {
  original: SplatMesh["onFrame"];
  hooks: Set<SparkOnFrame>;
  wrapper: SparkOnFrame;
}

const sparkFrameHooks = new WeakMap<SplatMesh, SparkFrameHookState>();

interface SparkWorldToViewState {
  original: boolean;
  users: number;
}

const sparkWorldToViewUsers = new WeakMap<SplatMesh, SparkWorldToViewState>();

interface SparkSurfaceTextures {
  data: DataTexture;
  dataUniform: dyno.DynoUniform<"sampler2D", string, Texture>;
  widthUniform: dyno.DynoInt<string>;
  enabledUniform: dyno.DynoBool<string>;
}

/** Bind every Gaussian to its closest authoritative Chitin hull surface. */
export function bindSparkToPhysSurface(
  visual: SplatMesh,
  phys: PhysFile,
): SparkPhysSurfaceBinding {
  const count =
    visual.splats?.getNumSplats() ??
    visual.packedSplats?.getNumSplats() ??
    visual.numSplats;
  const points = new Float32Array(count * 3);
  const normals = new Float32Array(count * 3);
  if (phys.hulls.length === 0) {
    throw new RangeError(
      "bindSparkToPhysSurface: the .phys artifact contains no hulls",
    );
  }
  const hulls = phys.hulls.map((hull) => {
    const center = new Vector3(
      (hull.aabbMin[0] + hull.aabbMax[0]) * 0.5,
      (hull.aabbMin[1] + hull.aabbMax[1]) * 0.5,
      (hull.aabbMin[2] + hull.aabbMax[2]) * 0.5,
    );
    const triangles = [] as Array<{ triangle: Triangle; normal: Vector3 }>;
    for (let offset = 0; offset < hull.indices.length; offset += 3) {
      const ia = hull.indices[offset] * 3;
      const ib = hull.indices[offset + 1] * 3;
      const ic = hull.indices[offset + 2] * 3;
      const triangle = new Triangle(
        new Vector3(
          hull.vertices[ia],
          hull.vertices[ia + 1],
          hull.vertices[ia + 2],
        ),
        new Vector3(
          hull.vertices[ib],
          hull.vertices[ib + 1],
          hull.vertices[ib + 2],
        ),
        new Vector3(
          hull.vertices[ic],
          hull.vertices[ic + 1],
          hull.vertices[ic + 2],
        ),
      );
      const normal = triangle.getNormal(new Vector3());
      const triangleCenter = triangle.getMidpoint(new Vector3());
      if (normal.dot(triangleCenter.sub(center)) < 0) normal.multiplyScalar(-1);
      triangles.push({ triangle, normal });
    }
    return { hull, triangles };
  });
  const source = new Vector3();
  const closest = new Vector3();
  const triangleClosest = new Vector3();
  const candidateIndices = new Int32Array(3);
  const candidateDistances = new Float64Array(3);
  visual.forEachSplat((index, center) => {
    source.copy(center);
    let bestDistance = Infinity;
    let bestNormalX = 0;
    let bestNormalY = 1;
    let bestNormalZ = 0;
    candidateIndices.fill(-1);
    candidateDistances.fill(Infinity);
    for (let hullIndex = 0; hullIndex < hulls.length; hullIndex++) {
      const hull = hulls[hullIndex].hull;
      const distance = distanceToAabbSquared(
        source,
        hull.aabbMin,
        hull.aabbMax,
      );
      for (let slot = 0; slot < candidateIndices.length; slot++) {
        if (distance >= candidateDistances[slot]) continue;
        for (let move = candidateIndices.length - 1; move > slot; move--) {
          candidateIndices[move] = candidateIndices[move - 1];
          candidateDistances[move] = candidateDistances[move - 1];
        }
        candidateIndices[slot] = hullIndex;
        candidateDistances[slot] = distance;
        break;
      }
    }
    for (const candidateIndex of candidateIndices) {
      if (candidateIndex < 0) continue;
      for (const { triangle, normal } of hulls[candidateIndex].triangles) {
        triangle.closestPointToPoint(source, triangleClosest);
        const distance = source.distanceToSquared(triangleClosest);
        if (distance >= bestDistance) continue;
        bestDistance = distance;
        closest.copy(triangleClosest);
        bestNormalX = normal.x;
        bestNormalY = normal.y;
        bestNormalZ = normal.z;
      }
    }
    const output = index * 3;
    points[output] = closest.x;
    points[output + 1] = closest.y;
    points[output + 2] = closest.z;
    normals[output] = bestNormalX;
    normals[output + 1] = bestNormalY;
    normals[output + 2] = bestNormalZ;
  });
  return { points, normals };
}

function createSurfaceTextures(
  visual: SplatMesh,
  binding?: SparkPhysSurfaceBinding,
): SparkSurfaceTextures {
  const count =
    visual.splats?.getNumSplats() ??
    visual.packedSplats?.getNumSplats() ??
    visual.numSplats;
  if (
    binding !== undefined &&
    (binding.points.length !== count * 3 || binding.normals.length !== count * 3)
  ) {
    throw new RangeError(
      `attachSparkDirectionalShadow: surface binding must contain ${count} point/normal triplets`,
    );
  }
  const width = Math.max(1, Math.min(2048, count));
  const height = Math.max(1, Math.ceil(count / width));
  const data = new Float32Array(width * height * 4);
  for (let index = 0; index < count; index++) {
    const source = index * 3;
    const target = index * 4;
    if (binding !== undefined) {
      data[target] = binding.points[source];
      data[target + 1] = binding.points[source + 1];
      data[target + 2] = binding.points[source + 2];
      data[target + 3] = encodeOctahedralNormal(
        binding.normals[source],
        binding.normals[source + 1],
        binding.normals[source + 2],
      );
    } else {
      data[target + 3] = encodeOctahedralNormal(0, 1, 0);
    }
  }
  const texture = new DataTexture(
    data,
    width,
    height,
    RGBAFormat,
    FloatType,
  );
  texture.minFilter = NearestFilter;
  texture.magFilter = NearestFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return {
    data: texture,
    dataUniform: new dyno.DynoUniform({
      type: "sampler2D" as const,
      value: texture,
    }),
    widthUniform: new dyno.DynoInt({ value: width }),
    enabledUniform: new dyno.DynoBool({ value: binding !== undefined }),
  };
}

function encodeOctahedralNormal(x: number, y: number, z: number): number {
  const inverseLength = 1 / Math.max(Math.abs(x) + Math.abs(y) + Math.abs(z), 1e-12);
  let octX = x * inverseLength;
  let octY = y * inverseLength;
  if (z < 0) {
    const previousX = octX;
    octX = (1 - Math.abs(octY)) * Math.sign(previousX || 1);
    octY = (1 - Math.abs(previousX)) * Math.sign(octY || 1);
  }
  const encodedX = Math.round((octX * 0.5 + 0.5) * 255);
  const encodedY = Math.round((octY * 0.5 + 0.5) * 255);
  return (encodedX + encodedY * 256) / 65535;
}

function distanceToAabbSquared(
  point: Vector3,
  minimum: [number, number, number],
  maximum: [number, number, number],
): number {
  let sum = 0;
  for (let axis = 0; axis < 3; axis++) {
    const value = point.getComponent(axis);
    const delta =
      value < minimum[axis]
        ? minimum[axis] - value
        : value > maximum[axis]
          ? value - maximum[axis]
          : 0;
    sum += delta * delta;
  }
  return sum;
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

/**
 * Make a Spark visual receive a Three.js directional-light shadow map.
 *
 * The light's Chitin hull proxies remain ordinary Three.js shadow casters. This
 * bridge samples Three's comparison depth texture at each world-space splat
 * center. Shadowing is gated by the view-facing Gaussian surface normal so an
 * occluder cannot project through a closed capture onto its far side. The
 * modifier composes with any Spark world modifiers already on the visual.
 */
export function attachSparkDirectionalShadow(
  options: AttachSparkDirectionalShadowOptions,
): SparkDirectionalShadowReceiver {
  const { visual, renderer, light } = options;
  if (!renderer.shadowMap.enabled) {
    throw new Error(
      "attachSparkDirectionalShadow: renderer.shadowMap.enabled must be true",
    );
  }
  if (renderer.shadowMap.type !== PCFShadowMap) {
    throw new Error(
      "attachSparkDirectionalShadow: renderer.shadowMap.type must be THREE.PCFShadowMap",
    );
  }
  if (!light.castShadow) {
    throw new Error(
      "attachSparkDirectionalShadow: the directional light must cast shadows",
    );
  }
  if (visual.covSplats) {
    throw new Error(
      "attachSparkDirectionalShadow: covariance splat pipelines are not supported yet",
    );
  }

  let strength = shadowStrength(options.strength ?? 0.65);
  let bias = shadowBias(options.bias ?? null);
  let normalBias = shadowNormalBias(options.normalBias ?? 0);
  let dynamic = options.dynamic ?? true;
  let disposed = false;

  // Three allocates the real depth texture lazily in its first shadow pass.
  // Keep a valid comparison sampler bound until that pass has completed.
  const placeholder = new DepthTexture(1, 1, UnsignedIntType);
  placeholder.compareFunction = LessEqualCompare;
  const mapUniform = new dyno.DynoUniform({
    type: "sampler2DShadow" as const,
    value: placeholder,
    // Dyno emits a uniform's globals before its declaration, which is where
    // GLSL ES requires the default sampler precision to be established.
    globals: () => ["precision highp sampler2DShadow;"],
  });
  const matrixUniform = new dyno.DynoMat4({ value: new Matrix4() });
  const strengthUniform = new dyno.DynoFloat({ value: strength });
  const biasUniform = new dyno.DynoFloat({ value: 0 });
  const normalBiasUniform = new dyno.DynoFloat({ value: normalBias });
  const lightDirectionUniform = new dyno.DynoVec3({ value: new Vector3() });
  const lightPosition = new Vector3();
  const lightTargetPosition = new Vector3();
  const surface = createSurfaceTextures(visual, options.surface);
  const surfaceMatrixUniform = new dyno.DynoMat4({ value: new Matrix4() });
  const releaseWorldToView = acquireSparkWorldToView(visual);
  const modifier = directionalShadowModifier({
    map: mapUniform,
    matrix: matrixUniform,
    strength: strengthUniform,
    bias: biasUniform,
    normalBias: normalBiasUniform,
    lightDirection: lightDirectionUniform,
    worldToView: visual.context.worldToView,
    surface,
    surfaceMatrix: surfaceMatrixUniform,
  });

  visual.worldModifiers = [...(visual.worldModifiers ?? []), modifier];
  visual.updateGenerator();

  const refresh = () => {
    if (disposed) return;
    const depthTexture = light.shadow.map?.depthTexture;
    if (depthTexture !== null && depthTexture !== undefined) {
      mapUniform.value = depthTexture;
    }
    matrixUniform.value.copy(light.shadow.matrix);
    strengthUniform.value = strength;
    biasUniform.value = bias ?? light.shadow.bias;
    normalBiasUniform.value = normalBias;
    lightDirectionUniform.value
      .copy(light.getWorldPosition(lightPosition))
      .sub(light.target.getWorldPosition(lightTargetPosition))
      .normalize();
    visual.updateWorldMatrix(true, false);
    surfaceMatrixUniform.value.copy(visual.matrixWorld);
    visual.updateVersion();
  };
  const removeFrameHook = addSparkFrameHook(visual, () => {
    if (dynamic) refresh();
  });
  refresh();

  return {
    visual,
    light,
    get strength() {
      return strength;
    },
    set strength(value: number) {
      strength = shadowStrength(value);
      refresh();
    },
    get bias() {
      return bias;
    },
    set bias(value: number | null) {
      bias = shadowBias(value);
      refresh();
    },
    get normalBias() {
      return normalBias;
    },
    set normalBias(value: number) {
      normalBias = shadowNormalBias(value);
      refresh();
    },
    get dynamic() {
      return dynamic;
    },
    set dynamic(value: boolean) {
      dynamic = value;
      refresh();
    },
    update: refresh,
    dispose() {
      if (disposed) return;
      disposed = true;
      const modifiers = visual.worldModifiers?.filter(
        (candidate) => candidate !== modifier,
      );
      visual.worldModifiers =
        modifiers === undefined || modifiers.length === 0
          ? undefined
          : modifiers;
      removeFrameHook();
      releaseWorldToView();
      visual.updateGenerator();
      surface.data.dispose();
      placeholder.dispose();
    },
  };
}

function addSparkFrameHook(visual: SplatMesh, hook: SparkOnFrame): () => void {
  let state = sparkFrameHooks.get(visual);
  if (state === undefined) {
    const original = visual.onFrame;
    const hooks = new Set<SparkOnFrame>();
    const wrapper: SparkOnFrame = (context) => {
      original?.(context);
      for (const candidate of hooks) candidate(context);
    };
    state = { original, hooks, wrapper };
    sparkFrameHooks.set(visual, state);
    visual.onFrame = wrapper;
  }
  state.hooks.add(hook);

  let removed = false;
  return () => {
    if (removed) return;
    removed = true;
    state.hooks.delete(hook);
    if (state.hooks.size !== 0) return;
    if (visual.onFrame === state.wrapper) visual.onFrame = state.original;
    sparkFrameHooks.delete(visual);
  };
}

function acquireSparkWorldToView(visual: SplatMesh): () => void {
  let state = sparkWorldToViewUsers.get(visual);
  if (state === undefined) {
    state = { original: visual.enableWorldToView, users: 0 };
    sparkWorldToViewUsers.set(visual, state);
  }
  state.users++;
  visual.enableWorldToView = true;

  let released = false;
  return () => {
    if (released) return;
    released = true;
    state.users--;
    if (state.users !== 0) return;
    visual.enableWorldToView = state.original;
    sparkWorldToViewUsers.delete(visual);
  };
}

function directionalShadowModifier({
  map,
  matrix,
  strength,
  bias,
  normalBias,
  lightDirection,
  worldToView,
  surface,
  surfaceMatrix,
}: {
  map: dyno.DynoUniform<"sampler2DShadow", string, Texture>;
  matrix: dyno.DynoMat4<string, Matrix4>;
  strength: dyno.DynoFloat;
  bias: dyno.DynoFloat;
  normalBias: dyno.DynoFloat;
  lightDirection: dyno.DynoVec3<Vector3>;
  worldToView: SplatMesh["context"]["worldToView"];
  surface: SparkSurfaceTextures;
  surfaceMatrix: dyno.DynoMat4<string, Matrix4>;
}): GsplatModifier {
  return dyno.dynoBlock(
    { gsplat: dyno.Gsplat },
    { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      if (gsplat === undefined) return undefined;
      const worldNormal = dyno.gsplatNormal(gsplat);
      const viewGsplat = worldToView.applyGsplat(gsplat);
      const viewCenter = dyno.splitGsplat(viewGsplat).outputs.center;
      const viewNormal = dyno.gsplatNormal(viewGsplat);
      const shaded = new dyno.Dyno({
        inTypes: {
          gsplat: dyno.Gsplat,
          shadowMap: "sampler2DShadow",
          shadowMatrix: "mat4",
          shadowStrength: "float",
          shadowBias: "float",
          shadowNormalBias: "float",
          lightDirection: "vec3",
          worldNormal: "vec3",
          viewCenter: "vec3",
          viewNormal: "vec3",
          surfaceData: "sampler2D",
          surfaceWidth: "int",
          usePhysSurface: "bool",
          surfaceMatrix: "mat4",
        },
        outTypes: { gsplat: dyno.Gsplat },
        inputs: {
          gsplat,
          shadowMap: map,
          shadowMatrix: matrix,
          shadowStrength: strength,
          shadowBias: bias,
          shadowNormalBias: normalBias,
          lightDirection,
          worldNormal,
          viewCenter,
          viewNormal,
          surfaceData: surface.dataUniform,
          surfaceWidth: surface.widthUniform,
          usePhysSurface: surface.enabledUniform,
          surfaceMatrix,
        },
        globals: () => [dyno.defineGsplat],
        statements: ({ inputs, outputs }) => {
          const source = inputs.gsplat;
          const target = outputs.gsplat;
          if (
            source === undefined ||
            target === undefined ||
            inputs.shadowMap === undefined ||
            inputs.shadowMatrix === undefined ||
            inputs.shadowStrength === undefined ||
            inputs.shadowBias === undefined ||
            inputs.shadowNormalBias === undefined ||
            inputs.lightDirection === undefined ||
            inputs.worldNormal === undefined ||
            inputs.viewCenter === undefined ||
            inputs.viewNormal === undefined ||
            inputs.surfaceData === undefined ||
            inputs.surfaceWidth === undefined ||
            inputs.usePhysSurface === undefined ||
            inputs.surfaceMatrix === undefined
          ) {
            return [];
          }
          return dyno.unindentLines(`
            ${target} = ${source};
            vec3 chitinSurfacePoint = ${source}.center;
            vec3 chitinSurfaceNormal = normalize(${inputs.worldNormal});
            if (${inputs.usePhysSurface}) {
              ivec2 chitinSurfaceCoord = ivec2(
                ${source}.index % ${inputs.surfaceWidth},
                ${source}.index / ${inputs.surfaceWidth}
              );
              vec4 chitinSurfaceData = texelFetch(
                ${inputs.surfaceData}, chitinSurfaceCoord, 0
              );
              vec3 chitinLocalSurfacePoint = chitinSurfaceData.xyz;
              uint chitinPackedNormal = uint(round(chitinSurfaceData.w * 65535.0));
              vec2 chitinOctNormal = vec2(
                float(chitinPackedNormal & 255u),
                float((chitinPackedNormal >> 8u) & 255u)
              ) / 255.0 * 2.0 - 1.0;
              vec3 chitinLocalSurfaceNormal = vec3(
                chitinOctNormal,
                1.0 - abs(chitinOctNormal.x) - abs(chitinOctNormal.y)
              );
              if (chitinLocalSurfaceNormal.z < 0.0) {
                chitinLocalSurfaceNormal.xy =
                  (1.0 - abs(chitinLocalSurfaceNormal.yx)) *
                  sign(chitinLocalSurfaceNormal.xy);
              }
              chitinLocalSurfaceNormal = normalize(chitinLocalSurfaceNormal);
              chitinSurfacePoint = (
                ${inputs.surfaceMatrix} * vec4(chitinLocalSurfacePoint, 1.0)
              ).xyz;
              chitinSurfaceNormal = normalize(
                transpose(inverse(mat3(${inputs.surfaceMatrix}))) *
                chitinLocalSurfaceNormal
              );
            }
            vec3 chitinLightDirection = normalize(${inputs.lightDirection});
            vec3 chitinLightNormal = dot(chitinSurfaceNormal, chitinLightDirection) >= 0.0
              ? chitinSurfaceNormal
              : -chitinSurfaceNormal;
            vec3 chitinShadowCenter = chitinSurfacePoint +
              chitinLightNormal * max(${inputs.shadowNormalBias}, 0.0);
            vec4 chitinShadowCoord4 = ${inputs.shadowMatrix} * vec4(chitinShadowCenter, 1.0);
            vec3 chitinShadowCoord = chitinShadowCoord4.xyz / chitinShadowCoord4.w;
            float chitinVisibility = 1.0;
            if (
              chitinShadowCoord.x >= 0.0 && chitinShadowCoord.x <= 1.0 &&
              chitinShadowCoord.y >= 0.0 && chitinShadowCoord.y <= 1.0 &&
              chitinShadowCoord.z >= 0.0 && chitinShadowCoord.z <= 1.0
            ) {
              chitinVisibility = texture(
                ${inputs.shadowMap},
                vec3(chitinShadowCoord.xy, chitinShadowCoord.z + ${inputs.shadowBias})
              );
            }
            float chitinShadowAmount =
              clamp(${inputs.shadowStrength}, 0.0, 1.0) * (1.0 - chitinVisibility);
            vec3 chitinReceiverNormal = chitinSurfaceNormal;
            if (
              !${inputs.usePhysSurface} &&
              dot(${inputs.viewCenter}, ${inputs.viewNormal}) >= 0.0
            ) {
              chitinReceiverNormal = -chitinReceiverNormal;
            }
            float chitinReceiverFacing = smoothstep(
              -0.05,
              0.15,
              dot(chitinReceiverNormal, chitinLightDirection)
            );
            chitinShadowAmount *= chitinReceiverFacing;
            ${target}.rgba.rgb *= 1.0 - chitinShadowAmount;
          `);
        },
      }).outputs.gsplat;
      return { gsplat: shaded };
    },
  );
}

function shadowStrength(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(
      "attachSparkDirectionalShadow: strength must be between 0 and 1",
    );
  }
  return value;
}

function shadowBias(value: number | null): number | null {
  if (value !== null && !Number.isFinite(value)) {
    throw new RangeError(
      "attachSparkDirectionalShadow: bias must be finite or null",
    );
  }
  return value;
}

function shadowNormalBias(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      "attachSparkDirectionalShadow: normalBias must be finite and non-negative",
    );
  }
  return value;
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
