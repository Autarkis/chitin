import { ChitinError } from "./errors.js";

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;
const TRIANGLES = 4;
const FLOAT = 5126;
const UNSIGNED_BYTE = 5121;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const MAX_INT32_INDEX = 0x7fffffff;

type JsonObject = Record<string, unknown>;

interface GltfBuffer {
  byteLength?: number;
  uri?: string;
}

interface GltfBufferView {
  buffer?: number;
  byteOffset?: number;
  byteLength?: number;
  byteStride?: number;
  extensions?: JsonObject;
}

interface GltfSparse {
  count?: number;
  indices?: { bufferView?: number; byteOffset?: number; componentType?: number };
  values?: { bufferView?: number; byteOffset?: number };
}

interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType?: number;
  count?: number;
  type?: string;
  normalized?: boolean;
  sparse?: GltfSparse;
}

interface GltfPrimitive {
  attributes?: Record<string, number>;
  indices?: number;
  mode?: number;
  targets?: unknown[];
  extensions?: JsonObject;
}

interface GltfMesh {
  name?: string;
  primitives?: GltfPrimitive[];
}

interface GltfNode {
  name?: string;
  mesh?: number;
  skin?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

interface GltfDocument {
  asset?: { version?: string };
  scene?: number;
  scenes?: Array<{ nodes?: number[] }>;
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
}

interface PrimitiveGeometry {
  vertices: Float64Array;
  faces: Int32Array;
}

export interface ParsedGlbMesh {
  /** Active-scene vertices after node transforms, flat xyz. */
  vertices: Float64Array;
  /** Active-scene triangle indices, flat triples. */
  faces: Int32Array;
  /** Number of active-scene nodes that instantiate a mesh. */
  mesh_count: number;
  /** Number of triangle primitives after node instancing. */
  primitive_count: number;
  /** Number of active-scene node occurrences visited. */
  node_count: number;
}

function invalid(message: string, context: Record<string, string | number> = {}): never {
  throw new ChitinError("INVALID_GLB", message, {
    stage: "parsing-input",
    suggestion: "Export a self-contained binary glTF 2.0 (.glb) file.",
    context,
  });
}

function unsupported(
  message: string,
  context: Record<string, string | number> = {},
): never {
  throw new ChitinError("UNSUPPORTED_GLTF", message, {
    stage: "parsing-input",
    suggestion: "Bake the feature into static triangle geometry before compiling.",
    context,
  });
}

function integer(value: unknown, label: string, fallback?: number): number {
  if (value === undefined && fallback !== undefined) return fallback;
  if (!Number.isInteger(value) || (value as number) < 0) invalid(`${label} must be a non-negative integer`);
  return value as number;
}

function checkedEnd(offset: number, length: number, limit: number, label: string): number {
  const end = offset + length;
  if (!Number.isSafeInteger(end) || offset < 0 || length < 0 || end > limit) {
    invalid(`${label} exceeds its containing byte range`);
  }
  return end;
}

function decodeDataUri(uri: string): Uint8Array {
  const match = /^data:([^,]*?),(.*)$/s.exec(uri);
  if (!match) unsupported("GLB references an external buffer URI", { uri });
  try {
    if (match[1].split(";").includes("base64")) {
      const binary = atob(match[2]);
      return Uint8Array.from(binary, (character) => character.charCodeAt(0));
    }
    return new TextEncoder().encode(decodeURIComponent(match[2]));
  } catch (cause) {
    throw new ChitinError("INVALID_GLB", "could not decode a buffer data URI", {
      stage: "parsing-input",
      context: { uri_prefix: uri.slice(0, 48) },
      cause,
    });
  }
}

function componentSize(componentType: number): number {
  if (componentType === UNSIGNED_BYTE) return 1;
  if (componentType === UNSIGNED_SHORT) return 2;
  if (componentType === UNSIGNED_INT || componentType === FLOAT) return 4;
  unsupported(`accessor componentType ${componentType} is not supported`);
}

function readComponent(view: DataView, offset: number, componentType: number): number {
  if (componentType === UNSIGNED_BYTE) return view.getUint8(offset);
  if (componentType === UNSIGNED_SHORT) return view.getUint16(offset, true);
  if (componentType === UNSIGNED_INT) return view.getUint32(offset, true);
  if (componentType === FLOAT) return view.getFloat32(offset, true);
  unsupported(`accessor componentType ${componentType} is not supported`);
}

class AccessorReader {
  constructor(
    private readonly doc: GltfDocument,
    private readonly buffers: Uint8Array[],
  ) {}

  private accessor(index: number): GltfAccessor {
    const accessor = this.doc.accessors?.[index];
    if (!accessor) invalid(`accessor ${index} does not exist`, { accessor_index: index });
    return accessor;
  }

  private bufferView(index: number): { definition: GltfBufferView; bytes: Uint8Array } {
    const definition = this.doc.bufferViews?.[index];
    if (!definition) invalid(`bufferView ${index} does not exist`, { buffer_view_index: index });
    if (definition.extensions?.EXT_meshopt_compression) {
      unsupported("EXT_meshopt_compression buffer views are not supported", {
        buffer_view_index: index,
      });
    }
    const bufferIndex = integer(definition.buffer, `bufferView ${index}.buffer`, 0);
    const bytes = this.buffers[bufferIndex];
    if (!bytes) invalid(`bufferView ${index} references missing buffer ${bufferIndex}`);
    const byteOffset = integer(definition.byteOffset, `bufferView ${index}.byteOffset`, 0);
    const byteLength = integer(definition.byteLength, `bufferView ${index}.byteLength`);
    checkedEnd(byteOffset, byteLength, bytes.byteLength, `bufferView ${index}`);
    return { definition, bytes: bytes.subarray(byteOffset, byteOffset + byteLength) };
  }

  private readValues(index: number, expectedType: "VEC3" | "SCALAR"): Float64Array {
    const accessor = this.accessor(index);
    if (accessor.type !== expectedType) {
      invalid(`accessor ${index} must be ${expectedType}, got ${String(accessor.type)}`, {
        accessor_index: index,
      });
    }
    const componentType = integer(accessor.componentType, `accessor ${index}.componentType`);
    if (expectedType === "VEC3" && componentType !== FLOAT) {
      unsupported(`POSITION accessor ${index} must use FLOAT components`, {
        accessor_index: index,
      });
    }
    if (
      expectedType === "SCALAR" &&
      ![UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT].includes(componentType)
    ) {
      unsupported(`index accessor ${index} must use an unsigned integer component type`, {
        accessor_index: index,
      });
    }
    if (accessor.normalized) unsupported(`accessor ${index} cannot be normalized`);

    const count = integer(accessor.count, `accessor ${index}.count`);
    const width = expectedType === "VEC3" ? 3 : 1;
    const size = componentSize(componentType);
    const values = new Float64Array(count * width);
    if (accessor.bufferView !== undefined) {
      const { definition, bytes } = this.bufferView(
        integer(accessor.bufferView, `accessor ${index}.bufferView`),
      );
      const stride = integer(definition.byteStride, `accessor ${index} byteStride`, size * width);
      if (stride < size * width || stride % size !== 0) {
        invalid(`accessor ${index} has invalid byteStride ${stride}`);
      }
      const start = integer(accessor.byteOffset, `accessor ${index}.byteOffset`, 0);
      if (count > 0) checkedEnd(start + (count - 1) * stride, size * width, bytes.byteLength, `accessor ${index}`);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let element = 0; element < count; element++) {
        for (let component = 0; component < width; component++) {
          values[element * width + component] = readComponent(
            view,
            start + element * stride + component * size,
            componentType,
          );
        }
      }
    } else if (!accessor.sparse) {
      invalid(`accessor ${index} has neither a bufferView nor sparse values`);
    }

    if (accessor.sparse) this.applySparse(index, accessor.sparse, values, count, width, componentType);
    return values;
  }

  private applySparse(
    accessorIndex: number,
    sparse: GltfSparse,
    values: Float64Array,
    accessorCount: number,
    width: number,
    valueComponentType: number,
  ): void {
    const count = integer(sparse.count, `accessor ${accessorIndex}.sparse.count`);
    if (count > accessorCount) invalid(`accessor ${accessorIndex} sparse count exceeds accessor count`);
    const indicesDefinition = sparse.indices;
    const valuesDefinition = sparse.values;
    if (!indicesDefinition || !valuesDefinition) invalid(`accessor ${accessorIndex} has incomplete sparse data`);
    const indexComponentType = integer(
      indicesDefinition.componentType,
      `accessor ${accessorIndex}.sparse.indices.componentType`,
    );
    if (![UNSIGNED_BYTE, UNSIGNED_SHORT, UNSIGNED_INT].includes(indexComponentType)) {
      invalid(`accessor ${accessorIndex} has invalid sparse index component type`);
    }
    const indexSize = componentSize(indexComponentType);
    const indexViewData = this.bufferView(
      integer(indicesDefinition.bufferView, `accessor ${accessorIndex}.sparse.indices.bufferView`),
    ).bytes;
    const indexOffset = integer(
      indicesDefinition.byteOffset,
      `accessor ${accessorIndex}.sparse.indices.byteOffset`,
      0,
    );
    checkedEnd(indexOffset, count * indexSize, indexViewData.byteLength, `accessor ${accessorIndex} sparse indices`);
    const sparseValues = this.bufferView(
      integer(valuesDefinition.bufferView, `accessor ${accessorIndex}.sparse.values.bufferView`),
    ).bytes;
    const valueOffset = integer(
      valuesDefinition.byteOffset,
      `accessor ${accessorIndex}.sparse.values.byteOffset`,
      0,
    );
    const valueSize = componentSize(valueComponentType);
    checkedEnd(valueOffset, count * width * valueSize, sparseValues.byteLength, `accessor ${accessorIndex} sparse values`);
    const indexView = new DataView(indexViewData.buffer, indexViewData.byteOffset, indexViewData.byteLength);
    const valueView = new DataView(sparseValues.buffer, sparseValues.byteOffset, sparseValues.byteLength);
    for (let entry = 0; entry < count; entry++) {
      const target = readComponent(indexView, indexOffset + entry * indexSize, indexComponentType);
      if (target >= accessorCount) invalid(`accessor ${accessorIndex} sparse index ${target} is out of range`);
      for (let component = 0; component < width; component++) {
        values[target * width + component] = readComponent(
          valueView,
          valueOffset + (entry * width + component) * valueSize,
          valueComponentType,
        );
      }
    }
  }

  positions(index: number): Float64Array {
    const values = this.readValues(index, "VEC3");
    for (let i = 0; i < values.length; i++) {
      if (!Number.isFinite(values[i])) invalid(`POSITION accessor ${index} contains a non-finite value`);
    }
    return values;
  }

  indices(index: number): Int32Array {
    const values = this.readValues(index, "SCALAR");
    const indices = new Int32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      if (values[i] > MAX_INT32_INDEX) unsupported(`index ${values[i]} exceeds Chitin's signed 32-bit mesh limit`);
      indices[i] = values[i];
    }
    return indices;
  }
}

type Matrix4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

const IDENTITY: Matrix4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

function multiply(a: Matrix4, b: Matrix4): Matrix4 {
  const out = new Array<number>(16).fill(0) as Matrix4;
  for (let column = 0; column < 4; column++) {
    for (let row = 0; row < 4; row++) {
      for (let k = 0; k < 4; k++) out[column * 4 + row] += a[k * 4 + row] * b[column * 4 + k];
    }
  }
  return out;
}

function finiteTuple(values: number[] | undefined, length: number, fallback: number[], label: string): number[] {
  const tuple = values ?? fallback;
  if (tuple.length !== length || tuple.some((value) => !Number.isFinite(value))) invalid(`${label} must contain ${length} finite numbers`);
  return tuple;
}

function nodeMatrix(node: GltfNode, nodeIndex: number): Matrix4 {
  if (node.matrix !== undefined) {
    if (node.translation || node.rotation || node.scale) invalid(`node ${nodeIndex} combines matrix with TRS properties`);
    return finiteTuple(node.matrix, 16, [], `node ${nodeIndex}.matrix`) as Matrix4;
  }
  const [tx, ty, tz] = finiteTuple(node.translation, 3, [0, 0, 0], `node ${nodeIndex}.translation`);
  const [x, y, z, w] = finiteTuple(node.rotation, 4, [0, 0, 0, 1], `node ${nodeIndex}.rotation`);
  const [sx, sy, sz] = finiteTuple(node.scale, 3, [1, 1, 1], `node ${nodeIndex}.scale`);
  const length = Math.hypot(x, y, z, w);
  if (length === 0) invalid(`node ${nodeIndex}.rotation is a zero quaternion`);
  const qx = x / length;
  const qy = y / length;
  const qz = z / length;
  const qw = w / length;
  const xx = qx * qx;
  const yy = qy * qy;
  const zz = qz * qz;
  const xy = qx * qy;
  const xz = qx * qz;
  const yz = qy * qz;
  const wx = qw * qx;
  const wy = qw * qy;
  const wz = qw * qz;
  return [
    (1 - 2 * (yy + zz)) * sx,
    (2 * (xy + wz)) * sx,
    (2 * (xz - wy)) * sx,
    0,
    (2 * (xy - wz)) * sy,
    (1 - 2 * (xx + zz)) * sy,
    (2 * (yz + wx)) * sy,
    0,
    (2 * (xz + wy)) * sz,
    (2 * (yz - wx)) * sz,
    (1 - 2 * (xx + yy)) * sz,
    0,
    tx,
    ty,
    tz,
    1,
  ];
}

function determinant3(matrix: Matrix4): number {
  return (
    matrix[0] * (matrix[5] * matrix[10] - matrix[9] * matrix[6]) -
    matrix[4] * (matrix[1] * matrix[10] - matrix[9] * matrix[2]) +
    matrix[8] * (matrix[1] * matrix[6] - matrix[5] * matrix[2])
  );
}

function transformed(vertices: Float64Array, matrix: Matrix4): Float64Array {
  const output = new Float64Array(vertices.length);
  for (let index = 0; index < vertices.length; index += 3) {
    const x = vertices[index];
    const y = vertices[index + 1];
    const z = vertices[index + 2];
    output[index] = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    output[index + 1] = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    output[index + 2] = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
  }
  return output;
}

function parseContainer(buffer: ArrayBuffer): { doc: GltfDocument; binary: Uint8Array | null } {
  if (buffer.byteLength < 20) invalid("file is too short to be a GLB");
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== GLB_MAGIC) invalid("input does not have the GLB magic header");
  if (view.getUint32(4, true) !== 2) unsupported(`GLB version ${view.getUint32(4, true)} is not supported`);
  const declaredLength = view.getUint32(8, true);
  if (declaredLength !== buffer.byteLength) invalid(`GLB declares ${declaredLength} bytes but received ${buffer.byteLength}`);
  let offset = 12;
  let json: Uint8Array | null = null;
  let binary: Uint8Array | null = null;
  while (offset < buffer.byteLength) {
    checkedEnd(offset, 8, buffer.byteLength, "GLB chunk header");
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    offset += 8;
    checkedEnd(offset, length, buffer.byteLength, "GLB chunk");
    const bytes = new Uint8Array(buffer, offset, length);
    if (type === JSON_CHUNK) {
      if (json) invalid("GLB contains multiple JSON chunks");
      json = bytes;
    } else if (type === BIN_CHUNK) {
      if (binary) invalid("GLB contains multiple BIN chunks");
      binary = bytes;
    }
    offset += length;
  }
  if (!json) invalid("GLB has no JSON chunk");
  let doc: GltfDocument;
  try {
    doc = JSON.parse(new TextDecoder().decode(json).replace(/[\u0000\u0020]+$/g, "")) as GltfDocument;
  } catch (cause) {
    throw new ChitinError("INVALID_GLB", "GLB JSON chunk is not valid JSON", {
      stage: "parsing-input",
      cause,
    });
  }
  if (doc.asset?.version !== "2.0") unsupported(`glTF asset.version ${String(doc.asset?.version)} is not supported`);
  return { doc, binary };
}

function resolveBuffers(doc: GltfDocument, binary: Uint8Array | null): Uint8Array[] {
  return (doc.buffers ?? []).map((definition, index) => {
    let bytes: Uint8Array;
    if (definition.uri !== undefined) bytes = decodeDataUri(definition.uri);
    else if (index === 0 && binary) bytes = binary;
    else invalid(`buffer ${index} has no embedded data`);
    const byteLength = integer(definition.byteLength, `buffer ${index}.byteLength`);
    if (bytes.byteLength < byteLength) invalid(`buffer ${index} is shorter than its declared byteLength`);
    return bytes.subarray(0, byteLength);
  });
}

function parseGlbUnchecked(buffer: ArrayBuffer): ParsedGlbMesh {
  const { doc, binary } = parseContainer(buffer);
  const buffers = resolveBuffers(doc, binary);
  const reader = new AccessorReader(doc, buffers);
  const nodes = doc.nodes ?? [];
  const meshes = doc.meshes ?? [];
  const scenes = doc.scenes ?? [];
  let roots: number[];
  if (scenes.length > 0) {
    const sceneIndex = integer(doc.scene, "scene", 0);
    const scene = scenes[sceneIndex];
    if (!scene) invalid(`scene ${sceneIndex} does not exist`);
    roots = scene.nodes ?? [];
  } else {
    const children = new Set((nodes.flatMap((node) => node.children ?? [])));
    roots = nodes.map((_, index) => index).filter((index) => !children.has(index));
  }

  const pieces: PrimitiveGeometry[] = [];
  let meshCount = 0;
  let primitiveCount = 0;
  let nodeCount = 0;

  function visit(rawNodeIndex: number, parent: Matrix4, ancestors: Set<number>): void {
    const nodeIndex = integer(rawNodeIndex, "scene node index");
    const node = nodes[nodeIndex];
    if (!node) invalid(`node ${nodeIndex} does not exist`, { node_index: nodeIndex });
    if (ancestors.has(nodeIndex)) invalid(`node hierarchy contains a cycle at node ${nodeIndex}`);
    nodeCount++;
    const world = multiply(parent, nodeMatrix(node, nodeIndex));
    if (node.skin !== undefined) unsupported(`skinned node ${nodeIndex} cannot be compiled as static geometry`, { node_index: nodeIndex });
    if (node.mesh !== undefined) {
      const meshIndex = integer(node.mesh, `node ${nodeIndex}.mesh`);
      const mesh = meshes[meshIndex];
      if (!mesh) invalid(`node ${nodeIndex} references missing mesh ${meshIndex}`);
      meshCount++;
      for (let primitiveIndex = 0; primitiveIndex < (mesh.primitives ?? []).length; primitiveIndex++) {
        const primitive = mesh.primitives![primitiveIndex];
        const context = { mesh_index: meshIndex, primitive_index: primitiveIndex };
        if (primitive.extensions?.KHR_draco_mesh_compression) unsupported("Draco-compressed primitives are not supported", context);
        if (primitive.targets?.length) unsupported("morph-targeted primitives are not supported", context);
        if ((primitive.mode ?? TRIANGLES) !== TRIANGLES) unsupported("only TRIANGLES glTF primitives are supported", context);
        const positionAccessor = primitive.attributes?.POSITION;
        if (!Number.isInteger(positionAccessor)) invalid("primitive has no POSITION accessor", context);
        const sourceVertices = reader.positions(positionAccessor!);
        const vertexCount = sourceVertices.length / 3;
        let faces: Int32Array;
        if (primitive.indices === undefined) {
          if (vertexCount % 3 !== 0) invalid("unindexed TRIANGLES primitive vertex count is not divisible by 3", context);
          faces = Int32Array.from({ length: vertexCount }, (_, index) => index);
        } else {
          faces = reader.indices(integer(primitive.indices, "primitive.indices"));
          if (faces.length % 3 !== 0) invalid("triangle index count is not divisible by 3", context);
          for (const index of faces) if (index >= vertexCount) invalid(`primitive index ${index} exceeds its ${vertexCount} vertices`, context);
        }
        if (determinant3(world) < 0) {
          for (let index = 0; index < faces.length; index += 3) {
            const swap = faces[index + 1];
            faces[index + 1] = faces[index + 2];
            faces[index + 2] = swap;
          }
        }
        pieces.push({ vertices: transformed(sourceVertices, world), faces });
        primitiveCount++;
      }
    }
    const nextAncestors = new Set(ancestors).add(nodeIndex);
    for (const child of node.children ?? []) visit(child, world, nextAncestors);
  }

  for (const root of roots) visit(root, IDENTITY, new Set());
  if (pieces.length === 0) {
    throw new ChitinError("INVALID_MESH", "the active GLB scene contains no triangle geometry", {
      stage: "validating-input",
      suggestion: "Select or export a scene containing at least one triangle mesh.",
    });
  }
  const vertexCount = pieces.reduce((total, piece) => total + piece.vertices.length / 3, 0);
  const faceIndexCount = pieces.reduce((total, piece) => total + piece.faces.length, 0);
  if (vertexCount > MAX_INT32_INDEX) unsupported(`combined scene has ${vertexCount} vertices, exceeding Chitin's signed 32-bit mesh limit`);
  const vertices = new Float64Array(vertexCount * 3);
  const faces = new Int32Array(faceIndexCount);
  let vertexOffset = 0;
  let faceOffset = 0;
  for (const piece of pieces) {
    vertices.set(piece.vertices, vertexOffset * 3);
    for (let index = 0; index < piece.faces.length; index++) faces[faceOffset + index] = piece.faces[index] + vertexOffset;
    vertexOffset += piece.vertices.length / 3;
    faceOffset += piece.faces.length;
  }
  return { vertices, faces, mesh_count: meshCount, primitive_count: primitiveCount, node_count: nodeCount };
}

/** Parse static triangle geometry from the active scene of a self-contained GLB 2.0 file. */
export function parseGlb(buffer: ArrayBuffer): ParsedGlbMesh {
  try {
    return parseGlbUnchecked(buffer);
  } catch (cause) {
    if (cause instanceof ChitinError) throw cause;
    throw new ChitinError("INVALID_GLB", "GLB contains malformed glTF data", {
      stage: "parsing-input",
      suggestion: "Validate and re-export the file as self-contained binary glTF 2.0.",
      cause,
    });
  }
}
