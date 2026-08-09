// Format-only entry point: parses/validates .phys with no physics-engine or
// Three.js dependency. Import the Rapier bindings from "@autarkis/chitin-web/rapier"
// and the Three.js debug meshes from "@autarkis/chitin-web/three".
export { parsePhys, selectLodHulls } from "./phys-parser.js";
export type {
  PhysFile,
  PhysHull,
  PhysBone,
  PhysLodTier,
} from "./phys-parser.js";
export {
  applyBindPose,
  calculateVertexBounds,
  combinePhysGeometry,
  materializePhysHulls,
  resolvePhysTransform,
  transformHullVertices,
} from "./geometry.js";
export type {
  CombinedHullRange,
  CombinedPhysGeometry,
  MaterializedPhysGeometry,
  MaterializedPhysHull,
  MaterializeOptions,
  PhysBounds,
  PhysTransform,
  Quaternion,
  ResolvedPhysTransform,
  Vec3,
} from "./geometry.js";
