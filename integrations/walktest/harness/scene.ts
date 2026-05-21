import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";

interface WalkFrame {
  step: number;
  x: number;
  y: number;
  z: number;
  vy: number;
  grounded: boolean;
  stuck: boolean;
}

interface WalktestAPI {
  ready: boolean;
  loadPhys: (url: string) => Promise<void>;
  runPlaybook: (waypoints: [number, number, number][], stepsPerWaypoint: number) => Promise<WalkFrame[]>;
  getSceneBounds: () => { min: [number, number, number]; max: [number, number, number] } | null;
}

const CAPSULE_RADIUS = 0.3;
const CAPSULE_HALF_HEIGHT = 0.5;
const MOVE_FORCE = 8.0;
const STUCK_VELOCITY_THRESHOLD = 0.01;
const FALL_THROUGH_Y_MARGIN = 2.0;

let scene: THREE.Scene;
let camera: THREE.PerspectiveCamera;
let renderer: THREE.WebGLRenderer;
let world: RAPIER.World;
let capsuleBody: RAPIER.RigidBody | null = null;
let sceneBoundsMin: THREE.Vector3 | null = null;
let sceneBoundsMax: THREE.Vector3 | null = null;

const status = document.getElementById("status")!;

function parsePhys(buffer: ArrayBuffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x53594850) throw new Error("bad magic");

  const flags = view.getUint16(6, true);
  const hullCount = view.getUint32(8, true);
  const hullTableOff = view.getUint32(20, true);
  const vertexDataOff = view.getUint32(24, true);
  const indexDataOff = view.getUint32(28, true);
  const hasBones = (flags & 0x01) !== 0;
  const descSize = hasBones ? 44 : 40;

  const hulls: { vertices: Float32Array; indices: Uint16Array }[] = [];

  for (let i = 0; i < hullCount; i++) {
    const off = hullTableOff + i * descSize;
    const vOff = view.getUint32(off, true);
    const vCount = view.getUint32(off + 4, true);
    const iOff = view.getUint32(off + 8, true);
    const iCount = view.getUint32(off + 12, true);
    const aabbMin = [
      view.getFloat32(off + 16, true),
      view.getFloat32(off + 20, true),
      view.getFloat32(off + 24, true),
    ];
    const aabbMax = [
      view.getFloat32(off + 28, true),
      view.getFloat32(off + 32, true),
      view.getFloat32(off + 36, true),
    ];

    const vertices = new Float32Array(vCount * 3);
    const qOff = vertexDataOff + vOff * 6;
    for (let v = 0; v < vCount; v++) {
      for (let c = 0; c < 3; c++) {
        const q = view.getInt16(qOff + (v * 3 + c) * 2, true);
        const extent = aabbMax[c] - aabbMin[c];
        const e = extent === 0 ? 1.0 : extent;
        vertices[v * 3 + c] = ((q + 32768) / 65535) * e + aabbMin[c];
      }
    }

    const indices = new Uint16Array(iCount);
    const idxByteOff = indexDataOff + iOff * 2;
    for (let t = 0; t < iCount; t++) {
      indices[t] = view.getUint16(idxByteOff + t * 2, true);
    }

    hulls.push({ vertices, indices });
  }
  return hulls;
}

async function init() {
  await RAPIER.init();
  world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 10, 15);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  const api: WalktestAPI = {
    ready: true,

    loadPhys: async (url: string) => {
      const resp = await fetch(url);
      const buffer = await resp.arrayBuffer();
      const hulls = parsePhys(buffer);

      const bMin = new THREE.Vector3(Infinity, Infinity, Infinity);
      const bMax = new THREE.Vector3(-Infinity, -Infinity, -Infinity);

      const fixedBody = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());

      for (const hull of hulls) {
        const desc = RAPIER.ColliderDesc.convexHull(hull.vertices);
        if (desc) {
          world.createCollider(desc, fixedBody);
        }

        for (let i = 0; i < hull.vertices.length; i += 3) {
          bMin.x = Math.min(bMin.x, hull.vertices[i]);
          bMin.y = Math.min(bMin.y, hull.vertices[i + 1]);
          bMin.z = Math.min(bMin.z, hull.vertices[i + 2]);
          bMax.x = Math.max(bMax.x, hull.vertices[i]);
          bMax.y = Math.max(bMax.y, hull.vertices[i + 1]);
          bMax.z = Math.max(bMax.z, hull.vertices[i + 2]);
        }

        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.Float32BufferAttribute(hull.vertices, 3));
        geo.setIndex(new THREE.Uint16BufferAttribute(hull.indices, 1));
        geo.computeVertexNormals();
        const mesh = new THREE.Mesh(
          geo,
          new THREE.MeshBasicMaterial({ color: 0x00ff88, wireframe: true, transparent: true, opacity: 0.2 })
        );
        scene.add(mesh);
      }

      sceneBoundsMin = bMin;
      sceneBoundsMax = bMax;

      camera.position.set(
        (bMin.x + bMax.x) / 2,
        bMax.y + (bMax.y - bMin.y),
        bMax.z + (bMax.z - bMin.z) * 0.5
      );
      camera.lookAt((bMin.x + bMax.x) / 2, (bMin.y + bMax.y) / 2, (bMin.z + bMax.z) / 2);

      status.textContent = `loaded: ${hulls.length} hulls`;
    },

    runPlaybook: async (
      waypoints: [number, number, number][],
      stepsPerWaypoint: number
    ): Promise<WalkFrame[]> => {
      if (!sceneBoundsMin || !sceneBoundsMax) {
        throw new Error("no scene loaded");
      }

      const spawnY = sceneBoundsMax.y + 2;
      const [sx, , sz] = waypoints[0] ?? [0, 0, 0];

      if (capsuleBody) {
        world.removeRigidBody(capsuleBody);
      }
      const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(sx, spawnY, sz)
        .lockRotations();
      capsuleBody = world.createRigidBody(bodyDesc);
      const colDesc = RAPIER.ColliderDesc.capsule(CAPSULE_HALF_HEIGHT, CAPSULE_RADIUS);
      world.createCollider(colDesc, capsuleBody);

      const capsuleMesh = new THREE.Mesh(
        new THREE.CapsuleGeometry(CAPSULE_RADIUS, CAPSULE_HALF_HEIGHT * 2, 4, 8),
        new THREE.MeshBasicMaterial({ color: 0xff4444 })
      );
      scene.add(capsuleMesh);

      const floorY = sceneBoundsMin.y - FALL_THROUGH_Y_MARGIN;
      const frames: WalkFrame[] = [];
      let prevPos = { x: sx, y: spawnY, z: sz };
      let globalStep = 0;

      for (let wp = 0; wp < waypoints.length; wp++) {
        const [tx, , tz] = waypoints[wp];
        const targetY = waypoints[wp][1];

        for (let s = 0; s < stepsPerWaypoint; s++) {
          const pos = capsuleBody.translation();
          const vel = capsuleBody.linvel();

          const dx = tx - pos.x;
          const dz = tz - pos.z;
          const dist = Math.sqrt(dx * dx + dz * dz);

          if (dist > 0.1) {
            const fx = (dx / dist) * MOVE_FORCE;
            const fz = (dz / dist) * MOVE_FORCE;
            capsuleBody.applyImpulse({ x: fx * 0.016, y: 0, z: fz * 0.016 }, true);
          }

          world.step();

          const newPos = capsuleBody.translation();
          const newVel = capsuleBody.linvel();
          const displacement = Math.sqrt(
            (newPos.x - prevPos.x) ** 2 + (newPos.z - prevPos.z) ** 2
          );
          const grounded = newPos.y < prevPos.y + 0.05 && Math.abs(newVel.y) < 0.5;
          const stuck = displacement < STUCK_VELOCITY_THRESHOLD && dist > 0.5 && grounded;

          frames.push({
            step: globalStep,
            x: newPos.x,
            y: newPos.y,
            z: newPos.z,
            vy: newVel.y,
            grounded,
            stuck,
          });

          capsuleMesh.position.set(newPos.x, newPos.y, newPos.z);
          renderer.render(scene, camera);

          prevPos = { x: newPos.x, y: newPos.y, z: newPos.z };
          globalStep++;

          if (newPos.y < floorY) break;
        }

        const finalPos = capsuleBody.translation();
        if (finalPos.y < floorY) break;
      }

      scene.remove(capsuleMesh);
      return frames;
    },

    getSceneBounds: () => {
      if (!sceneBoundsMin || !sceneBoundsMax) return null;
      return {
        min: [sceneBoundsMin.x, sceneBoundsMin.y, sceneBoundsMin.z],
        max: [sceneBoundsMax.x, sceneBoundsMax.y, sceneBoundsMax.z],
      };
    },
  };

  (window as any).__walktest = api;
  renderer.render(scene, camera);
}

init();
