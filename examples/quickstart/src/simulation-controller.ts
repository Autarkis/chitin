import RAPIER from "@dimforge/rapier3d-compat";
import { parsePhys } from "@autarkis/chitin-web";
import { addToWorld, applyBindPose } from "@autarkis/chitin-web/rapier";
import * as THREE from "three";

export interface SimulationApi {
  start(physBuffer: ArrayBuffer, scene: THREE.Scene): Promise<void>;
  stop(): void;
  tick(time: number): void;
  isRunning(): boolean;
  height(): number | null;
  dispose(): void;
}

export class SimulationController implements SimulationApi {
  private rapierReady: Promise<typeof RAPIER> | null = null;
  private world: RAPIER.World | null = null;
  private sphereBody: RAPIER.RigidBody | null = null;
  private group: THREE.Group | null = null;
  private sphereMesh: THREE.Mesh | null = null;
  private scene: THREE.Scene | null = null;
  private running = false;
  private lastTime = 0;
  private accumulator = 0;
  private settled = false;
  private settledFrames = 0;

  private ensureRapier(): Promise<typeof RAPIER> {
    if (!this.rapierReady) {
      this.rapierReady = RAPIER.init().then(() => RAPIER);
    }
    return this.rapierReady;
  }

  async start(physBuffer: ArrayBuffer, scene: THREE.Scene): Promise<void> {
    const rapier = await this.ensureRapier();

    if (this.world || this.group) {
      this.stop();
    }

    const phys = parsePhys(physBuffer);

    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    addToWorld(rapier, this.world, phys);

    const bbox = new THREE.Box3();
    for (const hull of phys.hulls) {
      const vertices = hull.boneIndex === null
        ? hull.vertices
        : applyBindPose(hull.vertices, phys.bones[hull.boneIndex].bindTransform);
      for (let i = 0; i < vertices.length; i += 3) {
        bbox.expandByPoint(new THREE.Vector3(vertices[i], vertices[i + 1], vertices[i + 2]));
      }
    }

    const bsphere = bbox.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(0.02, bsphere.radius * 0.05);

    // Drop from well above the collider so it visibly falls; CCD keeps the thin sphere from
    // tunnelling through the collider at the higher speeds that drop height produces.
    const dropHeight = bbox.max.y + radius * 10;
    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(bsphere.center.x, dropHeight, bsphere.center.z)
      .setCcdEnabled(true);
    this.sphereBody = this.world.createRigidBody(bodyDesc);
    // Restitution/friction chosen for a visible, believable bounce-then-settle rather than a
    // physically "correct" material.
    const colliderDesc = RAPIER.ColliderDesc.ball(radius).setRestitution(0.5).setFriction(0.4);
    this.world.createCollider(colliderDesc, this.sphereBody);

    this.group = new THREE.Group();
    this.group.name = "simulation_objects";

    const sphereGeo = new THREE.SphereGeometry(radius, 24, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
      color: 0xd7ff43,
      emissive: 0xd7ff43,
      emissiveIntensity: 0.3,
      roughness: 0.3,
      metalness: 0.1,
    });
    this.sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
    this.sphereMesh.position.set(bsphere.center.x, dropHeight, bsphere.center.z);
    this.group.add(this.sphereMesh);

    const ringGeo = new THREE.RingGeometry(radius * 0.8, radius * 1.2, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xd7ff43,
      transparent: true,
      opacity: 0.15,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(bsphere.center.x, bbox.min.y + 0.001, bsphere.center.z);
    ring.name = "ground_indicator";
    this.group.add(ring);

    scene.add(this.group);
    this.scene = scene;

    this.running = true;
    this.settled = false;
    this.settledFrames = 0;
    this.lastTime = 0;
    this.accumulator = 0;
  }

  tick(time: number): void {
    if (!this.running || !this.world || !this.sphereBody || !this.sphereMesh || this.settled) return;

    if (this.lastTime === 0) {
      this.lastTime = time;
      return;
    }

    const dt = Math.min(time - this.lastTime, 100) / 1000;
    this.lastTime = time;

    const fixedStep = 1 / 60;
    this.accumulator += dt;
    let steps = 0;
    while (this.accumulator >= fixedStep && steps < 4) {
      this.world.step();
      this.accumulator -= fixedStep;
      steps++;
    }

    if (steps === 0) return;

    const pos = this.sphereBody.translation();
    const rot = this.sphereBody.rotation();
    this.sphereMesh.position.set(pos.x, pos.y, pos.z);
    this.sphereMesh.quaternion.set(rot.x, rot.y, rot.z, rot.w);

    const vel = this.sphereBody.linvel();
    const speed = Math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    if (speed < 0.01) {
      this.settledFrames++;
      if (this.settledFrames >= 120) {
        this.settled = true;
      }
    } else {
      this.settledFrames = 0;
    }
  }

  stop(): void {
    if (this.group && this.scene) {
      this.scene.remove(this.group);
      this.group.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose());
          else obj.material.dispose();
        }
        if (obj instanceof THREE.LineSegments) {
          obj.geometry.dispose();
          (obj.material as THREE.Material).dispose();
        }
      });
    }
    if (this.world) {
      this.world.free();
    }
    this.world = null;
    this.sphereBody = null;
    this.sphereMesh = null;
    this.group = null;
    this.scene = null;
    this.running = false;
    this.settled = false;
    this.settledFrames = 0;
    this.lastTime = 0;
    this.accumulator = 0;
  }

  isRunning(): boolean {
    return this.running;
  }

  height(): number | null {
    return this.sphereBody?.translation().y ?? null;
  }

  isSettled(): boolean {
    return this.settled;
  }

  dispose(): void {
    this.stop();
    this.rapierReady = null;
  }
}
