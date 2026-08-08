import type { CompileGlbResult } from "@autarkis/chitin-lite";
import { parsePhys } from "@autarkis/chitin-web";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

type ColliderPart = {
  group: THREE.Group;
  fill: THREE.MeshStandardMaterial;
  wire: THREE.LineBasicMaterial;
  delay: number;
  basePosition: THREE.Vector3;
  explodeDirection: THREE.Vector3;
};

export type PreviewState = {
  sourceVisible: boolean;
  colliderVisible: boolean;
  sourceMeshes: number;
  sourceFilledMeshes: number;
  colliderRevealActive: boolean;
  colliderRevealCount: number;
  exploded: boolean;
  explosionAmount: number;
};

export interface PreviewApi {
  hasSource(): boolean;
  showSourcePreview(file: File): Promise<void>;
  showColliderPreview(result: CompileGlbResult): void;
  clearColliders(): void;
  updateLayers(): void;
  updateExplosionControls(): void;
  state(): PreviewState;
  dispose(): void;
}

type PreviewControls = {
  canvas: HTMLCanvasElement;
  viewportPanel: HTMLElement;
  showSource: HTMLInputElement;
  showColliders: HTMLInputElement;
  explodeColliders: HTMLInputElement;
  explodeDistance: HTMLInputElement;
};

function asMaterialArray(material: THREE.Material | THREE.Material[]): THREE.Material[] {
  return Array.isArray(material) ? material : [material];
}

function disposeMaterial(material: THREE.Material): void {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose();
  }
  material.dispose();
}

function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh || object instanceof THREE.LineSegments)) return;
    object.geometry.dispose();
    for (const material of asMaterialArray(object.material)) disposeMaterial(material);
  });
}

export class PreviewController implements PreviewApi {
  private readonly viewportPanel: HTMLElement;
  private readonly showSource: HTMLInputElement;
  private readonly showColliders: HTMLInputElement;
  private readonly explodeColliders: HTMLInputElement;
  private readonly explodeDistance: HTMLInputElement;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.01, 10_000);
  private readonly controls: OrbitControls;
  private readonly grid = new THREE.GridHelper(20, 28, 0x313831, 0x202520);
  private readonly resizeObserver: ResizeObserver;
  private sourceRoot = new THREE.Group();
  private colliderRoot = new THREE.Group();
  private colliderParts: ColliderPart[] = [];
  private colliderExplosionRadius = 1;
  private colliderExplosionCurrent = 0;
  private colliderExplosionTarget = 0;
  private colliderReveal: { startedAt: number | null; parts: ColliderPart[] } | null = null;
  private colliderRevealCount = 0;
  private animationFrame: number | null = null;
  private disposed = false;
  private readonly animate = (time: number): void => {
    if (this.disposed) return;
    this.controls.update();
    this.updateColliderReveal(time);
    this.updateColliderExplosion();
    this.renderer.render(this.scene, this.camera);
    this.animationFrame = requestAnimationFrame(this.animate);
  };

  constructor({
    canvas,
    viewportPanel,
    showSource,
    showColliders,
    explodeColliders,
    explodeDistance,
  }: PreviewControls) {
    this.viewportPanel = viewportPanel;
    this.showSource = showSource;
    this.showColliders = showColliders;
    this.explodeColliders = explodeColliders;
    this.explodeDistance = explodeDistance;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.scene.fog = new THREE.FogExp2(0x0e110f, 0.035);
    this.camera.position.set(3.8, 2.8, 4.8);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new THREE.HemisphereLight(0xe9f2df, 0x222820, 1.7));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(4, 7, 5);
    this.scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xd7ff43, 1.1);
    rimLight.position.set(-4, 2, -3);
    this.scene.add(rimLight);

    (this.grid.material as THREE.Material).transparent = true;
    (this.grid.material as THREE.Material).opacity = 0.55;
    this.scene.add(this.grid, this.sourceRoot, this.colliderRoot);

    this.resizeObserver = new ResizeObserver(([entry]) => {
      const width = Math.max(1, entry.contentRect.width);
      const height = Math.max(1, entry.contentRect.height);
      this.renderer.setSize(width, height, false);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    });
    this.resizeObserver.observe(viewportPanel);
    this.animationFrame = requestAnimationFrame(this.animate);
  }

  hasSource(): boolean {
    return this.sourceRoot.children.length > 0;
  }

  async showSourcePreview(file: File): Promise<void> {
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
      for (const material of sourceMaterials) disposeMaterial(material);
      const previewMaterials = sourceMaterials.map(() => new THREE.MeshBasicMaterial({
        color: 0xdce6df,
        transparent: true,
        opacity: this.showColliders.checked ? 0.18 : 0.84,
        depthWrite: !this.showColliders.checked,
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
          opacity: this.showColliders.checked ? 0.22 : 0.82,
        }),
      );
      outline.userData.sourceOutline = true;
      outline.renderOrder = 2;
      object.add(outline);
    }
    this.sourceRoot = this.replaceGroup(this.sourceRoot, next);
    this.updateLayers();
    this.fitCamera();
  }

  showColliderPreview(result: CompileGlbResult): void {
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
    this.colliderExplosionRadius = Math.max(
      bounds.getSize(new THREE.Vector3()).length() * 0.5,
      0.2,
    );
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
    this.colliderParts = revealParts;
    this.explodeColliders.disabled = false;
    this.explodeDistance.disabled = !this.explodeColliders.checked;
    this.colliderExplosionTarget = this.explodeColliders.checked
      ? Number(this.explodeDistance.value)
      : 0;
    this.colliderReveal = { startedAt: null, parts: revealParts };
    this.colliderRevealCount++;
    this.viewportPanel.dataset.colliderReveal = "pending";
    this.colliderRoot = this.replaceGroup(this.colliderRoot, next);
    this.colliderRoot.traverse((object) => { object.renderOrder = 2; });
    this.updateLayers();
    this.fitCamera();
  }

  clearColliders(): void {
    this.colliderParts = [];
    this.colliderExplosionCurrent = 0;
    this.colliderExplosionTarget = 0;
    this.explodeColliders.checked = false;
    this.explodeColliders.disabled = true;
    this.explodeDistance.disabled = true;
    this.colliderReveal = null;
    this.viewportPanel.dataset.colliderReveal = "idle";
    const emptyColliders = new THREE.Group();
    emptyColliders.name = "compiled_colliders";
    this.colliderRoot = this.replaceGroup(this.colliderRoot, emptyColliders);
    this.updateLayers();
  }

  updateLayers(): void {
    this.sourceRoot.visible = this.showSource.checked;
    this.colliderRoot.visible = this.showColliders.checked;
    if (this.colliderRoot.visible && this.colliderReveal?.startedAt === null) {
      this.viewportPanel.dataset.colliderReveal = "pending";
    }
    const sourceOpacity = this.showColliders.checked ? 0.18 : 0.84;
    const outlineOpacity = this.showColliders.checked ? 0.22 : 0.82;
    this.sourceRoot.traverse((object) => {
      if (object instanceof THREE.LineSegments && object.userData.sourceOutline) {
        for (const material of asMaterialArray(object.material)) material.opacity = outlineOpacity;
        return;
      }
      if (!(object instanceof THREE.Mesh) || !object.userData.sourceFill) return;
      for (const material of asMaterialArray(object.material)) {
        material.transparent = true;
        material.opacity = sourceOpacity;
        material.depthWrite = !this.showColliders.checked;
        material.side = THREE.DoubleSide;
      }
    });
  }

  updateExplosionControls(): void {
    this.explodeDistance.disabled =
      this.explodeColliders.disabled || !this.explodeColliders.checked;
    this.colliderExplosionTarget = this.explodeColliders.checked
      ? Number(this.explodeDistance.value)
      : 0;
    if (this.colliderParts.length === 0) return;
    const previousAmount = this.colliderExplosionCurrent;
    this.positionColliderParts(this.colliderExplosionTarget);
    this.fitCamera(
      this.explodeColliders.checked
        ? this.colliderRoot
        : (this.sourceRoot.children.length > 0 ? this.sourceRoot : this.colliderRoot),
      this.explodeColliders.checked ? 3.5 : 2.7,
    );
    this.positionColliderParts(previousAmount);
  }

  state(): PreviewState {
    let sourceMeshes = 0;
    let sourceFilledMeshes = 0;
    this.sourceRoot.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      sourceMeshes++;
      if (!Array.isArray(object.material) || object.geometry.groups.length > 0) {
        sourceFilledMeshes++;
      }
    });
    return {
      sourceVisible: this.sourceRoot.visible,
      colliderVisible: this.colliderRoot.visible,
      sourceMeshes,
      sourceFilledMeshes,
      colliderRevealActive: this.colliderReveal !== null,
      colliderRevealCount: this.colliderRevealCount,
      exploded: this.colliderExplosionCurrent > 0.001,
      explosionAmount: this.colliderExplosionCurrent,
    };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.animationFrame !== null) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = null;
    this.resizeObserver.disconnect();
    this.controls.dispose();
    disposeObject(this.scene);
    this.renderer.dispose();
  }

  private replaceGroup(current: THREE.Group, next: THREE.Group): THREE.Group {
    this.scene.remove(current);
    disposeObject(current);
    this.scene.add(next);
    return next;
  }

  private fitCamera(preferredRoot?: THREE.Object3D, distanceMultiplier = 2.7): void {
    const visible = new THREE.Group();
    if (preferredRoot) visible.add(preferredRoot.clone());
    else if (this.sourceRoot.children.length > 0) visible.add(this.sourceRoot.clone());
    else if (this.colliderRoot.children.length > 0) visible.add(this.colliderRoot.clone());
    const box = new THREE.Box3().setFromObject(visible);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.length() * 0.5, 0.2);
    this.controls.target.copy(center);
    this.camera.near = Math.max(radius / 1000, 0.001);
    this.camera.far = Math.max(radius * 100, 100);
    this.camera.position.copy(center).add(
      new THREE.Vector3(1.25, 0.9, 1.55).normalize().multiplyScalar(radius * distanceMultiplier),
    );
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.grid.position.y = box.min.y;
    this.grid.scale.setScalar(Math.max(radius / 4, 0.15));
  }

  private positionColliderParts(amount: number): void {
    for (const part of this.colliderParts) {
      part.group.position.copy(part.basePosition).addScaledVector(
        part.explodeDirection,
        amount * this.colliderExplosionRadius,
      );
    }
  }

  private updateColliderExplosion(): void {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const delta = this.colliderExplosionTarget - this.colliderExplosionCurrent;
    this.colliderExplosionCurrent = reducedMotion || Math.abs(delta) < 0.001
      ? this.colliderExplosionTarget
      : this.colliderExplosionCurrent + delta * 0.14;
    this.positionColliderParts(this.colliderExplosionCurrent);
    this.viewportPanel.dataset.exploded = String(this.colliderExplosionCurrent > 0.001);
  }

  private finishColliderReveal(): void {
    if (!this.colliderReveal) return;
    for (const part of this.colliderReveal.parts) {
      part.group.scale.setScalar(1);
      part.fill.opacity = 0.48;
      part.wire.opacity = 0.92;
    }
    this.colliderReveal = null;
    this.viewportPanel.dataset.colliderReveal = "complete";
  }

  private updateColliderReveal(time: number): void {
    if (!this.colliderReveal || !this.colliderRoot.visible) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      this.finishColliderReveal();
      return;
    }
    this.colliderReveal.startedAt ??= time;
    this.viewportPanel.dataset.colliderReveal = "running";
    const duration = 520;
    let complete = true;
    for (const part of this.colliderReveal.parts) {
      const linear = Math.min(
        1,
        Math.max(0, (time - this.colliderReveal.startedAt - part.delay) / duration),
      );
      const remaining = linear - 1;
      const eased = 1 + 2.2 * remaining ** 3 + 1.2 * remaining ** 2;
      part.group.scale.setScalar(0.78 + 0.22 * eased);
      part.fill.opacity = 0.48 * linear;
      part.wire.opacity = 0.92 * linear;
      if (linear < 1) complete = false;
    }
    if (complete) this.finishColliderReveal();
  }
}

export class NullPreviewController implements PreviewApi {
  hasSource(): boolean {
    return false;
  }

  showSourcePreview(): Promise<void> {
    return Promise.resolve();
  }

  showColliderPreview(): void {}

  clearColliders(): void {}

  updateLayers(): void {}

  updateExplosionControls(): void {}

  state(): PreviewState {
    return {
      sourceVisible: false,
      colliderVisible: false,
      sourceMeshes: 0,
      sourceFilledMeshes: 0,
      colliderRevealActive: false,
      colliderRevealCount: 0,
      exploded: false,
      explosionAmount: 0,
    };
  }

  dispose(): void {}
}
