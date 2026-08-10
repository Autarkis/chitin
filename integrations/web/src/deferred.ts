import {
  Camera,
  Color,
  DepthTexture,
  DirectionalLight,
  LessEqualCompare,
  LinearFilter,
  Matrix4,
  Mesh,
  NearestFilter,
  OrthographicCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  UnsignedIntType,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
  type Texture,
} from "three";

export interface DeferredHullLightOptions {
  light: DirectionalLight;
  /** Fraction of source RGB removed at full occlusion. */
  strength: number;
  /** Shadow-depth bias. Null follows `light.shadow.bias`. */
  bias?: number | null;
  /** World-space offset along the hull normal. */
  normalBias?: number;
}

export interface DeferredHullLightingOptions {
  renderer: WebGLRenderer;
  scene: Scene;
  camera: Camera;
  /** Layer containing color-invisible Chitin hull receivers. Defaults to 1. */
  hullLayer?: number;
  lights: DeferredHullLightOptions[];
}

export interface DeferredHullLight {
  readonly light: DirectionalLight;
  strength: number;
  bias: number | null;
  normalBias: number;
}

export interface DeferredHullLightingPass {
  readonly lights: DeferredHullLight[];
  /** Render scene color, the hidden hull G-buffer, then their final composite. */
  render(): void;
  /** Force render-target dimensions. Normal rendering also follows the renderer. */
  setSize(width: number, height: number): void;
  dispose(): void;
}

interface LightUniforms {
  shadowMap: { value: Texture };
  shadowMatrix: { value: Matrix4 };
  lightDirection: { value: Vector3 };
  strength: { value: number };
  bias: { value: number };
  normalBias: { value: number };
}

/**
 * Composite hull-derived lighting over an appearance render. The hull layer is
 * rendered only into camera depth/normal buffers; it never becomes visible RGB.
 */
export function createDeferredHullLightingPass(
  options: DeferredHullLightingOptions,
): DeferredHullLightingPass {
  if (options.lights.length === 0) {
    throw new RangeError(
      "createDeferredHullLightingPass: at least one light is required",
    );
  }
  const { renderer, scene, camera } = options;
  const hullLayer = options.hullLayer ?? 1;
  if (!Number.isInteger(hullLayer) || hullLayer < 0 || hullLayer > 31) {
    throw new RangeError(
      "createDeferredHullLightingPass: hullLayer must be an integer in [0, 31]",
    );
  }

  const size = renderer.getDrawingBufferSize(new Vector2());
  const colorTarget = renderTarget(size.x, size.y);
  colorTarget.texture.minFilter = LinearFilter;
  colorTarget.texture.magFilter = LinearFilter;
  const hullTarget = renderTarget(size.x, size.y);

  const hullMaterial = new ShaderMaterial({
    vertexShader: `
      varying vec3 chitinWorldNormal;
      void main() {
        chitinWorldNormal = normalize(
          transpose(inverse(mat3(modelMatrix))) * normal
        );
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec3 chitinWorldNormal;
      void main() {
        gl_FragColor = vec4(chitinWorldNormal * 0.5 + 0.5, 1.0);
      }
    `,
    depthTest: true,
    depthWrite: true,
    toneMapped: false,
  });

  const placeholder = new DepthTexture(1, 1, UnsignedIntType);
  placeholder.compareFunction = LessEqualCompare;
  const lightPosition = new Vector3();
  const targetPosition = new Vector3();
  const bindings = options.lights.map((candidate) => {
    if (!candidate.light.castShadow) {
      throw new Error(
        "createDeferredHullLightingPass: every light must cast shadows",
      );
    }
    let strength = shadowStrength(candidate.strength);
    let bias = shadowBias(candidate.bias ?? null);
    let normalBias = shadowNormalBias(candidate.normalBias ?? 0);
    return {
      light: candidate.light,
      get strength() {
        return strength;
      },
      set strength(value: number) {
        strength = shadowStrength(value);
      },
      get bias() {
        return bias;
      },
      set bias(value: number | null) {
        bias = shadowBias(value);
      },
      get normalBias() {
        return normalBias;
      },
      set normalBias(value: number) {
        normalBias = shadowNormalBias(value);
      },
    } satisfies DeferredHullLight;
  });
  const lightUniforms: LightUniforms[] = bindings.map(() => ({
    shadowMap: { value: placeholder },
    shadowMatrix: { value: new Matrix4() },
    lightDirection: { value: new Vector3() },
    strength: { value: 0 },
    bias: { value: 0 },
    normalBias: { value: 0 },
  }));
  const uniforms: Record<string, { value: unknown }> = {
    chitinSceneColor: { value: colorTarget.texture },
    chitinSceneDepth: { value: colorTarget.depthTexture },
    chitinHullNormal: { value: hullTarget.texture },
    chitinHullDepth: { value: hullTarget.depthTexture },
    chitinProjectionInverse: { value: new Matrix4() },
    chitinCameraWorld: { value: new Matrix4() },
  };
  lightUniforms.forEach((candidate, index) => {
    uniforms[`chitinShadowMap${index}`] = candidate.shadowMap;
    uniforms[`chitinShadowMatrix${index}`] = candidate.shadowMatrix;
    uniforms[`chitinLightDirection${index}`] = candidate.lightDirection;
    uniforms[`chitinStrength${index}`] = candidate.strength;
    uniforms[`chitinBias${index}`] = candidate.bias;
    uniforms[`chitinNormalBias${index}`] = candidate.normalBias;
  });

  const compositeMaterial = new ShaderMaterial({
    uniforms,
    vertexShader: `
      varying vec2 chitinUv;
      void main() {
        chitinUv = uv;
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragmentShader: compositeFragmentShader(bindings.length),
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const compositeScene = new Scene();
  const compositeCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new Mesh(new PlaneGeometry(2, 2), compositeMaterial);
  quad.frustumCulled = false;
  compositeScene.add(quad);

  let disposed = false;
  const resize = (width: number, height: number): void => {
    const resolvedWidth = Math.max(1, Math.floor(width));
    const resolvedHeight = Math.max(1, Math.floor(height));
    if (
      colorTarget.width === resolvedWidth &&
      colorTarget.height === resolvedHeight
    ) {
      return;
    }
    colorTarget.setSize(resolvedWidth, resolvedHeight);
    hullTarget.setSize(resolvedWidth, resolvedHeight);
  };

  return {
    lights: bindings,
    setSize: resize,
    render() {
      if (disposed) return;
      renderer.getDrawingBufferSize(size);
      resize(size.x, size.y);

      const previousTarget = renderer.getRenderTarget();
      const previousOverride = scene.overrideMaterial;
      const previousBackground = scene.background;
      const previousLayerMask = camera.layers.mask;
      const previousShadowAutoUpdate = renderer.shadowMap.autoUpdate;
      const previousClearColor = renderer.getClearColor(new Color()).clone();
      const previousClearAlpha = renderer.getClearAlpha();

      try {
        renderer.shadowMap.autoUpdate = true;
        camera.layers.set(0);
        renderer.setRenderTarget(colorTarget);
        renderer.clear(true, true, true);
        renderer.render(scene, camera);

        scene.overrideMaterial = hullMaterial;
        scene.background = null;
        camera.layers.set(hullLayer);
        renderer.setClearColor(0x000000, 0);
        renderer.setRenderTarget(hullTarget);
        renderer.clear(true, true, true);
        // Three filters shadow casters through the active render camera's
        // layers. This hull render therefore also refreshes shadow maps for
        // lights assigned to `hullLayer`, enabling true hull-on-hull shadows.
        renderer.render(scene, camera);

        camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
        uniforms.chitinProjectionInverse.value = camera.projectionMatrixInverse;
        uniforms.chitinCameraWorld.value = camera.matrixWorld;
        bindings.forEach((binding, index) => {
          const candidate = lightUniforms[index];
          const depthTexture = binding.light.shadow.map?.depthTexture;
          if (depthTexture !== null && depthTexture !== undefined) {
            candidate.shadowMap.value = depthTexture;
          }
          candidate.shadowMatrix.value.copy(binding.light.shadow.matrix);
          candidate.lightDirection.value
            .copy(binding.light.getWorldPosition(lightPosition))
            .sub(binding.light.target.getWorldPosition(targetPosition))
            .normalize();
          candidate.strength.value = binding.strength;
          candidate.bias.value = binding.bias ?? binding.light.shadow.bias;
          candidate.normalBias.value = binding.normalBias;
        });

        renderer.setRenderTarget(previousTarget);
        renderer.setClearColor(previousClearColor, previousClearAlpha);
        renderer.render(compositeScene, compositeCamera);
      } finally {
        renderer.setRenderTarget(previousTarget);
        renderer.setClearColor(previousClearColor, previousClearAlpha);
        renderer.shadowMap.autoUpdate = previousShadowAutoUpdate;
        scene.overrideMaterial = previousOverride;
        scene.background = previousBackground;
        camera.layers.mask = previousLayerMask;
      }
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      colorTarget.dispose();
      hullTarget.dispose();
      hullMaterial.dispose();
      compositeMaterial.dispose();
      quad.geometry.dispose();
      placeholder.dispose();
    },
  };
}

function renderTarget(width: number, height: number): WebGLRenderTarget {
  const target = new WebGLRenderTarget(width, height, {
    depthBuffer: true,
  });
  target.texture.generateMipmaps = false;
  target.depthTexture = new DepthTexture(width, height, UnsignedIntType);
  target.depthTexture.minFilter = NearestFilter;
  target.depthTexture.magFilter = NearestFilter;
  return target;
}

function compositeFragmentShader(lightCount: number): string {
  const declarations = Array.from(
    { length: lightCount },
    (_, index) => `
      uniform sampler2DShadow chitinShadowMap${index};
      uniform mat4 chitinShadowMatrix${index};
      uniform vec3 chitinLightDirection${index};
      uniform float chitinStrength${index};
      uniform float chitinBias${index};
      uniform float chitinNormalBias${index};
    `,
  ).join("\n");
  const samples = Array.from(
    { length: lightCount },
    (_, index) => `
      {
        vec3 lightDirection = normalize(chitinLightDirection${index});
        float receiverFacing = smoothstep(
          -0.05, 0.15, dot(viewDirection, lightDirection)
        );
        vec3 biasNormal = dot(hullNormal, lightDirection) >= 0.0
          ? hullNormal
          : -hullNormal;
        vec3 shadowPoint = worldPosition +
          biasNormal * max(chitinNormalBias${index}, 0.0);
        vec4 shadowCoord4 = chitinShadowMatrix${index} * vec4(shadowPoint, 1.0);
        vec3 shadowCoord = shadowCoord4.xyz / shadowCoord4.w;
        float visibility = 1.0;
        if (
          shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 &&
          shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0 &&
          shadowCoord.z >= 0.0 && shadowCoord.z <= 1.0
        ) {
          visibility = texture(
            chitinShadowMap${index},
            vec3(shadowCoord.xy, shadowCoord.z + chitinBias${index})
          );
        }
        float shadowAmount = clamp(chitinStrength${index}, 0.0, 1.0) *
          (1.0 - visibility) * receiverFacing;
        lighting *= 1.0 - shadowAmount;
      }
    `,
  ).join("\n");

  return `
    precision highp float;
    precision highp int;
    precision highp sampler2DShadow;
    varying vec2 chitinUv;
    uniform sampler2D chitinSceneColor;
    uniform sampler2D chitinSceneDepth;
    uniform sampler2D chitinHullNormal;
    uniform sampler2D chitinHullDepth;
    uniform mat4 chitinProjectionInverse;
    uniform mat4 chitinCameraWorld;
    ${declarations}

    vec3 reconstructWorldPosition(float depth) {
      vec4 clip = vec4(chitinUv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
      vec4 view = chitinProjectionInverse * clip;
      view /= view.w;
      return (chitinCameraWorld * view).xyz;
    }

    void writeOutput(vec4 color) {
      gl_FragColor = color;
      #include <colorspace_fragment>
    }

    void main() {
      vec4 source = texture2D(chitinSceneColor, chitinUv);
      vec4 hullSample = texture2D(chitinHullNormal, chitinUv);
      float hullDepth = texture2D(chitinHullDepth, chitinUv).x;
      float sceneDepth = texture2D(chitinSceneDepth, chitinUv).x;
      if (
        hullSample.a < 0.5 ||
        hullDepth >= 1.0 ||
        sceneDepth + 0.0002 < hullDepth
      ) {
        writeOutput(source);
        return;
      }

      vec3 worldPosition = reconstructWorldPosition(hullDepth);
      vec3 viewDirection = normalize(chitinCameraWorld[3].xyz - worldPosition);
      vec3 hullNormal = normalize(hullSample.xyz * 2.0 - 1.0);
      if (dot(hullNormal, viewDirection) < 0.0) {
        hullNormal = -hullNormal;
      }
      float lighting = 1.0;
      ${samples}
      writeOutput(vec4(source.rgb * lighting, source.a));
    }
  `;
}

function shadowStrength(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(
      "createDeferredHullLightingPass: strength must be between 0 and 1",
    );
  }
  return value;
}

function shadowBias(value: number | null): number | null {
  if (value !== null && !Number.isFinite(value)) {
    throw new RangeError(
      "createDeferredHullLightingPass: bias must be finite or null",
    );
  }
  return value;
}

function shadowNormalBias(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(
      "createDeferredHullLightingPass: normalBias must be finite and non-negative",
    );
  }
  return value;
}
