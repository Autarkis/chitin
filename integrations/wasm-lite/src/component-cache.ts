import type { ConvexHull } from "./types.js";

const MAX_CONFIGS_PER_COMPONENT = 6;

function cloneHulls(hulls: ConvexHull[]): ConvexHull[] {
  return hulls.map((hull) => ({
    vertices: hull.vertices.slice(),
    indices: hull.indices.slice(),
  }));
}

export class ComponentResultCache {
  private readonly results = new Map<string, ConvexHull[]>();
  private readonly keysByComponent = new Map<number, string[]>();

  get(componentIndex: number, key: string): ConvexHull[] | undefined {
    const hulls = this.results.get(key);
    if (!hulls) return undefined;
    const keys = this.keysByComponent.get(componentIndex);
    const previous = keys?.indexOf(key) ?? -1;
    if (keys && previous >= 0 && previous !== keys.length - 1) {
      keys.splice(previous, 1);
      keys.push(key);
    }
    return cloneHulls(hulls);
  }

  set(componentIndex: number, key: string, hulls: ConvexHull[]): void {
    this.results.set(key, cloneHulls(hulls));
    const keys = this.keysByComponent.get(componentIndex) ?? [];
    const previous = keys.indexOf(key);
    if (previous >= 0) keys.splice(previous, 1);
    keys.push(key);
    while (keys.length > MAX_CONFIGS_PER_COMPONENT) {
      const expired = keys.shift();
      if (expired) this.results.delete(expired);
    }
    this.keysByComponent.set(componentIndex, keys);
  }
}
