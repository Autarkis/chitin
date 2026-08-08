import type { PreviewState } from "./preview-controller";

export interface ChitinDemoState extends PreviewState {
  busy: boolean;
  hulls: number;
  verdict: string | null;
  reportVersion: number | null;
  appliedThreshold: number | null;
  qualityMeasured: boolean;
  reusedComponents: number;
}

export interface ChitinDemoApi {
  ready: boolean;
  state(): ChitinDemoState;
}
