import type { PreviewState } from "./preview-controller";

export interface ChitinDemoState extends PreviewState {
  busy: boolean;
  hulls: number;
  verdict: string | null;
  reportVersion: number | null;
  appliedThreshold: number | null;
  qualityMeasured: boolean;
  reusedComponents: number;
  simulationActive: boolean;
  simulationHeight: number | null;
  profile: string;
}

export interface ChitinDemoApi {
  ready: boolean;
  previewAvailable: boolean;
  state(): ChitinDemoState;
}
