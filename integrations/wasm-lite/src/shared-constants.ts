// Generated from docs/shared-constants.json; do not edit.

export const COACD_CONCAVITY_THRESHOLD = 0.05;
export const COACD_PREPROCESS_RESOLUTION = 50;
export const NATIVE_MIN_HULL_VERTICES = 4;
export const INTERACTIVE_MIN_HULL_VERTICES = 8;
export const PROFILE_NAMES = ["interactive", "walkable", "robotics"] as const;
export type ProfileName = (typeof PROFILE_NAMES)[number];
