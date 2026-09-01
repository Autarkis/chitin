import { describe, expect, it } from "vitest";
import { browserProfileVerdict } from "../src/profile-verdict.js";
import { metric } from "../src/report.js";
import { HULLS } from "./compiler-fixture.js";

describe("browser profile verdicts", () => {
  it("keeps interactive permissive", () => {
    expect(browserProfileVerdict("interactive", HULLS, undefined)).toBeUndefined();
  });

  it("evaluates available walkable metrics without claiming native probes", () => {
    const verdict = browserProfileVerdict("walkable", HULLS, {
      source_surface_coverage: metric(0.9, "ratio"),
      false_fill_fraction: metric(0.2, "ratio"),
    });
    expect(verdict?.status).toBe("not_evaluated");
    expect(verdict?.checks).toContainEqual(expect.objectContaining({ code: "source_surface_coverage", status: "pass" }));
    expect(verdict?.checks).toContainEqual(expect.objectContaining({ code: "walkable_probe", status: "not_evaluated" }));
  });

  it("rejects robotics artifacts that miss measured thresholds", () => {
    const verdict = browserProfileVerdict("robotics", HULLS, {
      source_surface_coverage: metric(0.8, "ratio"),
      worst_component_surface_coverage: metric(0.8, "ratio"),
      false_fill_fraction: metric(0.1, "ratio"),
      deep_false_fill_fraction: metric(0.1, "ratio"),
    });
    expect(verdict?.status).toBe("fail");
    expect(verdict?.reasons).toHaveLength(1);
  });
});
