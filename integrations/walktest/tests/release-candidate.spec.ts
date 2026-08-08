import { expect, test } from "@playwright/test";

interface ReleaseCandidateResult {
  packages: Record<string, string>;
  hashes: [string, string];
  deterministic: boolean;
  cancelledWith: string;
  recovered: boolean;
  inputPreserved: boolean;
  stages: string[];
  physVersion: number;
  physBytes: number;
  hullCount: number;
  rapierColliderCount: number;
  fallingBodyY: number;
  reportVersion: number;
  verdictStatus: string;
  reportHash: string | null;
  reportDeterministic: boolean | null;
  reportProblems: string[];
}

test("packed packages complete the real Worker/WASM/Rapier path", async ({
  page,
  browserName,
}) => {
  const browserErrors: string[] = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.goto("http://localhost:3219/release-candidate.html");
  await page.waitForFunction(() => window.__chitinReleaseCandidate?.ready);

  const result = await page.evaluate(
    () => window.__chitinReleaseCandidate.run(),
  ) as ReleaseCandidateResult;

  expect(browserErrors).toEqual([]);
  expect(Object.keys(result.packages).sort()).toEqual([
    "@autarkis/chitin-coacd-wasm",
    "@autarkis/chitin-lite",
    "@autarkis/chitin-web",
  ]);
  expect(result.packages["@autarkis/chitin-lite"]).toBe(
    result.packages["@autarkis/chitin-coacd-wasm"],
  );
  expect(result.packages["@autarkis/chitin-web"]).toMatch(/^\d+\.\d+\.\d+/);
  expect(result.cancelledWith).toBe("CANCELLED");
  expect(result.recovered).toBe(true);
  expect(result.inputPreserved).toBe(true);
  expect(result.stages).toContain("reading-input");
  expect(result.stages).toContain("parsing-input");
  expect(result.stages).toContain("loading-wasm");
  expect(result.stages).toContain("decomposing");
  expect(result.stages).toContain("writing-phys");
  expect(result.stages).toContain("done");
  expect(result.physVersion).toBe(3);
  expect(result.physBytes).toBeGreaterThan(32);
  expect(result.hullCount).toBeGreaterThanOrEqual(2);
  expect(result.rapierColliderCount).toBe(result.hullCount);
  expect(result.fallingBodyY).toBeGreaterThan(0.5);
  expect(result.fallingBodyY).toBeLessThan(1.5);
  expect(result.reportVersion).toBe(1);
  expect(result.verdictStatus).toBe("not_evaluated");
  expect(result.reportHash).toBe(result.hashes[0]);
  // A single compilation does not claim it measured repeatability; the gate
  // performs the separate two-run check below.
  expect(result.reportDeterministic).toBeNull();
  expect(result.reportProblems).toEqual([]);

  // Determinism is asserted within this browser/runtime. No Python↔WASM or
  // cross-browser byte-parity claim is implied by this check.
  expect(result.deterministic).toBe(true);
  expect(result.hashes[0]).toBe(result.hashes[1]);

  console.log(
    JSON.stringify({ runtime: browserName, ...result }, null, 2),
  );
});

declare global {
  interface Window {
    __chitinReleaseCandidate: {
      ready: boolean;
      run(): Promise<ReleaseCandidateResult>;
    };
  }
}
