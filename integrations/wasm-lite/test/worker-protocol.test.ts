import { describe, expect, it } from "vitest";

import { ChitinError } from "../src/errors.js";
import { mapWorkerError } from "../src/worker-protocol.js";

describe("mapWorkerError", () => {
  it("passes a ChitinError through with its code", () => {
    const err = new ChitinError("INVALID_MESH", "bad geometry");
    expect(mapWorkerError(err)).toEqual({
      code: "INVALID_MESH",
      message: "bad geometry",
      stage: null,
      suggestion: null,
      retryable: false,
      context: {},
    });
  });

  it("maps heap-exhaustion messages to OUT_OF_MEMORY", () => {
    for (const message of [
      "Cannot enlarge memory arrays",
      "abort: OOM",
      "std::bad_alloc",
      "allocation failed",
    ]) {
      expect(mapWorkerError(new Error(message)).code).toBe("OUT_OF_MEMORY");
    }
  });

  it("maps any other error to WORKER_ERROR", () => {
    expect(mapWorkerError(new Error("segfault in native code")).code).toBe("WORKER_ERROR");
    expect(mapWorkerError("string thrown").code).toBe("WORKER_ERROR");
  });
});
