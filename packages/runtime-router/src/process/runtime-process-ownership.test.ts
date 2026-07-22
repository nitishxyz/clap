import { describe, expect, test } from "bun:test";
import { runtimeProcessOwnershipViolations } from "../../../../scripts/check-runtime-process-ownership";

describe("native inference process ownership", () => {
  test("keeps native inference lifecycle in the unified resident process", async () => {
    expect(await runtimeProcessOwnershipViolations()).toEqual([]);
  });
});
