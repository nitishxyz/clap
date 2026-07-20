import { describe, expect, test } from "bun:test";
import { fmtMaxOutputTokens } from "./format";

describe("max output capability presentation", () => {
  test("shows a declared fixed generation cap", () => {
    expect(fmtMaxOutputTokens(8192, 131072)).toBe("8192");
  });

  test("shows prompt-dependent output when only context is bounded", () => {
    expect(fmtMaxOutputTokens(null, 131072)).toBe("context-bound");
  });

  test("reserves unknown for models with no output or context metadata", () => {
    expect(fmtMaxOutputTokens(null, null)).toBe("unknown");
  });
});
