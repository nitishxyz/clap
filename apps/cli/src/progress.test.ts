import { describe, expect, test } from "bun:test";
import { formatBytes, formatDownloadProgress } from "./progress";

describe("CLI download progress formatting", () => {
  test("formats known-size downloads with a progress bar", () => {
    expect(formatDownloadProgress({
      id: "pull_1",
      model: "acme/model",
      currentFile: "model.gguf",
      status: "running",
      bytesReceived: 50,
      totalBytes: 100,
      startedAt: "2026-01-01T00:00:00.000Z",
    }, 0, 10)).toBe("[=====-----]  50% 50 B/100 B model.gguf");
  });

  test("formats unknown-size downloads with a spinner and bytes", () => {
    expect(formatDownloadProgress({
      id: "pull_1",
      model: "acme/model",
      status: "running",
      bytesReceived: 1536,
      startedAt: "2026-01-01T00:00:00.000Z",
    }, 1)).toBe("\\ 1.50 KiB acme/model");
  });

  test("formats byte units", () => {
    expect(formatBytes(42)).toBe("42 B");
    expect(formatBytes(10 * 1024 * 1024)).toBe("10.0 MiB");
  });
});
