import { describe, expect, test } from "bun:test";
import { LegacyWorkerProtocol } from "./legacy-worker-protocol";

describe("explicit configured legacy worker adapter", () => {
  const protocol = new LegacyWorkerProtocol();

  test("decodes only structured object messages", () => {
    expect(protocol.decode('{"id":"req","token":"ok"}')).toEqual({
      kind: "message",
      message: { id: "req", token: "ok" },
    });
  });

  test("never treats malformed stdout as generated content", () => {
    expect(protocol.decode("raw model output")).toEqual({ kind: "malformed" });
    expect(protocol.decode("17")).toEqual({ kind: "malformed" });
    expect(protocol.decode("[]")).toEqual({ kind: "malformed" });
  });
});
