import { describe, expect, test } from "bun:test";
import { createPullCancellationHandler } from "./pull-cancellation";

describe("pull cancellation handler", () => {
  test("cancels once, prints a concise message, and exits non-zero", async () => {
    const calls: string[] = [];
    let exitCode: number | undefined;
    const handler = createPullCancellationHandler({
      cancel: async () => {
        calls.push("cancel");
      },
      write: (message) => calls.push(message),
      exit: (code) => {
        exitCode = code;
        return undefined as never;
      },
    });

    handler();
    handler();
    await Bun.sleep(0);

    expect(calls).toEqual(["cancel", "\nclap: pull cancelled\n"]);
    expect(exitCode).toBe(130);
  });
});
