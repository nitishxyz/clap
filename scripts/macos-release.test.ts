import { describe, expect, test } from "bun:test";
import { join } from "node:path";

const root = join(import.meta.dir, "..");

describe("macOS release signing plan", () => {
  test("verifies the notarized CLI before packaging the installer tarball", async () => {
    const process = Bun.spawn(["bash", "scripts/macos-release.sh"], {
      cwd: root,
      env: { ...Bun.env, CLAP_SIGNING_DRY_RUN: "1", CLAP_RELEASE_VERSION: "test" },
      stdout: "pipe",
      stderr: "pipe",
    });
    const output = await new Response(process.stdout).text();
    const error = await new Response(process.stderr).text();
    expect(await process.exited, error).toBe(0);

    const positions = [
      "libexec/clap-llama",
      "libexec/clap-mlx",
      "bun run build:binary",
      "dist/clap",
      "env CLAP_HOME=",
      "hdiutil create",
      "notarytool submit",
      "stapler staple",
      "spctl --assess --type open",
      "hdiutil attach",
      "spctl --assess --type execute",
      "cmp dist/clap",
      "hdiutil detach",
      "tar -czf dist/clap-test-darwin-arm64.tar.gz",
      "shasum -a 256",
    ].map((step) => output.indexOf(step));

    expect(positions.every((position) => position >= 0)).toBe(true);
    expect(positions).toEqual([...positions].sort((a, b) => a - b));
    expect(output).toContain("--options runtime");
    expect(output).toContain("config/macos/clap.entitlements.plist");
    expect(output).toContain("clap-test-darwin-arm64.dmg");
    expect(output).toContain("clap-test-darwin-arm64.tar.gz");
  });
});
