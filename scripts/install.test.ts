import { afterEach, describe, expect, test } from "bun:test";
import { chmod, mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const roots: string[] = [];

afterEach(async () => {
  await Promise.all(roots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe("install.sh", () => {
  test("installs the checksummed macOS tarball without a terminal", async () => {
    const root = await mkdtemp(join(tmpdir(), "clap-install-test-"));
    roots.push(root);
    const bin = join(root, "bin");
    const assets = join(root, "assets");
    const installDir = join(root, "installed");
    await mkdir(bin);
    await mkdir(assets);
    await writeFile(join(assets, "clap"), "#!/bin/sh\necho installed\n");
    await chmod(join(assets, "clap"), 0o755);

    const archive = "clap-v0.2.0-darwin-arm64.tar.gz";
    const tar = Bun.spawnSync(["tar", "-czf", join(assets, archive), "-C", assets, "clap"]);
    expect(tar.exitCode, tar.stderr.toString()).toBe(0);
    const checksum = Bun.spawnSync(["shasum", "-a", "256", join(assets, archive)]);
    expect(checksum.exitCode, checksum.stderr.toString()).toBe(0);
    await writeFile(join(assets, `${archive}.sha256`), checksum.stdout);

    await writeFile(join(bin, "uname"), '#!/bin/sh\n[ "${1:-}" = -s ] && echo Darwin || echo arm64\n');
    await writeFile(join(bin, "curl"), `#!/bin/sh\nout=\nurl=\nwhile [ $# -gt 0 ]; do\n  case "$1" in\n    -o) out=$2; shift 2 ;;\n    http*) url=$1; shift ;;\n    *) shift ;;\n  esac\ndone\ncp "${assets}/\${url##*/}" "$out"\n`);
    await chmod(join(bin, "uname"), 0o755);
    await chmod(join(bin, "curl"), 0o755);
    for (const command of ["tar", "shasum", "awk", "mkdir", "mktemp", "install", "rm", "cp"]) {
      const location = Bun.which(command);
      expect(location).not.toBeNull();
      await symlink(await realpath(location!), join(bin, command));
    }

    const process = Bun.spawn(["/bin/sh", "install.sh"], {
      cwd: join(import.meta.dir, ".."),
      env: {
        HOME: root,
        PATH: bin,
        CLAP_VERSION: "v0.2.0",
        CLAP_INSTALL_DIR: installDir,
      },
      stdin: "ignore",
      stdout: "pipe",
      stderr: "pipe",
    });
    const output = await new Response(process.stdout).text();
    const error = await new Response(process.stderr).text();
    expect(await process.exited, error).toBe(0);
    expect(output).toContain("downloading clap v0.2.0 (darwin-arm64)");
    expect(await readFile(join(installDir, "clap"), "utf8")).toContain("installed");
  });
});
