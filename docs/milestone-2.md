# Milestone 2: Background Server

Milestone 2 keeps `clap serve` as the foreground debugging server and adds a detached background server lifecycle in the CLI.

## Commands

```bash
clap serve
clap server start
clap server stop
clap server status
clap server restart
clap server logs
clap server install
```

`clap chat` and `clap run` call the same lifecycle helper and auto-start the background server when `/clap/v1/health` is not healthy.

## State And Logs

The default state path is `~/.clap`. Override it with `CLAP_HOME`.

```txt
~/.clap/server.json
~/.clap/server.log
~/.clap/server.err.log
```

`server.json` stores:

```json
{
  "pid": 12345,
  "port": 11435,
  "baseURL": "http://localhost:11435",
  "startedAt": "2026-06-27T00:00:00.000Z"
}
```

Status and duplicate-start prevention use `/clap/v1/health` instead of trusting the pid metadata alone.

## User Services

`clap server install` writes a pragmatic per-user service template for the current platform:

- macOS: `~/Library/LaunchAgents/dev.clap.server.plist`
- Linux: `~/.config/systemd/user/clap.service`

The install command prints the platform command to start or inspect the service after writing the template.
