import type { ClapModel, LoadedModel } from "@clap/api";

export function formatModelList(models: ClapModel[], aliases?: ClapModel[]): string {
  const lines = ["models:", ...formatRows(models)];
  if (aliases) {
    lines.push("aliases:", ...formatRows(aliases));
  }
  return lines.join("\n");
}

export function formatModelStatus(models: ClapModel[], loaded: LoadedModel[], aliases?: ClapModel[]): string {
  const lines = ["installed:", ...formatRows(models), "loaded:", ...formatLoadedRows(loaded)];
  if (aliases) {
    lines.push("aliases:", ...formatRows(aliases));
  }
  return lines.join("\n");
}

export function formatModelListJson(models: ClapModel[], aliases?: ClapModel[]): string {
  const body = aliases ? { models, aliases } : { models };
  return JSON.stringify(body, null, 2);
}

function formatRows(models: ClapModel[]): string[] {
  if (models.length === 0) return ["  (none)"];
  const rows = models.map((model) => ({
    id: model.id,
    backend: model.backend,
    format: model.format,
    status: model.status,
  }));
  const widths = {
    id: Math.max("id".length, ...rows.map((row) => row.id.length)),
    backend: Math.max("backend".length, ...rows.map((row) => row.backend.length)),
    format: Math.max("format".length, ...rows.map((row) => row.format.length)),
  };

  return [
    `  ${"id".padEnd(widths.id)}  ${"backend".padEnd(widths.backend)}  ${"format".padEnd(widths.format)}  status`,
    ...rows.map((row) => `  ${row.id.padEnd(widths.id)}  ${row.backend.padEnd(widths.backend)}  ${row.format.padEnd(widths.format)}  ${row.status}`),
  ];
}

function formatLoadedRows(models: LoadedModel[]): string[] {
  if (models.length === 0) return ["  (none)"];
  const rows = models.map((model) => ({
    id: model.id,
    backend: model.backend,
    state: model.state,
    active: String(model.activeRequests),
    keepAlive: model.keepAlive,
    expiresAt: model.expiresAt ?? "never",
  }));
  const widths = {
    id: Math.max("id".length, ...rows.map((row) => row.id.length)),
    backend: Math.max("backend".length, ...rows.map((row) => row.backend.length)),
    state: Math.max("state".length, ...rows.map((row) => row.state.length)),
    active: Math.max("active".length, ...rows.map((row) => row.active.length)),
    keepAlive: Math.max("keepAlive".length, ...rows.map((row) => row.keepAlive.length)),
  };

  return [
    `  ${"id".padEnd(widths.id)}  ${"backend".padEnd(widths.backend)}  ${"state".padEnd(widths.state)}  ${"active".padEnd(widths.active)}  ${"keepAlive".padEnd(widths.keepAlive)}  expiresAt`,
    ...rows.map((row) => `  ${row.id.padEnd(widths.id)}  ${row.backend.padEnd(widths.backend)}  ${row.state.padEnd(widths.state)}  ${row.active.padEnd(widths.active)}  ${row.keepAlive.padEnd(widths.keepAlive)}  ${row.expiresAt}`),
  ];
}
