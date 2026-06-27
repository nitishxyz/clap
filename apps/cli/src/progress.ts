import type { Download } from "@clap/api";

const spinnerFrames = ["-", "\\", "|", "/"];

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = units[0]!;
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index]!;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${unit}`;
}

export function formatDownloadProgress(download: Download, tick = 0, width = 24): string {
  const file = download.currentFile ?? download.file ?? download.model;
  if (download.totalBytes && download.totalBytes > 0) {
    const ratio = Math.min(1, download.bytesReceived / download.totalBytes);
    const filled = Math.max(0, Math.min(width, Math.round(ratio * width)));
    const bar = `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
    const percent = Math.floor(ratio * 100).toString().padStart(3, " ");
    return `[${bar}] ${percent}% ${formatBytes(download.bytesReceived)}/${formatBytes(download.totalBytes)} ${file}`;
  }
  const frame = spinnerFrames[tick % spinnerFrames.length]!;
  return `${frame} ${formatBytes(download.bytesReceived)} ${file}`;
}
