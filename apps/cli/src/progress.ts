import type { Download } from "@clap/api";

const spinnerFrames = ["-", "\\", "|", "/"];

type RateSample = { time: number; bytes: number };
const rateSamples = new Map<string, RateSample[]>();
const rateWindowMs = 5_000;

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

export function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.round(totalSeconds));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m${(seconds % 60).toString().padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h${(minutes % 60).toString().padStart(2, "0")}m`;
}

// Rolling-window transfer rate in bytes/second; returns null until enough
// samples arrive to be meaningful (avoids flashing bogus numbers).
export function trackRate(id: string, bytes: number, now = Date.now()): number | null {
  const samples = rateSamples.get(id) ?? [];
  samples.push({ time: now, bytes });
  while (samples.length > 2 && samples[0]!.time < now - rateWindowMs) samples.shift();
  rateSamples.set(id, samples);
  const first = samples[0]!;
  const elapsed = (now - first.time) / 1000;
  const delta = bytes - first.bytes;
  if (elapsed < 0.5 || delta <= 0) return null;
  return delta / elapsed;
}

export function formatDownloadProgress(download: Download, tick = 0, width = 24): string {
  const file = download.currentFile ?? download.file ?? download.model;
  const rate = trackRate(download.id, download.bytesReceived);
  const speed = rate ? ` ${formatBytes(rate)}/s` : "";
  if (download.totalBytes && download.totalBytes > 0) {
    const ratio = Math.min(1, download.bytesReceived / download.totalBytes);
    const filled = Math.max(0, Math.min(width, Math.round(ratio * width)));
    const bar = `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
    const percent = Math.floor(ratio * 100).toString().padStart(3, " ");
    const eta = rate ? ` eta ${formatDuration((download.totalBytes - download.bytesReceived) / rate)}` : "";
    return `[${bar}] ${percent}% ${formatBytes(download.bytesReceived)}/${formatBytes(download.totalBytes)}${speed}${eta} ${file}`;
  }
  const frame = spinnerFrames[tick % spinnerFrames.length]!;
  return `${frame} ${formatBytes(download.bytesReceived)}${speed} ${file}`;
}
