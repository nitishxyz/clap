import { assets } from "./web-assets.generated";

export type WebAsset = { bytes: ArrayBuffer; type: string };

const decoded = new Map<string, WebAsset>();

export function webAsset(path: string): WebAsset | undefined {
  const cached = decoded.get(path);
  if (cached) return cached;
  const entry = assets[path];
  if (!entry) return undefined;
  const buffer = Buffer.from(entry.base64, "base64");
  const bytes = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer;
  const asset: WebAsset = { bytes, type: entry.type };
  decoded.set(path, asset);
  return asset;
}
