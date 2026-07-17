import type { ModelResolveOption, ModelResolveResponse } from "@clap/api";
import { formatBytes } from "./progress";

export function supportedOptions(response: ModelResolveResponse): ModelResolveOption[] {
  return response.options.filter((option) => option.supported);
}

export function unsupportedOptions(response: ModelResolveResponse): ModelResolveOption[] {
  return response.options.filter((option) => !option.supported);
}

export function optionLabel(option: ModelResolveOption): string {
  const marker = option.recommended ? "recommended" : "available";
  const quant = option.quantization ? ` ${option.quantization}` : "";
  const size = option.sizeBytes ? ` ${formatBytes(option.sizeBytes)}` : "";
  const file = option.file ? ` ${option.file}` : "";
  return `${option.backend}/${option.format}${quant}${size}${file} (${marker})`;
}

export function formatResolveOptions(response: ModelResolveResponse): string {
  const lines = [`model: ${response.model}`];
  const supported = supportedOptions(response);
  if (supported.length) {
    lines.push("", "Supported runnable options:");
    supported.forEach((option, index) => {
      const star = option.recommended ? "*" : " ";
      lines.push(`${star} ${index + 1}. ${optionLabel(option)}`);
      lines.push(`     repo: ${option.repo}`);
      lines.push(`     reason: ${option.reason}`);
    });
  } else {
    lines.push("", "Supported runnable options: none");
  }

  const unsupported = unsupportedOptions(response);
  if (unsupported.length) {
    lines.push("", "Unsupported / guidance:");
    unsupported.forEach((option) => {
      const size = option.sizeBytes ? ` ${formatBytes(option.sizeBytes)}` : "";
      lines.push(`  - ${option.backend}/${option.format}${size} ${option.repo}`);
      lines.push(`    ${option.unsupportedReason ?? option.reason}`);
    });
  }
  return lines.join("\n");
}

export function defaultOptionIndex(options: ModelResolveOption[]): number {
  const recommended = options.findIndex((option) => option.recommended);
  return recommended >= 0 ? recommended : 0;
}

export function chooseOptionByInput(options: ModelResolveOption[], input: string): ModelResolveOption {
  if (!options.length) throw new Error("no supported model options are available");
  const trimmed = input.trim();
  if (!trimmed) return options[defaultOptionIndex(options)]!;
  const selected = Number(trimmed);
  if (!Number.isInteger(selected) || selected < 1 || selected > options.length) {
    throw new Error(`expected a choice from 1-${options.length}, got: ${input}`);
  }
  return options[selected - 1]!;
}

export function findOptionByQuant(options: ModelResolveOption[], quant: string): ModelResolveOption | undefined {
  const wanted = quant.toUpperCase();
  return options.find((option) => option.supported && option.backend === "gguf" && option.quantization?.toUpperCase() === wanted);
}
