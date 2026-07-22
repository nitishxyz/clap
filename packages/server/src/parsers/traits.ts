import type { ResolvedModel } from "@clap/models";
import { readFile, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { ParserTemplateInfo, ParserTraitSource } from "./types";

const metadataFiles = ["tokenizer_config.json", "tokenizer.json", "config.json", "generation_config.json", "chat_template.jinja"] as const;

export async function resolveParserTemplateInfo(model: ResolvedModel): Promise<ParserTemplateInfo | undefined> {
  const root = await metadataRoot(model);
  if (!root) return undefined;
  const sourceFiles: string[] = [];
  const sources: ParserTraitSource[] = [];
  const chunks: string[] = [];
  const nameChunks: string[] = [];
  const templateChunks: string[] = [];
  for (const file of metadataFiles) {
    try {
      const text = (await readFile(join(root, file), "utf8")).slice(0, 250_000);
      sourceFiles.push(file);
      chunks.push(text);
      const kind = file === "chat_template.jinja" || (file === "tokenizer_config.json" && /chat_template/.test(text)) ? "template"
        : file === "tokenizer.json" ? "tokenizer" : "config";
      sources.push({ file, kind });
      if (kind === "template") templateChunks.push(text);
      // Vocabulary can contain family words as ordinary tokens. Only
      // templates and configs can intentionally identify a family by name.
      if (file !== "tokenizer.json") nameChunks.push(text);
    } catch {
      // Model metadata is optional.
    }
  }
  if (!chunks.length) return undefined;
  const markerText = chunks.join("\n").toLowerCase();
  const familyHints = inferParserFamilies(markerText, nameChunks.join("\n").toLowerCase());
  return {
    familyHints,
    hasToolCalls: /tool_call|tool_calls|\[tool_calls\]|python_tag|call:/.test(markerText),
    hasReasoning: /enable_thinking|reasoning_effort|reasoning_content|<think>|analysis|commentary|final/.test(markerText),
    implicitThink: templatePrefillsThink(templateChunks.join("\n")),
    sourceFiles,
    sources,
    templateInferred: templateChunks.length > 0,
  };
}

export function inferParserFamilies(markerText: string, nameText: string): string[] {
  const hints: string[] = [];
  const add = (family: string, markers: RegExp, names?: RegExp) => {
    if ((markers.test(markerText) || (names?.test(nameText) ?? false)) && !hints.includes(family)) hints.push(family);
  };
  add("harmony", /<\|channel\|>/, /gpt-oss|harmony/);
  add("qwen", /<\|tool_call_start\|>|<function=/, /qwen/);
  add("deepseek", /<｜tool▁calls▁begin｜>/, /deepseek/);
  add("mistral", /\[tool_calls\]/, /mistral|mixtral/);
  add("llama", /<\|python_tag\|>/);
  add("gemma", /functiongemma/, /gemma/);
  add("hermes", /<tool_call>/, /hermes|xlam|functionary/);
  return hints;
}

function templatePrefillsThink(templateText: string): boolean {
  for (const match of templateText.matchAll(/\{\{-?\s*'([^']*)'\s*-?\}\}/g)) {
    const literal = match[1] ?? "";
    if (literal.includes("<think>") && !literal.includes("</think>")) return true;
  }
  return false;
}

async function metadataRoot(model: ResolvedModel): Promise<string | undefined> {
  const path = model.modelPath ?? model.input;
  try {
    return (await stat(path)).isDirectory() ? path : dirname(path);
  } catch {
    return undefined;
  }
}
