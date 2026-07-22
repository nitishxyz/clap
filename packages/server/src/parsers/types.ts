import type { ChatCompletionRequest, ChatToolCall } from "@clap/api";
import type { ModelProfileDefinition } from "../model-profiles";

export type ParsedAssistantOutput = {
  content: string | null;
  reasoning?: string;
  toolCalls?: ChatToolCall[];
  finishReason: "stop" | "tool_calls";
};

export type NormalizedOutput = {
  content: string;
  reasoning?: string;
  toolCalls?: ChatToolCall[];
};

export type ParserContext = {
  request: ChatCompletionRequest;
  toolMode: boolean;
};

export type ToolParser = (text: string, context: ParserContext) => ChatToolCall[];

export type AssistantOutputParser = {
  name: string;
  families: string[];
  toolParsers: ToolParser[];
  profile?: ModelProfileDefinition;
  parse: (text: string, context: ParserContext) => NormalizedOutput;
};

export type ParserTraitSource = {
  file: string;
  kind: "template" | "config" | "tokenizer";
};

export type ParserTemplateInfo = {
  familyHints?: string[];
  hasToolCalls?: boolean;
  hasReasoning?: boolean;
  implicitThink?: boolean;
  sourceFiles?: string[];
  sources?: ParserTraitSource[];
  templateInferred?: boolean;
};

export type ParserRegistryInput = {
  model: string;
  request: ChatCompletionRequest;
  traits?: ParserTemplateInfo;
  user: AssistantOutputParser[];
  builtin: AssistantOutputParser[];
  generic: AssistantOutputParser;
  plain: AssistantOutputParser;
};
