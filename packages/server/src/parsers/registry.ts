import type { AssistantOutputParser, ParserRegistryInput } from "./types";

// Selection is deliberately trait-driven. Built-in family names appearing in
// a model id are not evidence of an output protocol.
export function selectRegisteredParser(input: ParserRegistryInput): AssistantOutputParser {
  const exactModel = input.model.toLowerCase();
  const explicitUser = input.user.find((candidate) =>
    candidate.name.toLowerCase() === exactModel
    || candidate.families.some((family) => family.toLowerCase() === exactModel));
  if (explicitUser) return explicitUser;

  for (const hint of input.traits?.familyHints ?? []) {
    const normalizedHint = hint.toLowerCase();
    const parser = [...input.user, ...input.builtin].find((candidate) =>
      candidate.name.toLowerCase() === normalizedHint
      || candidate.families.some((family) => family.toLowerCase() === normalizedHint));
    if (parser) return parser;
  }

  // Generic structured parsing preserves best-effort JSON/tool handling when
  // the request or discovered template calls for it. Plain is the final text
  // fallback; it may share implementation primitives but remains identifiable.
  if (input.request.tools?.length || input.request.response_format || input.traits?.hasToolCalls || input.traits?.hasReasoning) {
    return input.generic;
  }
  return input.plain;
}
