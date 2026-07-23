import type { ModelProfileDefinition } from "../model-profiles";

export function applyProfileReplacements(text: string, profile?: ModelProfileDefinition): string {
  const replacements = profile?.markers?.replace;
  if (!replacements) return text;
  let output = text;
  for (const [from, to] of Object.entries(replacements)) output = output.split(from).join(to);
  return output;
}

export function applyProfileMarkers(text: string, profile?: ModelProfileDefinition): string {
  if (!profile?.markers) return text;
  let output = text;
  for (const marker of profile.markers.suppress ?? []) {
    const index = output.indexOf(marker);
    if (index >= 0) output = output.slice(0, index);
  }
  for (const marker of profile.markers.strip ?? []) output = output.split(marker).join("");
  return output;
}

export function suppressProtocolMarkers(text: string): string {
  return text
    .replace(/<\|?tool_call\|?>[\s\S]*?<\|?\/tool_call\|?>/g, "")
    .replace(/<\|?tool_call\|?>[\s\S]*$/g, "")
    .replace(/to=functions\.[\w.-]+[\s\S]*?<\|message\|>[\s\S]*?<\|call\|>/g, "")
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
    .replace(/<function=[\w.-]+>[\s\S]*?<\/function>/g, "")
    .replace(/<function=[\w.-]+>[\s\S]*$/g, "")
    .replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "")
    .replace(/<｜tool▁calls▁begin｜>[\s\S]*?<｜tool▁calls▁end｜>/g, "")
    .replace(/<\|python_tag\|>[\s\S]*$/g, "")
    .replace(/\[TOOL_CALLS\][\s\S]*$/g, "")
    .replace(/call:[\w.-]+\s*\{[\s\S]*?\}/g, "");
}

export function cleanupProtocolText(text: string): string {
  return text
    .replace(/<\|(?:message|call|\/tool_call|tool_call)\|>/g, "")
    .replace(/<\|?channel\|?>/g, "")
    .replace(/<\|(?:im_end|eot_id|end_of_text)\|>/g, "")
    .replace(/<\|turn>(?:assistant|user|model|system)?\s*$/g, "")
    .replace(/<turn\|>/g, "")
    .replace(/<\|turn>/g, "")
    .replace(/<\|tool_call_(?:start|end)\|>/g, "")
    .replace(/<\/?tool_call>/g, "")
    .replace(/<\/?function(?:=[\w.-]+)?>/g, "")
    .replace(/<\/?parameter(?:=[\w.-]+)?>/g, "")
    .replace(/<\|?tool_call\|?>[\s\S]*$/g, "")
    .replace(/call::?[\w.-]+[\s\S]*$/g, "")
    .replace(/<｜tool▁(?:calls▁begin|calls▁end|call▁begin|call▁end)｜>/g, "")
    .replace(/<\|python_tag\|>/g, "")
    .replace(/\[TOOL_CALLS\]/g, "")
    .replace(/<\/s>/g, "")
    .trim();
}
