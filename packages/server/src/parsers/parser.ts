// Stable facade while parser primitives are incrementally split out of the
// compatibility implementation.
export {
  parseAssistantOutput,
  prepareChatRequest,
  profileStreamExtras,
  rejectUnsupportedContentParts,
  remainingDelta,
  resetCompiledProfiles,
  selectParser,
  StreamingOutputFilter,
} from "../chat-compat";
export type { PrepareChatOptions, StreamDelta, StreamFilterOptions, StreamMarker, StreamingParserState } from "../chat-compat";
export type { AssistantOutputParser, ParsedAssistantOutput, ParserTemplateInfo } from "./types";
