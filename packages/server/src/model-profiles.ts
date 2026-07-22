import { readdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// Extensible model-format profiles.
//
// A profile teaches the server how to interpret a model family's raw output:
// which tool-call syntaxes to try (built-in primitives and/or custom regex
// formats), which protocol markers to hide from streamed content, and whether
// the chat template pre-fills an implicit <think> block.
//
// Built-in profiles cover the common families. Users can add or override
// profiles by dropping JSON files into ~/.clap/profiles/ (or $CLAP_HOME/profiles)
// without rebuilding clap — first matching user profile wins.

export type CustomParserSpec = {
  // Regex tried against the model output. Use named groups: (?<name>...) for
  // the tool name (or set `name` for a fixed tool) and (?<args>...) for the
  // argument payload (JSON or loose JSON).
  pattern: string;
  flags?: string;
  name?: string;
};

export type ProfileMarkers = {
  // Streamed/visible content is cut at these markers (tool-call preambles).
  suppress?: string[];
  // These markers are removed wherever they appear (end-of-turn tokens).
  strip?: string[];
  // Literal token replacements applied before parsing (e.g. gemma's special
  // quote token <|"|> becomes a plain double quote).
  replace?: Record<string, string>;
};

export type ModelProfileDefinition = {
  name: string;
  // User profiles may list an exact model id here. For all profiles these are
  // also matched exactly against family hints derived from model metadata.
  // Model ids are never substring-matched to infer a built-in family.
  families?: string[];
  // Ordered built-in tool parser primitives to try. Available names:
  // harmony, function-message, xml-function, tagged-json, qwen-bracket,
  // deepseek, mistral, python-tag, gemma-call, json.
  parsers?: string[];
  // Custom regex tool-call formats, tried before `parsers`.
  customParsers?: CustomParserSpec[];
  markers?: ProfileMarkers;
  implicitThink?: boolean;
};

export const builtinProfiles: ModelProfileDefinition[] = [
  { name: "harmony", families: ["harmony", "gpt-oss", "codex"], parsers: ["harmony", "function-message", "tagged-json", "json"] },
  { name: "hermes", families: ["hermes", "nous", "functionary", "xlam"], parsers: ["xml-function", "tagged-json", "json"] },
  { name: "qwen", families: ["qwen"], parsers: ["xml-function", "qwen-bracket", "tagged-json", "json"] },
  { name: "deepseek", families: ["deepseek"], parsers: ["deepseek", "tagged-json", "json"] },
  { name: "mistral", families: ["mistral", "mixtral"], parsers: ["mistral", "json"] },
  { name: "llama", families: ["llama", "granite"], parsers: ["python-tag", "json"] },
  {
    name: "gemma",
    families: ["gemma", "functiongemma"],
    parsers: ["gemma-call", "json"],
    markers: { replace: { '<|"|>': '"' } },
  },
];

export const genericProfile: ModelProfileDefinition = {
  name: "generic",
  parsers: [
    // Try the more specific `call:` envelopes before harmony's broad
    // call:name matcher so protocol markers can identify Gemma syntax without
    // relying on the model id.
    "gemma-call",
    "harmony",
    "function-message",
    "xml-function",
    "tagged-json",
    "qwen-bracket",
    "deepseek",
    "mistral",
    "python-tag",
    "json",
  ],
  markers: { replace: { '<|"|>': '"' } },
};

let cachedUserProfiles: ModelProfileDefinition[] | undefined;

export function loadUserProfiles(): ModelProfileDefinition[] {
  if (cachedUserProfiles) return cachedUserProfiles;
  const dir = join(process.env.CLAP_HOME ?? join(process.env.HOME ?? homedir(), ".clap"), "profiles");
  const profiles: ModelProfileDefinition[] = [];
  let files: string[] = [];
  try {
    files = readdirSync(dir).filter((file) => file.endsWith(".json")).sort();
  } catch {
    cachedUserProfiles = [];
    return cachedUserProfiles;
  }
  for (const file of files) {
    try {
      const parsed = JSON.parse(readFileSync(join(dir, file), "utf8"));
      const entries = Array.isArray(parsed) ? parsed : [parsed];
      for (const entry of entries) {
        if (entry && typeof entry.name === "string") profiles.push(entry as ModelProfileDefinition);
        else console.error(`[clap] ignoring invalid profile in ${file}: missing "name"`);
      }
    } catch (error) {
      console.error(`[clap] failed to load model profile ${file}: ${error}`);
    }
  }
  cachedUserProfiles = profiles;
  return profiles;
}

export function resetUserProfileCache(): void {
  cachedUserProfiles = undefined;
}
