import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import type { ActionState } from "@/hooks/useActions";
import type { DashboardDownload } from "@/lib/api";
import { Downloads, isHuggingFaceAuthFailure, latestActionableAuthFailure } from "./ModelTables";

const actions: ActionState = {
  busy: {},
  run: () => undefined,
  dismissError: () => undefined,
};

function download(extra: Partial<DashboardDownload> = {}): DashboardDownload {
  return {
    id: "pull_1",
    model: "acme/private-model",
    status: "failed",
    bytesReceived: 0,
    ...extra,
  };
}

describe("download Hugging Face authentication recovery", () => {
  test("recognizes structured and legacy authentication failures", () => {
    expect(isHuggingFaceAuthFailure(download({ errorCode: "hf_auth_required" }))).toBe(true);
    expect(isHuggingFaceAuthFailure("Hugging Face authentication failed (401)")).toBe(true);
    expect(isHuggingFaceAuthFailure(download({ error: "checksum mismatch" }))).toBe(false);
  });

  test("does not treat an old auth failure as actionable after that target is retried", () => {
    const failed = download({ targetKey: "repo:key", errorCode: "hf_auth_required" });
    const retry = download({ id: "pull_2", targetKey: "repo:key", status: "completed" });
    expect(latestActionableAuthFailure([failed, retry])).toBeUndefined();
  });

  test("renders a secure token prompt and retry action for authentication failures", () => {
    const html = renderToStaticMarkup(<Downloads downloads={[download({
      errorCode: "hf_auth_required",
      error: "Hugging Face authentication failed (401)",
    })]} actions={actions} />);

    expect(html).toContain("Hugging Face authentication required");
    expect(html).toContain('type="password"');
    expect(html).toContain("save + retry");
    expect(html).toContain("https://huggingface.co/settings/tokens");
  });

  test("does not ask for a token for unrelated failures", () => {
    const html = renderToStaticMarkup(<Downloads downloads={[download({ error: "checksum mismatch" })]} actions={actions} />);
    expect(html).not.toContain("Hugging Face authentication required");
  });
});
