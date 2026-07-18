import { useEffect, useState } from "react";
import { fetchRequestDetail, type DashboardRequest } from "@/lib/api";
import { fmtClock, fmtDuration, fmtTokens } from "@/lib/format";
import { Tag } from "./Shared";

const roleColor: Record<string, string> = {
  system: "text-thinking",
  user: "text-accent",
  assistant: "text-ok",
  tool: "text-warn",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="border border-soft-border bg-panel-strong px-2 py-1.5">
      <div className="text-[0.62rem] uppercase tracking-[0.06em] text-muted">{label}</div>
      <div className="mt-0.5 text-[0.8rem]">{children}</div>
    </div>
  );
}

function MessageCard({ role, content, truncated, toolCalls }: { role: string; content: string; truncated?: boolean; toolCalls?: Array<{ name: string; arguments: string }> }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = content.length > 600;
  const shown = expanded || !isLong ? content : `${content.slice(0, 600)}…`;
  return (
    <div className="border border-soft-border">
      <div className="flex items-baseline justify-between border-b border-soft-border bg-panel-strong px-2 py-1">
        <span className={`text-[0.68rem] uppercase tracking-[0.06em] ${roleColor[role] ?? "text-muted"}`}>{role}</span>
        <span className="text-[0.65rem] text-muted">
          {content.length ? `${fmtTokens(content.length)} chars${truncated ? " (truncated)" : ""}` : ""}
        </span>
      </div>
      {content ? (
        <pre className="m-0 max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words px-2 py-1.5 text-[0.74rem] leading-relaxed">{shown}</pre>
      ) : null}
      {toolCalls?.length ? (
        <div className="border-t border-soft-border px-2 py-1.5">
          {toolCalls.map((call, index) => (
            <div key={index} className="text-[0.74rem]">
              <span className="text-warn">{call.name}</span>
              <span className="text-muted">({call.arguments})</span>
            </div>
          ))}
        </div>
      ) : null}
      {isLong ? (
        <button
          type="button"
          onClick={() => setExpanded((value) => !value)}
          className="w-full cursor-pointer border-t border-soft-border bg-panel-strong px-2 py-1 text-left text-[0.65rem] uppercase tracking-[0.06em] text-muted hover:text-foreground"
        >
          {expanded ? "collapse" : `expand (${fmtTokens(content.length)} chars)`}
        </button>
      ) : null}
    </div>
  );
}

export function RequestDetailModal({ id, onClose }: { id: string; onClose: () => void }) {
  const [record, setRecord] = useState<DashboardRequest>();
  const [error, setError] = useState<string>();
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    let disposed = false;
    fetchRequestDetail(id)
      .then((data) => {
        if (!disposed) setRecord(data);
      })
      .catch((cause) => {
        if (!disposed) setError(cause instanceof Error ? cause.message : String(cause));
      });
    return () => {
      disposed = true;
    };
  }, [id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const detail = record?.detail;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 p-4 sm:p-8" onClick={onClose}>
      <div className="w-full max-w-[860px] border border-border bg-panel" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center justify-between border-b border-border px-3 py-2">
          <span className="text-[0.72rem] uppercase tracking-[0.08em] text-muted">
            request {id} {record ? `· ${fmtClock(record.startedAt)}` : ""}
          </span>
          <button type="button" onClick={onClose} className="cursor-pointer border border-soft-border px-2 text-[0.72rem] text-muted hover:text-foreground">
            esc
          </button>
        </div>

        {error ? <div className="p-3 text-[0.78rem] text-err">{error}</div> : null}
        {!record && !error ? <div className="p-3 text-[0.78rem] text-muted">loading…</div> : null}

        {record ? (
          <div className="grid gap-3 p-3">
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-4">
              <Field label="model">{record.model}</Field>
              <Field label="status">
                {record.status === "ok" ? <Tag tone="ok">ok</Tag> : record.status === "error" ? <Tag tone="err">error</Tag> : <Tag tone="warn">{record.status}</Tag>}
                {record.finishReason ? <span className="ml-2 text-muted">{record.finishReason}</span> : null}
              </Field>
              <Field label="endpoint">{record.endpoint} · {record.stream ? "sse" : "json"}</Field>
              <Field label="timing">
                {record.loadMs !== undefined && record.loadMs > 500 ? `load ${fmtDuration(record.loadMs)} · ` : ""}
                ttft {fmtDuration(record.ttftMs)} · total {fmtDuration(record.durationMs)}
              </Field>
              <Field label="tokens">
                in {fmtTokens(record.promptTokens)} · out {fmtTokens(record.completionTokens)}
                {record.tokensPerSecond ? ` · ${record.tokensPerSecond} tok/s` : ""}
              </Field>
              <Field label="kv cache">
                {record.sideRequest
                  ? "side request"
                  : record.cacheHit === true
                    ? `hit${record.reuseKind ? ` · ${record.reuseKind}` : ""}${record.reuseScope ? ` · ${record.reuseScope}` : ""}${record.reusedTokens ? ` · ${fmtTokens(record.reusedTokens)} reused` : ""}`
                    : record.cacheHit === false
                      ? "miss"
                      : "-"}
              </Field>
              <Field label="params">
                {detail ? [
                  detail.params.temperature !== undefined ? `temp ${detail.params.temperature}` : null,
                  detail.params.topP !== undefined ? `top_p ${detail.params.topP}` : null,
                  detail.params.maxTokens !== undefined ? `max ${fmtTokens(detail.params.maxTokens)}` : null,
                  detail.params.stop?.length ? `stop [${detail.params.stop.join(", ")}]` : null,
                ].filter(Boolean).join(" · ") || "-" : "-"}
              </Field>
              <Field label="tools offered">
                {detail?.toolNames.length ? `${detail.toolNames.length}: ${detail.toolNames.slice(0, 6).join(", ")}${detail.toolNames.length > 6 ? "…" : ""}` : "none"}
              </Field>
            </div>

            {record.error ? (
              <div className="border border-err/40 bg-panel-strong p-2 text-[0.76rem] text-err">{record.error}</div>
            ) : null}

            {detail ? (
              <>
                <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">
                  conversation ({record.messageCount ?? detail.messages.length} messages{detail.droppedMessages ? `, first ${detail.droppedMessages} hidden` : ""})
                </div>
                <div className="grid max-h-[400px] gap-1.5 overflow-y-auto">
                  {detail.messages.map((message, index) => (
                    <MessageCard key={index} {...message} />
                  ))}
                </div>

                {detail.response ? (
                  <>
                    <div className="text-[0.68rem] uppercase tracking-[0.08em] text-muted">response</div>
                    {detail.response.reasoning ? (
                      <MessageCard role="assistant" content={detail.response.reasoning} toolCalls={undefined} />
                    ) : null}
                    <MessageCard
                      role="assistant"
                      content={detail.response.content ?? ""}
                      toolCalls={detail.response.toolCalls}
                    />
                  </>
                ) : null}

                {detail.rawOutput ? (
                  <div>
                    <button
                      type="button"
                      onClick={() => setShowRaw((value) => !value)}
                      className="cursor-pointer border border-soft-border px-2 py-0.5 text-[0.68rem] uppercase tracking-[0.06em] text-muted hover:text-foreground"
                    >
                      {showRaw ? "hide" : "show"} raw model output
                    </button>
                    {showRaw ? (
                      <pre className="m-0 mt-1.5 max-h-[300px] overflow-y-auto whitespace-pre-wrap break-words border border-soft-border bg-panel-strong px-2 py-1.5 text-[0.72rem]">{detail.rawOutput}</pre>
                    ) : null}
                  </div>
                ) : null}
              </>
            ) : (
              <div className="text-[0.76rem] text-muted">no detail captured for this request</div>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}
