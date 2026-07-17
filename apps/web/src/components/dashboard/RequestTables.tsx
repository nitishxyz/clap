import type { DashboardRequest } from "@/lib/api";
import { fmtClock, fmtDuration, fmtTokens } from "@/lib/format";
import { Empty, Panel, Table, Tag, Td } from "./Shared";

function StatusTag({ status }: { status: DashboardRequest["status"] }) {
  if (status === "ok") return <Tag tone="ok">ok</Tag>;
  if (status === "error") return <Tag tone="err">error</Tag>;
  if (status === "cancelled") return <Tag tone="warn">cancelled</Tag>;
  return <Tag>{status}</Tag>;
}

function PhaseTag({ request }: { request: DashboardRequest }) {
  const { phase } = request;
  if (phase === "loading") return <Tag tone="warn">loading model</Tag>;
  if (phase === "queued") return <Tag>queued</Tag>;
  if (phase === "prefill") {
    const pct = request.prefillTotal
      ? ` ${Math.min(99, Math.round(((request.prefillDone ?? 0) / request.prefillTotal) * 100))}%`
      : "";
    return <Tag tone="pin">prefill{pct}</Tag>;
  }
  if (phase === "decode") return <Tag tone="ok">decoding</Tag>;
  return <Tag>{phase}</Tag>;
}

// Stable pastel from the conversation fingerprint so rows belonging to the
// same session share a color without a fixed palette.
function ConvTag({ conversation }: { conversation?: string }) {
  if (!conversation) return <span className="text-muted">-</span>;
  const hue = parseInt(conversation, 16) % 360;
  return (
    <span
      className="inline-block border px-1.5 text-[0.7rem] tabular-nums"
      style={{ borderColor: `hsl(${hue} 45% 45%)`, color: `hsl(${hue} 65% 70%)` }}
      title="conversation fingerprint (same session groups together)"
    >
      {conversation.slice(0, 4)}
    </span>
  );
}

const phaseOrder: Record<DashboardRequest["phase"], number> = { decode: 0, prefill: 1, loading: 2, queued: 3, done: 4 };

function CacheTag({ request }: { request: DashboardRequest }) {
  if (request.sideRequest) return <Tag tone="warn">side</Tag>;
  const slot = request.slot !== undefined ? ` s${request.slot}` : "";
  if (request.cacheHit === true) {
    return <Tag tone="hit">hit{request.reusedTokens ? ` ${fmtTokens(request.reusedTokens)}` : ""}{slot}</Tag>;
  }
  if (request.cacheHit === false) return <Tag>miss{slot}</Tag>;
  return <span className="text-muted">-</span>;
}

export function ActiveRequests({ requests, now, onSelect }: { requests: DashboardRequest[]; now: number; onSelect: (id: string) => void }) {
  const sorted = [...requests].sort((a, b) => (phaseOrder[a.phase] - phaseOrder[b.phase]) || a.startedAt - b.startedAt);
  return (
    <Panel title="active requests" count={requests.length || ""}>
      {requests.length ? (
        <Table headers={["started", "conv", "model", "phase", "elapsed", { label: "msgs", numeric: true }, "endpoint", "stream"]}>
          {sorted.map((request) => (
            <tr key={request.id} className="cursor-pointer hover:bg-panel-strong" onClick={() => onSelect(request.id)}>
              <Td>{fmtClock(request.startedAt)}</Td>
              <Td><ConvTag conversation={request.conversation} /></Td>
              <Td className="max-w-[260px] overflow-hidden text-ellipsis">{request.model}</Td>
              <Td><PhaseTag request={request} /></Td>
              <Td>{fmtDuration(now - request.startedAt)}</Td>
              <Td numeric>{request.messageCount ?? "-"}</Td>
              <Td>{request.endpoint}</Td>
              <Td>{request.stream ? "sse" : "json"}</Td>
            </tr>
          ))}
        </Table>
      ) : (
        <Empty>idle</Empty>
      )}
    </Panel>
  );
}

export function RecentRequests({ requests, onSelect }: { requests: DashboardRequest[]; onSelect: (id: string) => void }) {
  return (
    <Panel title="recent requests" count={requests.length ? `${requests.length} shown · click a row for detail` : ""}>
      {requests.length ? (
        <Table
          headers={[
            "time",
            "conv",
            "model",
            "status",
            { label: "queue", numeric: true },
            { label: "load", numeric: true },
            { label: "ttft", numeric: true },
            { label: "total", numeric: true },
            { label: "in", numeric: true },
            { label: "out", numeric: true },
            { label: "tok/s", numeric: true },
            "cache",
            "finish",
            { label: "tools", numeric: true },
          ]}
        >
          {requests.map((request) => (
            <tr key={request.id} className="cursor-pointer hover:bg-panel-strong" onClick={() => onSelect(request.id)}>
              <Td>{fmtClock(request.startedAt)}</Td>
              <Td><ConvTag conversation={request.conversation} /></Td>
              <Td className="max-w-[240px] overflow-hidden text-ellipsis" title={request.error}>
                {request.model}
              </Td>
              <Td><StatusTag status={request.status} /></Td>
              <Td numeric>{request.queuedMs !== undefined && request.queuedMs > 500 ? fmtDuration(request.queuedMs) : "-"}</Td>
              <Td numeric>{request.loadMs !== undefined && request.loadMs > 500 ? fmtDuration(request.loadMs) : "-"}</Td>
              <Td numeric>{fmtDuration(request.ttftMs)}</Td>
              <Td numeric>{fmtDuration(request.durationMs)}</Td>
              <Td numeric>{fmtTokens(request.promptTokens)}</Td>
              <Td numeric>{fmtTokens(request.completionTokens)}</Td>
              <Td numeric>{request.tokensPerSecond ?? "-"}</Td>
              <Td><CacheTag request={request} /></Td>
              <Td>{request.finishReason ?? "-"}</Td>
              <Td numeric>{request.toolCalls ?? "-"}</Td>
            </tr>
          ))}
        </Table>
      ) : (
        <Empty>no requests yet</Empty>
      )}
    </Panel>
  );
}
