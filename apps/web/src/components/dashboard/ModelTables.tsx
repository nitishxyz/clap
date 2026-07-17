import { useState } from "react";
import { cancelDownload, loadModel, pullModel, removeModel, unloadModel, type DashboardDownload, type DashboardLoadedModel, type DashboardModel } from "@/lib/api";
import { fmtBytes, fmtDuration, fmtTokens } from "@/lib/format";
import type { ActionState } from "@/hooks/useActions";
import { Empty, Panel, Table, Tag, Td } from "./Shared";

function ActionButton({ label, busy, danger, onClick }: { label: string; busy?: boolean; danger?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      disabled={busy}
      onClick={(event) => {
        event.stopPropagation();
        onClick();
      }}
      className={`cursor-pointer border px-1.5 py-0 text-[0.68rem] uppercase tracking-[0.04em] disabled:cursor-wait disabled:opacity-40 ${
        danger
          ? "border-err/50 text-err hover:bg-err hover:text-background"
          : "border-soft-border text-muted hover:bg-foreground hover:text-background"
      }`}
    >
      {busy ? "…" : label}
    </button>
  );
}

export function LoadedModels({ models, now, actions }: { models: DashboardLoadedModel[]; now: number; actions: ActionState }) {
  return (
    <Panel title="loaded models" count={models.length || ""}>
      {models.length ? (
        <Table headers={["model", "backend", "state", { label: "mem", numeric: true }, { label: "cpu", numeric: true }, { label: "reqs", numeric: true }, "keep-alive", "expires", "last used", { label: "pid", numeric: true }, ""]}>
          {models.map((entry) => (
            <tr key={entry.key}>
              <Td className="max-w-[260px] overflow-hidden text-ellipsis" title={entry.localPath}>{entry.id}</Td>
              <Td>{entry.backend}</Td>
              <Td>{entry.state === "active" ? <Tag tone="ok">active</Tag> : <Tag>{entry.state}</Tag>}</Td>
              <Td numeric>{entry.usage ? fmtBytes(entry.usage.rssBytes) : "-"}</Td>
              <Td numeric>{entry.usage ? `${entry.usage.cpuPercent.toFixed(0)}%` : "-"}</Td>
              <Td numeric>{entry.activeRequests}</Td>
              <Td>{entry.keepAlive}</Td>
              <Td>
                {entry.pinned ? (
                  <Tag tone="pin">pinned</Tag>
                ) : entry.expiresAt ? (
                  fmtDuration(new Date(entry.expiresAt).getTime() - now)
                ) : (
                  "-"
                )}
              </Td>
              <Td>{fmtDuration(now - new Date(entry.lastUsedAt).getTime())} ago</Td>
              <Td numeric>{entry.worker?.pid ?? "-"}</Td>
              <Td>
                <span className="flex gap-1">
                  {entry.pinned ? (
                    <ActionButton
                      label="unpin"
                      busy={actions.busy[`pin:${entry.id}`]}
                      onClick={() => actions.run(`pin:${entry.id}`, () => loadModel(entry.id, "5m"))}
                    />
                  ) : (
                    <ActionButton
                      label="pin"
                      busy={actions.busy[`pin:${entry.id}`]}
                      onClick={() => actions.run(`pin:${entry.id}`, () => loadModel(entry.id, "always"))}
                    />
                  )}
                  <ActionButton
                    label="unload"
                    danger
                    busy={actions.busy[`unload:${entry.id}`]}
                    onClick={() => actions.run(`unload:${entry.id}`, () => unloadModel(entry.id))}
                  />
                </span>
              </Td>
            </tr>
          ))}
        </Table>
      ) : (
        <Empty>no models resident — send a request, `clap load &lt;model&gt;`, or press load below</Empty>
      )}
    </Panel>
  );
}

export function Downloads({ downloads, actions }: { downloads: DashboardDownload[]; actions: ActionState }) {
  const [pullInput, setPullInput] = useState("");
  const active = downloads.filter((download) => download.status === "running" || download.status === "queued");
  const shown = active.length ? active : downloads.slice(-5);
  const submitPull = () => {
    const model = pullInput.trim();
    if (!model) return;
    actions.run(`pull:${model}`, () => pullModel(model));
    setPullInput("");
  };
  return (
    <Panel
      title="downloads"
      count={
        <span className="flex items-center gap-2 normal-case">
          <input
            value={pullInput}
            onChange={(event) => setPullInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") submitPull();
            }}
            placeholder="org/repo to pull…"
            className="w-56 border border-soft-border bg-background px-2 py-0.5 text-[0.72rem] text-foreground placeholder:text-muted focus:outline-none"
          />
          <ActionButton label="pull" onClick={submitPull} />
          {active.length ? <span className="text-foreground">{active.length} active</span> : null}
        </span>
      }
    >
      {shown.length ? (
        <Table headers={["model", "status", "progress", "size", ""]}>
          {shown.map((download) => {
            const pct = download.totalBytes
              ? Math.min(100, Math.round((download.bytesReceived / download.totalBytes) * 100))
              : null;
            const running = download.status === "running" || download.status === "queued";
            return (
              <tr key={download.id}>
                <Td className="max-w-[260px] overflow-hidden text-ellipsis" title={download.currentFile}>{download.model}</Td>
                <Td>
                  {download.status === "completed" ? (
                    <Tag tone="ok">completed</Tag>
                  ) : download.status === "failed" ? (
                    <Tag tone="err">failed</Tag>
                  ) : (
                    <Tag>{download.status}</Tag>
                  )}
                </Td>
                <Td>
                  {pct === null ? (
                    "-"
                  ) : (
                    <span className="flex items-center gap-2">
                      <span className="relative inline-block h-1.5 w-32 border border-soft-border bg-panel-strong">
                        <i className="absolute inset-y-0 left-0 bg-accent" style={{ width: `${pct}%` }} />
                      </span>
                      <span className="text-muted">{pct}%</span>
                    </span>
                  )}
                </Td>
                <Td>
                  {fmtBytes(download.bytesReceived)} / {fmtBytes(download.totalBytes)}
                </Td>
                <Td>
                  {running ? (
                    <ActionButton
                      label="cancel"
                      danger
                      busy={actions.busy[`cancel:${download.id}`]}
                      onClick={() => actions.run(`cancel:${download.id}`, () => cancelDownload(download.id))}
                    />
                  ) : null}
                </Td>
              </tr>
            );
          })}
        </Table>
      ) : (
        <Empty>no downloads — paste a HuggingFace repo above to pull one</Empty>
      )}
    </Panel>
  );
}

export function ModelCache({ models, loaded, actions }: { models: DashboardModel[]; loaded: DashboardLoadedModel[]; actions: ActionState }) {
  const cached = models.filter((model) => model.status === "available");
  const loadedIds = new Set(loaded.map((entry) => entry.id));
  const [confirmRemove, setConfirmRemove] = useState<string>();
  return (
    <Panel title="model cache" count={`${cached.length} on disk`}>
      {cached.length ? (
        <Table headers={["model", "backend", "format", "quant", { label: "context", numeric: true }, "capabilities", ""]}>
          {cached.map((model) => {
            const caps = [
              model.capabilities?.toolCall ? "tools" : null,
              model.capabilities?.reasoning ? "reasoning" : null,
              model.modalities?.input?.includes("image") ? "vision" : null,
            ]
              .filter(Boolean)
              .join(" · ");
            const isLoaded = loadedIds.has(model.id);
            return (
              <tr key={model.id}>
                <Td className="max-w-[300px] overflow-hidden text-ellipsis" title={model.id}>{model.displayName ?? model.id}</Td>
                <Td>{model.backend}</Td>
                <Td>{model.format}</Td>
                <Td>{model.quantization ?? "-"}</Td>
                <Td numeric>{model.limit?.context ? fmtTokens(model.limit.context) : "-"}</Td>
                <Td className="text-muted">{caps || "-"}</Td>
                <Td>
                  <span className="flex gap-1">
                    {isLoaded ? (
                      <Tag tone="ok">loaded</Tag>
                    ) : (
                      <ActionButton
                        label="load"
                        busy={actions.busy[`load:${model.id}`]}
                        onClick={() => actions.run(`load:${model.id}`, () => loadModel(model.id))}
                      />
                    )}
                    {confirmRemove === model.id ? (
                      <>
                        <ActionButton
                          label="confirm rm"
                          danger
                          busy={actions.busy[`rm:${model.id}`]}
                          onClick={() => {
                            setConfirmRemove(undefined);
                            actions.run(`rm:${model.id}`, () => removeModel(model.id));
                          }}
                        />
                        <ActionButton label="keep" onClick={() => setConfirmRemove(undefined)} />
                      </>
                    ) : (
                      <ActionButton label="rm" danger onClick={() => setConfirmRemove(model.id)} />
                    )}
                  </span>
                </Td>
              </tr>
            );
          })}
        </Table>
      ) : (
        <Empty>no models cached — `clap pull &lt;model&gt;` or use the pull box above</Empty>
      )}
    </Panel>
  );
}
