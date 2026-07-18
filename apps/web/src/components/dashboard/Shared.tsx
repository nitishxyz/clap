import type { ReactNode } from "react";

export function Panel({ title, count, children }: { title: string; count?: ReactNode; children: ReactNode }) {
  return (
    <section className="w-full min-w-0 border border-border bg-panel">
      <h2 className="m-0 flex items-baseline justify-between border-b border-border px-3 py-2 text-[0.72rem] uppercase tracking-[0.08em] text-muted">
        {title}
        {count !== undefined && count !== "" ? <span className="text-foreground">{count}</span> : null}
      </h2>
      <div className="overflow-x-auto">{children}</div>
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="p-3 text-[0.78rem] text-muted">{children}</div>;
}

export function Table({ headers, children }: { headers: Array<string | { label: string; numeric?: boolean }>; children: ReactNode }) {
  return (
    <table className="w-full border-collapse text-[0.78rem]">
      <thead>
        <tr>
          {headers.map((header) => {
            const config = typeof header === "string" ? { label: header, numeric: false } : header;
            return (
              <th
                key={config.label}
                className={`whitespace-nowrap border-b border-soft-border px-3 py-1.5 text-left text-[0.66rem] font-normal uppercase tracking-[0.06em] text-muted ${config.numeric ? "text-right" : ""}`}
              >
                {config.label}
              </th>
            );
          })}
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}

export function Td({ children, numeric, className, title }: { children: ReactNode; numeric?: boolean; className?: string; title?: string }) {
  return (
    <td
      title={title}
      className={`whitespace-nowrap border-b border-soft-border px-3 py-1.5 [tr:last-child_&]:border-b-0 ${numeric ? "text-right tabular-nums" : ""} ${className ?? ""}`}
    >
      {children}
    </td>
  );
}

const tagColors = {
  default: "border-soft-border text-muted",
  ok: "border-ok text-ok",
  err: "border-err text-err",
  warn: "border-warn text-warn",
  hit: "border-cache text-cache",
  pin: "border-thinking text-thinking",
} as const;

export function Tag({ tone = "default", children }: { tone?: keyof typeof tagColors; children: ReactNode }) {
  return <span className={`inline-block border px-1.5 text-[0.7rem] ${tagColors[tone]}`}>{children}</span>;
}
