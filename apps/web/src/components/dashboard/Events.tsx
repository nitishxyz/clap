import type { ServerEvent } from "@/lib/api";
import { fmtClock } from "@/lib/format";
import { Empty, Panel, Table, Tag, Td } from "./Shared";

function EventTag({ type }: { type: ServerEvent["type"] }) {
  if (type === "load") return <Tag tone="ok">load</Tag>;
  if (type === "unload") return <Tag tone="warn">unload</Tag>;
  if (type === "expire") return <Tag tone="warn">expire</Tag>;
  if (type === "error") return <Tag tone="err">error</Tag>;
  if (type === "download") return <Tag tone="hit">download</Tag>;
  return <Tag>server</Tag>;
}

export function Events({ events }: { events: ServerEvent[] }) {
  return (
    <Panel title="events" count={events.length || ""}>
      {events.length ? (
        <Table headers={["time", "type", "message"]}>
          {events.map((event) => (
            <tr key={event.id}>
              <Td>{fmtClock(event.at)}</Td>
              <Td><EventTag type={event.type} /></Td>
              <Td className="whitespace-normal break-words">{event.message}</Td>
            </tr>
          ))}
        </Table>
      ) : (
        <Empty>no events yet</Empty>
      )}
    </Panel>
  );
}
