import { useEffect, useState } from "react";
import { fetchDashboard, type DashboardData } from "@/lib/api";

const POLL_MS = 2000;
const SSE_RETRY_MS = 15_000;

export type DashboardState = {
  data?: DashboardData;
  connected: boolean;
  refreshedAt?: number;
  mode: "live" | "polling";
};

export function useDashboard(): DashboardState {
  const [state, setState] = useState<DashboardState>({ connected: false, mode: "live" });

  useEffect(() => {
    let disposed = false;
    let source: EventSource | undefined;
    let polling = false;
    let pollTimer: ReturnType<typeof setTimeout> | undefined;
    let retryTimer: ReturnType<typeof setTimeout> | undefined;

    const apply = (data: DashboardData, mode: "live" | "polling") => {
      if (!disposed) setState({ data, connected: true, refreshedAt: Date.now(), mode });
    };

    const poll = async () => {
      if (disposed || !polling) return;
      try {
        apply(await fetchDashboard(), "polling");
      } catch {
        if (!disposed) setState((previous) => ({ ...previous, connected: false }));
      }
      if (!disposed && polling) pollTimer = setTimeout(poll, POLL_MS);
    };

    const startPolling = () => {
      if (polling || disposed) return;
      polling = true;
      void poll();
    };

    const stopPolling = () => {
      polling = false;
      if (pollTimer) clearTimeout(pollTimer);
      pollTimer = undefined;
    };

    const connect = () => {
      if (disposed || typeof EventSource === "undefined") {
        startPolling();
        return;
      }
      source = new EventSource("/clap/v1/dashboard/stream");
      source.addEventListener("dashboard", (event) => {
        stopPolling();
        try {
          apply(JSON.parse((event as MessageEvent<string>).data) as DashboardData, "live");
        } catch {
          // ignore malformed frames; the next tick replaces them
        }
      });
      source.onerror = () => {
        if (disposed) return;
        setState((previous) => ({ ...previous, connected: false }));
        // Poll while the stream is down. EventSource retries on its own when
        // CONNECTING; if the browser gave up (CLOSED) schedule a fresh one.
        startPolling();
        if (source?.readyState === EventSource.CLOSED) {
          source.close();
          source = undefined;
          if (retryTimer) clearTimeout(retryTimer);
          retryTimer = setTimeout(connect, SSE_RETRY_MS);
        }
      };
    };

    connect();
    return () => {
      disposed = true;
      stopPolling();
      if (retryTimer) clearTimeout(retryTimer);
      source?.close();
    };
  }, []);

  return state;
}
