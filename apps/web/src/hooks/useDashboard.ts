import { useEffect, useRef, useState } from "react";
import { fetchDashboard, type DashboardData } from "@/lib/api";

const POLL_MS = 2000;

export type DashboardState = {
  data?: DashboardData;
  connected: boolean;
  refreshedAt?: number;
};

export function useDashboard(): DashboardState {
  const [state, setState] = useState<DashboardState>({ connected: false });
  const timer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    let disposed = false;
    const tick = async () => {
      try {
        const data = await fetchDashboard();
        if (!disposed) setState({ data, connected: true, refreshedAt: Date.now() });
      } catch {
        if (!disposed) setState((previous) => ({ ...previous, connected: false }));
      }
      if (!disposed) timer.current = setTimeout(tick, POLL_MS);
    };
    void tick();
    return () => {
      disposed = true;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  return state;
}
