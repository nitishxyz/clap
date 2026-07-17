import { useRef, useState } from "react";

export type ActionState = {
  busy: Record<string, boolean>;
  error?: string;
  run: (key: string, action: () => Promise<unknown>) => void;
  dismissError: () => void;
};

export function useActions(onDone?: () => void): ActionState {
  const [busy, setBusy] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string>();
  const errorTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const run = (key: string, action: () => Promise<unknown>) => {
    setBusy((current) => ({ ...current, [key]: true }));
    action()
      .then(() => onDone?.())
      .catch((cause) => {
        setError(cause instanceof Error ? cause.message : String(cause));
        if (errorTimer.current) clearTimeout(errorTimer.current);
        errorTimer.current = setTimeout(() => setError(undefined), 8000);
      })
      .finally(() => {
        setBusy((current) => {
          const next = { ...current };
          delete next[key];
          return next;
        });
      });
  };

  return { busy, error, run, dismissError: () => setError(undefined) };
}
