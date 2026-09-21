import { useEffect, useState } from "react";

interface State<T> { data: T | null; error: string | null; loading: boolean }

/** Minimal fetch hook. Re-runs when `deps` change. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[] = []): State<T> {
  const [s, set] = useState<State<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    set((p) => ({ ...p, loading: true, error: null }));
    fn().then(
      (d) => alive && set({ data: d, error: null, loading: false }),
      (e: Error) => alive && set({ data: null, error: e.message, loading: false }),
    );
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return s;
}
