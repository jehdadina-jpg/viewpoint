import { useEffect, useRef, useState } from "react";
import { wsUrl } from "../api/client";
import type { Tick, TickMsg } from "../types";

export type TickMap = Record<string, Tick>;
export type WsStatus = "connecting" | "live" | "down";

/**
 * Subscribes to the SIMULATED tick stream (/ws/ticks) with auto-reconnect.
 * Returns the latest tick per ticker; consumers diff against previous values
 * to drive the cell-flash animation.
 */
export function useTicks(): { ticks: TickMap; status: WsStatus; simulated: boolean } {
  const [ticks, setTicks] = useState<TickMap>({});
  const [status, setStatus] = useState<WsStatus>("connecting");
  const [simulated, setSimulated] = useState(true);
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(wsUrl());
      ws.onopen = () => { retry.current = 0; setStatus("live"); };
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data) as TickMsg;
        if (m.type !== "ticks") return;
        setSimulated(m.simulated);
        setTicks((prev) => {
          const next: TickMap = { ...prev };
          for (const t of m.ticks) next[t.ticker] = t;
          return next;
        });
      };
      ws.onclose = () => {
        if (closed) return;
        setStatus("down");
        const wait = Math.min(10000, 500 * 2 ** retry.current++);
        timer = window.setTimeout(connect, wait);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => { closed = true; window.clearTimeout(timer); ws?.close(); };
  }, []);

  return { ticks, status, simulated };
}
