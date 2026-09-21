import { useEffect, useRef, useState } from "react";

/**
 * Returns "flash-up" / "flash-down" for ~700ms whenever `value` changes.
 * Ticks arrive every ~1s so the class is cleared before the next update,
 * which lets the CSS animation restart cleanly.
 */
export function useFlash(value: number | null | undefined): string {
  const prev = useRef<number | null | undefined>(value);
  const [cls, setCls] = useState("");

  useEffect(() => {
    if (value == null || prev.current == null || value === prev.current) { prev.current = value; return; }
    const dir = value > prev.current ? "flash-up" : "flash-down";
    prev.current = value;
    setCls(dir);
    const t = window.setTimeout(() => setCls(""), 720);
    return () => window.clearTimeout(t);
  }, [value]);

  return cls;
}
