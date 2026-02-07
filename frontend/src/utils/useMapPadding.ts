import { useCallback, useEffect, useRef, useState } from "react";

export type MapPadding = { top: number; right: number; bottom: number; left: number };

const BASE_PADDING = 40;

/**
 * Dynamically measures UI overlays (sidebar, top bar) and returns
 * the padding the map should use so content isn't hidden behind them.
 * Uses ResizeObserver for live updates when panels expand/collapse.
 *
 * Returns callback refs — pass them as `ref={sidebarRef}` / `ref={topBarRef}`.
 */
export function useMapPadding() {
  const sidebarEl = useRef<HTMLElement | null>(null);
  const topBarEl = useRef<HTMLDivElement | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);

  const [padding, setPadding] = useState<MapPadding>({
    top: BASE_PADDING,
    right: BASE_PADDING,
    bottom: BASE_PADDING,
    left: BASE_PADDING,
  });

  const recalc = useCallback(() => {
    const sidebar = sidebarEl.current;
    const topBar = topBarEl.current;

    // Left: sidebar covers the full left edge
    const left = sidebar
      ? sidebar.offsetLeft + sidebar.offsetWidth + BASE_PADDING
      : BASE_PADDING;

    // Top: top bar covers the top — push content below it
    const top = topBar
      ? topBar.offsetTop + topBar.offsetHeight + BASE_PADDING
      : BASE_PADDING;

    // Right: the top bar only occupies the top-right *corner*, not the
    // full right edge. The top padding already pushes content below it, so
    // we only need a small right margin.  Don't use the top bar width
    // here — that makes the total horizontal padding exceed the map width.
    const right = BASE_PADDING;

    setPadding((prev) => {
      if (prev.left === left && prev.top === top && prev.right === right && prev.bottom === BASE_PADDING) return prev;
      return { top, right, bottom: BASE_PADDING, left };
    });
  }, []);

  // Rebuild observer whenever elements change
  const rebuildObserver = useCallback(() => {
    if (observerRef.current) observerRef.current.disconnect();
    const obs = new ResizeObserver(recalc);
    if (sidebarEl.current) obs.observe(sidebarEl.current);
    if (topBarEl.current) obs.observe(topBarEl.current);
    observerRef.current = obs;
    recalc();
  }, [recalc]);

  // Callback ref for sidebar
  const sidebarRef = useCallback(
    (node: HTMLElement | null) => {
      sidebarEl.current = node;
      rebuildObserver();
    },
    [rebuildObserver]
  );

  // Callback ref for top bar
  const topBarRef = useCallback(
    (node: HTMLDivElement | null) => {
      topBarEl.current = node;
      rebuildObserver();
    },
    [rebuildObserver]
  );

  // Also recalc on window resize
  useEffect(() => {
    window.addEventListener("resize", recalc);
    return () => window.removeEventListener("resize", recalc);
  }, [recalc]);

  // Cleanup observer on unmount
  useEffect(() => {
    return () => observerRef.current?.disconnect();
  }, []);

  return { sidebarRef, topBarRef, padding };
}
