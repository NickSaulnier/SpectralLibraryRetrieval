import { useEffect, useRef } from "react";

export const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

export type Trace = Record<string, unknown>;
export type Layout = Record<string, unknown>;

// One shared import: every chart waits on the same load
let plotly: Promise<typeof import("plotly.js-basic-dist-min")> | null = null;
export const loadPlotly = () => (plotly ??= import("plotly.js-basic-dist-min"));

/** Wavelengths with a null wherever bands are missing (a step over `maxStep` nm), so lines
 * break across gaps such as the water-vapour bands instead of joining over them. Use the
 * returned `y` on every series plotted against the returned `x`. */
export function gapped(x: number[], maxStep = 40): { x: (number | null)[]; y: (ys: number[]) => (number | null)[] } {
  const breaks = new Set(x.flatMap((w, i) => (i > 0 && w - x[i - 1] > maxStep ? [i] : [])));
  const fill = <T,>(a: T[]) => a.flatMap((v, i) => (breaks.has(i) ? [null, v] : [v]));
  return { x: fill(x), y: fill };
}

/** Shaded spectral windows, for the shapes of a layout. */
export const windowShapes = (windows: [number, number][]) =>
  windows.map(([lo, hi]) => ({
    type: "rect", xref: "x", yref: "paper", x0: lo, x1: hi, y0: 0, y1: 1,
    fillcolor: css("--window"), line: { width: 0 }, layer: "below",
  }));

/** A Plotly chart in the app's theme. Plotly is loaded on first use, so pages without a
 * chart don't pay for it. `layout` is merged over the theme defaults. */
export function Plot({
  traces,
  layout,
  label,
  className = "plot",
}: {
  traces: Trace[];
  layout?: Layout;
  label: string;
  className?: string;
}) {
  const el = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    loadPlotly().then(({ default: Plotly }) => {
      if (cancelled || !el.current) return;
      const axis = { gridcolor: css("--line"), zeroline: false };
      const l = layout ?? {};
      Plotly.react(
        el.current,
        traces,
        {
          margin: { l: 48, r: 10, t: 16, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: css("--ink"), size: 11 },
          legend: { orientation: "h", y: -0.3 },
          ...l,
          xaxis: { ...axis, ...(l.xaxis as object) },
          yaxis: { ...axis, ...(l.yaxis as object) },
        },
        { displayModeBar: false, responsive: true },
      );
    }).catch((err) => console.error(`chart "${label}" failed:`, err));
    return () => {
      cancelled = true;
    };
  }, [traces, layout, label]);

  return <div ref={el} className={className} role="img" aria-label={label} />;
}
