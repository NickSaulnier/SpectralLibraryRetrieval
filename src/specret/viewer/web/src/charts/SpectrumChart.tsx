import { useEffect, useRef } from "react";

import type { Identification } from "../api/types";
import { css, loadPlotly } from "./Plot";

export const MATCH_COLORS = ["--m1", "--m2", "--m3"];

const scaled = (a: number[]) => {
  const m = Math.max(...a.filter(Number.isFinite));
  return a.map((v) => v / m);
};

/** The query spectrum over its top three matches, with diagnostic windows shaded. Plotly is
 * loaded on first use (charts/Plot.tsx), so pages without a chart don't pay for it. */
export function SpectrumChart({ result }: { result: Identification }) {
  const el = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    loadPlotly().then(({ default: Plotly }) => {
      if (cancelled || !el.current) return;
      const wl = result.wavelengths_nm;
      const traces = [
        { x: wl, y: scaled(result.query), name: "query", mode: "lines", line: { color: css("--query"), width: 2.5 } },
        ...result.matches.slice(0, 3).map((m, i) => ({
          x: wl,
          y: scaled(m.spectrum ?? []),
          name: m.material_name,
          mode: "lines",
          line: { color: css(MATCH_COLORS[i]), width: 1.5 },
        })),
      ];
      const windows = Object.entries(result.windows);
      const shapes = windows.map(([, [lo, hi]]) => ({
        type: "rect", xref: "x", yref: "paper", x0: lo, x1: hi, y0: 0, y1: 1,
        fillcolor: css("--window"), line: { width: 0 },
      }));
      // Windows in 2.0-2.5 um overlap, so their labels are staggered vertically.
      const annotations = windows.map(([name, [lo, hi]], i) => ({
        x: (lo + hi) / 2, y: 1, yref: "paper", yshift: (i % 3) * 12, text: name.replaceAll("_", "-"),
        showarrow: false, font: { size: 10, color: css("--muted") }, yanchor: "bottom",
      }));
      Plotly.react(
        el.current,
        traces,
        {
          margin: { l: 44, r: 10, t: 48, b: 40 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: css("--ink"), size: 12 },
          shapes,
          annotations,
          xaxis: { title: { text: "Wavelength (nm)" }, gridcolor: css("--line"), zeroline: false },
          yaxis: { title: { text: "Scaled reflectance" }, gridcolor: css("--line"), zeroline: false },
          legend: { orientation: "h", y: -0.25 },
        },
        { displayModeBar: false, responsive: true },
      );
    });
    return () => {
      cancelled = true;
    };
  }, [result]);

  return <div ref={el} className="plot" role="img" aria-label="Spectrum and top matches" />;
}
