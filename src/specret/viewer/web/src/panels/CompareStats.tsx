import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { api } from "../api/client";
import type { Comparison, FitStats } from "../api/types";
import { css, Plot } from "../charts/Plot";
import { fmt } from "../lib";
import type { Bounds } from "../map/MapView";

/** The visible area, rounded so small pans reuse the cached comparison. */
export const roundBounds = (b: Bounds): Bounds => b.map((v) => Math.round(v * 1e4) / 1e4) as Bounds;

/** Ends of a line through the origin with `slope`, across the sample's range. */
export function lineThrough(slope: number, xs: number[]): { x: number[]; y: number[] } {
  const lo = Math.min(0, ...xs);
  const hi = Math.max(0, ...xs);
  return { x: [lo, hi], y: [slope * lo, slope * hi] };
}

/** Our enhancement against its published counterpart over the visible area: r, slopes and
 * noise on plume and scene pixels, with a scatter of the two maps. */
export function CompareStats({ sceneKey, layer, view }: { sceneKey: string; layer: string; view: Bounds | null }) {
  const [whole, setWhole] = useState(false);
  const bbox = whole || !view ? undefined : roundBounds(view);
  const q = useQuery({
    queryKey: ["compare", sceneKey, layer, bbox],
    queryFn: () => api.compare(sceneKey, layer, bbox),
    placeholderData: (prev) => prev,
  });
  const c = q.data;
  return (
    <div className="compare-stats">
      <div className="chips-head hint">
        <label>
          <input type="checkbox" checked={whole} onChange={(e) => setWhole(e.target.checked)} /> whole scene
        </label>
        {q.isFetching && " · updating…"}
      </div>
      {q.isError && <p className="hint">{(q.error as Error).message}</p>}
      {c && (
        <>
          <table className="values">
            <thead>
              <tr>
                <th>{whole ? "Scene" : "In view"}</th>
                <th className="num">n</th>
                <th className="num">r</th>
                <th className="num">Deming</th>
                <th className="num">OLS</th>
              </tr>
            </thead>
            <tbody>
              <Row name="Plume pixels" f={c.plume_pixels} />
              <Row name="All pixels" f={c.scene_pixels} />
            </tbody>
          </table>
          <p className="hint">
            Background σ: ours {fmt(c.background_sigma_ours)}, {c.ref} {fmt(c.background_sigma_ref)} {c.units}. Plume
            pixels: {c.recipe}.
          </p>
          <Scatter c={c} />
        </>
      )}
    </div>
  );
}

function Row({ name, f }: { name: string; f: FitStats }) {
  return (
    <tr>
      <td>{name}</td>
      <td className="num">{f.n.toLocaleString()}</td>
      <td className="num">{fmt(f.r, 3)}</td>
      <td className="num">{fmt(f.deming_slope, 2)}</td>
      <td className="num">{fmt(f.slope, 2)}</td>
    </tr>
  );
}

function Scatter({ c }: { c: Comparison }) {
  const traces = useMemo(() => {
    const all = [...c.sample.plume.ref, ...c.sample.background.ref];
    const one = lineThrough(1, all);
    const out: Record<string, unknown>[] = [
      {
        x: c.sample.background.ref, y: c.sample.background.ours, name: "background", mode: "markers",
        marker: { size: 3, color: css("--muted"), opacity: 0.35 },
      },
      {
        x: c.sample.plume.ref, y: c.sample.plume.ours, name: "plume pixels", mode: "markers",
        marker: { size: 4, color: css("--m1"), opacity: 0.6 },
      },
      { ...one, name: "1:1", mode: "lines", line: { color: css("--ink"), width: 1, dash: "dot" } },
    ];
    if (c.plume_pixels.deming_slope != null)
      out.push({
        ...lineThrough(c.plume_pixels.deming_slope, all),
        name: `Deming ${c.plume_pixels.deming_slope.toFixed(2)}`,
        mode: "lines",
        line: { color: css("--m2"), width: 1.5 },
      });
    return out;
  }, [c]);
  return (
    <Plot
      label="Ours against the published map"
      traces={traces}
      layout={{
        xaxis: { title: { text: `${c.ref} (${c.units})` } },
        yaxis: { title: { text: `${c.ours} (${c.units})` } },
      }}
    />
  );
}
