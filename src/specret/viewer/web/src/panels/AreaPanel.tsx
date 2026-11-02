import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api, type AreaSummary, type Polygon } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { isActive, useJob } from "../hooks/useJob";
import { download, fmt, pretty, toCsv } from "../lib";
import type { DrawMode } from "../map/MapView";

interface Props {
  sceneKey: string | null;
  canUnmix: boolean;
  drawMode: DrawMode;
  selection: Polygon | null;
  onDraw: (mode: DrawMode) => void;
  onClear: () => void;
  onUnmixed: () => void;
}

export function selectionGeojson(poly: Polygon, summary?: AreaSummary): string {
  return JSON.stringify(
    {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: summary
            ? { area_km2: summary.area_km2, minerals: summary.minerals?.groups, plumes: summary.plumes }
            : {},
          geometry: { type: "Polygon", coordinates: [[...poly, poly[0]]] },
        },
      ],
    },
    null,
    1,
  );
}

export function summaryCsv(s: AreaSummary): string {
  const rows: unknown[][] = [["section", "name", "pixels", "fraction_or_median", "p05", "p95", "units"]];
  for (const g of s.minerals?.groups ?? []) rows.push(["mineral", g.group, g.pixels, g.fraction.toFixed(4), "", "", ""]);
  for (const l of s.layers) rows.push(["layer", l.title, l.pixels, l.median, l.p05, l.p95, l.units]);
  for (const [gas, v] of Object.entries(s.plumes.by_gas)) rows.push(["plumes", gas, v.count, v.total_q_kg_h, "", "", "kg/h"]);
  return toCsv(rows);
}

/** Draw an area on the map, then see what is inside it, unmix it, or export it. */
export function AreaPanel(p: Props) {
  const [minP, setMinP] = useState(0.3);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const { onUnmixed } = p;
  useEffect(() => {
    if (job?.state === "done") onUnmixed();
  }, [job?.state, onUnmixed]);
  useEffect(() => setJobId(null), [p.sceneKey]);

  const summary = useQuery({
    queryKey: ["summary", p.sceneKey, p.selection, minP],
    queryFn: () => api.summary(p.sceneKey!, p.selection!, minP),
    enabled: !!p.sceneKey && !!p.selection,
  });
  const s = summary.data;

  return (
    <div className="area">
      <div className="controls">
        <button className={p.drawMode === "box" ? "primary" : ""} onClick={() => p.onDraw(p.drawMode === "box" ? null : "box")}>
          Draw box
        </button>
        <button
          className={p.drawMode === "polygon" ? "primary" : ""}
          onClick={() => p.onDraw(p.drawMode === "polygon" ? null : "polygon")}
        >
          Draw polygon
        </button>
        <button disabled={!p.selection} onClick={p.onClear}>
          Clear
        </button>
      </div>
      <p className="hint">
        {p.drawMode === "box"
          ? "Drag on the map to draw a box."
          : p.drawMode === "polygon"
            ? "Click the map to add corners; double-click to finish."
            : p.selection
              ? null
              : "Draw an area to summarize it."}
      </p>
      {summary.isError && <p className="hint">{(summary.error as Error).message}</p>}
      {s && p.selection && (
        <>
          <p>
            <b>{fmt(s.area_km2, 2)} km²</b>
            {s.plumes.count > 0 &&
              ` · ${s.plumes.count} plume${s.plumes.count > 1 ? "s" : ""}: ${Object.entries(s.plumes.by_gas)
                .map(([g, v]) => `${g} ${fmt(v.count)} (${fmt(v.total_q_kg_h)} kg/h)`)
                .join(", ")}`}
          </p>
          {s.minerals && (
            <>
              <div className="panel-head">
                <h3>Minerals ({s.minerals.pixels.toLocaleString()} px)</h3>
                <label className="hint">
                  unknown below P {minP.toFixed(2)}
                  <input type="range" min={0} max={0.9} step={0.05} value={minP} onChange={(e) => setMinP(Number(e.target.value))} />
                </label>
              </div>
              <div className="bars">
                {s.minerals.groups.slice(0, 12).map((g) => (
                  <div key={g.group} className="bar-row">
                    <span>{pretty(g.group)}</span>
                    <span className="bar">
                      <i style={{ width: `${100 * g.fraction}%` }} />
                    </span>
                    <span className="num">{(100 * g.fraction).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}
          {s.layers.some((l) => l.pixels > 0) && (
            <table className="values">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th className="num">Median</th>
                  <th className="num">5–95%</th>
                  <th>Histogram</th>
                </tr>
              </thead>
              <tbody>
                {s.layers.filter((l) => l.pixels > 0).map((l) => (
                  <tr key={l.id}>
                    <td>{l.title}</td>
                    <td className="num">{fmt(l.median, Math.abs(l.median ?? 0) < 10 ? 2 : 0)}</td>
                    <td className="num">
                      {fmt(l.p05, 1)} – {fmt(l.p95, 1)} {l.units}
                    </td>
                    <td>
                      <Spark counts={l.hist_counts} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="controls">
            <button onClick={() => download("area.geojson", selectionGeojson(p.selection!, s), "application/geo+json")}>
              GeoJSON
            </button>
            <button onClick={() => download("area_summary.csv", summaryCsv(s))}>Summary CSV</button>
            {p.canUnmix && (
              <button
                className="primary"
                disabled={isActive(job)}
                onClick={async () => setJobId((await api.unmixArea(p.sceneKey!, p.selection!)).job_id)}
              >
                Unmix this area
              </button>
            )}
          </div>
          {job && isActive(job) && (
            <>
              <JobProgress job={job} label="Unmixing" />
              <div className="hint">Up to 3,000 pixels; larger areas use a coarser grid.</div>
            </>
          )}
          {job?.state === "done" && <p className="hint">Done: the fraction layers are in the Layers tab, under Unmixing.</p>}
          {job?.state === "failed" && <p className="hint">Unmixing failed: {job.error?.split("\n")[0]}</p>}
        </>
      )}
    </div>
  );
}

function Spark({ counts }: { counts: number[] }) {
  const max = Math.max(1, ...counts);
  return (
    <svg className="spark" viewBox={`0 0 ${counts.length} 10`} preserveAspectRatio="none" aria-hidden>
      {counts.map((c, i) => (
        <rect key={i} x={i + 0.1} width={0.8} y={10 - (10 * c) / max} height={(10 * c) / max} />
      ))}
    </svg>
  );
}
