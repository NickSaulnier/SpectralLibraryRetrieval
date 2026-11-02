/** Exports (PLAN_VIEWER.md V5): the map as a PNG with legend and scale bar, the plume table
 * as CSV or GeoJSON, and a plume's evidence card as one self-contained HTML file. */
import type { Map as MlMap } from "maplibre-gl";

import type { PlumeExplanation, PlumeRow } from "./api/types";
import { loadPlotly } from "./charts/Plot";
import { download, fmt, pretty, toCsv } from "./lib";
import type { LayerState } from "./map/layerState";

const EARTH_M = 40_075_016.686;

/** Ground metres per CSS pixel at a latitude and Web Mercator zoom (512 px tiles). */
export const metresPerPixel = (lat: number, zoom: number) =>
  (EARTH_M * Math.cos((lat * Math.PI) / 180)) / 2 ** (zoom + 9);

/** A round scale-bar length (1, 2 or 5 × 10^n m) no longer than `maxPx`. */
export function scaleBar(mpp: number, maxPx: number): { metres: number; px: number; label: string } {
  const max = mpp * maxPx;
  const p = 10 ** Math.floor(Math.log10(max));
  const metres = [5, 2, 1].map((k) => k * p).find((m) => m <= max) ?? p;
  return { metres, px: metres / mpp, label: metres >= 1000 ? `${metres / 1000} km` : `${metres} m` };
}

const PLUME_COLUMNS: (keyof PlumeRow)[] = [
  "scene_id", "sensor", "plume_id", "gas", "confidence", "source_lat", "source_lon", "n_px",
  "max_alpha_ppmm", "q_kg_h", "sigma_q_kg_h", "q_csf_kg_h", "u10", "wind_source",
  "confusers_fitted", "carbonate_fraction", "overpasses_covering", "overpasses_seen",
];

export function plumesCsv(rows: PlumeRow[], scene?: string): string {
  return toCsv([
    PLUME_COLUMNS,
    ...rows.map((r) => PLUME_COLUMNS.map((k) => (k === "scene_id" ? (r.scene_id ?? scene) : r[k]))),
  ]);
}

/** Plume sources as points (outlines come with each scene's plumes.geojson). */
export function plumesGeojson(rows: PlumeRow[], scene?: string): string {
  const features = rows
    .filter((r) => r.source_lat != null && r.source_lon != null)
    .map((r) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [r.source_lon, r.source_lat] },
      properties: Object.fromEntries(PLUME_COLUMNS.map((k) => [k, k === "scene_id" ? (r.scene_id ?? scene) : (r[k] ?? null)])),
    }));
  return JSON.stringify({ type: "FeatureCollection", features }, null, 1);
}

/** The map as drawn now, with a title, a legend of the visible layers and a scale bar. */
export async function mapPng(m: MlMap, layers: LayerState[], ramps: Record<string, string[]>, title: string) {
  // The WebGL buffer is only readable during the frame that drew it
  const mapImg = await new Promise<string>((resolve) => {
    m.once("render", () => resolve(m.getCanvas().toDataURL("image/png")));
    m.triggerRepaint();
  });
  const dpr = m.getCanvas().width / m.getCanvas().clientWidth || 1;
  const img = new Image();
  await new Promise((r) => {
    img.onload = r;
    img.src = mapImg;
  });
  const shown = layers.filter((l) => l.visible && l.kind !== "rgb").reverse();
  const legendW = 260 * dpr;
  const c = document.createElement("canvas");
  c.width = img.width + legendW;
  c.height = Math.max(img.height, (60 + shown.length * 70) * dpr);
  const g = c.getContext("2d")!;
  g.fillStyle = "#ffffff";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(img, 0, 0);

  // Scale bar, bottom left of the map
  const ctr = m.getCenter();
  const bar = scaleBar(metresPerPixel(ctr.lat, m.getZoom()), 150);
  const x0 = 16 * dpr;
  const y0 = img.height - 20 * dpr;
  g.fillStyle = "rgba(255,255,255,0.85)";
  g.fillRect(x0 - 6 * dpr, y0 - 18 * dpr, bar.px * dpr + 12 * dpr, 26 * dpr);
  g.fillStyle = "#000";
  g.fillRect(x0, y0, bar.px * dpr, 3 * dpr);
  g.font = `${12 * dpr}px sans-serif`;
  g.fillText(bar.label, x0, y0 - 5 * dpr);

  // Legend
  let y = 24 * dpr;
  const lx = img.width + 14 * dpr;
  g.fillStyle = "#111";
  g.font = `bold ${13 * dpr}px sans-serif`;
  g.fillText(title, lx, y);
  g.font = `${11 * dpr}px sans-serif`;
  g.fillStyle = "#555";
  y += 16 * dpr;
  g.fillText(`${ctr.lat.toFixed(4)}°, ${ctr.lng.toFixed(4)}°  zoom ${m.getZoom().toFixed(1)}`, lx, y);
  y += 22 * dpr;
  for (const l of shown) {
    g.fillStyle = "#111";
    g.font = `bold ${11 * dpr}px sans-serif`;
    g.fillText(l.title, lx, y);
    y += 8 * dpr;
    g.font = `${10 * dpr}px sans-serif`;
    if (l.kind === "continuous" && ramps[l.cmap]) {
      const grad = g.createLinearGradient(lx, 0, lx + 220 * dpr, 0);
      ramps[l.cmap].forEach((col, i, a) => grad.addColorStop(i / (a.length - 1), col));
      g.fillStyle = grad;
      g.fillRect(lx, y, 220 * dpr, 10 * dpr);
      g.fillStyle = "#333";
      y += 22 * dpr;
      g.fillText(`${l.stretch[0]}`, lx, y);
      const hi = `${l.stretch[1]} ${l.units}`;
      g.fillText(hi, lx + 220 * dpr - g.measureText(hi).width, y);
      y += 18 * dpr;
    } else if (l.kind === "categorical") {
      const cats = l.legend.filter((e) => (e.fraction == null || e.fraction > 0) && (!l.keep || l.keep.includes(e.code)));
      for (const e of cats.slice(0, 14)) {
        y += 4 * dpr;
        g.fillStyle = `rgb(${e.rgb.join(",")})`;
        g.fillRect(lx, y, 10 * dpr, 10 * dpr);
        g.fillStyle = "#333";
        g.fillText(pretty(e.label), lx + 16 * dpr, y + 9 * dpr);
        y += 12 * dpr;
      }
      if (cats.length > 14) {
        y += 12 * dpr;
        g.fillText(`…and ${cats.length - 14} more`, lx, y);
      }
      y += 16 * dpr;
    } else y += 10 * dpr;
    if (y > c.height - 20 * dpr) break;
  }
  g.fillStyle = "#888";
  g.fillText("specret viewer", lx, c.height - 10 * dpr);
  const a = document.createElement("a");
  a.href = c.toDataURL("image/png");
  a.download = `${title.replace(/[^\w.-]+/g, "_")}_map.png`;
  a.click();
}

const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]!);

/** A plume's evidence as one HTML file: the numbers, and the charts as embedded SVG. */
export function evidenceCardHtml(
  e: PlumeExplanation,
  plume: PlumeRow,
  scene: string,
  charts: { title: string; src: string }[],
): string {
  const row = (k: string, v: unknown) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`;
  const r = e.rate;
  const csf = e.cross_sectional_flux;
  const numbers = [
    row("Scene", scene),
    row("Gas", e.gas),
    row("Confidence", fmt(e.confidence, 2)),
    row("Verdict", e.verdict),
    row("Source", `${fmt(plume.source_lat, 5)}, ${fmt(plume.source_lon, 5)}`),
    row("Fitted terms", Object.entries(e.fitted_terms).map(([k, v]) => `${k} ${fmt(v, 3)}`).join(", ")),
    ...(r.q_kg_h != null
      ? [
          row("Rate (IME)", `${fmt(r.q_kg_h)} ± ${fmt(r.sigma_q_kg_h)} kg/h`),
          row("IME, length", `${fmt(r.ime_kg, 1)} kg, ${fmt(r.length_m)} m`),
          row("Wind", `${fmt(r.u10, 2)} m/s (${r.wind_source}), U_eff ${fmt(r.u_eff, 2)} m/s`),
        ]
      : [row("Rate", "not estimated for this sensor")]),
    ...(csf?.q_csf_kg_h != null
      ? [row("Rate (cross-section)", `${fmt(csf.q_csf_kg_h)} kg/h, ${csf.transects} transects, IQR ${fmt(100 * (csf.csf_iqr_rel ?? NaN))}%`)]
      : []),
    ...Object.entries(e.relative_uncertainty).map(([k, v]) => row(`Uncertainty: ${k}`, `±${fmt(100 * v)}%`)),
    ...(e.carbonate_fraction != null ? [row("Carbonate under plume", `${fmt(100 * e.carbonate_fraction)}%`)] : []),
    ...(plume.overpasses_covering
      ? [row("Seen again", `${plume.overpasses_seen} of ${plume.overpasses_covering} other overpasses`)]
      : []),
  ].join("");
  const evidence = Object.entries(e.evidence)
    .map(([k, v]) => `<tr><th>${esc(k)}</th><td>${fmt(v, 2)}</td><td>${esc(e.evidence_meaning[k])}</td></tr>`)
    .join("");
  const figs = charts.map((c) => `<figure><img alt="${esc(c.title)}" src="${c.src}"><figcaption>${esc(c.title)}</figcaption></figure>`).join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plume ${e.plume_id} (${esc(e.gas)}) – ${esc(scene)}</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#1d1f22;background:#fff}
h1{font-size:20px;margin:0 0 4px}table{border-collapse:collapse;margin:12px 0;width:100%}
th,td{text-align:left;padding:3px 8px;border-bottom:1px solid #e3e3e3;vertical-align:top}th{font-weight:600;white-space:nowrap}
figure{margin:16px 0}img{max-width:100%;border:1px solid #eee}figcaption{color:#666;font-size:12px}
.note{color:#666;font-size:12px}
</style></head><body>
<h1>Plume ${e.plume_id}: ${esc(e.gas)}</h1>
<p class="note">Evidence card exported from the specret viewer on ${new Date().toISOString().slice(0, 10)}.</p>
<table>${numbers}</table>
${figs}
<h2 style="font-size:15px">Confidence features</h2>
<table>${evidence}</table>
</body></html>
`;
}

/** Render the plume panel's charts to SVG and download the card. */
export async function downloadEvidenceCard(root: HTMLElement, e: PlumeExplanation, plume: PlumeRow, scene: string) {
  const { default: Plotly } = await loadPlotly();
  const charts = [];
  for (const el of Array.from(root.querySelectorAll<HTMLElement>(".plot[role=img]"))) {
    if (!el.querySelector(".main-svg")) continue;
    const src = await Plotly.toImage(el, { format: "svg", width: 820, height: 380 });
    charts.push({ title: el.getAttribute("aria-label") ?? "chart", src });
  }
  download(`${scene}_plume${e.plume_id}_evidence.html`, evidenceCardHtml(e, plume, scene, charts), "text/html");
}
