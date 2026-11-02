/** A view as a URL (PLAN_VIEWER.md V5): scene, camera, layers, selection, so a copied link
 * restores the same view in any viewer running on the same project data.
 *
 *   ?s=<scene>&t=<tab>&v=<lon>,<lat>,<zoom>&p=<plume>&pt=<lat>,<lon>&l=<layers>&a=<area>
 *
 * `l` lists the visible layers bottom first as name:opacity, with :lo,hi:colormap for
 * continuous layers whose stretch changed and :k<codes>:p<minP> for filtered categories.
 * `a` is the drawn area's vertices, lon,lat;lon,lat;...
 */
import type { Polygon } from "./api/client";
import type { LayerState } from "./map/layerState";

export interface Camera {
  lon: number;
  lat: number;
  zoom: number;
}

export interface SharedLayer {
  name: string;
  opacity: number;
  stretch?: [number, number];
  cmap?: string;
  keep?: number[];
  minP?: number;
}

export interface ViewState {
  scene?: string;
  tab?: string;
  camera?: Camera;
  plume?: number;
  point?: { lat: number; lon: number };
  layers?: SharedLayer[];
  area?: Polygon;
}

const num = (v: number, d: number) => String(Number(v.toFixed(d)));
const nums = (s: string | null) => (s ? s.split(",").map(Number) : []);
const finite = (a: number[], n: number) => a.length === n && a.every(Number.isFinite);

function encodeLayer(l: LayerState): string {
  const parts = [l.name, num(l.opacity, 2)];
  if (l.kind === "continuous" && (l.stretch[0] !== l.rescale[0] || l.stretch[1] !== l.rescale[1] || l.cmap !== l.colormap))
    parts.push(`${num(l.stretch[0], 4)},${num(l.stretch[1], 4)}`, l.cmap);
  if (l.kind === "categorical" && l.keep !== null) parts.push(`k${l.keep.join(".")}`);
  if (l.minP > 0) parts.push(`p${num(l.minP, 2)}`);
  return parts.join(":");
}

function decodeLayer(s: string): SharedLayer | null {
  const [name, opacity, ...rest] = s.split(":");
  if (!name || !Number.isFinite(Number(opacity))) return null;
  const out: SharedLayer = { name, opacity: Number(opacity) };
  const KEEP = /^k[\d.]*$/;
  const MIN_P = /^p[\d.]+$/;
  for (let i = 0; i < rest.length; i++) {
    const t = rest[i];
    if (KEEP.test(t)) out.keep = t.length > 1 ? t.slice(1).split(".").map(Number) : [];
    else if (MIN_P.test(t)) out.minP = Number(t.slice(1));
    else if (t.includes(",")) {
      const r = nums(t);
      if (finite(r, 2)) out.stretch = [r[0], r[1]];
      const next = rest[i + 1];
      if (next && !KEEP.test(next) && !MIN_P.test(next)) out.cmap = rest[++i];
    }
  }
  return out;
}

export function encodeView(v: { scene: string | null; tab: string; camera: Camera | null; plume: number | null;
  point: { lat: number; lon: number } | null; layers: LayerState[]; area: Polygon | null }): string {
  const q = new URLSearchParams();
  if (v.scene) q.set("s", v.scene);
  if (v.tab && v.tab !== "layers") q.set("t", v.tab);
  if (v.camera) q.set("v", `${num(v.camera.lon, 5)},${num(v.camera.lat, 5)},${num(v.camera.zoom, 2)}`);
  if (v.plume !== null) q.set("p", String(v.plume));
  if (v.point) q.set("pt", `${num(v.point.lat, 6)},${num(v.point.lon, 6)}`);
  const shown = v.layers.filter((l) => l.visible);
  if (v.layers.length) q.set("l", shown.map(encodeLayer).join("~"));
  if (v.area) q.set("a", v.area.map(([x, y]) => `${num(x, 6)},${num(y, 6)}`).join(";"));
  return q.toString().replaceAll("%2C", ",").replaceAll("%3A", ":").replaceAll("%3B", ";").replaceAll("%7E", "~");
}

export function decodeView(search: string): ViewState {
  const q = new URLSearchParams(search);
  const out: ViewState = {};
  if (q.get("s")) out.scene = q.get("s")!;
  if (q.get("t")) out.tab = q.get("t")!;
  const v = nums(q.get("v"));
  if (finite(v, 3)) out.camera = { lon: v[0], lat: v[1], zoom: v[2] };
  if (q.get("p") && Number.isInteger(Number(q.get("p")))) out.plume = Number(q.get("p"));
  const pt = nums(q.get("pt"));
  if (finite(pt, 2)) out.point = { lat: pt[0], lon: pt[1] };
  if (q.has("l")) out.layers = (q.get("l") || "").split("~").filter(Boolean).map(decodeLayer).filter((l) => l !== null);
  const a = (q.get("a") || "").split(";").map((p) => nums(p));
  if (a.length >= 3 && a.every((p) => finite(p, 2))) out.area = a.map(([x, y]) => [x, y] as [number, number]);
  return out;
}

/** A scene's layers set up as a shared view says: those listed shown, in its order (bottom
 * first) under nothing else, with its settings; every other layer hidden. */
export function applyShared(layers: LayerState[], shared: SharedLayer[]): LayerState[] {
  const by = new Map(shared.map((s, i) => [s.name, { s, i }]));
  const out = layers.map((l) => {
    const hit = by.get(l.name);
    if (!hit) return { ...l, visible: false };
    const s = hit.s;
    return {
      ...l,
      visible: true,
      opacity: s.opacity,
      stretch: s.stretch ?? l.stretch,
      cmap: s.cmap ?? l.cmap,
      keep: s.keep ?? null,
      minP: s.minP ?? 0,
    };
  });
  // Hidden layers keep their places at the bottom; shown ones go on top in the shared order
  const hidden = out.filter((l) => !by.has(l.name));
  const shown = out.filter((l) => by.has(l.name)).sort((a, b) => by.get(a.name)!.i - by.get(b.name)!.i);
  return [...hidden, ...shown];
}
