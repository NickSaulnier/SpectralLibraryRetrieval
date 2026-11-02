import type { components, paths } from "./schema";
import type {
  Comparison,
  GasPlumes,
  GasScene,
  Identification,
  Job,
  LegendEntry,
  PlumeExplanation,
  PlumeRow,
  SceneInfo,
  SetupItem,
} from "./types";

type Path = keyof paths;
export type GeoScene = components["schemas"]["GeoScene"];
export type LayerInfo = components["schemas"]["LayerInfo"];
export type AreaSummary = components["schemas"]["AreaSummary"];
export type Polygon = [number, number][]; // [lon, lat] vertices, open ring

export interface Agreement {
  pixels_mapped: number;
  pixels_with_reference: number;
  agreement: number;
  depth_min: number;
  top_confusions: { ours: string; l2bmin: string; pixels: number }[];
}

/** Fill a typed route template ("/api/map/{scene_id}") with its parameters. */
export function route<P extends Path>(path: P, params: Record<string, string | number> = {}): string {
  return path.replace(/\{(\w+)\}/g, (_, k: string) => encodeURIComponent(String(params[k])));
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : `${res.status} ${res.statusText}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) throw new ApiError(await errorText(res), res.status);
  return (await res.json()) as T;
}

const query = (q: Record<string, string | number | undefined>) =>
  new URLSearchParams(
    Object.entries(q).flatMap(([k, v]) => (v === undefined ? [] : [[k, String(v)]])),
  ).toString();

export const api = {
  scenes: () => json<SceneInfo[]>(route("/api/scenes")),
  startMapping: (sceneId: string) =>
    json<{ job_id: string }>(route("/api/map/{scene_id}", { scene_id: sceneId }), { method: "POST" }),
  job: (jobId: string) => json<Job>(route("/api/jobs/{job_id}", { job_id: jobId })),
  jobs: () => json<Job[]>(route("/api/jobs")),
  cancelJob: (jobId: string) =>
    json<Job>(route("/api/jobs/{job_id}/cancel", { job_id: jobId }), { method: "POST" }),
  startPersistence: (scenes: string[]) =>
    json<{ job_id: string }>(route("/api/jobs/persistence"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scenes }),
    }),
  setup: () => json<SetupItem[]>(route("/api/setup")),
  runSetupStep: (argv: string[]) =>
    json<{ job_id: string }>(route("/api/setup/run"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ argv }),
    }),
  mapUrl: (sceneId: string) => route("/api/map/{scene_id}.png", { scene_id: sceneId }),
  legend: (sceneId: string) => json<LegendEntry[]>(route("/api/legend/{scene_id}", { scene_id: sceneId })),
  pixel: (sceneId: string, q: { x?: number; y?: number; lat?: number; lon?: number }) =>
    json<Identification>(`${route("/api/pixel/{scene_id}", { scene_id: sceneId })}?${query(q)}`),
  field: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return json<Identification[]>(route("/api/field"), { method: "POST", body });
  },
  gasScenes: () => json<GasScene[]>(route("/api/gas/scenes")),
  startGas: (sceneId: string) =>
    json<{ job_id: string }>(route("/api/gas/{scene_id}", { scene_id: sceneId }), { method: "POST" }),
  gasMapUrl: (sceneId: string, gas: string) =>
    route("/api/gas/{scene_id}/{gas}.png", { scene_id: sceneId, gas }),
  gasPlumes: (sceneId: string) => json<GasPlumes>(route("/api/gas/{scene_id}/plumes", { scene_id: sceneId })),
  geoScenes: () => json<GeoScene[]>(route("/api/geo/scenes")),
  layers: (key: string) => json<LayerInfo[]>(route("/api/geo/{key}/layers", { key })),
  values: (key: string, lat: number, lon: number) =>
    json<Record<string, number | string | null>>(`${route("/api/geo/{key}/values", { key })}?${query({ lat, lon })}`),
  plumesGeojson: (key: string) =>
    json<GeoJSON.FeatureCollection>(route("/api/geo/{key}/plumes.geojson", { key })),
  colormaps: () => json<Record<string, string[]>>(route("/api/colormaps")),
  /** MapLibre tile template for a layer, with its current stretch, colormap and filters. */
  tileUrl: (layerId: string, q: Record<string, string | number | undefined>) =>
    `${location.origin}/api/tiles/${encodeURIComponent(layerId)}/{z}/{x}/{y}.png?${query(q)}`,
  summary: (key: string, polygon: Polygon, minP: number) =>
    json<AreaSummary>(route("/api/geo/{key}/summary", { key }), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon, min_p: minP }),
    }),
  unmixArea: (key: string, polygon: Polygon, maxPixels = 3000) =>
    json<{ job_id: string }>(route("/api/geo/{key}/unmix", { key }), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ polygon, max_pixels: maxPixels }),
    }),
  allPlumes: () => json<PlumeRow[]>(route("/api/gas/plumes")),
  compare: (key: string, layer: string, bbox?: [number, number, number, number]) =>
    json<Comparison>(
      `${route("/api/geo/{key}/compare", { key })}?${query({ layer, bbox: bbox?.map((v) => v.toFixed(5)).join(",") })}`,
    ),
  agreement: (key: string) => json<Agreement>(route("/api/geo/{key}/agreement", { key })),
  plume: (sceneId: string, plumeId: number) =>
    json<PlumeExplanation>(
      route("/api/gas/{scene_id}/plume/{plume_id}", { scene_id: sceneId, plume_id: plumeId }),
    ),
};
