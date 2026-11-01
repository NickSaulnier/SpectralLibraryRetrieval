import type { paths } from "./schema";
import type {
  GasPlumes,
  GasScene,
  Identification,
  Job,
  LegendEntry,
  PlumeExplanation,
  SceneInfo,
} from "./types";

type Path = keyof paths;

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
  plume: (sceneId: string, plumeId: number) =>
    json<PlumeExplanation>(
      route("/api/gas/{scene_id}/plume/{plume_id}", { scene_id: sceneId, plume_id: plumeId }),
    ),
};
