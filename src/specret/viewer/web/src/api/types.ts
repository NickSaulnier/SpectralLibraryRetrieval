// Payloads of the viewer API (specret/viewer/app.py). Route paths come from the generated
// OpenAPI schema (schema.d.ts); these endpoints return plain dicts, so their bodies are typed
// here by hand until they get Pydantic response models.

export interface SceneInfo {
  scene_id: string;
  kind: string;
  mapped: boolean;
  job: string | null;
}

export interface LegendEntry {
  group: string;
  code: number;
  rgb: [number, number, number];
  fraction: number;
}

/** A background job (scene/jobs.py): one queue, run one at a time. */
export interface Job {
  job_id: string;
  kind: "map" | "gas" | "unmix" | "persistence" | "command";
  scene_id: string;
  title: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number | null; // null: unknown (a command's progress)
  elapsed_s: number;
  created: number;
  finished: number | null;
  position: number | null; // place in the queue while queued
  cancelling: boolean;
  log: string[];
  error?: string;
  [key: string]: unknown;
}

export interface SetupStep {
  argv: string[];
  command: string;
  runnable: boolean; // false: a download that needs credentials (run it in a terminal)
}

export interface SetupItem {
  key: string;
  level: "required" | "recommended" | "optional";
  name: string;
  ok: boolean;
  detail: string;
  hint: string | null;
  steps: SetupStep[];
}

/** One diagnostic window's part in a match score (rerank/sff.explain). */
export interface FeatureEvidence {
  window: string;
  weight: number;
  score: number;
  query_depth: number;
  query_center_nm: number;
  cand_depth: number;
  cand_center_nm: number;
  shape_r: number;
}

/** Continuum-removed band depth in one window: the query, then the top matches in order. */
export interface WindowProfile {
  window: string;
  wavelengths_nm: number[];
  query: number[];
  matches: number[][];
  depth: number[];
  center_nm: number[];
}

export interface Match {
  spectrum_id: string;
  material_name: string;
  group: string | null;
  score: number;
  purity_flag: string | null;
  features?: FeatureEvidence[];
  spectrum?: number[];
}

export interface Summary {
  verdict: string;
  best_material?: string;
  best_group?: string;
  score?: number;
  p_group_correct?: number | null;
  confidence?: "confident" | "uncertain" | "likely unknown" | "uncalibrated";
  reason?: string;
}

export interface UnmixComponent {
  spectrum_id?: string;
  material_name: string;
  group?: string | null;
  fraction: number;
}

export interface Unmixing {
  components: UnmixComponent[];
  scale?: number;
  residual_rms?: number;
  reduced_chi2?: number | null;
  /** Per band (the good bands): endmembers scaled by their coefficients, in component order. */
  spectra?: { components: number[][]; baseline: number[]; model: number[]; residual: number[] };
}

export interface PixelInfo {
  row: number;
  col: number;
  x: number;
  y: number;
  lat?: number | null;
  lon?: number | null;
}

export interface Identification {
  sensor: string;
  summary: Summary;
  matches: Match[];
  unmixing?: Unmixing | null;
  wavelengths_nm: number[];
  query: number[];
  windows: Record<string, [number, number]>;
  profiles?: WindowProfile[];
  pixel?: PixelInfo;
  name?: string;
}

export interface GasScene {
  scene_id: string;
  analyzed: boolean;
  job: string | null;
}

export interface PlumeRow {
  plume_id: number;
  scene_id?: string; // in the all-scenes list
  sensor?: string; // EMIT, AVIRIS-NG or HyTES
  detection_gas?: string;
  q_csf_kg_h?: number | null;
  // From `specret gas-persistence` (EMIT): other overpasses over the source, and those that saw it
  overpasses_covering?: number;
  overpasses_seen?: number;
  covered_by?: string[];
  seen_in?: string[];
  gas: string;
  confidence: number;
  source_lat: number | null;
  source_lon: number | null;
  n_px: number;
  max_alpha_ppmm: number;
  q_kg_h: number | null;
  sigma_q_kg_h: number | null;
  u10: number | null;
  wind_source: string | null;
  confusers_fitted: string | null;
  carbonate_fraction: number | null;
  x?: number;
  y?: number;
}

export interface GasSummaryPart {
  plumes: number;
  total_q_kg_h: number | null;
  total_sigma_q_kg_h: number | null;
  largest_q_kg_h: number | null;
}

export interface GasPlumes {
  scene_id: string;
  plumes: PlumeRow[];
  summary: {
    scene_id: string;
    plumes: number;
    candidates?: number;
    candidate_verdicts?: Record<string, number>;
    [gas: string]: unknown;
  };
}

export interface PlumeExplanation {
  plume_id: number;
  gas: string;
  confidence: number;
  verdict: string;
  fitted_terms: Record<string, number>;
  evidence: Record<string, number>;
  evidence_meaning: Record<string, string>;
  carbonate_fraction: number | null;
  rate: {
    ime_kg: number;
    length_m: number;
    u10: number;
    u_eff: number;
    wind_source: string;
    q_kg_h: number;
    sigma_q_kg_h: number;
  };
  relative_uncertainty: Record<string, number>;
  cross_sectional_flux?: { q_csf_kg_h?: number | null; transects?: number | null; csf_iqr_rel?: number | null };
  source?: { source_lat: number | null; source_lon: number | null };
  wind?: { wind_u?: number | null; wind_v?: number | null };
  spectra?: PlumeSpectra | null;
}

/** A plume's stored fit (gas/outputs.write_spectra), in log-radiance units. */
export interface PlumeSpectra {
  units: string;
  wavelengths_nm: number[];
  windows: [number, number][];
  observed: number[];
  null_sd: number[];
  baseline: number[];
  terms: Record<string, number[]>;
  model: number[];
  remainder: number[];
  signatures_per_ppmm: Record<string, number[]>;
}

/** Our enhancement against a published map over the visible area (/api/geo/{key}/compare). */
export interface FitStats {
  n: number;
  r?: number;
  slope?: number;
  intercept?: number;
  mean_diff?: number;
  deming_slope?: number;
}

export interface Comparison {
  ours: string;
  ref: string;
  units: string;
  recipe: string;
  view_pixels: number;
  plume_pixels: FitStats;
  scene_pixels: FitStats;
  background_sigma_ours: number;
  background_sigma_ref: number;
  sample: { plume: { ref: number[]; ours: number[] }; background: { ref: number[]; ours: number[] } };
}
