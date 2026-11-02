import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import type { GasSummaryPart, PlumeExplanation, PlumeRow } from "../api/types";
import { useJob } from "../hooks/useJob";
import { fmt } from "../lib";
import { FitChart, Rates, SignatureChart } from "./PlumeEvidence";

interface Props {
  gasScene: string | null; // gas source id (EMIT scene, AVIRIS-NG line, HyTES flight), or null
  sensor: string;
  analyzed: boolean;
  selected: number | null;
  onSelect: (plume: PlumeRow) => void; // a row of another scene opens that scene
  onChanged: () => void; // a detection job finished
}

export type SortKey = "plume_id" | "gas" | "confidence" | "q_kg_h" | "q_csf_kg_h" | "u10" | "repeats" | "scene_id";

export interface PlumeFilter {
  gas: string; // "" = any
  sensor: string;
  minConfidence: number;
  confusers: "any" | "with" | "without";
}

export const NO_FILTER: PlumeFilter = { gas: "", sensor: "", minConfidence: 0, confusers: "any" };

const sortValue = (p: PlumeRow, k: SortKey): number | string =>
  k === "repeats" ? (p.overpasses_seen ?? -Infinity) : ((p[k] as number | string | null | undefined) ?? -Infinity);

/** The rows that pass the filter, sorted (missing values last either way). */
export function filterSort(rows: PlumeRow[], f: PlumeFilter, key: SortKey, desc: boolean): PlumeRow[] {
  const keep = rows.filter(
    (p) =>
      (!f.gas || p.gas === f.gas) &&
      (!f.sensor || p.sensor === f.sensor) &&
      (p.confidence ?? 0) >= f.minConfidence &&
      (f.confusers === "any" || (f.confusers === "with") === !!p.confusers_fitted),
  );
  return keep.sort((a, b) => {
    const x = sortValue(a, key);
    const y = sortValue(b, key);
    if (x === y) return 0;
    if (x === -Infinity) return 1;
    if (y === -Infinity) return -1;
    return (x < y ? -1 : 1) * (desc ? -1 : 1);
  });
}

/** "2023-06-08" from an EMIT scene id, an AVIRIS-NG line id or a HyTES flight name. */
export function sceneDate(id: string): string {
  const m = id.match(/^(\d{4})(\d{2})(\d{2})T/) ?? id.match(/^ang(\d{4})(\d{2})(\d{2})/) ?? id.match(/^(\d{4})-(\d{2})-(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]}` : id;
}

/** Plumes of the active scene, or of every analyzed scene, line and flight: summary, a
 * sortable and filterable table, and the selected plume's evidence. */
export function PlumesPanel({ gasScene, sensor, analyzed, selected, onSelect, onChanged }: Props) {
  const qc = useQueryClient();
  const [scope, setScope] = useState<"scene" | "all">("scene");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, 2000);
  useEffect(() => setJobId(null), [gasScene]);
  useEffect(() => {
    if (job?.state === "done") {
      qc.invalidateQueries({ queryKey: ["gasPlumes", gasScene] });
      qc.invalidateQueries({ queryKey: ["allPlumes"] });
      onChanged();
    }
  }, [job?.state, qc, gasScene, onChanged]);

  const plumes = useQuery({
    queryKey: ["gasPlumes", gasScene],
    queryFn: () => api.gasPlumes(gasScene!),
    enabled: !!gasScene && analyzed,
  });
  const all = useQuery({ queryKey: ["allPlumes"], queryFn: api.allPlumes, enabled: scope === "all" });
  const detail = useQuery({
    queryKey: ["plume", gasScene, selected],
    queryFn: () => api.plume(gasScene!, selected!),
    enabled: !!gasScene && analyzed && selected !== null,
  });

  let body;
  if (scope === "all") {
    body = <PlumeTable rows={all.data ?? []} selected={selected} currentScene={gasScene} onSelect={onSelect} allScenes />;
  } else if (!gasScene) {
    body = <p className="hint">This scene has no radiance for gas detection.</p>;
  } else if (job?.state === "running") {
    body = (
      <div>
        <progress max={1} value={job.progress} />
        <div className="hint">
          Detecting… {Math.round(100 * job.progress)}% · {job.elapsed_s} s
        </div>
      </div>
    );
  } else if (!analyzed) {
    body = (
      <div className="empty">
        <p>Not analyzed yet: detection takes {sensor === "HyTES" ? "a few minutes per flight" : "a minute or two"}.</p>
        <button className="primary" onClick={async () => setJobId((await api.startGas(gasScene)).job_id)}>
          Detect plumes
        </button>
        {job?.state === "failed" && <div className="hint">Detection failed: {job.error?.split("\n")[0]}</div>}
      </div>
    );
  } else {
    const rows = plumes.data?.plumes ?? [];
    const summary = plumes.data?.summary;
    const parts = Object.entries(summary ?? {}).flatMap(([g, v]) => {
      const part = v as GasSummaryPart | null;
      if (!part || typeof part !== "object" || !part.plumes) return [];
      return [`${g}: ${part.plumes}${part.total_q_kg_h != null ? `, ${fmt(part.total_q_kg_h)} kg/h in total` : ""}`];
    });
    const sel = rows.find((p) => p.plume_id === selected) ?? null;
    body = (
      <>
        {summary && (
          <p className="hint">
            {summary.plumes} plumes from {summary.candidates ?? "?"} candidates. {parts.join("; ")}
          </p>
        )}
        {detail.data && sel && <PlumeDetail plume={sel} e={detail.data} scene={gasScene} />}
        <PlumeTable rows={rows} selected={selected} currentScene={gasScene} onSelect={onSelect} />
      </>
    );
  }
  return (
    <div>
      <div className="controls" role="radiogroup" aria-label="Plumes of">
        {(["scene", "all"] as const).map((s) => (
          <button
            key={s}
            role="radio"
            aria-checked={scope === s}
            className={scope === s ? "primary" : ""}
            onClick={() => setScope(s)}
          >
            {s === "scene" ? "This scene" : "All scenes"}
          </button>
        ))}
      </div>
      {body}
    </div>
  );
}

function PlumeTable({
  rows,
  selected,
  currentScene,
  onSelect,
  allScenes = false,
}: {
  rows: PlumeRow[];
  selected: number | null;
  currentScene: string | null;
  onSelect: (p: PlumeRow) => void;
  allScenes?: boolean;
}) {
  const [f, setF] = useState<PlumeFilter>(NO_FILTER);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "confidence", desc: true });
  const shown = useMemo(() => filterSort(rows, f, sort.key, sort.desc), [rows, f, sort]);
  const gases = [...new Set(rows.map((r) => r.gas))].sort();
  const sensors = [...new Set(rows.map((r) => r.sensor).filter(Boolean))] as string[];
  const head = (key: SortKey, label: string, num = true) => (
    <th
      className={`sortable${num ? " num" : ""}`}
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}
      onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : num }))}
    >
      {label}
      {sort.key === key ? (sort.desc ? " ▾" : " ▴") : ""}
    </th>
  );
  return (
    <>
      <div className="filters">
        <select aria-label="Gas" value={f.gas} onChange={(e) => setF({ ...f, gas: e.target.value })}>
          <option value="">all gases</option>
          {gases.map((g) => (
            <option key={g}>{g}</option>
          ))}
        </select>
        {sensors.length > 1 && (
          <select aria-label="Sensor" value={f.sensor} onChange={(e) => setF({ ...f, sensor: e.target.value })}>
            <option value="">all sensors</option>
            {sensors.map((g) => (
              <option key={g}>{g}</option>
            ))}
          </select>
        )}
        <select
          aria-label="Confusers"
          value={f.confusers}
          onChange={(e) => setF({ ...f, confusers: e.target.value as PlumeFilter["confusers"] })}
        >
          <option value="any">any confusers</option>
          <option value="with">confuser fitted</option>
          <option value="without">no confuser</option>
        </select>
        <label className="hint">
          conf. ≥ {f.minConfidence.toFixed(1)}
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.1}
            value={f.minConfidence}
            aria-label="Minimum confidence"
            onChange={(e) => setF({ ...f, minConfidence: Number(e.target.value) })}
          />
        </label>
        <span className="hint">
          {shown.length} of {rows.length}
        </span>
      </div>
      <div className="table-scroll">
        <table className="plume-table">
          <thead>
            <tr>
              {allScenes && head("scene_id", "Scene", false)}
              {head("plume_id", "#")}
              {head("gas", "Gas", false)}
              {head("confidence", "Conf.")}
              {head("q_kg_h", "Q IME")}
              {head("q_csf_kg_h", "Q CSF")}
              {head("u10", "Wind")}
              <th>Confusers</th>
              {head("repeats", "Seen again")}
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => {
              const here = !allScenes || p.scene_id === currentScene;
              return (
                <tr
                  key={`${p.scene_id ?? ""}-${p.plume_id}`}
                  className={`clickable${here && p.plume_id === selected ? " sel" : ""}`}
                  onClick={() => onSelect(p)}
                >
                  {allScenes && (
                    <td title={p.scene_id}>
                      <span className="tag">{p.sensor}</span> {sceneDate(p.scene_id ?? "")}
                    </td>
                  )}
                  <td>{p.plume_id}</td>
                  <td>{p.gas}</td>
                  <td className="num">{fmt(p.confidence, 2)}</td>
                  <td className="num" title={p.sigma_q_kg_h != null ? `±${fmt(p.sigma_q_kg_h)} kg/h (1σ)` : undefined}>
                    {fmt(p.q_kg_h)}
                  </td>
                  <td className="num">{fmt(p.q_csf_kg_h)}</td>
                  <td className="num" title={p.wind_source ?? undefined}>
                    {fmt(p.u10, 1)}
                    {p.wind_source && <span className="hint"> {p.wind_source.split(" ")[0]}</span>}
                  </td>
                  <td className="ev">
                    {(p.confusers_fitted ?? "").replaceAll("confuser:", "").replaceAll(";", ", ") || "–"}
                  </td>
                  <td className="num">
                    {p.overpasses_covering ? (
                      <span
                        className={`badge${p.overpasses_seen ? " on" : ""}`}
                        title={`seen in ${p.overpasses_seen} of ${p.overpasses_covering} other overpasses`}
                      >
                        {p.overpasses_seen}/{p.overpasses_covering}
                      </span>
                    ) : (
                      "–"
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

/** Every overpass that covered the source, oldest first, and whether it saw the plume. */
export function Timeline({ plume, scene }: { plume: PlumeRow; scene: string }) {
  if (!plume.covered_by?.length) return null;
  const seen = new Set(plume.seen_in ?? []);
  const items = [...plume.covered_by, scene].sort();
  return (
    <div className="timeline">
      <div className="hint">
        Other overpasses over the source: seen in {seen.size} of {plume.covered_by.length}
      </div>
      <ol>
        {items.map((id) => (
          <li key={id} className={id === scene ? "this" : seen.has(id) ? "seen" : "unseen"} title={id}>
            <i />
            {sceneDate(id)}
            <span className="hint">{id === scene ? " this scene" : seen.has(id) ? " seen" : " not seen"}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Direction the air moves towards, as a compass point. */
export function compass(u: number, v: number): string {
  const deg = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
  return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(deg / 45) % 8];
}

function PlumeDetail({ plume, e, scene }: { plume: PlumeRow; e: PlumeExplanation; scene: string }) {
  const terms = Object.entries(e.fitted_terms)
    .map(([k, v]) => `${k.replace("confuser:", "confuser ")} ${fmt(v, k.startsWith("confuser") ? 3 : 0)}`)
    .join(", ");
  const u = e.wind?.wind_u;
  const v = e.wind?.wind_v;
  const hasWind = u != null && v != null && Number.isFinite(u) && Number.isFinite(v);
  const hasRate = e.rate.q_kg_h != null && Number.isFinite(e.rate.q_kg_h);
  return (
    <div className="plume-detail">
      <b>
        Plume {e.plume_id}: {e.gas}
      </b>
      , confidence {fmt(e.confidence, 2)} · source {fmt(plume.source_lat, 5)}, {fmt(plume.source_lon, 5)}
      <div className="ev">Fit: {terms}</div>
      {hasRate && (
        <div className="ev">
          IME {fmt(e.rate.ime_kg, 1)} kg, L {fmt(e.rate.length_m)} m, wind {fmt(e.rate.u10, 2)} m/s
          {hasWind && ` towards ${compass(u!, v!)}`} ({e.rate.wind_source}), U<sub>eff</sub> {fmt(e.rate.u_eff, 2)} m/s
          {hasWind && " · arrow on the map"}
        </div>
      )}
      {e.carbonate_fraction != null && (
        <div className="ev">carbonate under plume {fmt(100 * e.carbonate_fraction)}%</div>
      )}
      <Timeline plume={plume} scene={scene} />
      {hasRate ? (
        <Rates e={e} />
      ) : (
        <p className="hint">No emission rate: rates are estimated for EMIT and AVIRIS-NG plumes only.</p>
      )}
      {e.spectra ? (
        <>
          <h3>Fit</h3>
          <p className="hint">
            The plume's mean log-radiance residual against the fitted terms; the band is the spread of the same mask
            shifted to plume-free places.
          </p>
          <FitChart s={e.spectra} />
          <h3>Candidate gases</h3>
          <SignatureChart s={e.spectra} fitted={Object.keys(e.fitted_terms)} />
        </>
      ) : (
        <p className="hint">This scene was analyzed before fit spectra were stored; detect again to see the fit.</p>
      )}
      <h3>Confidence features</h3>
      <table>
        <tbody>
          {Object.entries(e.evidence).map(([k, v]) => (
            <tr key={k}>
              <td>{k}</td>
              <td className="num">{fmt(v, 2)}</td>
              <td className="ev">{e.evidence_meaning[k] ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
