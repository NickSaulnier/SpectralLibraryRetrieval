import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { GasSummaryPart, PlumeExplanation, PlumeRow } from "../api/types";
import { ZoomImage } from "../components/ZoomImage";
import { useJob } from "../hooks/useJob";
import { fmt } from "../lib";

const GASES = ["CH4", "CO2"];

/** Gas plumes found in EMIT radiance: enhancement map, plume table and evidence. */
export function GasPanel() {
  const qc = useQueryClient();
  const scenes = useQuery({ queryKey: ["gasScenes"], queryFn: api.gasScenes });
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [gas, setGas] = useState("CH4");
  const [plumeId, setPlumeId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const list = scenes.data ?? [];
  const scene = list.find((s) => s.scene_id === sceneId) ?? null;
  useEffect(() => {
    if (!sceneId && list.length) setSceneId((list.find((s) => s.analyzed) ?? list[0]).scene_id);
  }, [list, sceneId]);
  useEffect(() => {
    setPlumeId(null);
    setJobId(null);
  }, [sceneId]);

  const job = useJob(jobId ?? scene?.job ?? null, 2000);
  useEffect(() => {
    if (job?.state === "done") qc.invalidateQueries({ queryKey: ["gasScenes"] });
  }, [job?.state, qc]);

  const plumes = useQuery({
    queryKey: ["gasPlumes", sceneId],
    queryFn: () => api.gasPlumes(sceneId!),
    enabled: !!scene?.analyzed,
  });
  const detail = useQuery({
    queryKey: ["plume", sceneId, plumeId],
    queryFn: () => api.plume(sceneId!, plumeId!),
    enabled: !!scene?.analyzed && plumeId !== null,
  });

  if (scenes.isSuccess && !list.length) return null;
  const rows = plumes.data?.plumes ?? [];
  const selected = rows.find((p) => p.plume_id === plumeId) ?? null;
  const summary = plumes.data?.summary;
  const parts = GASES.flatMap((g) => {
    const v = summary?.[g] as GasSummaryPart | undefined;
    return v?.plumes ? [`${g}: ${v.plumes} plumes, ${fmt(v.total_q_kg_h)} kg/h in total`] : [];
  });

  let status = "";
  if (!scene) status = "";
  else if (job?.state === "running") status = `Detecting… ${Math.round(100 * job.progress)}% · ${job.elapsed_s} s`;
  else if (job?.state === "failed") status = `Detection failed: ${job.error?.split("\n")[0] ?? ""}`;
  else if (!scene.analyzed) status = "Not analyzed yet: detection takes a minute or two.";
  else if (summary)
    status = `${summary.plumes} plumes from ${summary.candidates ?? "?"} candidates. ${parts.join("; ")}`;

  return (
    <section className="panel gas" aria-label="Gas plumes">
      <h2>Gas plumes (EMIT radiance)</h2>
      <div className="controls">
        <select aria-label="Radiance scene" value={sceneId ?? ""} onChange={(e) => setSceneId(e.target.value)}>
          {list.map((s) => (
            <option key={s.scene_id} value={s.scene_id}>
              {s.scene_id}
              {s.analyzed ? "" : " (not analyzed)"}
            </option>
          ))}
        </select>
        <select aria-label="Gas" value={gas} onChange={(e) => setGas(e.target.value)}>
          {GASES.map((g) => (
            <option key={g}>{g}</option>
          ))}
        </select>
        {scene && !scene.analyzed && job?.state !== "running" && (
          <button
            className="primary"
            onClick={async () => setJobId((await api.startGas(scene.scene_id)).job_id)}
          >
            Detect plumes
          </button>
        )}
        <span className="hint">{status}</span>
      </div>
      <div className="grid">
        <div>
          <ZoomImage
            src={scene?.analyzed ? api.gasMapUrl(scene.scene_id, gas) : null}
            alt="Gas enhancement map"
            dark
            marker={selected?.x != null && selected.y != null ? { x: selected.x, y: selected.y } : null}
            markerClass="cyan"
          />
          <div className="hint">
            Enhancement from 0 to its 99.5th percentile (ppm·m); cyan outlines are reported plumes.
          </div>
        </div>
        <div>
          {rows.length > 0 && <PlumeTable rows={rows} selected={plumeId} onSelect={setPlumeId} />}
          {detail.data && selected && <PlumeDetail plume={selected} e={detail.data} />}
        </div>
      </div>
    </section>
  );
}

function PlumeTable({ rows, selected, onSelect }: { rows: PlumeRow[]; selected: number | null; onSelect: (id: number) => void }) {
  return (
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Gas</th>
          <th className="num">Conf.</th>
          <th className="num">Q (kg/h)</th>
          <th className="num">±1σ</th>
          <th className="num">Wind (m/s)</th>
          <th>Fitted confusers</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((p) => (
          <tr
            key={p.plume_id}
            className={`clickable${p.plume_id === selected ? " sel" : ""}`}
            onClick={() => onSelect(p.plume_id)}
          >
            <td>{p.plume_id}</td>
            <td>{p.gas}</td>
            <td className="num">{fmt(p.confidence, 2)}</td>
            <td className="num">{fmt(p.q_kg_h)}</td>
            <td className="num">{fmt(p.sigma_q_kg_h)}</td>
            <td className="num">{fmt(p.u10, 1)}</td>
            <td className="ev">
              {(p.confusers_fitted ?? "").replaceAll("confuser:", "").replaceAll(";", ", ") || "–"}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function PlumeDetail({ plume, e }: { plume: PlumeRow; e: PlumeExplanation }) {
  const terms = Object.entries(e.fitted_terms)
    .map(([k, v]) => `${k.replace("confuser:", "confuser ")} ${fmt(v, k.startsWith("confuser") ? 3 : 0)}`)
    .join(", ");
  const unc = Object.entries(e.relative_uncertainty)
    .map(([k, v]) => `${k} ${fmt(100 * v)}%`)
    .join(", ");
  return (
    <div className="plume-detail">
      <b>
        Plume {e.plume_id}: {e.gas}
      </b>
      , confidence {fmt(e.confidence, 2)} · source {fmt(plume.source_lat, 5)}, {fmt(plume.source_lon, 5)}
      <div className="ev">Fit: {terms}</div>
      <div className="ev">
        IME {fmt(e.rate.ime_kg, 1)} kg, L {fmt(e.rate.length_m)} m, wind {fmt(e.rate.u10, 2)} m/s ({e.rate.wind_source}),
        U<sub>eff</sub> {fmt(e.rate.u_eff, 2)} m/s
      </div>
      <div className="ev">
        Uncertainty: {unc}
        {e.carbonate_fraction != null && ` · carbonate under plume ${fmt(100 * e.carbonate_fraction)}%`}
      </div>
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
