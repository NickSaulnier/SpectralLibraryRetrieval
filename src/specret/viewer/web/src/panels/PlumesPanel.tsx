import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { GasSummaryPart, PlumeExplanation, PlumeRow } from "../api/types";
import { useJob } from "../hooks/useJob";
import { fmt } from "../lib";

const GASES = ["CH4", "CO2"];

interface Props {
  gasScene: string | null; // gas scene id (EMIT), or null when the scene has no radiance
  analyzed: boolean;
  selected: number | null;
  onSelect: (plume: PlumeRow) => void;
  onChanged: () => void; // a detection job finished
}

/** Plumes of the active scene: summary, sortable table and the selected plume's evidence. */
export function PlumesPanel({ gasScene, analyzed, selected, onSelect, onChanged }: Props) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId, 2000);
  useEffect(() => setJobId(null), [gasScene]);
  useEffect(() => {
    if (job?.state === "done") {
      qc.invalidateQueries({ queryKey: ["gasPlumes", gasScene] });
      onChanged();
    }
  }, [job?.state, qc, gasScene, onChanged]);

  const plumes = useQuery({
    queryKey: ["gasPlumes", gasScene],
    queryFn: () => api.gasPlumes(gasScene!),
    enabled: !!gasScene && analyzed,
  });
  const detail = useQuery({
    queryKey: ["plume", gasScene, selected],
    queryFn: () => api.plume(gasScene!, selected!),
    enabled: !!gasScene && analyzed && selected !== null,
  });

  if (!gasScene) return <p className="hint">This scene has no radiance granule for gas detection.</p>;
  if (job?.state === "running")
    return (
      <div>
        <progress max={1} value={job.progress} />
        <div className="hint">
          Detecting… {Math.round(100 * job.progress)}% · {job.elapsed_s} s
        </div>
      </div>
    );
  if (!analyzed)
    return (
      <div className="empty">
        <p>Not analyzed yet: detection takes a minute or two.</p>
        <button className="primary" onClick={async () => setJobId((await api.startGas(gasScene)).job_id)}>
          Detect plumes
        </button>
        {job?.state === "failed" && <div className="hint">Detection failed: {job.error?.split("\n")[0]}</div>}
      </div>
    );

  const rows = plumes.data?.plumes ?? [];
  const summary = plumes.data?.summary;
  const parts = GASES.flatMap((g) => {
    const v = summary?.[g] as GasSummaryPart | undefined;
    return v?.plumes ? [`${g}: ${v.plumes} plumes, ${fmt(v.total_q_kg_h)} kg/h in total`] : [];
  });
  const sel = rows.find((p) => p.plume_id === selected) ?? null;
  return (
    <div>
      {summary && (
        <p className="hint">
          {summary.plumes} plumes from {summary.candidates ?? "?"} candidates. {parts.join("; ")}
        </p>
      )}
      {detail.data && sel && <PlumeDetail plume={sel} e={detail.data} />}
      <PlumeTable rows={rows} selected={selected} onSelect={onSelect} />
    </div>
  );
}

function PlumeTable({ rows, selected, onSelect }: { rows: PlumeRow[]; selected: number | null; onSelect: (p: PlumeRow) => void }) {
  return (
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Gas</th>
          <th className="num">Conf.</th>
          <th className="num">Q (kg/h)</th>
          <th className="num">±1σ</th>
          <th className="num">Wind</th>
          <th>Confusers</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((p) => (
          <tr
            key={p.plume_id}
            className={`clickable${p.plume_id === selected ? " sel" : ""}`}
            onClick={() => onSelect(p)}
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
        Q {fmt(e.rate.q_kg_h)} ± {fmt(e.rate.sigma_q_kg_h)} kg/h · IME {fmt(e.rate.ime_kg, 1)} kg, L {fmt(e.rate.length_m)} m,
        wind {fmt(e.rate.u10, 2)} m/s ({e.rate.wind_source}), U<sub>eff</sub> {fmt(e.rate.u_eff, 2)} m/s
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
