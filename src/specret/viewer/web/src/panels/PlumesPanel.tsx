import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { GasSummaryPart, PlumeExplanation, PlumeRow } from "../api/types";
import { useJob } from "../hooks/useJob";
import { fmt } from "../lib";
import { FitChart, Rates, SignatureChart } from "./PlumeEvidence";

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

/** Direction the air moves towards, as a compass point. */
export function compass(u: number, v: number): string {
  const deg = ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
  return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.round(deg / 45) % 8];
}

function PlumeDetail({ plume, e }: { plume: PlumeRow; e: PlumeExplanation }) {
  const terms = Object.entries(e.fitted_terms)
    .map(([k, v]) => `${k.replace("confuser:", "confuser ")} ${fmt(v, k.startsWith("confuser") ? 3 : 0)}`)
    .join(", ");
  const u = e.wind?.wind_u;
  const v = e.wind?.wind_v;
  const hasWind = u != null && v != null && Number.isFinite(u) && Number.isFinite(v);
  return (
    <div className="plume-detail">
      <b>
        Plume {e.plume_id}: {e.gas}
      </b>
      , confidence {fmt(e.confidence, 2)} · source {fmt(plume.source_lat, 5)}, {fmt(plume.source_lon, 5)}
      <div className="ev">Fit: {terms}</div>
      <div className="ev">
        IME {fmt(e.rate.ime_kg, 1)} kg, L {fmt(e.rate.length_m)} m, wind {fmt(e.rate.u10, 2)} m/s
        {hasWind && ` towards ${compass(u!, v!)}`} ({e.rate.wind_source}), U<sub>eff</sub> {fmt(e.rate.u_eff, 2)} m/s
        {hasWind && " · arrow on the map"}
        {e.carbonate_fraction != null && ` · carbonate under plume ${fmt(100 * e.carbonate_fraction)}%`}
      </div>
      <Rates e={e} />
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
