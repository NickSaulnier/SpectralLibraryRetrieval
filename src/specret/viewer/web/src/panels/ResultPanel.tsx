import { useEffect, useState } from "react";

import type { Identification } from "../api/types";
import { MATCH_COLORS, SpectrumChart } from "../charts/SpectrumChart";
import { download, pretty, toCsv } from "../lib";
import { UnmixChart, WindowCards } from "./PixelEvidence";

const TIER: Record<string, string> = { confident: "high", uncertain: "medium", "likely unknown": "low" };

const evidence = (f: { window: string; query_center_nm: number; cand_center_nm: number; shape_r: number }) =>
  `${f.window.replaceAll("_", "-")}: ${f.query_center_nm.toFixed(0)}→${f.cand_center_nm.toFixed(0)} nm, r ${f.shape_r.toFixed(2)}`;

export function matchesCsv(r: Identification, name: string): string {
  const s = r.summary;
  const px = r.pixel;
  const rows: unknown[][] = [
    ["source", "row", "col", "lat", "lon", "verdict", "p_group_correct", "rank", "spectrum_id",
     "material_name", "group", "score", "purity", "evidence"],
  ];
  r.matches.forEach((m, i) =>
    rows.push([
      name, px?.row, px?.col, px?.lat, px?.lon, s.verdict, s.p_group_correct, i + 1, m.spectrum_id,
      m.material_name, m.group, m.score.toFixed(4), m.purity_flag,
      (m.features ?? [])
        .map((f) => `${f.window} ${f.query_center_nm.toFixed(0)}->${f.cand_center_nm.toFixed(0)} nm r ${f.shape_r.toFixed(2)}`)
        .join("; "),
    ]),
  );
  return toCsv(rows);
}

export function spectraCsv(r: Identification): string {
  const top = r.matches.slice(0, 3);
  const rows: unknown[][] = [["wavelength_nm", "query", ...top.map((m) => `${m.material_name} (${m.spectrum_id})`)]];
  r.wavelengths_nm.forEach((w, i) => rows.push([w, r.query[i], ...top.map((m) => m.spectrum?.[i])]));
  return toCsv(rows);
}

interface Props {
  status: string;
  result: Identification | null;
  name: string;
  onPin?: () => void;
}

/** Verdict, spectrum against matches, the matches table, the selected match's per-window
 * evidence, the unmixing decomposition and CSV exports. */
export function ResultPanel({ status, result, name, onPin }: Props) {
  const [sel, setSel] = useState(0);
  useEffect(() => setSel(0), [result]);
  const s = result?.summary;
  const tier = TIER[s?.confidence ?? ""] ?? "medium";
  const mix =
    result?.unmixing && result.unmixing.components.length > 1
      ? result.unmixing.components.map((c) => `${c.material_name} ${(100 * c.fraction).toFixed(0)}%`).join(" + ")
      : null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Spectrum and matches</h2>
        <div className="export">
          <button disabled={!result} onClick={() => download(`${name}_matches.csv`, matchesCsv(result!, name))}>
            Matches CSV
          </button>
          <button disabled={!result} onClick={() => download(`${name}_spectra.csv`, spectraCsv(result!))}>
            Spectra CSV
          </button>
          {onPin && (
            <button disabled={!result} onClick={onPin} title="Overlay this spectrum with others (up to 5)">
              Pin
            </button>
          )}
        </div>
      </div>
      <div className="status">{status}</div>
      {result && s && (
        <>
          {s.best_material ? (
            <div className="verdict">
              <b>{s.best_material}</b>
              <span>{pretty(s.best_group)}</span>
              <span className={`pill ${tier}`}>{s.confidence}</span>
              {s.p_group_correct != null && <span className="hint">P(group right) {s.p_group_correct.toFixed(2)}</span>}
            </div>
          ) : (
            <div className="verdict">
              <b>No match</b>
              <span className="hint">{s.reason}</span>
            </div>
          )}
          {mix && <div className="hint">Unmixed: {mix}</div>}
          <SpectrumChart result={result} />
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Material</th>
                <th>Group</th>
                <th className="num">Score</th>
                <th>Purity</th>
              </tr>
            </thead>
            <tbody>
              {result.matches.map((m, i) => (
                <tr
                  key={m.spectrum_id}
                  className={`clickable${i === sel ? " sel" : ""}`}
                  onClick={() => setSel(i)}
                  title="Show this match's evidence"
                >
                  <td>
                    {i < 3 && <span className="swatch" style={{ background: `var(${MATCH_COLORS[i]})` }} />}
                    {i + 1}
                  </td>
                  <td>
                    {m.material_name}
                    <div className="ev">{(m.features ?? []).slice(0, 2).map(evidence).join(" · ")}</div>
                  </td>
                  <td>{pretty(m.group)}</td>
                  <td className="num">{m.score.toFixed(2)}</td>
                  <td>{m.purity_flag ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {result.matches[sel] && (
            <>
              <h3>
                Evidence: {result.matches[sel].material_name} <span className="hint">(pixel → library)</span>
              </h3>
              <WindowCards result={result} index={sel} />
            </>
          )}
          <UnmixChart result={result} />
        </>
      )}
    </section>
  );
}
