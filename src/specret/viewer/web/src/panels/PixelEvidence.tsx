import { useMemo } from "react";

import type { Identification } from "../api/types";
import { css, gapped, Plot, windowShapes } from "../charts/Plot";
import { MATCH_COLORS } from "../charts/SpectrumChart";
import { fmt, pretty } from "../lib";

export const MAX_PINS = 5;

export interface Pin {
  name: string;
  result: Identification;
}

/** Add a spectrum to the comparison: replaces a pin of the same name, keeps the newest five. */
export function addPin(pins: Pin[], pin: Pin): Pin[] {
  return [...pins.filter((p) => p.name !== pin.name), pin].slice(-MAX_PINS);
}

/** A pin's label: "r703 c525" for a scene pixel, else the spectrum's own name. */
export const pinLabel = (name: string) => name.match(/r\d+_c\d+$/)?.[0].replace("_", " ") ?? name;

const PIN_COLORS = ["--m1", "--m2", "--m3", "--query", "--muted"];

/** Per-window evidence for one match, as the feature-fit score sees it: band centres, depths
 * and shape correlation, with the continuum-removed query and match in each window. */
export function WindowCards({ result, index }: { result: Identification; index: number }) {
  const m = result.matches[index];
  const profiles = Object.fromEntries((result.profiles ?? []).map((p) => [p.window, p]));
  const feats = m?.features ?? [];
  if (!feats.length) return <p className="hint">No diagnostic window has a feature in this pixel or {m?.material_name}.</p>;
  return (
    <div className="cards">
      {feats.map((f) => {
        const p = profiles[f.window];
        return (
          <div key={f.window} className="card">
            <div className="card-head">
              <b>{f.window.replaceAll("_", "-")}</b>
              <span className="hint">
                weight {fmt(f.weight, 3)} · score {fmt(f.score, 2)}
              </span>
            </div>
            <table className="values">
              <tbody>
                <tr>
                  <td>Band centre</td>
                  <td className="num">
                    {fmt(f.query_center_nm, 0)} → {fmt(f.cand_center_nm, 0)} nm
                  </td>
                </tr>
                <tr>
                  <td>Depth</td>
                  <td className="num">
                    {fmt(f.query_depth, 3)} / {fmt(f.cand_depth, 3)}
                  </td>
                </tr>
                <tr>
                  <td>Shape r</td>
                  <td className="num">{fmt(f.shape_r, 2)}</td>
                </tr>
              </tbody>
            </table>
            {p && index < p.matches.length && <ProfileInset p={p} index={index} name={m.material_name} />}
          </div>
        );
      })}
    </div>
  );
}

function ProfileInset({ p, index, name }: { p: NonNullable<Identification["profiles"]>[number]; index: number; name: string }) {
  const traces = useMemo(
    () => [
      { x: p.wavelengths_nm, y: p.query.map((d) => 1 - d), name: "pixel", mode: "lines", line: { color: css("--query"), width: 2 } },
      {
        x: p.wavelengths_nm,
        y: p.matches[index].map((d) => 1 - d),
        name,
        mode: "lines",
        line: { color: css(MATCH_COLORS[index]), width: 1.5 },
      },
    ],
    [p, index, name],
  );
  return (
    <Plot
      className="plot inset"
      label={`Continuum-removed ${p.window}`}
      traces={traces}
      layout={{ margin: { l: 36, r: 6, t: 4, b: 24 }, showlegend: false, yaxis: { title: { text: "CR" } } }}
    />
  );
}

/** The unmixing fit: each endmember scaled by its coefficient, the baseline, their sum against
 * the pixel, and the residual. */
export function UnmixChart({ result }: { result: Identification }) {
  const u = result.unmixing;
  const traces = useMemo(() => {
    if (!u?.spectra) return [];
    const { x: wl, y: g } = gapped(result.wavelengths_nm);
    return [
      { x: wl, y: g(result.query), name: "pixel", mode: "lines", line: { color: css("--query"), width: 2.5 } },
      { x: wl, y: g(u.spectra.model), name: "model", mode: "lines", line: { color: css("--ink"), width: 1.5, dash: "dot" } },
      ...u.spectra.components.map((c, i) => ({
        x: wl,
        y: g(c),
        name: `${u.components[i].material_name} ${(100 * u.components[i].fraction).toFixed(0)}%`,
        mode: "lines",
        line: { color: css(MATCH_COLORS[i % 3]), width: 1.2 },
      })),
      { x: wl, y: g(u.spectra.baseline), name: "baseline", mode: "lines", line: { color: css("--muted"), width: 1, dash: "dash" } },
      { x: wl, y: g(u.spectra.residual), name: "residual", mode: "lines", yaxis: "y2", line: { color: css("--muted"), width: 1 } },
    ];
  }, [u, result]);
  if (!u?.spectra) return null;
  return (
    <>
      <h3>Unmixing</h3>
      <p className="hint">
        RMS residual {fmt(u.residual_rms, 4)}
        {u.reduced_chi2 != null && ` · reduced χ² ${fmt(u.reduced_chi2, 2)}`}. The residual is on the lower axis.
      </p>
      <Plot
        label="Unmixing decomposition"
        traces={traces}
        layout={{
          xaxis: { title: { text: "Wavelength (nm)" }, anchor: "y2" },
          yaxis: { title: { text: "Reflectance" }, domain: [0.32, 1] },
          yaxis2: { title: { text: "Resid." }, domain: [0, 0.24], gridcolor: css("--line"), zeroline: true, zerolinecolor: css("--line") },
          shapes: windowShapes(Object.values(result.windows)),
        }}
      />
    </>
  );
}

/** Pinned spectra overlaid (each scaled to its maximum), with a table of their verdicts. */
export function PinnedSpectra({ pins, onRemove, onClear }: { pins: Pin[]; onRemove: (name: string) => void; onClear: () => void }) {
  const traces = useMemo(
    () =>
      pins.map((p, i) => {
        const m = Math.max(...p.result.query.filter(Number.isFinite));
        const g = gapped(p.result.wavelengths_nm);
        return {
          x: g.x,
          y: g.y(p.result.query.map((v) => v / m)),
          name: pinLabel(p.name),
          mode: "lines",
          line: { color: css(PIN_COLORS[i]), width: 1.8 },
        };
      }),
    [pins],
  );
  if (!pins.length) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Compare ({pins.length}/{MAX_PINS})</h2>
        <button onClick={onClear}>Clear</button>
      </div>
      <Plot
        label="Pinned spectra"
        traces={traces}
        layout={{ xaxis: { title: { text: "Wavelength (nm)" } }, yaxis: { title: { text: "Scaled reflectance" } } }}
      />
      <table>
        <thead>
          <tr>
            <th>Spectrum</th>
            <th>Best match</th>
            <th>Group</th>
            <th className="num">P</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {pins.map((p, i) => (
            <tr key={p.name}>
              <td title={p.name}>
                <span className="swatch" style={{ background: `var(${PIN_COLORS[i]})` }} />
                {pinLabel(p.name)}
              </td>
              <td>{p.result.summary.best_material ?? "no match"}</td>
              <td>{pretty(p.result.summary.best_group)}</td>
              <td className="num">{fmt(p.result.summary.p_group_correct, 2)}</td>
              <td>
                <button className="link" aria-label={`Unpin ${pinLabel(p.name)}`} onClick={() => onRemove(p.name)}>
                  ×
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {new Set(pins.map((p) => p.result.summary.best_group)).size === 1 && pins.length > 1 && (
        <p className="hint">All pinned spectra agree on {pretty(pins[0].result.summary.best_group)}.</p>
      )}
    </section>
  );
}
