import { useMemo } from "react";

import type { PlumeExplanation, PlumeSpectra } from "../api/types";
import { css, gapped, Plot, windowShapes } from "../charts/Plot";
import { fmt } from "../lib";

const TERM_COLORS = ["--m1", "--m2", "--m3", "--muted"];
const BUDGET_COLORS = ["--m1", "--m2", "--m3", "--query", "--muted"];

/** Each source's share of the rate's variance: the relative uncertainties add in quadrature. */
export function budgetShares(rel: Record<string, number>): { name: string; rel: number; share: number }[] {
  const parts = Object.entries(rel).filter(([, v]) => Number.isFinite(v) && v > 0);
  const total = parts.reduce((s, [, v]) => s + v * v, 0);
  return parts
    .map(([name, v]) => ({ name, rel: v, share: total > 0 ? (v * v) / total : 0 }))
    .sort((a, b) => b.share - a.share);
}

const label = (t: string) => t.replace("confuser:", "");

/** The plume's mean log-radiance residual against the fit: the shifted-mask null spread as a
 * band, each fitted term, their sum, and the remainder on the lower axis. */
export function FitChart({ s }: { s: PlumeSpectra }) {
  const traces = useMemo(() => {
    const { x: wl, y: g } = gapped(s.wavelengths_nm);
    // The null band, one filled pair per run of contiguous bands so it does not bridge gaps
    const runs: number[][] = [[]];
    wl.forEach((w, i) => (w === null ? runs.push([]) : runs[runs.length - 1].push(i)));
    const xs = wl.filter((w): w is number => w !== null);
    const bands = (yaxis: string, legend: boolean) =>
      runs.flatMap((run, r) => {
        const idx = run.map((i) => i - r); // positions in the gap-free arrays
        const base = { x: idx.map((i) => xs[i]), mode: "lines", line: { width: 0 }, hoverinfo: "skip", yaxis, showlegend: false };
        return [
          { ...base, y: idx.map((i) => -s.null_sd[i]) },
          {
            ...base,
            y: idx.map((i) => s.null_sd[i]),
            fill: "tonexty",
            fillcolor: css("--window"),
            name: "null ±1σ",
            legendgroup: "null",
            showlegend: legend && r === 0,
          },
        ];
      });
    return [
      ...bands("y", true),
      { x: wl, y: g(s.observed), name: "plume", mode: "lines", line: { color: css("--query"), width: 2 } },
      { x: wl, y: g(s.model), name: "fit", mode: "lines", line: { color: css("--ink"), width: 1.5, dash: "dot" } },
      ...Object.entries(s.terms).map(([t, y], i) => ({
        x: wl,
        y: g(y),
        name: label(t),
        mode: "lines",
        line: { color: css(TERM_COLORS[i % TERM_COLORS.length]), width: 1.2 },
      })),
      {
        x: wl,
        y: g(s.remainder),
        name: "remainder",
        mode: "lines",
        yaxis: "y2",
        line: { color: css("--muted"), width: 1 },
      },
      ...bands("y2", false),
    ];
  }, [s]);
  return (
    <Plot
      label="Plume fit"
      traces={traces}
      layout={{
        xaxis: { title: { text: "Wavelength (nm)" }, anchor: "y2" },
        yaxis: { title: { text: "ln ratio" }, domain: [0.34, 1] },
        yaxis2: { title: { text: "Rem." }, domain: [0, 0.26], gridcolor: css("--line"), zeroline: true, zerolinecolor: css("--line") },
        shapes: windowShapes(s.windows),
      }}
    />
  );
}

/** Each candidate gas's absorption per ppm·m, scaled to its own maximum, with the fit windows. */
export function SignatureChart({ s, fitted }: { s: PlumeSpectra; fitted: string[] }) {
  const traces = useMemo(
    () =>
      Object.entries(s.signatures_per_ppmm).map(([g, k], i) => {
        const m = Math.max(...k.map(Math.abs)) || 1;
        const gap = gapped(s.wavelengths_nm);
        return {
          x: gap.x,
          y: gap.y(k.map((v) => v / m)),
          name: fitted.includes(g) ? `${g} (fitted)` : g,
          mode: "lines",
          line: { color: css(TERM_COLORS[i % TERM_COLORS.length]), width: fitted.includes(g) ? 2 : 1 },
          opacity: fitted.includes(g) ? 1 : 0.6,
        };
      }),
    [s, fitted],
  );
  return (
    <Plot
      label="Gas signatures"
      traces={traces}
      layout={{
        xaxis: { title: { text: "Wavelength (nm)" } },
        yaxis: { title: { text: "Absorption (scaled)" } },
        shapes: windowShapes(s.windows),
      }}
    />
  );
}

/** Both rate estimates on one scale, IME ± 1σ and the cross-sectional flux with its spread
 * between transects, then the uncertainty budget of the IME rate. */
export function Rates({ e }: { e: PlumeExplanation }) {
  const q = e.rate.q_kg_h;
  const sq = e.rate.sigma_q_kg_h;
  const csf = e.cross_sectional_flux?.q_csf_kg_h ?? null;
  const half = csf != null && e.cross_sectional_flux?.csf_iqr_rel != null ? (Math.abs(csf) * e.cross_sectional_flux.csf_iqr_rel) / 2 : 0;
  const rows = [
    { name: "IME", q, lo: q - sq, hi: q + sq, note: `±1σ ${fmt(sq)}` },
    ...(csf != null && Number.isFinite(csf)
      ? [{ name: "Cross-section", q: csf, lo: csf - half, hi: csf + half, note: `${e.cross_sectional_flux?.transects ?? "?"} transects, IQR ${fmt(100 * (e.cross_sectional_flux?.csf_iqr_rel ?? NaN))}%` }]
      : []),
  ].filter((r) => Number.isFinite(r.q));
  const max = Math.max(1, ...rows.map((r) => r.hi));
  const pct = (v: number) => `${(100 * Math.max(0, v)) / max}%`;
  const budget = budgetShares(e.relative_uncertainty);
  return (
    <div className="rates">
      <h3>Emission rate (kg/h)</h3>
      {rows.map((r) => (
        <div key={r.name} className="bar-row">
          <span>{r.name}</span>
          <span className="bar rate" aria-label={`${r.name} ${fmt(r.q)} kg/h`}>
            <i className="range" style={{ left: pct(r.lo), width: `calc(${pct(r.hi)} - ${pct(r.lo)})` }} />
            <i className="point" style={{ left: pct(r.q) }} />
          </span>
          <span className="num">{fmt(r.q)}</span>
        </div>
      ))}
      <div className="hint">{rows.map((r) => `${r.name}: ${r.note}`).join(" · ")}</div>
      {budget.length > 0 && (
        <>
          <div className="hint">Uncertainty budget (share of the IME rate's variance)</div>
          <div className="budget" role="img" aria-label="Uncertainty budget">
            {budget.map((b, i) => (
              <i
                key={b.name}
                style={{ width: `${100 * b.share}%`, background: `var(${BUDGET_COLORS[i % BUDGET_COLORS.length]})` }}
                title={`${b.name}: ±${fmt(100 * b.rel)}%`}
              />
            ))}
          </div>
          <div className="legend">
            {budget.map((b, i) => (
              <span key={b.name}>
                <i style={{ background: `var(${BUDGET_COLORS[i % BUDGET_COLORS.length]})` }} />
                {b.name} ±{fmt(100 * b.rel)}% ({fmt(100 * b.share)}%)
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
