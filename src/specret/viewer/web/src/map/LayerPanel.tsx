import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { pretty } from "../lib";
import type { LayerState } from "./layerState";

const COLORMAPS = ["viridis", "magma", "inferno", "cividis", "rdylgn", "rdbu_r", "greys_r", "turbo"];

interface Props {
  sceneKey: string | null;
  compareId: string | null;
  onCompare: (id: string | null) => void;
  layers: LayerState[];
  onChange: (id: string, patch: Partial<LayerState>) => void;
  onMove: (id: string, dir: -1 | 1) => void;
  showPlumes: boolean;
  hasPlumes: boolean;
  onShowPlumes: (v: boolean) => void;
}

/** Layer list, top of the drawing order first, grouped by kind. */
export function LayerPanel(props: Props) {
  const { layers, onChange, onMove, showPlumes, hasPlumes, onShowPlumes, compareId, onCompare } = props;
  const ramps = useQuery({ queryKey: ["colormaps"], queryFn: api.colormaps, staleTime: Infinity });
  const topFirst = layers.slice().reverse();
  if (!layers.length) return <p className="hint">This scene has no layers yet: map it or detect plumes.</p>;
  return (
    <div className="layers">
      {hasPlumes && (
        <label className="layer-row">
          <input type="checkbox" checked={showPlumes} onChange={(e) => onShowPlumes(e.target.checked)} />
          <span className="layer-title">Plume outlines</span>
          <span className="tag">vector</span>
        </label>
      )}
      {topFirst.map((l, i) => (
        <div key={l.id} className={`layer${l.visible ? "" : " off"}`}>
          <div className="layer-row">
            <input
              type="checkbox"
              aria-label={`Show ${l.title}`}
              checked={l.visible}
              onChange={(e) => onChange(l.id, { visible: e.target.checked })}
            />
            <span className="layer-title" title={l.note || l.title}>
              {l.title}
            </span>
            <span className="tag">{l.group}</span>
            <button
              type="button"
              className={compareId === l.id ? "on" : ""}
              title="Swipe-compare this layer against the map"
              aria-label={`Swipe-compare ${l.title}`}
              onClick={() => onCompare(compareId === l.id ? null : l.id)}
            >
              ⇆
            </button>
            <button type="button" aria-label={`Move ${l.title} up`} disabled={i === 0} onClick={() => onMove(l.id, 1)}>
              ↑
            </button>
            <button
              type="button"
              aria-label={`Move ${l.title} down`}
              disabled={i === topFirst.length - 1}
              onClick={() => onMove(l.id, -1)}
            >
              ↓
            </button>
          </div>
          {l.visible && (
            <div className="layer-body">
              <label className="hint">
                Opacity
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={l.opacity}
                  aria-label={`${l.title} opacity`}
                  onChange={(e) => onChange(l.id, { opacity: Number(e.target.value) })}
                />
              </label>
              {!l.ready && l.buildable && <div className="hint">Built from the source granule on first view (a few seconds).</div>}
              {l.kind === "continuous" && (
                <>
                  <div className="stretch">
                    <select
                      aria-label={`${l.title} colormap`}
                      value={l.cmap}
                      onChange={(e) => onChange(l.id, { cmap: e.target.value })}
                    >
                      {COLORMAPS.map((c) => (
                        <option key={c}>{c}</option>
                      ))}
                    </select>
                    <input
                      type="number"
                      aria-label={`${l.title} minimum`}
                      value={l.stretch[0]}
                      onChange={(e) => onChange(l.id, { stretch: [Number(e.target.value), l.stretch[1]] })}
                    />
                    <input
                      type="number"
                      aria-label={`${l.title} maximum`}
                      value={l.stretch[1]}
                      onChange={(e) => onChange(l.id, { stretch: [l.stretch[0], Number(e.target.value)] })}
                    />
                    <span className="hint">{l.units}</span>
                  </div>
                  <Ramp colors={ramps.data?.[l.cmap]} lo={l.stretch[0]} hi={l.stretch[1]} units={l.units} />
                </>
              )}
              {l.kind === "categorical" && <Categories layer={l} onChange={onChange} />}
              {l.name === "agreement" && l.ready && props.sceneKey && <AgreementStats sceneKey={props.sceneKey} />}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function Ramp({ colors, lo, hi, units }: { colors?: string[]; lo: number; hi: number; units: string }) {
  if (!colors) return null;
  return (
    <div className="ramp">
      <div className="ramp-bar" style={{ background: `linear-gradient(to right, ${colors.join(", ")})` }} />
      <div className="ramp-labels">
        <span>{lo}</span>
        <span>
          {hi} {units}
        </span>
      </div>
    </div>
  );
}

/** Legend chips that filter the categories shown; for mineral groups, a confidence mask. */
function Categories({ layer: l, onChange }: { layer: LayerState; onChange: Props["onChange"] }) {
  const present = l.legend.filter((e) => e.fraction == null || e.fraction > 0);
  const shown = (code: number) => l.keep === null || l.keep.includes(code);
  const toggle = (code: number) => {
    const all = present.map((e) => e.code);
    const cur = l.keep ?? all;
    const next = cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code];
    onChange(l.id, { keep: next.length === all.length ? null : next });
  };
  return (
    <>
      {l.name === "mineral_group" && (
        <label className="hint">
          Hide P(group correct) below {l.minP.toFixed(2)}
          <input
            type="range"
            min={0}
            max={0.9}
            step={0.05}
            value={l.minP}
            aria-label="Confidence threshold"
            onChange={(e) => onChange(l.id, { minP: Number(e.target.value) })}
          />
        </label>
      )}
      <div className="chips-head hint">
        Click to filter ·{" "}
        <button type="button" className="link" onClick={() => onChange(l.id, { keep: null })}>
          all
        </button>{" "}
        <button type="button" className="link" onClick={() => onChange(l.id, { keep: [] })}>
          none
        </button>
      </div>
      <div className="legend chips">
        {present.map((e) => (
          <button
            type="button"
            key={e.code}
            className={shown(e.code) ? "" : "off"}
            aria-pressed={shown(e.code)}
            onClick={() => toggle(e.code)}
          >
            <i style={{ background: `rgb(${e.rgb.join(",")})` }} />
            {pretty(e.label)}
            {e.fraction != null && <span className="frac">{(100 * e.fraction).toFixed(e.fraction < 0.01 ? 1 : 0)}%</span>}
          </button>
        ))}
      </div>
    </>
  );
}

function AgreementStats({ sceneKey }: { sceneKey: string }) {
  const q = useQuery({ queryKey: ["agreement", sceneKey], queryFn: () => api.agreement(sceneKey) });
  if (!q.data) return null;
  const a = q.data;
  return (
    <div className="hint">
      <b>{(100 * a.agreement).toFixed(1)}%</b> agree over {a.pixels_with_reference.toLocaleString()} pixels where L2B MIN
      reports a ~2 µm mineral (band depth ≥ {a.depth_min}).
      <table className="values">
        <thead>
          <tr>
            <th>Ours</th>
            <th>L2B MIN</th>
            <th className="num">Pixels</th>
          </tr>
        </thead>
        <tbody>
          {a.top_confusions.slice(0, 5).map((c) => (
            <tr key={`${c.ours}-${c.l2bmin}`}>
              <td>{pretty(c.ours)}</td>
              <td>{pretty(c.l2bmin)}</td>
              <td className="num">{c.pixels.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
