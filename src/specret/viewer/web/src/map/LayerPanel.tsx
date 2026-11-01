import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { pretty } from "../lib";
import type { LayerState } from "./layerState";

const COLORMAPS = ["viridis", "magma", "inferno", "cividis", "rdylgn", "rdbu_r", "greys_r", "turbo"];

interface Props {
  layers: LayerState[];
  onChange: (id: string, patch: Partial<LayerState>) => void;
  onMove: (id: string, dir: -1 | 1) => void;
  showPlumes: boolean;
  hasPlumes: boolean;
  onShowPlumes: (v: boolean) => void;
}

/** Layer list, top of the drawing order first, grouped by kind. */
export function LayerPanel({ layers, onChange, onMove, showPlumes, hasPlumes, onShowPlumes }: Props) {
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
              {l.kind === "categorical" && (
                <div className="legend">
                  {l.legend.map((e) => (
                    <span key={e.code}>
                      <i style={{ background: `rgb(${e.rgb.join(",")})` }} />
                      {pretty(e.label)}
                    </span>
                  ))}
                </div>
              )}
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
