import type { LayerInfo } from "../api/client";

/** A layer as the user has set it up: shown or not, how opaque, and how it is stretched. */
export interface LayerState extends LayerInfo {
  visible: boolean;
  opacity: number;
  stretch: [number, number];
  cmap: string;
  keep: number[] | null; // categorical: codes shown in colour (null = all)
  minP: number; // mineral groups: hide pixels below this P(group correct)
}

/** Tile query for a layer's current settings. */
export function tileQuery(l: LayerState): Record<string, string | number | undefined> {
  if (l.kind === "categorical")
    return { groups: l.keep ? l.keep.join(",") : undefined, min_p: l.minP > 0 ? l.minP : undefined };
  if (l.kind === "continuous") return { rescale: l.stretch.join(","), colormap: l.cmap };
  return {};
}

/** Which layers start visible for a newly opened scene: the true-colour context and the main
 * product (mineral groups, else CH4 enhancement). */
export function initialLayers(layers: LayerInfo[]): LayerState[] {
  const has = (name: string) => layers.some((l) => l.name === name);
  const main = has("mineral_group") ? "mineral_group" : has("ch4_enh") ? "ch4_enh" : null;
  return layers.map((l) => ({
    ...l,
    visible: l.name === "rgb" || l.name === main,
    opacity: l.name === "rgb" ? 1 : 0.8,
    stretch: [l.rescale[0], l.rescale[1]] as [number, number],
    cmap: l.colormap,
    keep: null,
    minP: 0,
  }));
}

/** Move a layer one step up (drawn later, on top) or down in the drawing order. */
export function moveLayer(layers: LayerState[], id: string, dir: -1 | 1): LayerState[] {
  const i = layers.findIndex((l) => l.id === id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= layers.length) return layers;
  const out = layers.slice();
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

/** Fresh layer metadata (a newly built legend, say) merged into the user's current settings. */
export function mergeLayers(fresh: LayerInfo[], current: LayerState[]): LayerState[] {
  const byId = new Map(current.map((l) => [l.id, l]));
  const init = initialLayers(fresh);
  if (!current.length || current[0].scene !== fresh[0]?.scene) return init;
  const order = current.map((l) => l.id);
  return init
    .map((l) => {
      const c = byId.get(l.id);
      return c
        ? { ...l, visible: c.visible, opacity: c.opacity, stretch: c.stretch, cmap: c.cmap, keep: c.keep, minP: c.minP }
        : l;
    })
    .sort((a, b) => (order.indexOf(a.id) + 1 || 1e9) - (order.indexOf(b.id) + 1 || 1e9));
}
