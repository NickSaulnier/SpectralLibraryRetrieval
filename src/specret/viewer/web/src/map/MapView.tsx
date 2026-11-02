import maplibregl, { type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef, useState } from "react";

import { api, type GeoScene, type Polygon } from "../api/client";
import { tileQuery, type LayerState } from "./layerState";
import { windArrow, type Wind } from "./windArrow";

const OSM = {
  tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
  attribution: "© OpenStreetMap contributors",
};
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export type Bounds = [number, number, number, number];
export type DrawMode = "box" | "polygon" | null;

interface Props {
  scenes: GeoScene[];
  activeKey: string | null;
  layers: LayerState[];
  plumes: GeoJSON.FeatureCollection | null;
  showPlumes: boolean;
  selectedPlume: number | null;
  point: { lat: number; lon: number } | null;
  basemap: boolean;
  fitTo: { bounds: Bounds; seq: number } | null;
  drawMode: DrawMode;
  selection: Polygon | null;
  compare: LayerState | null;
  wind: Wind | null; // the selected plume's wind at its source
  onPoint: (lat: number, lon: number) => void;
  onPlume: (plumeId: number) => void;
  onScene: (key: string) => void;
  onTileLoaded: (layerId: string) => void;
  onSelection: (polygon: Polygon) => void;
  onView?: (bounds: Bounds) => void; // the visible area, after each move
}

const footprints = (scenes: GeoScene[], active: string | null): GeoJSON.FeatureCollection => ({
  type: "FeatureCollection",
  features: scenes
    .filter((s) => s.bounds)
    .map((s) => {
      const [w, so, e, n] = s.bounds as Bounds;
      return {
        type: "Feature",
        properties: { key: s.key, active: s.key === active },
        geometry: { type: "Polygon", coordinates: [[[w, so], [e, so], [e, n], [w, n], [w, so]]] },
      };
    }),
});

export const ring = (poly: Polygon): GeoJSON.Feature => ({
  type: "Feature",
  properties: {},
  geometry: { type: "Polygon", coordinates: [[...poly, poly[0]]] },
});

const boxOf = (a: maplibregl.LngLat, b: maplibregl.LngLat): Polygon => [
  [a.lng, a.lat],
  [b.lng, a.lat],
  [b.lng, b.lat],
  [a.lng, b.lat],
];

/** Add (or update) a raster layer for each entry of `layers`, in order, below `before`. */
function syncRasters(m: MlMap, layers: LayerState[], before: string) {
  const wanted = new Set(layers.map((l) => `lyr-${l.id}`));
  for (const layer of m.getStyle().layers) {
    if (layer.id.startsWith("lyr-") && !wanted.has(layer.id)) {
      m.removeLayer(layer.id);
      m.removeSource(layer.id.replace("lyr-", "src-"));
    }
  }
  for (const l of layers) {
    const src = `src-${l.id}`;
    const lyr = `lyr-${l.id}`;
    const tiles = [api.tileUrl(l.id, tileQuery(l))];
    const s = m.getSource(src) as maplibregl.RasterTileSource | undefined;
    if (!s) {
      m.addSource(src, { type: "raster", tiles, tileSize: 256, maxzoom: 18 });
      m.addLayer({ id: lyr, type: "raster", source: src, paint: { "raster-resampling": "nearest" } }, before);
    } else if (JSON.stringify(s.tiles) !== JSON.stringify(tiles)) {
      s.setTiles(tiles);
    }
    m.setLayoutProperty(lyr, "visibility", l.visible ? "visible" : "none");
    m.setPaintProperty(lyr, "raster-opacity", l.opacity);
    m.moveLayer(lyr, before); // in list order: later layers end up on top
  }
}

/** The MapLibre map: raster layers from the tile API, plume outlines, scene footprints, area
 * drawing (box or polygon), and a swipe comparison against one other layer. */
export function MapView(p: Props) {
  const el = useRef<HTMLDivElement>(null);
  const cmpEl = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const cmp = useRef<MlMap | null>(null);
  const ready = useRef(false);
  const marker = useRef<maplibregl.Marker | null>(null);
  const handlers = useRef(p);
  handlers.current = p;
  const draft = useRef<{ start?: maplibregl.LngLat; verts: Polygon }>({ verts: [] });
  const [swipe, setSwipe] = useState(50);

  const setSelectionData = (poly: Polygon | null) => {
    const s = map.current?.getSource("selection") as GeoJSONSource | undefined;
    s?.setData(poly && poly.length >= 2 ? { type: "FeatureCollection", features: [ring(poly)] } : EMPTY);
  };

  // Create the map once
  useEffect(() => {
    const m = new maplibregl.Map({
      container: el.current!,
      style: {
        version: 8,
        sources: {
          osm: { type: "raster", tileSize: 256, ...OSM },
          footprints: { type: "geojson", data: EMPTY },
          plumes: { type: "geojson", data: EMPTY },
          selection: { type: "geojson", data: EMPTY },
          wind: { type: "geojson", data: EMPTY },
        },
        layers: [
          { id: "background", type: "background", paint: { "background-color": "#20242a" } },
          { id: "osm", type: "raster", source: "osm", layout: { visibility: "none" } },
          { id: "rasters-top", type: "background", paint: { "background-opacity": 0 } },
          {
            id: "footprints-fill", type: "fill", source: "footprints",
            paint: { "fill-color": "#6aa5e6", "fill-opacity": ["case", ["get", "active"], 0, 0.06] },
          },
          {
            id: "footprints-line", type: "line", source: "footprints",
            paint: {
              "line-color": ["case", ["get", "active"], "#ffd166", "#6aa5e6"],
              "line-width": ["case", ["get", "active"], 2, 1],
              "line-dasharray": [3, 2],
            },
          },
          { id: "plumes-fill", type: "fill", source: "plumes", paint: { "fill-color": "#00e5ff", "fill-opacity": 0.12 } },
          { id: "plumes-line", type: "line", source: "plumes", paint: { "line-color": "#00e5ff", "line-width": 1.5 } },
          {
            id: "plume-selected", type: "line", source: "plumes", filter: ["==", ["get", "plume_id"], -1],
            paint: { "line-color": "#ffffff", "line-width": 3 },
          },
          {
            id: "wind-casing", type: "line", source: "wind",
            layout: { "line-cap": "round", "line-join": "round" },
            paint: { "line-color": "#000000", "line-width": 5, "line-opacity": 0.6 },
          },
          {
            id: "wind-line", type: "line", source: "wind",
            layout: { "line-cap": "round", "line-join": "round" },
            paint: { "line-color": "#ffffff", "line-width": 2.5 },
          },
          { id: "selection-fill", type: "fill", source: "selection", paint: { "fill-color": "#ffd166", "fill-opacity": 0.1 } },
          {
            id: "selection-line", type: "line", source: "selection",
            paint: { "line-color": "#ffd166", "line-width": 2, "line-dasharray": [2, 1] },
          },
        ],
      },
      center: [0, 20],
      zoom: 1.5,
      attributionControl: { compact: true },
    });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    const view = () => {
      const b = m.getBounds();
      handlers.current.onView?.([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
    };
    m.on("load", () => {
      ready.current = true;
      m.fire("specret:sync");
      view();
    });
    m.on("moveend", view);

    // Drawing: a box by dragging, a polygon by clicking vertices and double-clicking to close
    m.on("mousedown", (e) => {
      if (handlers.current.drawMode !== "box") return;
      e.preventDefault();
      draft.current = { start: e.lngLat, verts: [] };
    });
    m.on("mousemove", (e) => {
      const d = draft.current;
      if (handlers.current.drawMode === "box" && d.start) setSelectionData(boxOf(d.start, e.lngLat));
      if (handlers.current.drawMode === "polygon" && d.verts.length)
        setSelectionData([...d.verts, [e.lngLat.lng, e.lngLat.lat]]);
    });
    m.on("mouseup", (e) => {
      const d = draft.current;
      if (handlers.current.drawMode !== "box" || !d.start) return;
      const poly = boxOf(d.start, e.lngLat);
      draft.current = { verts: [] };
      if (Math.abs(poly[1][0] - poly[0][0]) > 1e-6 && Math.abs(poly[2][1] - poly[0][1]) > 1e-6)
        handlers.current.onSelection(poly);
    });
    m.on("dblclick", (e) => {
      if (handlers.current.drawMode !== "polygon") return;
      e.preventDefault();
      const verts = draft.current.verts;
      draft.current = { verts: [] };
      if (verts.length >= 3) handlers.current.onSelection(verts);
    });

    m.on("click", (e) => {
      const h = handlers.current;
      if (h.drawMode === "polygon") {
        const v = draft.current.verts;
        const last = v[v.length - 1];
        if (!last || last[0] !== e.lngLat.lng || last[1] !== e.lngLat.lat) v.push([e.lngLat.lng, e.lngLat.lat]);
        setSelectionData(v);
        return;
      }
      if (h.drawMode === "box") return;
      const plume = m.queryRenderedFeatures(e.point, { layers: ["plumes-fill"] })[0];
      if (plume) h.onPlume(Number(plume.properties.plume_id));
      // Inside the active scene, a click is a pixel query even where other footprints overlap;
      // outside it, clicking another scene's footprint opens that scene.
      const under = m.queryRenderedFeatures(e.point, { layers: ["footprints-fill"] });
      const inActive = under.some((f) => f.properties.key === h.activeKey);
      if (!inActive && under.length) {
        h.onScene(String(under[0].properties.key));
        return;
      }
      h.onPoint(e.lngLat.lat, e.lngLat.lng);
    });
    m.on("data", (e) => {
      const id = (e as { sourceId?: string }).sourceId;
      if (id?.startsWith("src-") && (e as { tile?: unknown }).tile) handlers.current.onTileLoaded(id.slice(4));
    });
    for (const id of ["plumes-fill", "footprints-fill"]) {
      m.on("mouseenter", id, () => {
        if (!handlers.current.drawMode) m.getCanvas().style.cursor = "pointer";
      });
      m.on("mouseleave", id, () => {
        if (!handlers.current.drawMode) m.getCanvas().style.cursor = "";
      });
    }
    // The container's size changes with the layout (sidebar, narrow screens), not just the window
    const resize = new ResizeObserver(() => {
      m.resize();
      cmp.current?.resize();
    });
    resize.observe(el.current!);
    map.current = m;
    return () => {
      resize.disconnect();
      m.remove();
    };
  }, []);

  // While drawing, the map does not pan on drag or zoom on double-click
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    draft.current = { verts: [] };
    if (p.drawMode === "box") m.dragPan.disable();
    else m.dragPan.enable();
    if (p.drawMode === "polygon") m.doubleClickZoom.disable();
    else m.doubleClickZoom.enable();
    m.getCanvas().style.cursor = p.drawMode ? "crosshair" : "";
  }, [p.drawMode]);

  useEffect(() => {
    const m = map.current;
    const set = () => ready.current && setSelectionData(p.selection);
    set();
    m?.on("specret:sync", set);
    return () => {
      m?.off("specret:sync", set);
    };
  }, [p.selection]);

  // Raster layers: add, remove, reorder, restyle to match the layer list
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const sync = () => ready.current && syncRasters(m, p.layers, "rasters-top");
    sync();
    m.on("specret:sync", sync);
    return () => {
      m.off("specret:sync", sync);
    };
  }, [p.layers]);

  useEffect(() => {
    const m = map.current;
    const set = () => {
      if (!m || !ready.current) return;
      (m.getSource("footprints") as GeoJSONSource).setData(footprints(p.scenes, p.activeKey));
      (m.getSource("plumes") as GeoJSONSource).setData(p.showPlumes && p.plumes ? p.plumes : EMPTY);
      m.setFilter("plume-selected", ["==", ["get", "plume_id"], p.selectedPlume ?? -1]);
      m.setLayoutProperty("osm", "visibility", p.basemap ? "visible" : "none");
      (m.getSource("wind") as GeoJSONSource).setData(p.wind ? windArrow(p.wind) : EMPTY);
    };
    set();
    m?.on("specret:sync", set);
    return () => {
      m?.off("specret:sync", set);
    };
  }, [p.scenes, p.activeKey, p.plumes, p.showPlumes, p.selectedPlume, p.basemap, p.wind]);

  useEffect(() => {
    const m = map.current;
    if (!m || !p.fitTo) return;
    const [w, s, e, n] = p.fitTo.bounds;
    m.fitBounds([[w, s], [e, n]], { padding: 30, duration: 600, maxZoom: 15 });
  }, [p.fitTo]);

  useEffect(() => {
    const m = map.current;
    if (!m) return;
    marker.current?.remove();
    marker.current = p.point ? new maplibregl.Marker({ color: "#ffd166" }).setLngLat([p.point.lon, p.point.lat]).addTo(m) : null;
  }, [p.point]);

  // Swipe comparison: a second map, kept in step with the first, showing one layer on the
  // right of the divider
  const compareId = p.compare?.id ?? null;
  useEffect(() => {
    const m = map.current;
    if (!m || !compareId || !cmpEl.current) return;
    const c = new maplibregl.Map({
      container: cmpEl.current,
      style: {
        version: 8,
        sources: {},
        layers: [
          { id: "background", type: "background", paint: { "background-color": "#20242a" } },
          { id: "top", type: "background", paint: { "background-opacity": 0 } },
        ],
      },
      center: m.getCenter(),
      zoom: m.getZoom(),
      interactive: false,
      attributionControl: false,
    });
    const follow = () => c.jumpTo({ center: m.getCenter(), zoom: m.getZoom(), bearing: m.getBearing(), pitch: m.getPitch() });
    m.on("move", follow);
    c.on("load", () => c.fire("specret:sync"));
    cmp.current = c;
    return () => {
      m.off("move", follow);
      c.remove();
      cmp.current = null;
    };
  }, [compareId]);

  useEffect(() => {
    const c = cmp.current;
    if (!c || !p.compare) return;
    const layer = { ...p.compare, visible: true, opacity: 1 };
    const sync = () => c.isStyleLoaded() && syncRasters(c, [layer], "top");
    sync();
    c.on("specret:sync", sync);
    return () => {
      c.off("specret:sync", sync);
    };
  }, [p.compare]);

  return (
    <div className="map-wrap">
      <div ref={el} className="map" role="region" aria-label="Map" />
      {p.compare && (
        <>
          <div ref={cmpEl} className="map compare" style={{ clipPath: `inset(0 0 0 ${swipe}%)` }} aria-hidden />
          <div
            className="swipe"
            style={{ left: `${swipe}%` }}
            role="slider"
            aria-label="Comparison divider"
            aria-valuenow={Math.round(swipe)}
            onPointerDown={(e) => {
              const box = (e.currentTarget.parentElement as HTMLElement).getBoundingClientRect();
              const move = (ev: PointerEvent) =>
                setSwipe(Math.min(98, Math.max(2, ((ev.clientX - box.left) / box.width) * 100)));
              const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
              };
              window.addEventListener("pointermove", move);
              window.addEventListener("pointerup", up);
            }}
          >
            <span className="swipe-label left">Map layers</span>
            <span className="swipe-label right">{p.compare.title}</span>
          </div>
        </>
      )}
    </div>
  );
}
