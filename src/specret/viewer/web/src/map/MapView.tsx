import maplibregl, { type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";

import { api, type GeoScene } from "../api/client";
import type { LayerState } from "./layerState";

const OSM = {
  tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
  attribution: "© OpenStreetMap contributors",
};
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export type Bounds = [number, number, number, number];

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
  onPoint: (lat: number, lon: number) => void;
  onPlume: (plumeId: number) => void;
  onScene: (key: string) => void;
  onTileLoaded: (layerId: string) => void;
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

/** The MapLibre map: raster layers from the tile API, plume outlines, scene footprints. */
export function MapView(p: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const ready = useRef(false);
  const marker = useRef<maplibregl.Marker | null>(null);
  const handlers = useRef(p);
  handlers.current = p;

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
        ],
      },
      center: [0, 20],
      zoom: 1.5,
      attributionControl: { compact: true },
    });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: false }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    m.on("load", () => {
      ready.current = true;
      m.fire("specret:sync");
    });
    m.on("click", (e) => {
      const plume = m.queryRenderedFeatures(e.point, { layers: ["plumes-fill"] })[0];
      if (plume) handlers.current.onPlume(Number(plume.properties.plume_id));
      // Inside the active scene, a click is a pixel query even where other footprints overlap;
      // outside it, clicking another scene's footprint opens that scene.
      const under = m.queryRenderedFeatures(e.point, { layers: ["footprints-fill"] });
      const inActive = under.some((f) => f.properties.key === handlers.current.activeKey);
      if (!inActive && under.length) {
        handlers.current.onScene(String(under[0].properties.key));
        return;
      }
      handlers.current.onPoint(e.lngLat.lat, e.lngLat.lng);
    });
    m.on("data", (e) => {
      const id = (e as { sourceId?: string }).sourceId;
      if (id?.startsWith("src-") && (e as { tile?: unknown }).tile) handlers.current.onTileLoaded(id.slice(4));
    });
    for (const id of ["plumes-fill", "footprints-fill"]) {
      m.on("mouseenter", id, () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", id, () => (m.getCanvas().style.cursor = ""));
    }
    // The container's size changes with the layout (sidebar, narrow screens), not just the window
    const resize = new ResizeObserver(() => m.resize());
    resize.observe(el.current!);
    map.current = m;
    return () => {
      resize.disconnect();
      m.remove();
    };
  }, []);

  // Raster layers: add, remove, reorder, restyle to match the layer list
  useEffect(() => {
    const m = map.current;
    if (!m) return;
    const sync = () => {
      if (!ready.current) return;
      const wanted = new Set(p.layers.map((l) => `lyr-${l.id}`));
      for (const layer of m.getStyle().layers) {
        if (layer.id.startsWith("lyr-") && !wanted.has(layer.id)) {
          m.removeLayer(layer.id);
          m.removeSource(layer.id.replace("lyr-", "src-"));
        }
      }
      for (const l of p.layers) {
        const src = `src-${l.id}`;
        const lyr = `lyr-${l.id}`;
        const tiles = [api.tileUrl(l.id, l.stretch, l.cmap)];
        const s = m.getSource(src) as maplibregl.RasterTileSource | undefined;
        if (!s) {
          m.addSource(src, { type: "raster", tiles, tileSize: 256, maxzoom: 18 });
          m.addLayer(
            { id: lyr, type: "raster", source: src, paint: { "raster-resampling": "nearest" } },
            "rasters-top",
          );
        } else if (JSON.stringify(s.tiles) !== JSON.stringify(tiles)) {
          s.setTiles(tiles);
        }
        m.setLayoutProperty(lyr, "visibility", l.visible ? "visible" : "none");
        m.setPaintProperty(lyr, "raster-opacity", l.opacity);
        m.moveLayer(lyr, "rasters-top"); // in list order: later layers end up on top
      }
    };
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
    };
    set();
    m?.on("specret:sync", set);
    return () => {
      m?.off("specret:sync", set);
    };
  }, [p.scenes, p.activeKey, p.plumes, p.showPlumes, p.selectedPlume, p.basemap]);

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

  return <div ref={el} className="map" role="region" aria-label="Map" />;
}
