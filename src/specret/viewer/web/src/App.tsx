import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";

import { api } from "./api/client";
import type { Identification, PlumeRow } from "./api/types";
import { parseLatLon } from "./lib";
import { LayerPanel } from "./map/LayerPanel";
import { mergeLayers, moveLayer, type LayerState } from "./map/layerState";
import { MapView, type Bounds } from "./map/MapView";
import { PixelPanel } from "./panels/PixelPanel";
import { PlumesPanel } from "./panels/PlumesPanel";

type Tab = "layers" | "pixel" | "plumes";

export function App() {
  const qc = useQueryClient();
  const scenes = useQuery({ queryKey: ["geoScenes"], queryFn: api.geoScenes });
  const [key, setKey] = useState<string | null>(null);
  const [layers, setLayers] = useState<LayerState[]>([]);
  const [tab, setTab] = useState<Tab>("layers");
  const [point, setPoint] = useState<{ lat: number; lon: number } | null>(null);
  const [result, setResult] = useState<Identification | null>(null);
  const [name, setName] = useState("result");
  const [status, setStatus] = useState("No pixel selected.");
  const [plume, setPlume] = useState<number | null>(null);
  const [showPlumes, setShowPlumes] = useState(true);
  const [basemap, setBasemap] = useState(false);
  const [fitTo, setFitTo] = useState<{ bounds: Bounds; seq: number } | null>(null);
  const [latlon, setLatlon] = useState("");

  const list = scenes.data ?? [];
  const scene = list.find((s) => s.key === key) ?? null;

  const layerInfo = useQuery({ queryKey: ["layers", key], queryFn: () => api.layers(key!), enabled: !!key });
  const plumes = useQuery({
    queryKey: ["plumesGeojson", key],
    queryFn: () => api.plumesGeojson(key!),
    enabled: !!scene?.analyzed,
  });
  const values = useQuery({
    queryKey: ["values", key, point?.lat, point?.lon],
    queryFn: () => api.values(key!, point!.lat, point!.lon),
    enabled: !!key && !!point,
  });

  // Open the first scene with a product, and fly to each scene when it is opened
  useEffect(() => {
    if (!key && list.length) setKey((list.find((s) => s.mapped || s.analyzed) ?? list[0]).key);
  }, [list, key]);
  useEffect(() => {
    if (scene?.bounds) setFitTo({ bounds: scene.bounds as Bounds, seq: Date.now() });
    setPoint(null);
    setPlume(null);
    setResult(null);
    setStatus("No pixel selected.");
  }, [scene?.key]);
  useEffect(() => {
    if (layerInfo.data) setLayers((current) => mergeLayers(layerInfo.data, current));
  }, [layerInfo.data]);

  const identify = async (lat: number, lon: number) => {
    setPoint({ lat, lon });
    setTab((t) => (t === "plumes" ? t : "pixel"));
    if (!scene?.mineral_scene || !scene.mapped) {
      setResult(null);
      setStatus(scene?.mineral_scene ? "Minerals not mapped for this scene." : "No reflectance for this scene.");
      return;
    }
    setStatus("Identifying…");
    try {
      const d = await api.pixel(scene.mineral_scene, { lat, lon });
      const px = d.pixel!;
      setStatus(`Pixel row ${px.row}, col ${px.col}`);
      setName(`${scene.key}_r${px.row}_c${px.col}`);
      setResult(d);
    } catch (e) {
      setResult(null);
      setStatus((e as Error).message);
    }
  };

  const onField = async (file: File) => {
    setStatus(`Identifying ${file.name}…`);
    try {
      const results = await api.field(file);
      const more = results.length > 1 ? ` (+${results.length - 1} more in file)` : "";
      setStatus(`Field spectrum: ${results[0].name}${more}`);
      setName(results[0].name || file.name);
      setResult(results[0]);
      setTab("pixel");
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  const refresh = useCallback(() => {
    qc.invalidateQueries({ queryKey: ["geoScenes"] });
    qc.invalidateQueries({ queryKey: ["layers", key] });
    qc.invalidateQueries({ queryKey: ["plumesGeojson", key] });
  }, [qc, key]);

  const selectPlume = (p: PlumeRow) => {
    setPlume(p.plume_id);
    if (p.source_lat != null && p.source_lon != null) {
      const d = 0.01;
      setFitTo({ bounds: [p.source_lon - d, p.source_lat - d, p.source_lon + d, p.source_lat + d], seq: Date.now() });
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Spectral Match Viewer</h1>
        <select aria-label="Scene" value={key ?? ""} onChange={(e) => setKey(e.target.value)}>
          {list.map((s) => (
            <option key={s.key} value={s.key}>
              {s.sensor} {s.key}
              {s.mapped ? " · minerals" : ""}
              {s.analyzed ? " · gas" : ""}
            </option>
          ))}
        </select>
        <form
          className="goto"
          onSubmit={(e) => {
            e.preventDefault();
            const ll = parseLatLon(latlon);
            if (!ll) return setStatus('Enter a location as "lat, lon", e.g. 37.53, -117.22');
            const d = 0.02;
            setFitTo({ bounds: [ll.lon - d, ll.lat - d, ll.lon + d, ll.lat + d], seq: Date.now() });
            identify(ll.lat, ll.lon);
          }}
        >
          <input
            value={latlon}
            onChange={(e) => setLatlon(e.target.value)}
            placeholder="lat, lon  e.g. 37.53, -117.22"
            aria-label="Latitude and longitude"
          />
          <button type="submit">Go</button>
        </form>
        <label className="hint basemap" title="Fetches tiles from tile.openstreetmap.org">
          <input type="checkbox" checked={basemap} onChange={(e) => setBasemap(e.target.checked)} />
          Online basemap (OpenStreetMap)
        </label>
      </header>
      <main className="workspace">
        <MapView
          scenes={list}
          activeKey={key}
          layers={layers}
          plumes={plumes.data ?? null}
          showPlumes={showPlumes}
          selectedPlume={plume}
          point={point}
          basemap={basemap}
          fitTo={fitTo}
          onPoint={identify}
          onPlume={(id) => {
            setPlume(id);
            setTab("plumes");
          }}
          onScene={setKey}
          onTileLoaded={(id) => {
            // A layer built on first view now exists: refetch its metadata (legend, ready)
            if (layers.some((l) => l.id === id && !l.ready)) qc.invalidateQueries({ queryKey: ["layers", key] });
          }}
        />
        <aside className="sidebar">
          <nav className="tabs" role="tablist">
            {(["layers", "pixel", "plumes"] as Tab[]).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>
                {t === "layers" ? "Layers" : t === "pixel" ? "Pixel" : "Plumes"}
              </button>
            ))}
          </nav>
          <div className="tab-body">
            {tab === "layers" && (
              <LayerPanel
                layers={layers}
                onChange={(id, patch) => setLayers((ls) => ls.map((l) => (l.id === id ? { ...l, ...patch } : l)))}
                onMove={(id, dir) => setLayers((ls) => moveLayer(ls, id, dir))}
                showPlumes={showPlumes}
                hasPlumes={!!plumes.data?.features.length}
                onShowPlumes={setShowPlumes}
              />
            )}
            {tab === "pixel" && (
              <PixelPanel
                point={point}
                values={values.data}
                layers={layers}
                mineralScene={scene?.mineral_scene ?? null}
                mapped={!!scene?.mapped}
                status={status}
                result={result}
                name={name}
                onField={onField}
                onMapped={refresh}
              />
            )}
            {tab === "plumes" && (
              <PlumesPanel
                gasScene={scene?.gas_scene ?? null}
                analyzed={!!scene?.analyzed}
                selected={plume}
                onSelect={selectPlume}
                onChanged={refresh}
              />
            )}
          </div>
        </aside>
      </main>
    </div>
  );
}
