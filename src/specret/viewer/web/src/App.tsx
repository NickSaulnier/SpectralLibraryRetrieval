import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "./api/client";
import type { Identification } from "./api/types";
import type { Point } from "./components/ZoomImage";
import { parseLatLon } from "./lib";
import { GasPanel } from "./panels/GasPanel";
import { MineralPanel } from "./panels/MineralPanel";
import { ResultPanel } from "./panels/ResultPanel";

export function App() {
  const scenes = useQuery({ queryKey: ["scenes"], queryFn: api.scenes });
  const [sceneId, setSceneId] = useState<string | null>(null);
  const [result, setResult] = useState<Identification | null>(null);
  const [name, setName] = useState("result");
  const [status, setStatus] = useState("No pixel selected.");
  const [marker, setMarker] = useState<Point | null>(null);
  const [latlon, setLatlon] = useState("");

  const list = scenes.data ?? [];
  const scene = list.find((s) => s.scene_id === sceneId) ?? null;
  useEffect(() => {
    if (!sceneId && list.length) setSceneId((list.find((s) => s.mapped) ?? list[0]).scene_id);
  }, [list, sceneId]);

  const identify = async (q: { x?: number; y?: number; lat?: number; lon?: number }) => {
    if (!scene) return;
    setStatus("Identifying…");
    try {
      const d = await api.pixel(scene.scene_id, q);
      const px = d.pixel!;
      setMarker({ x: px.x, y: px.y });
      const where = px.lat == null ? "" : ` · ${px.lat.toFixed(4)}°, ${px.lon!.toFixed(4)}°`;
      setStatus(`Pixel row ${px.row}, col ${px.col}${where}`);
      setName(`${scene.scene_id}_r${px.row}_c${px.col}`);
      setResult(d);
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  const onField = async (file: File) => {
    setStatus(`Identifying ${file.name}…`);
    try {
      const results = await api.field(file);
      setMarker(null);
      const more = results.length > 1 ? ` (+${results.length - 1} more in file)` : "";
      setStatus(`Field spectrum: ${results[0].name}${more}`);
      setName(results[0].name || file.name);
      setResult(results[0]);
    } catch (e) {
      setStatus((e as Error).message);
    }
  };

  return (
    <>
      <header>
        <h1>Spectral Match Viewer</h1>
        {list.length > 0 && (
          <select
            aria-label="Scene"
            value={sceneId ?? ""}
            onChange={(e) => {
              setSceneId(e.target.value);
              setMarker(null);
            }}
          >
            {list.map((s) => (
              <option key={s.scene_id} value={s.scene_id}>
                {s.scene_id}
                {s.mapped ? "" : " (not mapped)"}
              </option>
            ))}
          </select>
        )}
        <form
          className="goto"
          onSubmit={(e) => {
            e.preventDefault();
            if (!scene) return setStatus("No scene to look in.");
            const ll = parseLatLon(latlon);
            if (!ll) return setStatus('Enter a location as "lat, lon", e.g. 37.53, -117.22');
            identify(ll);
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
        <span className="hint">Click a pixel on the map, enter a location, or drop a field spectrum below.</span>
      </header>
      <main className="grid">
        <MineralPanel scene={scene} marker={marker} onPick={(p) => identify(p)} onField={onField} />
        <ResultPanel status={status} result={result} name={name} />
      </main>
      <GasPanel />
    </>
  );
}
