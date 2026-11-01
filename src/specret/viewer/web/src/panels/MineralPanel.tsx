import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { SceneInfo } from "../api/types";
import { ZoomImage, type Point } from "../components/ZoomImage";
import { useJob } from "../hooks/useJob";
import { pretty } from "../lib";
import { FieldDrop } from "./FieldDrop";

interface Props {
  scene: SceneInfo | null;
  marker: Point | null;
  onPick: (p: Point) => void;
  onField: (file: File) => void;
}

/** The best-match mineral group map of a scene, or a button to map it. */
export function MineralPanel({ scene, marker, onPick, onField }: Props) {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const job = useJob(jobId ?? scene?.job ?? null);

  useEffect(() => {
    setJobId(null);
    setError(null);
  }, [scene?.scene_id]);
  useEffect(() => {
    if (job?.state === "done") qc.invalidateQueries({ queryKey: ["scenes"] });
  }, [job?.state, qc]);

  const legend = useQuery({
    queryKey: ["legend", scene?.scene_id],
    queryFn: () => api.legend(scene!.scene_id),
    enabled: !!scene?.mapped,
  });

  const start = async () => {
    try {
      setJobId((await api.startMapping(scene!.scene_id)).job_id);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <section className="panel">
      <h2>Best-match mineral group</h2>
      {scene?.mapped ? (
        <>
          <ZoomImage src={api.mapUrl(scene.scene_id)} alt="Mineral group map" marker={marker} onPick={onPick} />
          <div className="legend">
            {(legend.data ?? [])
              .filter((l) => l.fraction >= 0.002)
              .map((l) => (
                <span key={l.code}>
                  <i style={{ background: `rgb(${l.rgb.join(",")})` }} />
                  {pretty(l.group)} {(100 * l.fraction).toFixed(1)}%
                </span>
              ))}
          </div>
        </>
      ) : scene ? (
        <div className="empty">
          <p>
            This {scene.kind.toUpperCase()} scene is not mapped yet. Mapping takes about 2 minutes for EMIT. You can
            still look up a location.
          </p>
          {job?.state === "running" ? (
            <div>
              <progress max={1} value={job.progress} />
              <div className="hint">
                Mapping… {Math.round(100 * job.progress)}% · {job.elapsed_s} s
              </div>
            </div>
          ) : (
            <button className="primary" onClick={start}>
              {job?.state === "failed" ? "Try again" : "Map this scene"}
            </button>
          )}
          {(error || job?.state === "failed") && (
            <div className="hint">Mapping failed: {error ?? job?.error?.split("\n")[0]}</div>
          )}
        </div>
      ) : (
        <p className="hint">No scenes in data/scenes. Drop a field spectrum below, or download a scene first.</p>
      )}
      <FieldDrop onFile={onField} />
    </section>
  );
}
