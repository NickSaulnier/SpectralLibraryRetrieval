import { useEffect, useState } from "react";

import type { LayerState } from "../map/layerState";
import type { Identification } from "../api/types";
import { api } from "../api/client";
import { JobProgress } from "../components/JobProgress";
import { isActive, useJob } from "../hooks/useJob";
import { fmt, pretty } from "../lib";
import { FieldDrop } from "./FieldDrop";
import { addPin, PinnedSpectra, type Pin } from "./PixelEvidence";
import { ResultPanel } from "./ResultPanel";

interface Props {
  point: { lat: number; lon: number } | null;
  values: Record<string, number | string | null> | undefined;
  layers: LayerState[];
  mineralScene: string | null;
  mapped: boolean;
  status: string;
  result: Identification | null;
  name: string;
  onField: (file: File) => void;
  onMapped: () => void;
  pins: Pin[];
  onPins: (pins: Pin[]) => void;
}

/** What is at the clicked point: every layer's value, then the mineral identification. */
export function PixelPanel(p: Props) {
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  const { onMapped } = p;
  useEffect(() => {
    if (job?.state === "done") {
      setJobId(null);
      onMapped();
    }
  }, [job?.state, onMapped]);
  const title = (id: string) => p.layers.find((l) => l.id === id);
  const rows = Object.entries(p.values ?? {}).filter(([, v]) => v !== null);
  return (
    <div>
      {p.point ? (
        <p className="hint">
          {p.point.lat.toFixed(5)}°, {p.point.lon.toFixed(5)}°
        </p>
      ) : (
        <p className="hint">Click the map, or enter a location above.</p>
      )}
      {rows.length > 0 && (
        <table className="values">
          <tbody>
            {rows.map(([id, v]) => {
              const l = title(id);
              return (
                <tr key={id}>
                  <td>{l?.title ?? id}</td>
                  <td className="num">{typeof v === "number" ? `${fmt(v, Math.abs(v) < 10 ? 3 : 0)} ${l?.units ?? ""}` : pretty(v)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {p.mineralScene && !p.mapped && (
        <div className="empty">
          <p>This scene's minerals are not mapped yet (about 2 minutes for EMIT).</p>
          {job && isActive(job) ? (
            <JobProgress job={job} label="Mapping" />
          ) : (
            <button
              className="primary"
              onClick={async () => setJobId((await api.startMapping(p.mineralScene!)).job_id)}
            >
              Map minerals
            </button>
          )}
        </div>
      )}
      <ResultPanel
        status={p.status}
        result={p.result}
        name={p.name}
        onPin={p.result ? () => p.onPins(addPin(p.pins, { name: p.name, result: p.result! })) : undefined}
      />
      <PinnedSpectra
        pins={p.pins}
        onRemove={(n) => p.onPins(p.pins.filter((x) => x.name !== n))}
        onClear={() => p.onPins([])}
      />
      <FieldDrop onFile={p.onField} />
    </div>
  );
}
