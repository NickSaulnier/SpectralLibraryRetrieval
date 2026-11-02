import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api, type GeoScene } from "../api/client";
import type { Job, SetupItem } from "../api/types";
import { JobProgress } from "../components/JobProgress";
import { isActive } from "../hooks/useJob";

const STATE_PILL: Record<Job["state"], string> = {
  queued: "medium",
  running: "medium",
  done: "high",
  failed: "low",
  cancelled: "low",
};

/** Ids of jobs that were active at the last look and have finished since. */
export function newlyFinished(before: Job[], now: Job[]): string[] {
  const was = new Set(before.filter(isActive).map((j) => j.job_id));
  return now.filter((j) => was.has(j.job_id) && !isActive(j)).map((j) => j.job_id);
}

/** Every long operation in one queue, a persistence run over chosen scenes, and the
 * `specret doctor` checklist with a Run button for each fix the viewer can do. */
export function JobsPanel({ scenes, onChanged }: { scenes: GeoScene[]; onChanged: () => void }) {
  const qc = useQueryClient();
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    refetchInterval: (q) => ((q.state.data ?? []).some(isActive) ? 1500 : 10000),
  });
  const list = jobs.data ?? [];

  // When a job finishes, what it wrote shows up in the scene list, layers and plumes
  const last = useRef<Job[]>([]);
  useEffect(() => {
    if (newlyFinished(last.current, list).length) {
      onChanged();
      qc.invalidateQueries({ queryKey: ["setup"] });
      qc.invalidateQueries({ queryKey: ["allPlumes"] });
      qc.invalidateQueries({ queryKey: ["gasPlumes"] });
    }
    last.current = list;
  }, [list, onChanged, qc]);

  const queue = async (start: Promise<{ job_id: string }>) => {
    await start;
    qc.invalidateQueries({ queryKey: ["jobs"] });
  };

  return (
    <div className="jobs">
      <h3>Queue</h3>
      {list.length === 0 && <p className="hint">No jobs yet. Mapping, detection and setup steps queue here.</p>}
      {list.map((j) => (
        <div key={j.job_id} className="job">
          <div className="card-head">
            <b>{j.title}</b>
            <span className={`pill ${STATE_PILL[j.state]}`}>{j.state}</span>
          </div>
          {isActive(j) ? (
            <JobProgress job={j} label={j.kind === "command" ? "Running" : "Working"} />
          ) : (
            <div className="hint">
              {j.elapsed_s} s{j.error && ` · ${j.error}`}
            </div>
          )}
          {j.log.length > 0 && (
            <details>
              <summary className="hint">Log</summary>
              <pre className="log">{j.log.join("\n")}</pre>
            </details>
          )}
        </div>
      ))}
      <Persistence scenes={scenes} onRun={(ids) => queue(api.startPersistence(ids))} />
      <Setup onRun={(argv) => queue(api.runSetupStep(argv))} />
    </div>
  );
}

function Persistence({ scenes, onRun }: { scenes: GeoScene[]; onRun: (ids: string[]) => void }) {
  const analyzed = scenes.filter((s) => s.sensor === "EMIT" && s.analyzed && s.gas_scene);
  const [picked, setPicked] = useState<string[] | null>(null);
  const chosen = picked ?? analyzed.map((s) => s.gas_scene!);
  if (analyzed.length < 2) return null;
  const toggle = (id: string) => setPicked(chosen.includes(id) ? chosen.filter((x) => x !== id) : [...chosen, id]);
  return (
    <>
      <h3>Repeat sightings</h3>
      <p className="hint">Count, for each plume, the other overpasses that covered its source and saw it again.</p>
      <div className="scene-picks">
        {analyzed.map((s) => (
          <label key={s.key}>
            <input type="checkbox" checked={chosen.includes(s.gas_scene!)} onChange={() => toggle(s.gas_scene!)} />
            {s.key}
          </label>
        ))}
      </div>
      <button className="primary" disabled={chosen.length < 2} onClick={() => onRun(chosen)}>
        Run persistence over {chosen.length} scenes
      </button>
    </>
  );
}

function Setup({ onRun }: { onRun: (argv: string[]) => void }) {
  const [open, setOpen] = useState(false);
  const setup = useQuery({ queryKey: ["setup"], queryFn: api.setup, enabled: open });
  return (
    <>
      <h3>
        <button type="button" className="link" onClick={() => setOpen(!open)} aria-expanded={open}>
          {open ? "▾" : "▸"} Project setup
        </button>
      </h3>
      {open && setup.isLoading && <p className="hint">Checking the project…</p>}
      {open && setup.data && <Checklist items={setup.data} onRun={onRun} />}
    </>
  );
}

export function Checklist({ items, onRun }: { items: SetupItem[]; onRun: (argv: string[]) => void }) {
  return (
    <ul className="checklist">
      {items.map((it) => (
        <li key={it.key} className={it.ok ? "ok" : it.level}>
          <span className="mark" aria-label={it.ok ? "done" : "to do"}>
            {it.ok ? "✓" : it.level === "required" ? "!" : "–"}
          </span>
          <div>
            <b>{it.name}</b> <span className="hint">{it.detail}</span>
            {!it.ok &&
              it.steps.map((s) =>
                s.runnable ? (
                  <div key={s.command} className="step">
                    <code>{s.command}</code>
                    <button type="button" onClick={() => onRun(s.argv)}>
                      Run
                    </button>
                  </div>
                ) : (
                  <div key={s.command} className="step">
                    <code>{s.command}</code> <span className="hint">needs your Earthdata login: run it in a terminal</span>
                  </div>
                ),
              )}
            {!it.ok && it.hint && <div className="hint">{it.hint}</div>}
          </div>
        </li>
      ))}
    </ul>
  );
}
