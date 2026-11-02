import { useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { Job } from "../api/types";

/** A job's place in the queue or its progress, elapsed time, and a cancel button. */
export function JobProgress({ job, label }: { job: Job; label: string }) {
  const qc = useQueryClient();
  const cancel = async () => {
    await api.cancelJob(job.job_id);
    qc.invalidateQueries({ queryKey: ["job", job.job_id] });
    qc.invalidateQueries({ queryKey: ["jobs"] });
  };
  return (
    <div className="job-progress">
      {job.state === "queued" ? (
        <div className="hint">
          {label}: queued{job.position ? ` (${job.position} in line)` : ""}. Jobs run one at a time.
        </div>
      ) : (
        <>
          {job.progress == null ? <progress /> : <progress max={1} value={job.progress} />}
          <div className="hint">
            {label}… {job.progress != null && `${Math.round(100 * job.progress)}% · `}
            {job.elapsed_s} s{job.cancelling && " · stopping"}
          </div>
        </>
      )}
      <button type="button" className="link" onClick={cancel} disabled={job.cancelling}>
        Cancel
      </button>
    </div>
  );
}
