import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import type { Job } from "../api/types";

/** Queued or running: not finished yet. */
export const isActive = (j?: Job | null): boolean => j?.state === "queued" || j?.state === "running";

/** Poll a background job until it finishes; undefined while there is no job. */
export function useJob(jobId: string | null, intervalMs = 1500): Job | undefined {
  const { data } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (q) => (q.state.data && !isActive(q.state.data) ? false : intervalMs),
  });
  return data;
}
