import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import type { Job } from "../api/types";

/** Poll a background job until it finishes; null while there is no job. */
export function useJob(jobId: string | null, intervalMs = 1500): Job | undefined {
  const { data } = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (q) => (q.state.data && q.state.data.state !== "running" ? false : intervalMs),
  });
  return data;
}
