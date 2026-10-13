import { useEffect, useState } from "react";
import { cancelJob, listJobs, onJob } from "./commands";
import type { JobInfo } from "./commands";

/** Jobs that are queued or running, along the bottom of every view. */
export default function JobStrip({ workspace }: { workspace: string | null }) {
  const [jobs, setJobs] = useState<Map<string, JobInfo>>(new Map());

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    listJobs()
      .then((all) => setJobs(new Map(all.map((j) => [j.id, j]))))
      .catch(() => {});
    onJob((job) => {
      setJobs((before) => new Map(before).set(job.id, job));
    }).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, []);

  const active = [...jobs.values()].filter(
    (j) =>
      (j.state === "queued" || j.state === "running") &&
      (workspace === null || j.workspace_id === workspace),
  );
  if (active.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2 border-t border-gray-200 bg-gray-50 px-3 py-1 text-xs dark:border-gray-800 dark:bg-gray-900">
      {active.map((j) => (
        <span
          key={j.id}
          className="flex items-center gap-2 rounded bg-white px-2 py-1 dark:bg-gray-800"
        >
          <span className="font-medium">{j.kind}</span>
          <span className="max-w-64 truncate" title={j.label}>
            {j.label}
          </span>
          {j.progress ? (
            <span className="text-gray-500">
              {j.progress.done}/{j.progress.total}
            </span>
          ) : (
            <span className="text-gray-500">{j.state}</span>
          )}
          {j.status && <span className="text-gray-500">{j.status}</span>}
          <button
            type="button"
            className="text-red-600 hover:underline"
            disabled={j.cancel_requested}
            onClick={() => cancelJob(j.id).catch(() => {})}
          >
            Cancel
          </button>
        </span>
      ))}
    </div>
  );
}
