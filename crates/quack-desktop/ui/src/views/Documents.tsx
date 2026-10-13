import { useCallback, useEffect, useState } from "react";
import { confirm, open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { addPaths, deleteDocument, listDocuments, messageOf, onJob } from "../commands";
import type { DocumentInfo, WorkspaceRow } from "../commands";
import type { MenuBus } from "../menu";

interface Props {
  workspace: WorkspaceRow | null;
  menu: MenuBus;
}

/** One page of the listing, files and folders added by path. */
export default function Documents({ workspace, menu }: Props) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [next, setNext] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [all, setAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const workspaceId = workspace?.id ?? null;

  const load = useCallback(
    (after: string | null) => {
      if (!workspaceId) return;
      listDocuments({ workspace: workspaceId, after, all })
        .then((page) => {
          setDocs((before) => (after ? [...before, ...page.documents] : page.documents));
          setNext(page.next ?? null);
          setTotal(page.total);
          setError(null);
        })
        .catch((e) => setError(messageOf(e)));
    },
    [workspaceId, all],
  );

  useEffect(() => load(null), [load]);

  // A finished ingest changes the listing.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    onJob((job) => {
      if (job.kind === "ingest" && job.workspace_id === workspaceId && job.finished_at) load(null);
    }).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, [workspaceId, load]);

  const add = useCallback(
    async (paths: string[]) => {
      if (!workspaceId || paths.length === 0) return;
      try {
        await addPaths(workspaceId, paths);
        setError(null);
      } catch (e) {
        setError(messageOf(e));
      }
    },
    [workspaceId],
  );

  const pick = useCallback(
    async (directory: boolean) => {
      const chosen = await open({ multiple: true, directory });
      if (!chosen) return;
      await add(Array.isArray(chosen) ? chosen : [chosen]);
    },
    [add],
  );

  useEffect(
    () =>
      menu.subscribe((id) => {
        if (id === "file:add-files") void pick(false);
        if (id === "file:add-folder") void pick(true);
      }),
    [menu, pick],
  );

  // An OS drop brings paths, never bytes.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        if (event.payload.type === "over") setDragging(true);
        else if (event.payload.type === "drop") {
          setDragging(false);
          void add(event.payload.paths);
        } else setDragging(false);
      })
      .then((fn) => {
        unlisten = fn;
      });
    return () => unlisten?.();
  }, [add]);

  const remove = async (doc: DocumentInfo) => {
    if (!workspaceId) return;
    const sure = await confirm(`Delete "${doc.filename}" and what was built from it?`, {
      title: "Delete document",
      kind: "warning",
      okLabel: "Delete",
      cancelLabel: "Keep",
    });
    if (!sure) return;
    try {
      await deleteDocument(workspaceId, doc.id);
      load(null);
    } catch (e) {
      setError(messageOf(e));
    }
  };

  if (!workspace) {
    return <p className="p-6 text-sm text-gray-500">Open a workspace first.</p>;
  }
  return (
    <div className={`min-h-full p-6 ${dragging ? "bg-blue-50 dark:bg-blue-950" : ""}`}>
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-lg font-semibold">Documents</h1>
        <span className="text-sm text-gray-500">{total}</span>
        <button
          type="button"
          className="rounded border px-3 py-1 text-sm"
          onClick={() => void pick(false)}
        >
          Add files…
        </button>
        <button
          type="button"
          className="rounded border px-3 py-1 text-sm"
          onClick={() => void pick(true)}
        >
          Add folder…
        </button>
        <label className="ml-auto flex items-center gap-1 text-sm text-gray-500">
          <input type="checkbox" checked={all} onChange={(e) => setAll(e.target.checked)} />
          show replaced
        </label>
      </div>
      {error && <p className="mb-3 text-sm text-red-600">{error}</p>}
      <table className="w-full text-sm">
        <thead className="text-left text-gray-500">
          <tr>
            <th className="py-1">File</th>
            <th className="py-1">Status</th>
            <th className="py-1">Chunks</th>
            <th className="py-1">Source</th>
            <th className="py-1" />
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.id} className="border-t border-gray-100 dark:border-gray-800">
              <td className="max-w-md truncate py-2" title={d.title ?? d.filename}>
                {d.filename}
                {d.pinned && <span className="ml-2 text-xs text-gray-500">pinned</span>}
              </td>
              <td className="py-2 text-gray-500">
                {d.status}
                {d.error_message && (
                  <span className="ml-1 text-red-600" title={d.error_message}>
                    !
                  </span>
                )}
              </td>
              <td className="py-2 text-gray-500">{d.chunk_count ?? ""}</td>
              <td className="py-2 text-gray-500">{d.source}</td>
              <td className="py-2 text-right">
                <button
                  type="button"
                  className="text-red-600 hover:underline"
                  onClick={() => void remove(d)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {docs.length === 0 && !error && (
        <p className="text-sm text-gray-500">
          No documents yet. Drop files here, or add them above.
        </p>
      )}
      {next && (
        <button type="button" className="mt-3 text-sm hover:underline" onClick={() => load(next)}>
          More
        </button>
      )}
    </div>
  );
}
