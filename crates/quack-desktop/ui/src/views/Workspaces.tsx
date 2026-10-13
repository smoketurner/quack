import { useCallback, useEffect, useState } from "react";
import { confirm } from "@tauri-apps/plugin-dialog";
import {
  closeWorkspace,
  createWorkspace,
  deleteWorkspace,
  listWorkspaces,
  messageOf,
  openWorkspace,
  renameWorkspace,
} from "../commands";
import type { WorkspaceRow, WorkspaceSummary } from "../commands";
import type { MenuBus } from "../menu";

interface Props {
  current: WorkspaceRow | null;
  onOpen: (workspace: WorkspaceRow) => void;
  menu: MenuBus;
}

/** The workspaces: open one, make, rename, or delete one. */
export default function Workspaces({ current, onOpen, menu }: Props) {
  const [rows, setRows] = useState<WorkspaceSummary[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  const refresh = useCallback(() => {
    listWorkspaces()
      .then(setRows)
      .catch((e) => setError(messageOf(e)));
  }, []);

  useEffect(refresh, [refresh]);
  useEffect(
    () =>
      menu.subscribe((id) => {
        if (id === "file:new-workspace") setCreating(true);
      }),
    [menu],
  );

  const create = async () => {
    try {
      const row = await createWorkspace(name);
      setName("");
      setCreating(false);
      setError(null);
      refresh();
      onOpen(await openWorkspace(row.id));
    } catch (e) {
      setError(messageOf(e));
    }
  };

  const open = async (row: WorkspaceRow) => {
    try {
      onOpen(await openWorkspace(row.id));
      setError(null);
    } catch (e) {
      setError(messageOf(e));
    }
  };

  const rename = async (row: WorkspaceRow) => {
    const next = window.prompt(`Rename "${row.name}" to:`, row.name);
    if (!next || next === row.name) return;
    try {
      await renameWorkspace(row.id, next);
      setError(null);
      refresh();
    } catch (e) {
      setError(messageOf(e));
    }
  };

  const remove = async (row: WorkspaceRow) => {
    const sure = await confirm(
      `Delete workspace "${row.name}", its documents, tables, sessions, and graph? This cannot be undone.`,
      { title: "Delete workspace", kind: "warning", okLabel: "Delete", cancelLabel: "Keep" },
    );
    if (!sure) return;
    try {
      await deleteWorkspace(row.id);
      setError(null);
      refresh();
    } catch (e) {
      setError(messageOf(e));
    }
  };

  const close = async (row: WorkspaceRow) => {
    await closeWorkspace(row.id).catch(() => {});
    refresh();
  };

  return (
    <div className="p-6">
      <div className="mb-4 flex items-center gap-3">
        <h1 className="text-lg font-semibold">Workspaces</h1>
        <button
          type="button"
          className="rounded bg-gray-900 px-3 py-1 text-sm text-white dark:bg-gray-100 dark:text-gray-900"
          onClick={() => setCreating(true)}
        >
          New
        </button>
      </div>
      {creating && (
        <form
          className="mb-4 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void create();
          }}
        >
          <input
            autoFocus
            className="rounded border border-gray-300 px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-900"
            placeholder="Workspace name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" className="rounded border px-3 py-1 text-sm">
            Create
          </button>
          <button type="button" className="px-2 text-sm" onClick={() => setCreating(false)}>
            Cancel
          </button>
        </form>
      )}
      {error && <p className="mb-3 text-sm text-red-600">{error}</p>}
      <table className="w-full text-sm">
        <thead className="text-left text-gray-500">
          <tr>
            <th className="py-1">Name</th>
            <th className="py-1">Providers</th>
            <th className="py-1">Held</th>
            <th className="py-1" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-t border-gray-100 dark:border-gray-800">
              <td className="py-2">
                <button
                  type="button"
                  className="font-medium hover:underline"
                  onClick={() => open(r)}
                >
                  {r.name}
                </button>
                {current?.id === r.id && (
                  <span className="ml-2 text-xs text-gray-500">current</span>
                )}
              </td>
              <td className="py-2 text-gray-500">
                {r.allowed_providers.length === 0 ? "any" : r.allowed_providers.join(", ")}
              </td>
              <td className="py-2 text-gray-500">{r.open ? "open" : ""}</td>
              <td className="space-x-3 py-2 text-right">
                <button type="button" className="hover:underline" onClick={() => rename(r)}>
                  Rename
                </button>
                {r.open && (
                  <button type="button" className="hover:underline" onClick={() => close(r)}>
                    Close
                  </button>
                )}
                <button
                  type="button"
                  className="text-red-600 hover:underline"
                  onClick={() => remove(r)}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length === 0 && !error && (
        <p className="text-sm text-gray-500">No workspaces yet. Make one to start.</p>
      )}
    </div>
  );
}
