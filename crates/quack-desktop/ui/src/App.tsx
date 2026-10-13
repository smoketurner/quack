import { useCallback, useEffect, useState } from "react";
import { onMenu } from "./commands";
import { menuBus } from "./menu";
import type { WorkspaceRow } from "./commands";
import JobStrip from "./JobStrip";
import Chat from "./views/Chat";
import Documents from "./views/Documents";
import Settings from "./views/Settings";
import Workspaces from "./views/Workspaces";

export type View = "workspaces" | "chat" | "documents" | "settings";

const VIEWS: { id: View; label: string }[] = [
  { id: "workspaces", label: "Workspaces" },
  { id: "chat", label: "Chat" },
  { id: "documents", label: "Documents" },
  { id: "settings", label: "Settings" },
];

/** The window: a sidebar, the view, and the job strip along the bottom. */
export default function App() {
  const [view, setView] = useState<View>("workspaces");
  const [workspace, setWorkspace] = useState<WorkspaceRow | null>(null);
  // What the native menu asked for, handed to the view that acts on it.
  const [bus] = useState(menuBus);

  const open = useCallback((ws: WorkspaceRow) => {
    setWorkspace(ws);
    setView("chat");
  }, []);

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    onMenu((id) => {
      if (id.startsWith("view:")) {
        setView(id.slice("view:".length) as View);
      } else if (id === "settings") {
        setView("settings");
      } else {
        if (id === "file:add-files" || id === "file:add-folder") setView("documents");
        if (id === "file:new-session") setView("chat");
        if (id === "file:new-workspace") setView("workspaces");
        bus.emit(id);
      }
    }).then((fn) => {
      unlisten = fn;
    });
    return () => unlisten?.();
  }, [bus]);

  return (
    <div className="flex h-full">
      <nav className="flex w-48 shrink-0 flex-col border-r border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-900">
        <div className="px-4 py-3 text-sm font-semibold">quack</div>
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setView(v.id)}
            className={`px-4 py-2 text-left text-sm hover:bg-gray-200 dark:hover:bg-gray-800 ${
              view === v.id ? "bg-gray-200 font-medium dark:bg-gray-800" : ""
            }`}
          >
            {v.label}
          </button>
        ))}
        <div className="mt-auto truncate px-4 py-3 text-xs text-gray-500" title={workspace?.name}>
          {workspace ? workspace.name : "No workspace open"}
        </div>
      </nav>
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-auto">
          {view === "workspaces" && <Workspaces current={workspace} onOpen={open} menu={bus} />}
          {view === "chat" && <Chat key={workspace?.id ?? ""} workspace={workspace} menu={bus} />}
          {view === "documents" && (
            <Documents key={workspace?.id ?? ""} workspace={workspace} menu={bus} />
          )}
          {view === "settings" && <Settings />}
        </div>
        <JobStrip workspace={workspace?.id ?? null} />
      </main>
    </div>
  );
}
