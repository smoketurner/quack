// The app's API: one wrapper per `#[tauri::command]`, typed with the
// schema `quack-desktop --schema` writes (`make ui-types`), so every type
// here is the engine's own.
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { components } from "./schema";

type S = components["schemas"];
export type WorkspaceRow = S["WorkspaceRow"];
export type WorkspaceSummary = S["WorkspaceSummary"];
export type SessionRow = S["SessionRow"];
export type MessageRow = S["MessageRow"];
export type ChatMode = S["ChatMode"];
export type TurnEvent = S["TurnEvent"];
export type AgentResponseBody = S["AgentResponseBody"];
export type ToolStep = S["ToolStep"];
export type DocumentPage = S["DocumentPage"];
export type DocumentInfo = S["DocumentInfo"];
export type DocumentFilter = S["DocumentFilter"];
export type JobInfo = S["JobInfo"];
export type JobState = S["JobState"];
export type SettingsView = S["SettingsView"];
export type CommandError = S["CommandError"];
export type ErrorKind = S["ErrorKind"];

/** A failed command, as `CommandError` arrives from Rust. */
export function isCommandError(e: unknown): e is CommandError {
  return typeof e === "object" && e !== null && "kind" in e && "message" in e;
}

/** The message to show for any failure a command threw. */
export function messageOf(e: unknown): string {
  if (isCommandError(e)) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

// Workspaces
export const listWorkspaces = () => invoke<WorkspaceSummary[]>("list_workspaces");
export const createWorkspace = (name: string) => invoke<WorkspaceRow>("create_workspace", { name });
export const renameWorkspace = (id: string, name: string) =>
  invoke<WorkspaceRow>("rename_workspace", { id, name });
export const deleteWorkspace = (id: string) => invoke<void>("delete_workspace", { id });
export const openWorkspace = (id: string) => invoke<WorkspaceRow>("open_workspace", { id });
export const closeWorkspace = (id: string) => invoke<boolean>("close_workspace", { id });

// Chat
export const listSessions = (workspace: string, limit?: number) =>
  invoke<SessionRow[]>("list_sessions", { workspace, limit });
export const newSession = (workspace: string, mode?: ChatMode) =>
  invoke<SessionRow>("new_session", { workspace, mode });
export const sessionMessages = (workspace: string, session: string) =>
  invoke<MessageRow[]>("session_messages", { workspace, session });
export const setSessionMode = (workspace: string, session: string, mode: ChatMode) =>
  invoke<void>("set_session_mode", { workspace, session, mode });
export const setAllowWrites = (workspace: string, allow: boolean) =>
  invoke<void>("set_allow_writes", { workspace, allow });
export const cancelTurn = (workspace: string, session: string) =>
  invoke<boolean>("cancel_turn", { workspace, session });

/** Ask `message`; every event of the turn reaches `onEvent` in order. */
export function startTurn(
  args: { workspace: string; session: string; message: string; documents: string[] },
  onEvent: (event: TurnEvent) => void,
): Promise<JobInfo> {
  const channel = new Channel<TurnEvent>();
  channel.onmessage = onEvent;
  return invoke<JobInfo>("start_turn", { ...args, onEvent: channel });
}

// Documents
export const listDocuments = (args: {
  workspace: string;
  after?: string | null;
  all?: boolean;
  filter?: DocumentFilter | null;
  limit?: number;
}) =>
  invoke<DocumentPage>("list_documents", {
    workspace: args.workspace,
    after: args.after ?? null,
    all: args.all ?? false,
    filter: args.filter ?? null,
    limit: args.limit,
  });
export const addPaths = (workspace: string, paths: string[]) =>
  invoke<JobInfo[]>("add_paths", { workspace, paths });
export const deleteDocument = (workspace: string, document: string) =>
  invoke<boolean>("delete_document", { workspace, document });

// Jobs
export const listJobs = (workspace?: string) =>
  invoke<JobInfo[]>("list_jobs", { workspace: workspace ?? null });
export const cancelJob = (job: string) => invoke<boolean>("cancel_job", { job });

/** Every job snapshot the queue broadcasts (the Tauri event `job`). */
export const onJob = (handler: (job: JobInfo) => void): Promise<UnlistenFn> =>
  listen<JobInfo>("job", (event) => handler(event.payload));

/** A native menu item chosen, by id (`view:chat`, `file:add-files`, ...). */
export const onMenu = (handler: (id: string) => void): Promise<UnlistenFn> =>
  listen<string>("menu", (event) => handler(event.payload));

// Settings
export const settings = () => invoke<SettingsView>("settings");
