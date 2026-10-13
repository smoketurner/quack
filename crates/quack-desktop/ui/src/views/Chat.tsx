import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelTurn,
  listSessions,
  messageOf,
  newSession,
  sessionMessages,
  setAllowWrites,
  setSessionMode,
  startTurn,
} from "../commands";
import type { ChatMode, MessageRow, SessionRow, WorkspaceRow } from "../commands";
import type { MenuBus } from "../menu";
import { empty, reduce } from "../transcript/reducer";
import type { Transcript } from "../transcript/reducer";

interface Props {
  workspace: WorkspaceRow | null;
  menu: MenuBus;
}

/** Sessions on the left, the transcript and the composer on the right. */
export default function Chat({ workspace, menu }: Props) {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [session, setSession] = useState<SessionRow | null>(null);
  const [history, setHistory] = useState<MessageRow[]>([]);
  const [turn, setTurn] = useState<Transcript | null>(null);
  const [question, setQuestion] = useState("");
  const [allow, setAllow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const workspaceId = workspace?.id ?? null;

  const refreshSessions = useCallback(() => {
    if (!workspaceId) return;
    listSessions(workspaceId)
      .then(setSessions)
      .catch((e) => setError(messageOf(e)));
  }, [workspaceId]);

  // The view is keyed by the workspace, so a switch starts it fresh.
  useEffect(refreshSessions, [refreshSessions]);

  const select = useCallback(
    (row: SessionRow) => {
      if (!workspaceId) return;
      setSession(row);
      setTurn(null);
      sessionMessages(workspaceId, row.id)
        .then(setHistory)
        .catch((e) => setError(messageOf(e)));
    },
    [workspaceId],
  );

  const create = useCallback(async () => {
    if (!workspaceId) return;
    try {
      const row = await newSession(workspaceId);
      setError(null);
      refreshSessions();
      select(row);
    } catch (e) {
      setError(messageOf(e));
    }
  }, [workspaceId, refreshSessions, select]);

  useEffect(
    () =>
      menu.subscribe((id) => {
        if (id === "file:new-session") void create();
      }),
    [menu, create],
  );

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [turn, history]);

  const ask = async () => {
    if (!workspaceId || !session || question.trim() === "" || (turn && !turn.done)) return;
    const message = question;
    setQuestion("");
    setHistory((h) => [...h, userMessage(session.id, message, h.length)]);
    setTurn(empty);
    try {
      await startTurn(
        { workspace: workspaceId, session: session.id, message, documents: [] },
        (event) => setTurn((t) => reduce(t ?? empty, event)),
      );
      setError(null);
    } catch (e) {
      setError(messageOf(e));
      setTurn(null);
    }
  };

  const cancel = () => {
    if (workspaceId && session) cancelTurn(workspaceId, session.id).catch(() => {});
  };

  const toggleAllow = async (on: boolean) => {
    if (!workspaceId) return;
    setAllow(on);
    await setAllowWrites(workspaceId, on).catch((e) => setError(messageOf(e)));
  };

  const changeMode = async (mode: ChatMode) => {
    if (!workspaceId || !session) return;
    await setSessionMode(workspaceId, session.id, mode).catch((e) => setError(messageOf(e)));
    setSession({ ...session, mode });
    refreshSessions();
  };

  if (!workspace) {
    return <p className="p-6 text-sm text-gray-500">Open a workspace first.</p>;
  }
  return (
    <div className="flex h-full">
      <aside className="flex w-56 shrink-0 flex-col border-r border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between px-3 py-2">
          <span className="text-sm font-semibold">Sessions</span>
          <button type="button" className="text-sm hover:underline" onClick={() => void create()}>
            New
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          {sessions.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => select(s)}
              className={`block w-full truncate px-3 py-1.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-800 ${
                session?.id === s.id ? "bg-gray-100 dark:bg-gray-800" : ""
              }`}
              title={s.title ?? s.id}
            >
              {s.title ?? s.id.slice(0, 8)}
            </button>
          ))}
        </div>
      </aside>
      <section className="flex min-w-0 flex-1 flex-col">
        {error && <p className="px-4 pt-2 text-sm text-red-600">{error}</p>}
        <div className="min-h-0 flex-1 overflow-auto px-4 py-3">
          {!session && <p className="text-sm text-gray-500">Pick a session or start a new one.</p>}
          {history.map((m) => (
            <Message key={m.id} row={m} />
          ))}
          {turn && <Turn t={turn} />}
          <div ref={bottom} />
        </div>
        {session && (
          <form
            className="border-t border-gray-200 p-3 dark:border-gray-800"
            onSubmit={(e) => {
              e.preventDefault();
              void ask();
            }}
          >
            <div className="mb-2 flex items-center gap-4 text-xs text-gray-500">
              <label className="flex items-center gap-1">
                mode
                <select
                  value={session.mode}
                  onChange={(e) => void changeMode(e.target.value as ChatMode)}
                  className="rounded border border-gray-300 bg-transparent px-1 dark:border-gray-700"
                >
                  <option value="chat">chat</option>
                  <option value="query">query</option>
                </select>
              </label>
              <label className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={allow}
                  onChange={(e) => void toggleAllow(e.target.checked)}
                />
                Run changes without asking
              </label>
              {turn && !turn.done && (
                <button type="button" className="text-red-600 hover:underline" onClick={cancel}>
                  Cancel
                </button>
              )}
            </div>
            <div className="flex gap-2">
              <textarea
                className="flex-1 resize-none rounded border border-gray-300 px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-900"
                rows={2}
                placeholder="Ask about the workspace…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void ask();
                  }
                }}
              />
              <button
                type="submit"
                disabled={Boolean(turn && !turn.done)}
                className="rounded bg-gray-900 px-3 text-sm text-white disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900"
              >
                Ask
              </button>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}

/** The question as the transcript shows it before the turn records it. */
function userMessage(session: string, content: string, seq: number): MessageRow {
  return {
    id: `pending-${seq}`,
    session_id: session,
    seq,
    role: "user",
    content,
    metadata: null,
    created_at: new Date().toISOString(),
  } as MessageRow;
}

function Message({ row }: { row: MessageRow }) {
  if (row.role === "tool") {
    return (
      <div className="my-1 text-xs text-gray-500">
        <span className="font-medium">tool</span> {firstLine(row.content)}
      </div>
    );
  }
  return (
    <div className={`my-2 ${row.role === "user" ? "text-right" : ""}`}>
      <div
        className={`inline-block max-w-prose rounded px-3 py-2 text-left text-sm whitespace-pre-wrap ${
          row.role === "user" ? "bg-gray-100 dark:bg-gray-800" : ""
        }`}
      >
        {row.content}
      </div>
    </div>
  );
}

/** The turn in progress or just ended: text, steps, prompts, outcome. */
function Turn({ t }: { t: Transcript }) {
  return (
    <div className="my-2 text-sm">
      {t.status && <p className="text-xs text-gray-500">{t.status}</p>}
      {t.reasoning && !t.text && <p className="text-xs text-gray-500">thinking…</p>}
      {t.steps.map((s, i) => (
        <details key={i} className="my-1 text-xs text-gray-600 dark:text-gray-400">
          <summary>
            <span className="font-medium">{s.tool}</span> {s.step ? s.step.summary : s.detail}
            {s.step && <span className="text-gray-400"> {s.step.duration_ms} ms</span>}
          </summary>
          {s.step?.sql && (
            <pre className="mt-1 overflow-auto rounded bg-gray-50 p-2 dark:bg-gray-900">
              {s.step.sql}
            </pre>
          )}
          {s.step?.result && (
            <pre className="mt-1 max-h-48 overflow-auto rounded bg-gray-50 p-2 dark:bg-gray-900">
              {JSON.stringify(s.step.result, null, 1)}
            </pre>
          )}
        </details>
      ))}
      {t.prompts.map((p, i) => (
        <div
          key={i}
          className="my-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs dark:bg-amber-950"
        >
          <p className="font-medium">{p.heading}</p>
          {p.notice && <p className="text-gray-600 dark:text-gray-300">{p.notice}</p>}
          <pre className="mt-1 whitespace-pre-wrap">{p.sql}</pre>
          <p className="mt-1 text-gray-600 dark:text-gray-300">
            {p.reply ?? "Answer in the dialog."}
          </p>
        </div>
      ))}
      {t.text && <div className="whitespace-pre-wrap">{t.text}</div>}
      {t.response && t.response.citations.length > 0 && (
        <ol className="mt-2 list-decimal pl-5 text-xs text-gray-600 dark:text-gray-400">
          {t.response.citations.map((c, i) => (
            <li key={i}>{c.label}</li>
          ))}
        </ol>
      )}
      {t.response?.write_refused && (
        <p className="text-xs text-amber-700">A write was refused during this turn.</p>
      )}
      {t.response?.cancelled && <p className="text-xs text-gray-500">Cancelled.</p>}
      {t.failure && <p className="text-red-600">{t.failure.message}</p>}
    </div>
  );
}

function firstLine(text: string): string {
  return text.split("\n").find((l) => l.trim() !== "") ?? "";
}
