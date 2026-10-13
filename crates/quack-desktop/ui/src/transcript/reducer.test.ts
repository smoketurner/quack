import { describe, expect, it } from "vitest";
import type { TurnEvent } from "../commands";
import { empty, reduce } from "./reducer";

const step = (tool: "run_sql", summary: string): TurnEvent["step" & keyof never] & object =>
  ({ tool, summary, duration_ms: 3 }) as unknown as object;

function run(events: TurnEvent[]) {
  return events.reduce(reduce, empty);
}

describe("the transcript reducer", () => {
  it("streams text, drops the status once text arrives, and takes the final answer", () => {
    const t = run([
      { type: "status", text: "loading the model" },
      { type: "reasoning" },
      { type: "text", delta: "Hel" },
      { type: "text", delta: "lo" },
    ]);
    expect(t.text).toBe("Hello");
    expect(t.status).toBeNull();
    expect(t.reasoning).toBe(false);
    expect(t.done).toBe(false);
  });

  it("pairs a finished step with the step that started it", () => {
    const finished = step("run_sql", "3 rows") as unknown as Extract<
      TurnEvent,
      { type: "tool_finished" }
    >["step"];
    const t = run([
      { type: "tool_started", tool: "run_sql", detail: "SELECT 1" },
      { type: "tool_started", tool: "search_documents", detail: "q" },
      { type: "tool_finished", step: finished },
    ]);
    expect(t.steps).toHaveLength(2);
    expect(t.steps[0]?.step).toBe(finished);
    expect(t.steps[1]?.step).toBeUndefined();
  });

  it("records a write prompt and the answer the dialog gave it", () => {
    const t = run([
      {
        type: "permission_required",
        sql: "DELETE FROM t",
        reason: "read_documents",
        notice: "This turn read document text.",
        heading: "This statement changes the workspace:",
      },
      { type: "permission_answered", decision: "deny", reply: "Not run." },
    ]);
    expect(t.prompts).toEqual([
      {
        sql: "DELETE FROM t",
        reason: "read_documents",
        notice: "This turn read document text.",
        heading: "This statement changes the workspace:",
        decision: "deny",
        reply: "Not run.",
      },
    ]);
  });

  it("ends on a failure with its kind", () => {
    const t = run([
      { type: "text", delta: "part" },
      { type: "failed", message: "no chat model is configured", kind: "no_chat_model" },
    ]);
    expect(t.done).toBe(true);
    expect(t.failure?.kind).toBe("no_chat_model");
    expect(t.text).toBe("part");
  });

  it("never mutates the transcript it was given", () => {
    const before = { ...empty, steps: [] };
    const after = reduce(before, { type: "text", delta: "x" });
    expect(before.text).toBe("");
    expect(after.text).toBe("x");
  });
});
