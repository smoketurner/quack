// A turn's events folded into what the transcript shows: the streamed
// text, each step as it starts and finishes, the write prompts and their
// answers, and how the turn ended.
import type { AgentResponseBody, ErrorKind, ToolStep, TurnEvent } from "../commands";

export interface Step {
  tool: string;
  detail: string;
  /** Set once the tool finished. */
  step?: ToolStep;
}

export interface Prompt {
  sql: string;
  reason: string;
  notice: string | null;
  heading: string;
  /** Set once the dialog was answered. */
  decision?: string;
  reply?: string;
}

export interface Transcript {
  text: string;
  status: string | null;
  reasoning: boolean;
  steps: Step[];
  prompts: Prompt[];
  response: AgentResponseBody | null;
  failure: { message: string; kind: ErrorKind } | null;
  done: boolean;
}

export const empty: Transcript = {
  text: "",
  status: null,
  reasoning: false,
  steps: [],
  prompts: [],
  response: null,
  failure: null,
  done: false,
};

/** The transcript after `event`. Never mutates `t`. */
export function reduce(t: Transcript, event: TurnEvent): Transcript {
  switch (event.type) {
    case "status":
      return { ...t, status: event.text };
    case "reasoning":
      return { ...t, reasoning: true, status: null };
    case "text":
      return { ...t, text: t.text + event.delta, status: null, reasoning: false };
    case "tool_started":
      return {
        ...t,
        status: null,
        reasoning: false,
        steps: [...t.steps, { tool: event.tool, detail: event.detail }],
      };
    case "tool_finished": {
      // The finished step closes the last started one of the same tool;
      // a finish with no start (a repaired call) stands on its own.
      const steps = [...t.steps];
      const open = steps
        .map((s, i) => (s.step === undefined && s.tool === event.step.tool ? i : -1))
        .filter((i) => i >= 0)
        .pop();
      if (open === undefined) {
        steps.push({ tool: event.step.tool, detail: "", step: event.step });
      } else {
        const started = steps[open];
        if (started) steps[open] = { ...started, step: event.step };
      }
      return { ...t, steps };
    }
    case "permission_required":
      return {
        ...t,
        prompts: [
          ...t.prompts,
          {
            sql: event.sql,
            reason: event.reason,
            notice: event.notice ?? null,
            heading: event.heading,
          },
        ],
      };
    case "permission_answered": {
      const prompts = [...t.prompts];
      const open = prompts
        .map((p, i) => (p.decision === undefined ? i : -1))
        .filter((i) => i >= 0)[0];
      if (open !== undefined) {
        const prompt = prompts[open];
        if (prompt) prompts[open] = { ...prompt, decision: event.decision, reply: event.reply };
      }
      return { ...t, prompts };
    }
    case "complete":
      // The answer carries the whole text, with citations validated.
      return {
        ...t,
        text: event.response.answer,
        response: event.response,
        status: null,
        done: true,
      };
    case "failed":
      return {
        ...t,
        failure: { message: event.message, kind: event.kind },
        status: null,
        done: true,
      };
  }
}
