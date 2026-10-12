// The SQL page's editor: CodeMirror over the plain textarea, which stays in
// the form (hidden) so the htmx run and the CSV download post it as before.
// Without JavaScript the textarea is the editor.
//
// Completion comes from the server: the statement and the cursor go to
// POST .../sql/complete, and the completer the terminal uses answers with
// the word's span and the items (quack_core::completion). The browser holds
// no schema and no keyword list.
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, placeholder } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, closeBrackets, completionKeymap } from "@codemirror/autocomplete";
import { bracketMatching } from "@codemirror/language";
import { PostgreSQL, sql } from "@codemirror/lang-sql";
import { oneDark } from "@codemirror/theme-one-dark";

// The server counts characters (Unicode scalar values); CodeMirror counts
// UTF-16 code units. A statement with an emoji or a CJK name differs.
function charsBefore(text, index) {
  return Array.from(text.slice(0, index)).length;
}

function indexOfChar(text, chars) {
  let index = 0;
  for (const c of text) {
    if (chars === 0) break;
    index += c.length;
    chars -= 1;
  }
  return index;
}

const KIND_TYPES = { table: "type", column: "property", keyword: "keyword", function: "function" };

// Ask the server what the statement can take at the cursor.
function serverSource(url) {
  return async (context) => {
    const text = context.state.doc.toString();
    const found = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sql: text, cursor: charsBefore(text, context.pos) }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .catch(() => null);
    if (!found || found.items.length === 0) return null;
    return {
      from: indexOfChar(text, found.start),
      to: indexOfChar(text, found.end),
      options: found.items.map((item) => ({
        label: item.label,
        apply: item.word,
        type: KIND_TYPES[item.kind] ?? "text",
        detail: item.kind === "column" ? item.table : item.kind,
      })),
      // The server ranked and cut the list; let it answer again on each change.
      filter: false,
    };
  };
}

// The editor is built at the textarea's size and on its background, so the
// box does not flash.
function attach(textarea) {
  if (textarea.dataset.editor) return;
  textarea.dataset.editor = "on";
  const form = textarea.closest("form");
  // An out-of-band swap replaces the textarea; its old editor goes with it.
  for (const stale of document.querySelectorAll(".quack-sql-editor")) stale.remove();
  const holder = document.createElement("div");
  holder.className = "quack-sql-editor overflow-hidden rounded border border-slate-700 text-sm";
  const height = textarea.offsetHeight;
  textarea.after(holder);
  textarea.hidden = true;
  textarea.required = false;
  const view = new EditorView({
    parent: holder,
    state: EditorState.create({
      doc: textarea.value,
      extensions: [
        history(),
        keymap.of([
          { key: "Mod-Enter", run: () => (form.requestSubmit(), true) },
          ...completionKeymap,
          ...defaultKeymap,
          ...historyKeymap,
        ]),
        sql({ dialect: PostgreSQL, upperCaseKeywords: true }),
        autocompletion({
          override: [serverSource(form.dataset.complete)],
          activateOnTypingDelay: 250,
        }),
        closeBrackets(),
        bracketMatching(),
        oneDark,
        placeholder(textarea.placeholder),
        EditorView.lineWrapping,
        EditorView.contentAttributes.of({ "aria-label": textarea.getAttribute("aria-label") }),
        EditorView.theme({
          "&": { backgroundColor: "transparent" },
          ".cm-gutters": { backgroundColor: "transparent" },
          ".cm-content, .cm-gutter": { minHeight: `${Math.max(height - 2, 0)}px` },
        }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) textarea.value = update.state.doc.toString();
        }),
      ],
    }),
  });
  view.focus();
}

function attachAll() {
  for (const textarea of document.querySelectorAll("form[data-complete] textarea#sql-input")) {
    attach(textarea);
  }
}

document.addEventListener("DOMContentLoaded", attachAll);
document.addEventListener("htmx:after:swap", attachAll);
