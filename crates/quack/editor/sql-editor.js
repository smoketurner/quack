// The SQL page's editor: CodeMirror over the plain textarea, which stays in
// the form (hidden) so the htmx run and the CSV download post it as before.
// Without JavaScript the textarea is the editor.
import { EditorState } from "@codemirror/state";
import { EditorView, keymap, placeholder } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, closeBrackets, completionKeymap } from "@codemirror/autocomplete";
import { bracketMatching } from "@codemirror/language";
import { PostgreSQL, keywordCompletionSource, schemaCompletionSource, sql } from "@codemirror/lang-sql";
import { oneDark } from "@codemirror/theme-one-dark";

// The schema each form's tables load from, fetched once per page.
const schemas = new Map();

function loadSchema(url) {
  if (!schemas.has(url)) {
    schemas.set(
      url,
      fetch(url, { credentials: "same-origin" })
        .then((r) => (r.ok ? r.json() : { tables: [] }))
        .catch(() => ({ tables: [] })),
    );
  }
  return schemas.get(url);
}

// lang-sql's namespace: each table with its columns, every name inserted the
// way a statement writes it (quoted when DuckDB needs that).
function namespace(schema) {
  const tables = {};
  for (const table of schema.tables) {
    tables[table.name.name] = {
      self: { label: table.name.name, apply: table.name.sql, type: "type" },
      children: table.columns.map((c) => ({ label: c.name, apply: c.sql, type: "property" })),
    };
  }
  return tables;
}

// Names from the schema everywhere; keywords too, except after `name.`,
// where only that table's columns belong. Prefix matches only, as in the
// terminal.
function completion(schema) {
  const names = schemaCompletionSource({ dialect: PostgreSQL, schema: namespace(schema) });
  const keywords = keywordCompletionSource(PostgreSQL, true);
  return autocompletion({
    filterStrict: true,
    override: [names, (context) => (context.matchBefore(/\.[\w$]*$/) ? null : keywords(context))],
  });
}

function attach(textarea) {
  if (textarea.dataset.editor) return;
  textarea.dataset.editor = "on";
  const form = textarea.closest("form");
  // An out-of-band swap replaces the textarea; its old editor goes with it.
  for (const stale of document.querySelectorAll(".quack-sql-editor")) stale.remove();
  const holder = document.createElement("div");
  holder.className = "quack-sql-editor overflow-hidden rounded border border-slate-700 text-sm";
  textarea.after(holder);
  textarea.hidden = true;
  textarea.required = false;
  loadSchema(form.dataset.schema).then((schema) => {
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
          completion(schema),
          closeBrackets(),
          bracketMatching(),
          oneDark,
          placeholder(textarea.placeholder),
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ "aria-label": textarea.getAttribute("aria-label") }),
          EditorView.theme({ ".cm-content, .cm-gutter": { minHeight: "9rem" } }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) textarea.value = update.state.doc.toString();
          }),
        ],
      }),
    });
    view.focus();
  });
}

function attachAll() {
  for (const textarea of document.querySelectorAll("form[data-schema] textarea#sql-input")) {
    attach(textarea);
  }
}

document.addEventListener("DOMContentLoaded", attachAll);
document.addEventListener("htmx:after:swap", attachAll);
