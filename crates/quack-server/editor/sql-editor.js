// The SQL page's editor: CodeMirror over the plain textarea, which stays in
// the form (hidden) so the htmx run and the CSV download post it as before.
// Without JavaScript the textarea is the editor.
import { Compartment, EditorState } from "@codemirror/state";
import { EditorView, keymap, placeholder } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { autocompletion, closeBrackets, completionKeymap } from "@codemirror/autocomplete";
import { bracketMatching, syntaxTree } from "@codemirror/language";
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
// way a statement writes it (quoted when DuckDB needs that). Its source
// completes `name.` with that table's (or alias's) columns.
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

// Words after which a table name comes, and words that start a clause of
// expressions: the terminal's rules (`terminal/sql.rs`).
const TABLE_WORDS = new Set(["from", "join", "describe", "summarize", "update", "into"]);
const EXPRESSION_WORDS = new Set([
  "select", "where", "by", "on", "having", "qualify", "set", "using", "returning",
]);
// After these an expression must come, so a blank word is offered names
// (not `*`, which may be `SELECT *`).
const EXPRESSION_STARTS = new Set([
  ...EXPRESSION_WORDS, "and", "or", "not", "case", "when", "then", "else",
  ",", "(", "=", "<>", "!=", "<", ">", "<=", ">=", "+", "-", "/",
]);

// The tokens of the statement around `pos`, as CodeMirror's SQL parser read
// them: each with its node name and lowercased text.
function statementTokens(state, pos) {
  const tree = syntaxTree(state);
  let statement = tree.resolveInner(pos, -1);
  while (statement.parent && statement.name !== "Statement") statement = statement.parent;
  const tokens = [];
  tree.iterate({
    from: statement.from,
    to: statement.to,
    enter(node) {
      if (node.node.firstChild) return;
      tokens.push({
        name: node.name,
        from: node.from,
        to: node.to,
        text: state.doc.sliceString(node.from, node.to).toLowerCase(),
      });
    },
  });
  return tokens;
}

// Table and column names where the statement takes them, then keywords
// and functions in expressions; nothing at an alias or right after a
// table name. Prefix matches only, as in the terminal.
function namesSource(schema) {
  const tables = new Map(schema.tables.map((t) => [t.name.name.toLowerCase(), t]));
  const tableOption = (t) => ({ label: t.name.name, apply: t.name.sql, type: "type", boost: 1 });
  const keywords = keywordCompletionSource(PostgreSQL, true);
  return (context) => {
    // `name.`: lang-sql's schema source answers with that table's columns.
    if (context.matchBefore(/\.[\w$]*$/)) return null;
    const word = context.matchBefore(/[\w$]*$/);
    const tokens = statementTokens(context.state, context.pos);
    const before = tokens.filter((t) => t.to <= word.from);
    const previous = before[before.length - 1];
    if (!previous) return null;
    if (TABLE_WORDS.has(previous.text)) {
      return { from: word.from, options: schema.tables.map(tableOption), validFor: /^[\w$]*$/ };
    }
    const opener = [...before].reverse().find((t) => TABLE_WORDS.has(t.text) || EXPRESSION_WORDS.has(t.text));
    if (!opener) return null;
    if (TABLE_WORDS.has(opener.text)) {
      // In a FROM list a name follows a comma; anything else is an alias.
      if (previous.text !== ",") return null;
      return { from: word.from, options: schema.tables.map(tableOption), validFor: /^[\w$]*$/ };
    }
    // An alias after AS or after a name, and a blank word, get nothing.
    if (previous.text === "as" || /Identifier$/.test(previous.name)) return null;
    // A blank word gets names only where an expression must follow.
    if (word.from === word.to && !context.explicit && !EXPRESSION_STARTS.has(previous.text)) {
      return null;
    }
    // The columns of every table the statement names, before or after the
    // cursor, first; then table names; then keywords and functions.
    const named = new Set();
    for (const t of tokens) {
      const name = t.name === "QuotedIdentifier" ? t.text.slice(1, -1) : t.text;
      if (/Identifier$/.test(t.name) && tables.has(name)) named.add(name);
    }
    const seen = new Set();
    const options = [];
    for (const name of named) {
      for (const column of tables.get(name).columns) {
        if (seen.has(column.name)) continue;
        seen.add(column.name);
        options.push({ label: column.name, apply: column.sql, type: "property", detail: name, boost: 2 });
      }
    }
    options.push(...schema.tables.map(tableOption));
    const words = keywords(context);
    if (words) options.push(...words.options);
    return { from: word.from, options, validFor: /^[\w$]*$/ };
  };
}

function completion(schema) {
  const qualified = schemaCompletionSource({ dialect: PostgreSQL, schema: namespace(schema) });
  return [
    sql({ dialect: PostgreSQL, upperCaseKeywords: true }),
    autocompletion({
      filterStrict: true,
      override: [
        // lang-sql's own source only after `name.`; everywhere else it
        // would repeat the table names and offer them at an alias.
        (context) => (context.matchBefore(/\.[\w$]*$/) ? qualified(context) : null),
        namesSource(schema),
      ],
    }),
  ];
}

// The editor is built at once over an empty schema, at the textarea's size
// and on its background, so the box does not flash; the schema plugs in
// when it arrives.
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
  const names = new Compartment();
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
        names.of(completion({ tables: [] })),
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
  loadSchema(form.dataset.schema).then((schema) => {
    view.dispatch({ effects: names.reconfigure(completion(schema)) });
  });
}

function attachAll() {
  for (const textarea of document.querySelectorAll("form[data-schema] textarea#sql-input")) {
    attach(textarea);
  }
}

document.addEventListener("DOMContentLoaded", attachAll);
document.addEventListener("htmx:after:swap", attachAll);
