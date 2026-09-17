// Chat streaming over the SSE endpoint, and chart rendering from the
// stored ChartSpec. Everything else on the site is a plain form or htmx.
(function () {
  "use strict";

  function chartOption(spec) {
    if (spec.kind === "pie") {
      var first = (spec.series && spec.series[0]) || { values: [] };
      return {
        title: { text: spec.title },
        tooltip: {},
        series: [{ type: "pie", radius: "60%", data: spec.x.values.map(function (l, i) { return { name: l, value: first.values[i] }; }) }]
      };
    }
    return {
      title: { text: spec.title },
      tooltip: {},
      legend: { bottom: 0 },
      xAxis: { type: "category", name: spec.x.label, data: spec.x.values },
      yAxis: { type: "value" },
      series: spec.series.map(function (s) { return { name: s.name, type: spec.kind, data: s.values }; })
    };
  }

  function renderChart(el, spec) {
    if (!window.echarts || !spec) return;
    var chart = window.echarts.init(el);
    chart.setOption(chartOption(spec));
    window.addEventListener("resize", function () { chart.resize(); });
  }

  function renderStoredCharts() {
    document.querySelectorAll(".chart[data-chart]").forEach(function (el) {
      try { renderChart(el, JSON.parse(el.getAttribute("data-chart"))); } catch (e) { /* malformed spec: leave empty */ }
    });
  }

  // A GraphResult ({nodes, edges, roots}) as an ECharts force graph, nodes
  // coloured by class; clicking a node scrolls its inspector entry into view.
  function graphOption(result) {
    var classes = [];
    result.nodes.forEach(function (n) { if (classes.indexOf(n.class_id) < 0) classes.push(n.class_id); });
    var roots = result.roots || [];
    return {
      tooltip: { formatter: function (p) { return p.dataType === "edge" ? p.data.label.formatter : p.data.name + " (" + classes[p.data.category] + ")"; } },
      legend: [{ data: classes, bottom: 0 }],
      series: [{
        type: "graph",
        layout: "force",
        roam: true,
        draggable: true,
        force: { repulsion: 220, edgeLength: 90 },
        categories: classes.map(function (c) { return { name: c }; }),
        label: { show: true, position: "right", fontSize: 11 },
        edgeSymbol: ["none", "arrow"],
        edgeSymbolSize: 7,
        edgeLabel: { show: true, fontSize: 9, formatter: "{c}" },
        lineStyle: { color: "#94a3b8", curveness: 0.1 },
        data: result.nodes.map(function (n) {
          return { id: n.id, name: n.label, category: classes.indexOf(n.class_id), symbolSize: roots.indexOf(n.id) >= 0 ? 26 : 14, itemStyle: n.provisional ? { opacity: 0.5 } : {} };
        }),
        links: result.edges.map(function (e) {
          return { source: e.source_node_id, target: e.target_node_id, value: e.relation_id, label: { formatter: e.relation_id } };
        })
      }]
    };
  }

  function renderGraphs() {
    document.querySelectorAll(".graph[data-graph]").forEach(function (el) {
      if (!window.echarts) return;
      var result;
      try { result = JSON.parse(el.getAttribute("data-graph")); } catch (e) { return; }
      var chart = window.echarts.init(el);
      chart.setOption(graphOption(result));
      chart.on("click", function (p) {
        if (p.dataType !== "node") return;
        var row = document.getElementById("node-" + p.data.id);
        if (row) { row.scrollIntoView({ block: "center", behavior: "smooth" }); row.classList.add("ring-2", "ring-blue-400"); }
      });
      window.addEventListener("resize", function () { chart.resize(); });
    });
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function citationLabel(c) {
    var label = c.filename || "";
    if (c.page) label += ", page " + c.page;
    if (c.heading) label += ', under "' + c.heading + '"';
    return label;
  }

  function startAssistant(messages) {
    var article = el("article", "rounded border border-slate-200 bg-white p-4 msg-assistant");
    article.appendChild(el("div", "text-xs font-semibold uppercase text-slate-500", "assistant"));
    var details = el("details", "mt-2 text-sm hidden");
    var summary = el("summary", "cursor-pointer text-slate-600", "0 steps");
    details.appendChild(summary);
    var steps = el("ol", "mt-1 space-y-1 font-mono text-xs");
    details.appendChild(steps);
    article.appendChild(details);
    var body = el("div", "answer mt-2 whitespace-pre-wrap");
    article.appendChild(body);
    var working = el("div", "mt-2 flex items-center gap-2 text-sm text-slate-500");
    var dot = el("span", "inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-blue-600");
    working.appendChild(dot);
    working.appendChild(el("span", null, "Thinking…"));
    article.appendChild(working);
    messages.appendChild(article);
    article.scrollIntoView({ block: "end" });
    return { article: article, details: details, summary: summary, steps: steps, body: body, working: working, count: 0 };
  }

  // The sidebar lists sessions at render time; a session started from this
  // page appears once its first answer completes.
  function addSessionEntry(chat, sessionId, prompt) {
    var list = document.getElementById("sessions");
    if (!list) return;
    var ws = chat.getAttribute("data-workspace");
    var li = el("li", "group flex items-start gap-1");
    var a = el("a", "block min-w-0 flex-1 truncate rounded bg-slate-100 px-2 py-1");
    a.href = "/w/" + ws + "/chat?session=" + sessionId;
    a.textContent = prompt.length > 80 ? prompt.slice(0, 80) : prompt;
    a.appendChild(el("span", "block text-xs text-slate-500", "just now · " + document.getElementById("ask").mode.value));
    li.appendChild(a);
    var form = el("form");
    form.method = "post";
    form.action = "/w/" + ws + "/chat/" + sessionId + "/delete";
    form.onsubmit = function () { return confirm("Delete this session?"); };
    var button = el("button", "rounded px-2 py-1 text-slate-400 hover:bg-red-50 hover:text-red-700", "×");
    button.type = "submit";
    button.title = "Delete session";
    form.appendChild(button);
    li.appendChild(form);
    list.insertBefore(li, list.firstChild);
  }

  function setWorking(view, text) {
    if (!view.working) return;
    view.working.lastChild.textContent = text;
  }

  function finishWorking(view) {
    if (view.working && view.working.parentNode) view.working.parentNode.removeChild(view.working);
    view.working = null;
  }

  function setBusy(form, busy) {
    var button = form.querySelector("button[type=submit]");
    if (button) {
      button.disabled = busy;
      button.textContent = busy ? "Working…" : "Send";
      button.classList.toggle("opacity-50", busy);
      button.classList.toggle("cursor-not-allowed", busy);
      button.classList.toggle("ml-auto", !busy);
    }
    var stop = form.querySelector("#stop");
    if (stop) stop.classList.toggle("hidden", !busy);
    form.prompt.disabled = busy;
  }

  function parseSse(buffer, onEvent) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop();
    parts.forEach(function (block) {
      var event = "message", data = [];
      block.split("\n").forEach(function (line) {
        if (line.indexOf("event:") === 0) event = line.slice(6).trim();
        else if (line.indexOf("data:") === 0) data.push(line.slice(5).replace(/^ /, ""));
      });
      if (data.length) onEvent(event, data.join("\n"));
    });
    return rest;
  }

  function submitAsk(form, chat) {
    var ws = chat.getAttribute("data-workspace");
    var messages = document.getElementById("messages");
    var status = document.getElementById("status");
    var prompt = form.prompt.value.trim();
    if (!prompt) return;
    var user = el("article", "rounded border border-slate-200 bg-white p-4 msg-user");
    user.appendChild(el("div", "text-xs font-semibold uppercase text-slate-500", "user"));
    user.appendChild(el("div", "mt-2 whitespace-pre-wrap", prompt));
    messages.appendChild(user);
    form.prompt.value = "";
    var view = startAssistant(messages);
    view.prompt = prompt;
    setBusy(form, true);
    status.textContent = "";
    var body = { prompt: prompt, mode: form.mode.value };
    if (chat.getAttribute("data-session")) body.session_id = chat.getAttribute("data-session");
    if (form.allow_write && form.allow_write.checked) body.allow_write = true;

    // Stop aborts the request; the server sees the stream close and
    // cancels the turn, recording what streamed so far.
    var controller = new AbortController();
    var stop = form.querySelector("#stop");
    var onStop = function () { controller.abort(); };
    if (stop) stop.addEventListener("click", onStop);

    fetch("/api/v1/workspaces/" + ws + "/query/stream", {
      method: "POST",
      headers: { "content-type": "application/json", "accept": "text/event-stream" },
      body: JSON.stringify(body),
      credentials: "same-origin",
      signal: controller.signal
    }).then(function (res) {
      if (!res.ok) return res.json().then(function (j) { throw new Error(j.error || res.statusText); });
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) return;
          buffer += decoder.decode(r.value, { stream: true });
          buffer = parseSse(buffer, function (event, data) { handle(event, data, view, chat, status); });
          return pump();
        });
      }
      return pump();
    }).catch(function (err) {
      if (err.name === "AbortError") {
        view.body.textContent += (view.body.textContent ? "\n\n" : "") + "(Cancelled.)";
        return;
      }
      view.body.textContent = "Error: " + err.message;
      view.article.classList.add("border-red-300");
    }).then(function () {
      if (stop) stop.removeEventListener("click", onStop);
      finishWorking(view);
      setBusy(form, false);
      form.prompt.focus();
    });
  }

  function handle(event, data, view, chat, status) {
    if (event === "text") {
      view.body.textContent += data;
      setWorking(view, "Answering…");
    } else if (event === "tool_started") {
      var s = JSON.parse(data);
      view.count += 1;
      view.details.classList.remove("hidden");
      view.summary.textContent = view.count + " steps";
      var li = el("li");
      li.appendChild(el("span", "font-semibold", s.tool + " "));
      li.appendChild(document.createTextNode(s.detail));
      li.setAttribute("data-pending", "1");
      view.steps.appendChild(li);
      setWorking(view, "Running " + s.tool + "…");
    } else if (event === "tool_finished") {
      var f = JSON.parse(data);
      var pending = view.steps.querySelector("li[data-pending]");
      if (pending) {
        pending.removeAttribute("data-pending");
        pending.appendChild(el("span", "text-slate-500", " → " + f.summary + ", " + f.duration_ms + " ms"));
      }
    } else if (event === "complete") {
      var r = JSON.parse(data);
      if (r.answer_html) {
        // Rendered server-side from the Markdown, with raw HTML escaped.
        view.body.classList.remove("whitespace-pre-wrap");
        view.body.innerHTML = r.answer_html;
      } else {
        view.body.textContent = r.answer;
      }
      if (r.chart) {
        var c = el("div", "chart mt-3 h-72");
        view.article.appendChild(c);
        renderChart(c, r.chart);
      }
      if (r.citations && r.citations.length) {
        var ol = el("ol", "mt-3 space-y-1 text-sm text-slate-600");
        r.citations.forEach(function (cit) {
          var li = el("li", null, "[" + cit.n + "] ");
          var a = el("a", "text-blue-700 hover:underline", citationLabel(cit));
          a.href = "/w/" + chat.getAttribute("data-workspace") + "/documents#doc-" + cit.document_id;
          li.appendChild(a);
          ol.appendChild(li);
        });
        view.article.appendChild(ol);
      }
      if (r.write_refused) {
        view.article.appendChild(el("p", "mt-2 text-sm text-amber-800", "A change to the tables was refused. Tick “Allow the agent to change tables” and ask again to permit it."));
      }
      if (!chat.getAttribute("data-session") && r.session_id) {
        chat.setAttribute("data-session", r.session_id);
        history.replaceState(null, "", "?session=" + r.session_id);
        addSessionEntry(chat, r.session_id, view.prompt);
      }
      finishWorking(view);
    } else if (event === "error") {
      view.body.textContent = "Error: " + data;
      view.article.classList.add("border-red-300");
      finishWorking(view);
    } else if (event === "permission_denied") {
      status.textContent = data;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderStoredCharts();
    renderGraphs();
    var chat = document.getElementById("chat");
    var form = document.getElementById("ask");
    if (chat && form) {
      // A session's mode is fixed when it is created; the selector only
      // chooses the mode of a new session.
      var mode = chat.getAttribute("data-mode");
      if (mode) { form.mode.value = mode; form.mode.disabled = true; form.mode.title = "Set when the session was created"; }
      form.addEventListener("submit", function (ev) { ev.preventDefault(); submitAsk(form, chat); });
    }
  });
})();
