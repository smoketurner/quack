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
    var body = el("div", "mt-2 whitespace-pre-wrap");
    article.appendChild(body);
    messages.appendChild(article);
    return { article: article, details: details, summary: summary, steps: steps, body: body, count: 0 };
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
    status.textContent = "Thinking…";
    var body = { prompt: prompt, mode: form.mode.value };
    if (chat.getAttribute("data-session")) body.session_id = chat.getAttribute("data-session");
    if (form.allow_write && form.allow_write.checked) body.allow_write = true;

    fetch("/api/v1/workspaces/" + ws + "/query/stream", {
      method: "POST",
      headers: { "content-type": "application/json", "accept": "text/event-stream" },
      body: JSON.stringify(body),
      credentials: "same-origin"
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
      status.textContent = "";
      view.body.textContent = "Error: " + err.message;
      view.article.classList.add("border-red-300");
    });
  }

  function handle(event, data, view, chat, status) {
    if (event === "text") {
      view.body.textContent += data;
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
      status.textContent = "Running " + s.tool + "…";
    } else if (event === "tool_finished") {
      var f = JSON.parse(data);
      var pending = view.steps.querySelector("li[data-pending]");
      if (pending) {
        pending.removeAttribute("data-pending");
        pending.appendChild(el("span", "text-slate-500", " → " + f.summary + ", " + f.duration_ms + " ms"));
      }
    } else if (event === "complete") {
      var r = JSON.parse(data);
      view.body.textContent = r.answer;
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
      }
      status.textContent = "";
    } else if (event === "error") {
      view.body.textContent = "Error: " + data;
      view.article.classList.add("border-red-300");
      status.textContent = "";
    } else if (event === "permission_denied") {
      status.textContent = data;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderStoredCharts();
    var chat = document.getElementById("chat");
    var form = document.getElementById("ask");
    if (chat && form) {
      var mode = chat.getAttribute("data-mode");
      if (mode) form.mode.value = mode;
      form.addEventListener("submit", function (ev) { ev.preventDefault(); submitAsk(form, chat); });
    }
  });
})();
