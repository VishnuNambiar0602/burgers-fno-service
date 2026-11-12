/* Agent mode: the machine-readable version of the page. This file is loaded only when the user switches Agent mode ON, and everything it
   adds to the page is removed again when the mode is switched OFF. Nothing machine-oriented exists in the normal page. */
(function () {
  "use strict";
  var root = null, listener = null, manifest = null, B = null, lastRunId = "";

  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function pretty(o) { return JSON.stringify(o, null, 2); }
  function jsonScript(id, obj) {                      // an inert JSON data block, easy for parsers to find
    var s = document.createElement("script"); s.type = "application/json"; s.id = id; s.textContent = JSON.stringify(obj).replace(/</g, "\\u003c"); return s;
  }
  var r4 = function (v) { return Math.round(v * 1e4) / 1e4; };

  function snapshot() {
    var S = B.state, I = S.input, R = S.result, st = { mode: "agent", page: location.origin + location.pathname, busy: !!S.busy, offline: !!S.offline };
    st.input = I ? { label: I.label, kind: I.kind, points: I.u.length, max_abs: r4(Array.prototype.reduce.call(I.u, function (m, v) { return Math.max(m, Math.abs(v)); }, 0)),
      mean: r4(Array.prototype.reduce.call(I.u, function (s, v) { return s + v; }, 0) / I.u.length), u0: Array.prototype.map.call(I.u, function (v) { return Math.round(v * 1e6) / 1e6; }) } : null;
    if (R) {
      var d = R.resp;
      st.last_result = { received_at: R.at.toISOString(), input_label: R.label, verdict: d.verdict || null, strip_text: document.getElementById("strip-text").textContent,
        difference_percent: r4(100 * d.solver.rel_l2_fno_vs_solver_whole), ood: d.ood, self_check: d.self_check || null,
        timings_ms: { ai: r4(d.fno.seconds * 1000), solver: Math.round(d.solver.seconds * 1000), round_trip: Math.round(R.rtMs) },
        mass_drift: { ai: d.fno.mass_drift, solver: d.solver.mass_drift }, model: d.model, arrays: "not repeated here; call tool 'predict' (profiles) or POST /predict (full 201 x 256 arrays)" };
    } else st.last_result = null;
    st.human_ui_controls = [
      { selector: "button.pre[data-key]", does: "choose a preset wave", keys: ["sine", "two", "bump", "square", "random", "tall", "noise", "wiggly"] },
      { selector: "#input-plot", does: "draw a wave with pointer events (one stroke, left to right)" },
      { selector: "#advanced", does: "open the dialog for sine components or 256 pasted numbers" },
      { selector: "#run", does: "run the forecast (POST /predict)" },
      { selector: "#strip-text", does: "the verdict sentence" }, { selector: "#nums", does: "key numbers" },
      { selector: "#dl-csv, #dl-json, #dl-pdf", does: "downloads" }, { selector: "#agent-toggle", does: "leave agent mode (back to the dashboard)" }];
    st.note = "In agent mode the human dashboard is hidden. Prefer the MCP tools, the REST API, or the form on this page.";
    return st;
  }

  function refresh() {
    if (!root) return;
    var st = snapshot(), old = root.querySelector("#agent-state");
    if (old) old.textContent = JSON.stringify(st).replace(/</g, "\\u003c");
    var pre = root.querySelector("#agent-state-pre"); if (pre) pre.textContent = pretty(st);
  }

  function build() {
    var base = location.origin + "/", mcpOn = !!(manifest && manifest.interfaces.mcp.enabled !== false);
    root.textContent = "";
    root.appendChild(el("h2", null, "Machine view"));
    root.appendChild(el("p", "note", "The same application in a form built for AI agents and scripts. Humans: switch Agent mode off (button in the header) to return to the dashboard."));
    root.appendChild(jsonScript("agent-manifest", manifest || { error: "could not load /agent.json" }));
    root.appendChild(jsonScript("agent-state", snapshot()));
    if (!manifest) { root.appendChild(el("p", "warn", "The manifest could not be loaded from /agent.json. This page needs the server (" + base + ").")); }

    root.appendChild(el("h3", null, "Connect"));
    var c = el("pre", "mjson"); c.setAttribute("data-agent-section", "connect");
    var lines = [];
    if (mcpOn) {
      lines.push("MCP    POST " + base + "mcp   (Streamable HTTP, JSON-RPC 2.0, JSON responses, no sessions)");
      lines.push("       client config: " + JSON.stringify({ mcpServers: { burgers: { url: base + "mcp" } } }));
    } else lines.push("MCP    switched off on this server (the operator can enable it with MCP_ENABLED=1); the REST API below still works");
    lines.push("REST   POST " + base + "predict   body {\"u0\": [256 numbers], \"include_solver\": true}");
    lines.push("Other  GET " + base + "agent.json (this manifest)   GET " + base + "openapi.json   GET " + base + "health   GET " + base + "model-info");
    c.textContent = lines.join("\n");
    root.appendChild(c);

    if (manifest) {
      root.appendChild(el("h3", null, "Rules for agents"));
      var ul = el("ul"); manifest.rules_for_agents.forEach(function (r) { ul.appendChild(el("li", null, r)); }); root.appendChild(ul);
      if (mcpOn) {
        root.appendChild(el("h3", null, "Tools (MCP)"));
      var tbl = el("table", "mtable"), hd = el("tr"); ["name", "what it does", "arguments (JSON Schema)"].forEach(function (h) { hd.appendChild(el("th", null, h)); }); tbl.appendChild(hd);
      manifest.interfaces.mcp.tools.forEach(function (t) {
        var tr = el("tr"); tr.setAttribute("data-agent-tool", t.name);
        tr.appendChild(el("td", null, t.name)); tr.appendChild(el("td", null, t.description)); var td = el("td"); td.appendChild(el("code", null, JSON.stringify(t.inputSchema))); tr.appendChild(td); tbl.appendChild(tr);
      });
      var wrap = el("div", "tablewrap"); wrap.appendChild(tbl); root.appendChild(wrap);
      }
    }

    root.appendChild(el("h3", null, "Current state of the page"));
    var sp = el("pre", "mjson"); sp.id = "agent-state-pre"; sp.setAttribute("data-agent-section", "state"); sp.tabIndex = 0; sp.textContent = pretty(snapshot()); root.appendChild(sp);

    if (!mcpOn) { root.appendChild(el("p", "warn", "MCP is switched off on this server, so there are no tools to call here.")); return; }
    root.appendChild(el("h3", null, "Try it (calls the MCP endpoint above)"));
    var form = el("form", "aform"); form.id = "agent-form"; form.setAttribute("data-agent-form", "tool-call");
    var l1 = el("label", null, "Tool"); l1.setAttribute("for", "agent-tool");
    var sel = el("select"); sel.id = "agent-tool"; sel.setAttribute("data-agent-field", "tool");
    var DEFAULTS = { predict: { preset: "sine" }, make_wave: { preset: "bump" }, list_presets: {}, get_profile: { run_id: "", times: [0, 1, 2] }, get_model_info: {}, get_status: {}, explain: { topic: "limits" } };
    ((manifest && manifest.interfaces.mcp.tools) || []).forEach(function (t) { var o = el("option", null, t.name); o.value = t.name; sel.appendChild(o); });
    var l2 = el("label", null, "Arguments (JSON)"); l2.setAttribute("for", "agent-args");
    var ta = el("textarea"); ta.id = "agent-args"; ta.rows = 5; ta.spellcheck = false; ta.setAttribute("data-agent-field", "arguments");
    function fillDefault() { var d = JSON.parse(JSON.stringify(DEFAULTS[sel.value] || {})); if (sel.value === "get_profile") d.run_id = lastRunId; ta.value = pretty(d); }
    sel.value = "predict"; fillDefault(); sel.addEventListener("change", fillDefault);
    var btn = el("button", "btn primary", "Call tool"); btn.type = "submit"; btn.id = "agent-call"; btn.setAttribute("data-agent-action", "call-tool");
    var out = el("pre", "mjson"); out.id = "agent-out"; out.setAttribute("data-agent-output", "result"); out.setAttribute("aria-live", "polite"); out.tabIndex = 0; out.textContent = "(no call yet)";
    form.appendChild(l1); form.appendChild(sel); form.appendChild(l2); form.appendChild(ta); form.appendChild(btn);
    form.addEventListener("submit", async function (ev) {
      ev.preventDefault(); var args;
      try { args = JSON.parse(ta.value || "{}"); } catch (e) { out.textContent = pretty({ error: "arguments are not valid JSON: " + e.message }); return; }
      out.textContent = "calling " + sel.value + " ..."; btn.disabled = true;
      try {
        var r = await fetch("/mcp", { method: "POST", headers: { "content-type": "application/json", accept: "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/call", params: { name: sel.value, arguments: args } }) });
        var j = await r.json(), res = j.result;
        if (res && res.structuredContent && res.structuredContent.run_id) lastRunId = res.structuredContent.run_id;
        out.textContent = pretty(res && res.structuredContent ? res.structuredContent : (res || j));
      } catch (e) { out.textContent = pretty({ error: "request failed: " + e.message }); }
      btn.disabled = false;
    });
    root.appendChild(form); root.appendChild(out);
  }

  async function enable(burgers) {
    if (root) return;
    B = burgers;
    root = el("section", "agent-view"); root.id = "agent-view"; root.setAttribute("data-agent-view", "true"); root.setAttribute("aria-label", "Machine-readable view of this application");
    document.querySelector(".app").appendChild(root);
    root.appendChild(el("p", "note", "Loading the machine-readable manifest…"));
    try { var r = await fetch("/agent.json"); manifest = r.ok ? await r.json() : null; } catch (e) { manifest = null; }
    if (!root) return;                                  // switched off while loading
    build();
    listener = function () { refresh(); };
    document.addEventListener("burgers:state", listener);
  }

  function disable() {
    if (listener) document.removeEventListener("burgers:state", listener);
    listener = null; if (root) root.remove(); root = null; manifest = null;
  }

  window.BurgersAgent = { enable: enable, disable: disable };
})();
