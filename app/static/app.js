/* Dashboard logic. Every result shown comes from a successful POST /predict response; presets/drawings only build the INPUT. */
(function () {
  "use strict";
  var BL = window.BL, NX = BL.NX, NT = BL.NT, YR = 3.6;                       // fixed vertical range of the input plot
  var IN_M = { l: 30, r: 8, t: 6, b: 22 };                                     // input plot margins (also used to map pointer -> data)
  var $ = function (id) { return document.getElementById(id); };
  var state = { offline: false, input: null, preset: "sine", seed: 1, stroke: null, rawStroke: null, drawInfo: null, snapOn: true, result: null, busy: false, tIndex: 100, model: null, modelErr: null, why: null };
  var SHORT = { sine: "Smooth wave", two: "Two waves", bump: "Bump", square: "Square", random: "Random mix", tall: "Too tall", noise: "Noise", wiggly: "Very wiggly" };

  function cssv(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || "#888"; }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function card2(k, v, s) { var c = el("div", "card2"); c.appendChild(el("div", "k", k)); c.appendChild(el("div", "v", v)); if (s) c.appendChild(el("div", "s", s)); return c; }

  // ---------- theme ----------
  (function () {
    var r = document.documentElement;
    try { var s = localStorage.getItem("theme"); if (s) r.setAttribute("data-theme", s); } catch (e) { /* storage may be blocked */ }
    $("theme").addEventListener("click", function () {
      var dark = r.getAttribute("data-theme") === "dark" || (!r.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches);
      var n = dark ? "light" : "dark"; r.setAttribute("data-theme", n);
      try { localStorage.setItem("theme", n); } catch (e) { /* ignore */ }
      redrawAll();
    });
  })();

  // ---------- dialogs (all explanations are shown only when asked for) ----------
  function wireDialog(d) { d.addEventListener("click", function (e) { if (e.target === d) d.close(); }); }
  ["dlg", "dlg-advanced", "dlg-why", "dlg-model"].forEach(function (id) { wireDialog($(id)); });
  function openHelp(topic) {
    var t = $("help-" + topic); if (!t) return;
    var frag = t.content.cloneNode(true), h = frag.querySelector("h2");
    $("dlg-title").textContent = h ? h.textContent : "Help"; if (h) h.remove();
    var body = $("dlg-body"); body.textContent = ""; body.appendChild(frag); body.scrollTop = 0; $("dlg").showModal();
  }
  Array.prototype.forEach.call(document.querySelectorAll("[data-help]"), function (b) { b.addEventListener("click", function () { openHelp(b.getAttribute("data-help")); }); });

  // ---------- canvas helpers ----------
  function setup(c) {
    var r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, w = Math.max(60, Math.round(r.width)), h = Math.max(40, Math.round(r.height));
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    var ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h); ctx.font = "11px system-ui, sans-serif"; return { ctx: ctx, w: w, h: h };
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag, step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function tickLabel(v) { return Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : String(parseFloat(v.toPrecision(3))); }
  function placeholder(canvas, text) {
    var s = setup(canvas); s.ctx.fillStyle = cssv("--muted"); s.ctx.textAlign = "center"; s.ctx.textBaseline = "middle"; s.ctx.font = "13px system-ui, sans-serif";
    s.ctx.fillText(text, s.w / 2, s.h / 2);
  }

  function plotLines(canvas, o) {
    var s = setup(canvas), ctx = s.ctx, m = o.m || { l: 38, r: 8, t: 8, b: 32 }; if (s.h < 150 && m.b > 24) m = { l: m.l, r: m.r, t: m.t, b: 24 }; var ml = m.l, mt = m.t, pw = s.w - m.l - m.r, ph = s.h - m.t - m.b;
    var X = function (x) { return ml + (x - o.xmin) / (o.xmax - o.xmin) * pw; }, Y = function (y) { return mt + (1 - (y - o.ymin) / (o.ymax - o.ymin)) * ph; };
    var ink = cssv("--ink"), muted = cssv("--muted"), line = cssv("--line");
    ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.lineWidth = 1; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    niceTicks(o.ymin, o.ymax, o.yticks || 4).forEach(function (v) { ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke(); ctx.fillText(tickLabel(v), ml - 4, Y(v)); });
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    niceTicks(o.xmin, o.xmax, 4).forEach(function (v) { ctx.fillText(tickLabel(v), X(v), mt + ph + 4); });
    ctx.fillStyle = ink; if (o.xlabel && m.b >= 30) ctx.fillText(o.xlabel, ml + pw / 2, mt + ph + 17);
    if (o.ylabel) { ctx.save(); ctx.translate(9, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "top"; ctx.fillText(o.ylabel, 0, -6); ctx.restore(); }
    ctx.strokeStyle = muted; ctx.strokeRect(ml, mt, pw, ph);
    (o.hlines || []).forEach(function (h) { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = muted; ctx.beginPath(); ctx.moveTo(ml, Y(h.y)); ctx.lineTo(ml + pw, Y(h.y)); ctx.stroke(); ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; if (h.label) ctx.fillText(h.label, ml + 4, Y(h.y) - 1); ctx.restore(); });
    o.series.forEach(function (se) {
      ctx.save(); ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip(); ctx.strokeStyle = se.color; ctx.lineWidth = se.width || 2; ctx.setLineDash(se.dash || []); ctx.beginPath();
      for (var i = 0; i < se.x.length; i++) { var px = X(se.x[i]), py = Y(se.y[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke(); ctx.restore();
    });
    if (o.legend) { var lx = ml + pw - 6; ctx.textAlign = "right"; ctx.textBaseline = "top"; o.series.forEach(function (se, i) { if (!se.label) return; ctx.fillStyle = se.color; ctx.fillRect(lx - ctx.measureText(se.label).width - 15, mt + 5 + i * 14 + 2, 9, 9); ctx.fillStyle = ink; ctx.fillText(se.label, lx, mt + 5 + i * 14); }); }
  }

  function drawHeatmap(canvas, flat, vmin, vmax, cmap, name) {
    var s = setup(canvas), ctx = s.ctx, small = s.h < 215, ml = 26, mr = 6, mt = 4, mb = small ? 40 : 52, pw = s.w - ml - mr, ph = s.h - mt - mb;
    var off = document.createElement("canvas"); off.width = NX; off.height = NT;
    off.getContext("2d").putImageData(new ImageData(BL.heatmapRGBA(flat, NT, NX, vmin, vmax, cmap), NX, NT), 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(off, ml, mt, pw, ph);
    var ink = cssv("--ink"), muted = cssv("--muted");
    ctx.strokeStyle = muted; ctx.strokeRect(ml, mt, pw, ph); ctx.fillStyle = muted; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    [0, 1, 2].forEach(function (t) { ctx.fillText(String(t), ml - 4, mt + ph - t / 2 * ph); });
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    [0, 0.5, 1].forEach(function (x) { ctx.fillText(String(x), ml + x * pw, mt + ph + 3); });
    ctx.fillStyle = ink; ctx.fillText(small ? "position x (time up)" : "position x  (time t goes up)", ml + pw / 2, mt + ph + (small ? 13 : 15));
    var cy = mt + ph + (small ? 25 : 29), ch = small ? 6 : 8; for (var i = 0; i < pw; i++) { var c = BL.colorAt(cmap, i / (pw - 1)); ctx.fillStyle = "rgb(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + ")"; ctx.fillRect(ml + i, cy, 1, ch); }
    ctx.strokeStyle = muted; ctx.strokeRect(ml, cy, pw, ch); ctx.fillStyle = muted; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText(BL.fmt(vmin, 2), ml, cy + ch + 2); ctx.textAlign = "right"; ctx.fillText(BL.fmt(vmax, 2), ml + pw, cy + ch + 2);
    if (cmap === "rdbu") { ctx.textAlign = "center"; ctx.fillText("0", ml + pw / 2, cy + ch + 2); }
    canvas.setAttribute("aria-label", name + ". Colour scale from " + BL.fmt(vmin, 2) + " to " + BL.fmt(vmax, 2) + ". Position across, time up from 0 to 2.");
  }

  // ---------- starting wave: presets, drawing, advanced ----------
  function thumb(canvas, u) {            // all thumbnails share one vertical scale (+-1.5), so "too tall" really looks too tall
    var dpr = window.devicePixelRatio || 1, w = 60, h = 30; canvas.width = w * dpr; canvas.height = h * dpr;
    var ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip();
    ctx.strokeStyle = cssv("--fno"); ctx.lineWidth = 1.5; ctx.beginPath();
    for (var i = 0; i < u.length; i++) { var x = i / (u.length - 1) * (w - 4) + 2, y = h / 2 - (u[i] / 1.5) * (h / 2 - 2); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
    ctx.stroke();
  }
  function buildPresets() {
    function add(box, key) {
      var p = BL.PRESETS[key], b = el("button", "pre"); b.type = "button"; b.setAttribute("data-key", key); b.setAttribute("aria-pressed", "false");
      b.title = p.label + ". " + p.note; b.setAttribute("aria-label", p.label);
      var c = el("canvas"); c.setAttribute("aria-hidden", "true"); b.appendChild(c); b.appendChild(el("span", null, SHORT[key]));
      b.addEventListener("click", function () { choosePreset(key, key === state.preset && key === "random"); }); box.appendChild(b); thumb(c, p.make(3));
    }
    ["sine", "two", "bump", "square", "random"].forEach(function (k) { add($("presets"), k); });
    ["tall", "noise", "wiggly"].forEach(function (k) { add($("presets-stress"), k); });
  }
  function markPreset(key) { Array.prototype.forEach.call(document.querySelectorAll(".pre"), function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-key") === key ? "true" : "false"); }); }
  function choosePreset(key, reroll) {
    if (reroll) state.seed = (state.seed % 1000000) + 1;
    state.preset = key; state.stroke = state.rawStroke = state.drawInfo = null; markPreset(key);
    var p = BL.PRESETS[key]; setInput(p.make(state.seed), key === "random" ? SHORT[key] + " #" + state.seed : SHORT[key], key);
  }
  function setInput(u, label, kind) {
    state.input = { u: u, label: label, kind: kind }; updateInputNote(); drawInput(); $("run").disabled = state.busy || state.offline;
  }
  function updateInputNote() {
    var i = state.input, warn = $("input-warn"), note = $("input-note"); warn.hidden = true; warn.textContent = "";
    if (!i) { note.textContent = "Tip: draw on the plot above to make your own wave."; return; }
    var m = BL.maxAbs(i.u);
    if (state.drawInfo) {
      var d = state.drawInfo;
      note.textContent = d.snapOn ? "Your drawing, smoothed (changed by " + (100 * d.snap.change).toFixed(0) + "%). Highest point: " + BL.fmt(m, 2) + "." : "Your drawing as drawn. Highest point: " + BL.fmt(m, 2) + ".";
      warn.hidden = false; warn.textContent = d.snap.outside ? "This drawing is outside what the AI was trained on, so expect bigger differences." : "Hand-drawn waves are usually harder for the AI: check the solver next to it.";
    } else {
      note.textContent = i.label + " · highest point " + BL.fmt(m, 2) + ". Draw on the plot to replace it.";
      if (m > BL.getTrainMaxAbs()) { warn.hidden = false; warn.textContent = "Higher than anything the AI was trained on (limit about " + BL.fmt(BL.getTrainMaxAbs(), 2) + "). Expect a warning and a wrong forecast."; }
    }
  }
  function drawInput() {
    var c = $("input-plot"), tm = BL.getTrainMaxAbs(), x = BL.grid(NX), series = [];
    if (state.input) series.push({ x: x, y: state.input.u, color: cssv("--fno"), width: 2.5 });
    var pts = state.stroke || state.rawStroke; if (pts) series.push({ x: pts.map(function (p) { return p.x; }), y: pts.map(function (p) { return p.y; }), color: cssv("--solver"), width: 1.5 });
    plotLines(c, { xmin: 0, xmax: 1, ymin: -YR, ymax: YR, xlabel: "", ylabel: "", m: IN_M, yticks: 4, hlines: [{ y: tm, label: "training limit" }, { y: -tm, label: "" }], series: series });
    c.setAttribute("aria-label", state.input ? "Starting wave: " + state.input.label + ". Highest point " + BL.fmt(BL.maxAbs(state.input.u), 2) + ". You can draw on this plot." : "No starting wave yet. You can draw on this plot.");
  }
  function pointerPos(ev) {
    var c = $("input-plot"), r = c.getBoundingClientRect(), pw = r.width - IN_M.l - IN_M.r, ph = r.height - IN_M.t - IN_M.b;
    var x = (ev.clientX - r.left - IN_M.l) / pw, y = (1 - (ev.clientY - r.top - IN_M.t) / ph) * 2 * YR - YR;
    return { x: Math.min(Math.max(x, 0), 1), y: Math.min(Math.max(y, -YR), YR) };
  }
  function finishStroke() {
    if (!state.stroke) return; var pts = state.stroke; state.stroke = null;
    var raw = BL.pointsToGrid(pts);
    if (!raw) { state.rawStroke = null; $("input-note").textContent = "That stroke was too short. Press and drag from left to right."; drawInput(); return; }
    var snap = BL.snapDrawing(raw); state.rawStroke = pts; state.drawInfo = { raw: raw, snap: snap, snapOn: state.snapOn }; markPreset("");
    setInput(state.snapOn ? snap.u : raw, state.snapOn ? "Your drawing (smoothed)" : "Your drawing", "drawn");
  }
  (function wireDraw() {
    var c = $("input-plot");
    c.addEventListener("pointerdown", function (ev) { ev.preventDefault(); try { c.setPointerCapture(ev.pointerId); } catch (e) { /* capture is optional */ } state.stroke = [pointerPos(ev)]; state.rawStroke = null; drawInput(); });
    c.addEventListener("pointermove", function (ev) { if (!state.stroke) return; state.stroke.push(pointerPos(ev)); drawInput(); });
    c.addEventListener("pointerup", finishStroke); c.addEventListener("pointercancel", finishStroke);
  })();

  // advanced dialog
  function buildMix() {
    var g = $("mixgrid"), defaults = [[1, 1, 0], [0.5, 3, 1], [0, 2, 0]];
    for (var i = 0; i < 3; i++) {
      var fs = el("fieldset"); fs.appendChild(el("legend", null, "Wave " + (i + 1)));
      [["amp", "Amplitude (height)", defaults[i][0], "0.1", "-6", "6"], ["k", "Waves (1-60)", defaults[i][1], "1", "1", "60"], ["ph", "Phase (shift)", defaults[i][2], "0.1", "-7", "7"]].forEach(function (f) {
        var row = el("div", "row"), id = "mix-" + f[0] + i, lb = el("label", null, f[1]); lb.setAttribute("for", id);
        var inp = el("input"); inp.type = "number"; inp.id = id; inp.step = f[3]; inp.min = f[4]; inp.max = f[5]; inp.value = f[2]; row.appendChild(lb); row.appendChild(inp); fs.appendChild(row);
      });
      g.appendChild(fs);
    }
  }
  $("advanced").addEventListener("click", function () { $("dlg-advanced").showModal(); });
  $("snap").addEventListener("change", function () {
    state.snapOn = $("snap").checked;
    if (state.drawInfo) { state.drawInfo.snapOn = state.snapOn; setInput(state.snapOn ? state.drawInfo.snap.u : state.drawInfo.raw, state.snapOn ? "Your drawing (smoothed)" : "Your drawing", "drawn"); }
  });
  $("apply-mix").addEventListener("click", function () {
    var comps = [], mean = parseFloat($("mix-mean").value) || 0, msg = $("mix-msg");
    for (var i = 0; i < 3; i++) {
      var a = parseFloat($("mix-amp" + i).value), k = Math.round(parseFloat($("mix-k" + i).value)), p = parseFloat($("mix-ph" + i).value);
      if (!isFinite(a) || !isFinite(k) || !isFinite(p)) { msg.textContent = "Wave " + (i + 1) + ": please fill in all three numbers."; return; }
      if (k < 1 || k > 60) { msg.textContent = "Wave " + (i + 1) + ": waves must be between 1 and 60."; return; }
      if (a !== 0) comps.push({ amp: a, k: k, phase: p });
    }
    if (!comps.length && mean === 0) { msg.textContent = "Everything is zero, so nothing would happen. Give at least one wave an amplitude."; return; }
    msg.textContent = ""; state.stroke = state.rawStroke = state.drawInfo = null; markPreset(""); setInput(BL.sineMix(comps, mean), "Your sine mix", "mix"); $("dlg-advanced").close();
  });
  $("apply-paste").addEventListener("click", function () {
    var r = BL.parseNumbers($("paste").value), msg = $("paste-msg");
    if (r.error) { msg.textContent = r.error; return; }
    msg.textContent = ""; state.stroke = state.rawStroke = state.drawInfo = null; markPreset(""); setInput(r.u, "Pasted numbers", "paste"); $("dlg-advanced").close();
  });

  // ---------- status strip ----------
  function setStrip(kind, html, why) {
    var s = $("strip"); s.className = "strip " + kind; $("strip-icon").textContent = kind === "good" ? "✓" : (kind === "bad" || kind === "warn") ? "!" : kind === "busy" ? "" : "i";
    var t = $("strip-text"); t.textContent = ""; if (typeof html === "string") t.textContent = html; else t.appendChild(html);
    state.why = why || null; $("strip-why").hidden = !why;
  }
  $("strip-why").addEventListener("click", function () { if (!state.why) return; var b = $("why-body"); b.textContent = ""; b.appendChild(state.why()); $("dlg-why").showModal(); });

  // ---------- running ----------
  function setBusy(b) { state.busy = b; $("run").disabled = b || !state.input || state.offline; $("run").setAttribute("aria-busy", b ? "true" : "false"); }
  function errMsg(status, data) {
    var d = data && data.detail;
    if (status === 422 && Array.isArray(d) && d.length) return "The server rejected this wave: " + d.map(function (e) { return e.msg; }).join("; ");
    if (status === 503) return "The AI model is not available on the server right now.";
    return "The server returned an error (HTTP " + status + ").";
  }
  function clearResults() {
    state.result = null; $("nums").textContent = ""; var a = el("dt", null, "Press Predict"), b = el("dd", null, "to see how close the AI is, and how fast."); $("nums").appendChild(a); $("nums").appendChild(b);
    ["dl-csv", "dl-json"].forEach(function (i) { $(i).disabled = true; }); $("t-slider").disabled = true; $("t-readout").textContent = "t = 1.00";
    [["hm-fno", "AI forecast map. Press Predict to fill it."], ["hm-solver", "Classic solver map. Press Predict to fill it."], ["hm-error", "Difference map. Press Predict to fill it."],
     ["profile-plot", "Snapshot. Press Predict to fill it."], ["err-time-plot", "Gap over time. Press Predict to fill it."]].forEach(function (p) { placeholder($(p[0]), "Press Predict"); $(p[0]).setAttribute("aria-label", p[1]); });
  }
  function validate(d) {
    function mat(m) { return Array.isArray(m) && m.length === NT && m.every(function (r) { return Array.isArray(r) && r.length === NX; }); }
    return d && d.fno && mat(d.fno.u) && d.solver && mat(d.solver.u) && Array.isArray(d.t) && d.t.length === NT && Array.isArray(d.x) && d.x.length === NX &&
      d.ood && typeof d.ood.is_ood === "boolean" && Array.isArray(d.solver.rel_l2_fno_vs_solver_per_t) && d.solver.rel_l2_fno_vs_solver_per_t.length === NT;
  }
  async function run() {
    if (!state.input) { setStrip("warn", "Pick or draw a starting wave first."); return; }
    clearResults(); setBusy(true); setStrip("busy", "Running the AI and the classic solver on the server… about a second.");
    var ctrl = new AbortController(), to = setTimeout(function () { ctrl.abort(); }, 120000), t0 = performance.now(), sent = new Float64Array(state.input.u), label = state.input.label;
    try {
      var r = await fetch("/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ u0: Array.from(sent), include_solver: true }), signal: ctrl.signal });
      var txt = await r.text(), data = null; try { data = JSON.parse(txt); } catch (e) { data = null; }
      if (!r.ok) throw new Error(errMsg(r.status, data));
      if (!validate(data)) throw new Error("The server's reply was not in the expected format, so nothing is shown.");
      var fno = BL.flatten(data.fno.u), solver = BL.flatten(data.solver.u);
      if (!fno || !solver || !Array.prototype.every.call(fno, isFinite) || !Array.prototype.every.call(solver, isFinite)) throw new Error("The server's reply contained invalid numbers, so nothing is shown.");
      var err = new Float64Array(fno.length), emax = 0; for (var i = 0; i < fno.length; i++) { err[i] = Math.abs(fno[i] - solver[i]); if (err[i] > emax) emax = err[i]; }
      state.result = { resp: data, fno: fno, solver: solver, err: err, emax: emax, rtMs: performance.now() - t0, at: new Date(), u0: sent, label: label };
      renderResults();
    } catch (e) {
      clearResults(); setStrip("bad", e.name === "AbortError" ? "The request took too long and was cancelled." : (e instanceof TypeError ? "Could not reach the server. Check your connection and try again." : e.message));
    } finally { clearTimeout(to); setBusy(false); }
  }

  // ---------- results ----------
  function para(text) { return el("p", null, text); }
  function renderResults() {
    var R = state.result, d = R.resp, rel = d.solver.rel_l2_fno_vs_solver_whole, pct = (100 * rel).toFixed(1);
    var sc = d.self_check, scFlag = !!(sc && sc.flagged), flagged = d.ood.is_ood || scFlag;
    if (d.ood.is_ood) {
      setStrip("bad", "Unfamiliar starting wave: do not trust the AI forecast here. The classic solver is still reliable.", function () {
        var w = el("div"); w.appendChild(para("The AI was trained on a particular family of waves. This one is outside it, so the AI's answer may be wrong without looking wrong:"));
        var ul = el("ul"); d.ood.reasons.forEach(function (r) { ul.appendChild(el("li", null, r)); }); w.appendChild(ul);
        w.appendChild(para("In this run the AI and the solver differ by " + pct + "%. Compare the two maps and believe the solver.")); return w; });
    } else if (scFlag) {
      setStrip("bad", "The AI's forecast fails its own sanity checks, so it is probably wrong here. Use the classic solver's answer.", function () {
        var w = el("div"); w.appendChild(para("After every forecast the page runs three quick checks on the AI's own answer, without needing the solver: does it reproduce the wave you gave it, does it keep the total amount of “stuff” constant, and does it obey the equation? This forecast failed:"));
        var ul = el("ul"); sc.reasons.forEach(function (r) { ul.appendChild(el("li", null, r.charAt(0).toUpperCase() + r.slice(1) + ".")); }); w.appendChild(ul);
        w.appendChild(para("In this run the AI and the solver differ by " + pct + "%. In our tests these checks caught about half of the cases where the AI was more than 6% off, with very few false alarms. They can miss problems, so keep comparing with the solver.")); return w; });
    } else if (rel > 0.06) {
      setStrip("warn", "The AI and the solver disagree by " + pct + "%. Treat the AI forecast with caution.", function () {
        var w = el("div"); w.appendChild(para("No warning was raised about the starting wave and the AI's forecast passed its own sanity checks, but it differs from the solver's by " + pct + "% over the whole map. For waves like its training data this is typically 1–3%."));
        w.appendChild(para("Hand-drawn waves, square waves and very sharp fronts often cause this. The solver is the reference: believe it over the AI.")); return w; });
    } else {
      setStrip("good", "The AI matches the classic solver within " + pct + "%. No warning was raised. (That is not a guarantee for every wave.)");
    }
    var nums = $("nums"); nums.textContent = "";
    [["Best answer", (flagged || rel > 0.06) ? "Classic solver" : "AI is fine"], ["Difference", pct + " %"], ["AI took", BL.fmt(d.fno.seconds * 1000, 1) + " ms"], ["Solver took", BL.fmt(d.solver.seconds, 2) + " s"], ["Round trip", BL.fmt(R.rtMs, 0) + " ms"],
     ["Mass drift AI", BL.fmt(d.fno.mass_drift, 4)], ["Mass drift solver", BL.fmt(d.solver.mass_drift, 2)]].forEach(function (p) { nums.appendChild(el("dt", null, p[0])); nums.appendChild(el("dd", null, p[1])); });
    ["dl-csv", "dl-json"].forEach(function (i) { $(i).disabled = false; }); $("t-slider").disabled = false;
    requestAnimationFrame(drawResultPlots);
  }
  function drawResultPlots() {
    var R = state.result; if (!R) return;
    var mx = 0; for (var i = 0; i < R.fno.length; i++) mx = Math.max(mx, Math.abs(R.fno[i]), Math.abs(R.solver[i])); mx = mx || 1;
    drawHeatmap($("hm-fno"), R.fno, -mx, mx, "rdbu", "AI forecast map");
    drawHeatmap($("hm-solver"), R.solver, -mx, mx, "rdbu", "Classic solver map");
    drawHeatmap($("hm-error"), R.err, 0, R.emax || 1e-12, "viridis", "Difference map, size of the gap between AI and solver");
    drawProfile();
    var per = R.resp.solver.rel_l2_fno_vs_solver_per_t;
    plotLines($("err-time-plot"), { xmin: 0, xmax: 2, ymin: 0, ymax: Math.max.apply(null, per) * 1.1 || 1e-6, xlabel: "time t", ylabel: "gap (relative)", m: { l: 44, r: 8, t: 8, b: 32 }, series: [{ x: R.resp.t, y: per, color: cssv("--fno") }] });
    $("err-time-plot").setAttribute("aria-label", "Gap between AI and solver over time, from " + BL.fmt(per[0], 3) + " at t=0 to " + BL.fmt(per[NT - 1], 3) + " at t=2.");
  }
  function drawProfile() {
    var R = state.result; if (!R) return;
    var k = state.tIndex, t = R.resp.t[k], x = R.resp.x, f = R.fno.subarray(k * NX, (k + 1) * NX), s = R.solver.subarray(k * NX, (k + 1) * NX);
    var lo = Infinity, hi = -Infinity; for (var i = 0; i < R.fno.length; i++) { lo = Math.min(lo, R.fno[i], R.solver[i]); hi = Math.max(hi, R.fno[i], R.solver[i]); }
    var pad = (hi - lo) * 0.08 || 0.1;
    plotLines($("profile-plot"), { xmin: 0, xmax: 1, ymin: lo - pad, ymax: hi + pad, xlabel: "position x", ylabel: "wave height u", legend: true, m: { l: 44, r: 8, t: 8, b: 32 },
      series: [{ x: x, y: s, color: cssv("--solver"), width: 3, label: "Solver" }, { x: x, y: f, color: cssv("--fno"), width: 2, dash: [6, 3], label: "AI" }] });
    var rl = BL.relL2(f, s); $("t-readout").textContent = "t = " + t.toFixed(2);
    $("profile-plot").setAttribute("aria-label", "Snapshot at t = " + t.toFixed(2) + ": AI and solver wave height along x. Gap at this moment " + (100 * rl).toFixed(1) + " percent.");
    $("t-slider").setAttribute("aria-valuetext", "t = " + t.toFixed(2) + ", gap " + (100 * rl).toFixed(1) + " percent");
  }
  $("t-slider").addEventListener("input", function () { state.tIndex = parseInt(this.value, 10); drawProfile(); });
  $("run").addEventListener("click", run);

  // ---------- downloads ----------
  function download(name, text, type) { var a = el("a"), url = URL.createObjectURL(new Blob([text], { type: type })); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(url); }, 1000); }
  $("dl-json").addEventListener("click", function () { var R = state.result; if (!R) return; var d = R.resp; download("burgers_result.json", JSON.stringify({ created: R.at.toISOString(), input_label: R.label, u0: Array.from(R.u0), x: d.x, t: d.t, fno_u: d.fno.u, solver_u: d.solver.u, ood: d.ood, model: d.model, equation: d.equation, fno_vs_solver_relL2: d.solver.rel_l2_fno_vs_solver_whole }), "application/json"); });
  $("dl-csv").addEventListener("click", function () { var R = state.result; if (!R) return; download("burgers_result.csv", BL.csvFromResponse(R.resp, R.fno, R.solver), "text/csv"); });

  // ---------- model info + health ----------
  function showOffline(why) {
    state.offline = true; $("run").disabled = true;
    var msg = location.protocol === "file:"
      ? "You opened this file straight from disk. The AI runs on the server, so please open http://127.0.0.1:8000/ instead (start the server first if it is not running)."
      : "This page cannot reach the Burgers server (" + why + "), so Predict is switched off. Start it with: python -m uvicorn app.main:app --port 8000, then open http://127.0.0.1:8000/";
    setStrip("bad", msg);
  }
  async function loadMeta() {
    var b = $("api-badge");
    try {
      var h = await fetch("/health");
      if (h.ok) { b.textContent = "Server: ready"; b.className = "badge ok"; }
      else if (h.status === 404 || h.status === 405) { b.textContent = "Server: not connected"; b.className = "badge bad"; showOffline("HTTP " + h.status); return; }
      else { b.textContent = "Server: not ready"; b.className = "badge bad"; }
    } catch (e) { b.textContent = "Server: unreachable"; b.className = "badge bad"; showOffline("network error"); return; }
    try {
      var r = await fetch("/model-info"); if (!r.ok) throw new Error("HTTP " + r.status); state.model = await r.json(); BL.setTrainMaxAbs(state.model.runtime.ood_train_range.max_abs.max); updateInputNote(); drawInput();
    } catch (e) { state.modelErr = e.message; }
  }
  $("btn-model").addEventListener("click", function () {
    var box = $("model-body"); box.textContent = "";
    if (!state.model) { box.appendChild(el("p", "warn", "Model information is unavailable" + (state.modelErr ? " (" + state.modelErr + ")" : "") + ".")); $("dlg-model").showModal(); return; }
    var m = state.model, mt = m.metrics, g = el("div", "cards");
    g.appendChild(card2("Model", m.name, m.type)); g.appendChild(card2("Size", m.parameters.toLocaleString() + " parameters", "weights sha256 " + m.runtime.served_sha256.slice(0, 12) + "…"));
    g.appendChild(card2("Test error of the AI", BL.fmt(mt.fno_test_mean_3_seeds, 4) + " ± " + mt.fno_test_sd_3_seeds.toFixed(4), "relative L2 on 1,000 unseen simulations; mean ± spread of 3 training runs"));
    g.appendChild(card2("This deployed copy (seed 1)", BL.fmt(mt.deployed_seed1_test, 4), "one of the 3 runs, not the best-of"));
    g.appendChild(card2("Simpler AI for comparison (U-Net)", BL.fmt(mt.unet_test_mean_3_seeds, 4), "the FNO is about 3.4× more accurate"));
    g.appendChild(card2("Near vs away from shocks", BL.fmt(mt.near_shock_pooled_relL2_test_seed1, 3) + " vs " + BL.fmt(mt.away_from_shock_pooled_relL2_test_seed1, 3), "error next to a shock vs elsewhere"));
    g.appendChild(card2("Mass drift (test)", BL.fmt(mt.mass_drift_pred_test_seed1, 4) + " vs " + BL.fmt(mt.mass_drift_truth_test, 2), "AI vs reference data"));
    g.appendChild(card2("Trained on", m.training.dataset, m.training.train_samples + " train / " + m.training.val + " validation / " + m.training.test + " test; " + m.training.epochs + " epochs"));
    box.appendChild(g); box.appendChild(el("p", "note", "Equation used: " + m.equation + ". Run on the server's CPU with " + m.runtime.threads + " threads (torch " + m.runtime.torch + ")."));
    $("dlg-model").showModal();
  });

  // ---------- init ----------
  function redrawAll() { drawInput(); if (state.result) drawResultPlots(); else clearResults(); Array.prototype.forEach.call(document.querySelectorAll(".pre"), function (b) { thumb(b.querySelector("canvas"), BL.PRESETS[b.getAttribute("data-key")].make(3)); }); }
  var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(redrawAll, 100); });
  (function init() {
    var q = new URLSearchParams(location.search);
    if (q.get("theme") === "light" || q.get("theme") === "dark") document.documentElement.setAttribute("data-theme", q.get("theme"));
    if (q.get("seed")) state.seed = parseInt(q.get("seed"), 10) || 1;
    buildPresets(); buildMix();
    var key = q.get("preset") && BL.PRESETS[q.get("preset")] ? q.get("preset") : "sine";
    clearResults(); choosePreset(key, false); loadMeta();
    if (q.get("run") === "1") run();
  })();
})();
