/* Page logic. Every result shown comes from a successful POST /predict response; presets/drawings only build the INPUT. */
(function () {
  "use strict";
  var BL = window.BL, NX = BL.NX, NT = BL.NT;
  var $ = function (id) { return document.getElementById(id); };
  var state = { input: null, drawing: null, pasted: null, seed: 1, result: null, busy: false, tIndex: 100 };

  function cssv(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888"; }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function card(k, v, s) { var c = el("div", "card"); c.appendChild(el("div", "k", k)); c.appendChild(el("div", "v", v)); if (s) c.appendChild(el("div", "s", s)); return c; }

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

  // ---------- canvas helpers ----------
  function setup(c) {
    var r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    var w = Math.max(220, Math.round(r.width)), h = Math.max(140, Math.round(r.height));
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    var ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    ctx.font = "12px system-ui, sans-serif"; return { ctx: ctx, w: w, h: h };
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag, step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
    for (var v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function tickLabel(v) { return Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : String(parseFloat(v.toPrecision(3))); }

  function plotLines(canvas, o) {
    var s = setup(canvas), ctx = s.ctx, ml = 46, mr = 12, mt = 12, mb = 38, pw = s.w - ml - mr, ph = s.h - mt - mb;
    var X = function (x) { return ml + (x - o.xmin) / (o.xmax - o.xmin) * pw; }, Y = function (y) { return mt + (1 - (y - o.ymin) / (o.ymax - o.ymin)) * ph; };
    var ink = cssv("--ink"), muted = cssv("--muted"), line = cssv("--line");
    ctx.strokeStyle = line; ctx.fillStyle = muted; ctx.lineWidth = 1; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    niceTicks(o.ymin, o.ymax, 5).forEach(function (v) { ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke(); ctx.fillText(tickLabel(v), ml - 6, Y(v)); });
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    niceTicks(o.xmin, o.xmax, 5).forEach(function (v) { ctx.fillText(tickLabel(v), X(v), mt + ph + 6); });
    ctx.fillStyle = ink; ctx.fillText(o.xlabel || "", ml + pw / 2, mt + ph + 22);
    ctx.save(); ctx.translate(12, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "top"; ctx.fillText(o.ylabel || "", 0, -8); ctx.restore();
    ctx.strokeStyle = muted; ctx.strokeRect(ml, mt, pw, ph);
    (o.hlines || []).forEach(function (h) { ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = muted; ctx.beginPath(); ctx.moveTo(ml, Y(h.y)); ctx.lineTo(ml + pw, Y(h.y)); ctx.stroke(); ctx.fillStyle = muted; ctx.textAlign = "left"; ctx.textBaseline = "bottom"; ctx.fillText(h.label, ml + 4, Y(h.y) - 2); ctx.restore(); });
    o.series.forEach(function (se) {
      ctx.save(); ctx.beginPath(); ctx.rect(ml, mt, pw, ph); ctx.clip();
      ctx.strokeStyle = se.color; ctx.lineWidth = se.width || 2; ctx.setLineDash(se.dash || []); ctx.beginPath();
      for (var i = 0; i < se.x.length; i++) { var px = X(se.x[i]), py = Y(se.y[i]); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }
      ctx.stroke(); ctx.restore();
    });
    if (o.legend) { var lx = ml + pw - 8; ctx.textAlign = "right"; ctx.textBaseline = "top"; o.series.forEach(function (se, i) { if (!se.label) return; ctx.fillStyle = se.color; ctx.fillRect(lx - ctx.measureText(se.label).width - 18, mt + 6 + i * 16 + 2, 10, 10); ctx.fillStyle = ink; ctx.fillText(se.label, lx, mt + 6 + i * 16); }); }
    return { X: X, Y: Y, ml: ml, mt: mt, pw: pw, ph: ph };
  }

  function drawHeatmap(canvas, flat, vmin, vmax, cmap, name) {
    var s = setup(canvas), ctx = s.ctx, ml = 40, mr = 10, mt = 8, mb = 70, pw = s.w - ml - mr, ph = s.h - mt - mb;
    var off = document.createElement("canvas"); off.width = NX; off.height = NT;
    off.getContext("2d").putImageData(new ImageData(BL.heatmapRGBA(flat, NT, NX, vmin, vmax, cmap), NX, NT), 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(off, ml, mt, pw, ph);
    var ink = cssv("--ink"), muted = cssv("--muted");
    ctx.strokeStyle = muted; ctx.strokeRect(ml, mt, pw, ph); ctx.fillStyle = muted; ctx.textBaseline = "middle"; ctx.textAlign = "right";
    [0, 0.5, 1, 1.5, 2].forEach(function (t) { ctx.fillText(String(t), ml - 5, mt + ph - t / 2 * ph); });
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    [0, 0.5, 1].forEach(function (x) { ctx.fillText(String(x), ml + x * pw, mt + ph + 5); });
    ctx.fillStyle = ink; ctx.fillText("x", ml + pw / 2, mt + ph + 18);
    ctx.save(); ctx.translate(11, mt + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "top"; ctx.fillText("t", 0, -6); ctx.restore();
    var cy = mt + ph + 36, ch = 12; for (var i = 0; i < pw; i++) { var c = BL.colorAt(cmap, i / (pw - 1)); ctx.fillStyle = "rgb(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + ")"; ctx.fillRect(ml + i, cy, 1, ch); }
    ctx.strokeStyle = muted; ctx.strokeRect(ml, cy, pw, ch); ctx.fillStyle = muted; ctx.textBaseline = "top";
    ctx.textAlign = "left"; ctx.fillText(BL.fmt(vmin, 2), ml, cy + ch + 3);
    ctx.textAlign = "right"; ctx.fillText(BL.fmt(vmax, 2), ml + pw, cy + ch + 3);
    if (cmap === "rdbu") { ctx.textAlign = "center"; ctx.fillText("0", ml + pw / 2, cy + ch + 3); }
    canvas.setAttribute("aria-label", name + ". Colour scale from " + BL.fmt(vmin, 2) + " to " + BL.fmt(vmax, 2) + ". x across, time up from 0 to 2.");
  }

  // ---------- inputs ----------
  function buildMixUI() {
    var g = $("mixgrid"), defaults = [[1, 1, 0], [0.5, 3, 1], [0, 2, 0]];
    for (var i = 0; i < 3; i++) {
      var fs = el("fieldset"), lg = el("legend", null, "Wave " + (i + 1)); fs.appendChild(lg);
      [["amp", "Amplitude", defaults[i][0], "0.1", "-6", "6"], ["k", "Waves", defaults[i][1], "1", "1", "60"], ["ph", "Phase", defaults[i][2], "0.1", "-7", "7"]].forEach(function (f) {
        var row = el("div", "row"), id = "mix-" + f[0] + i, lb = el("label", null, f[1]); lb.setAttribute("for", id);
        var inp = el("input"); inp.type = "number"; inp.id = id; inp.step = f[3]; inp.min = f[4]; inp.max = f[5]; inp.value = f[2];
        row.appendChild(lb); row.appendChild(inp); fs.appendChild(row); inp.addEventListener("input", refreshInput);
      });
      g.appendChild(fs);
    }
    $("mix-mean").addEventListener("input", refreshInput);
  }
  function mixInput() {
    var comps = [], mean = parseFloat($("mix-mean").value) || 0;
    for (var i = 0; i < 3; i++) {
      var a = parseFloat($("mix-amp" + i).value), k = Math.round(parseFloat($("mix-k" + i).value)), p = parseFloat($("mix-ph" + i).value);
      if (!isFinite(a) || !isFinite(k) || !isFinite(p)) return { error: "Wave " + (i + 1) + ": enter numbers in all three fields." };
      if (k < 1 || k > 60) return { error: "Wave " + (i + 1) + ": waves must be between 1 and 60." };
      if (a !== 0) comps.push({ amp: a, k: k, phase: p });
    }
    if (!comps.length && mean === 0) return { error: "All amplitudes and the mean are zero: the solution would be flat." };
    return { u: BL.sineMix(comps, mean), label: "Sine mix" };
  }
  function activeSource() { return document.querySelector('input[name="src"]:checked').value; }

  function refreshInput() {
    var src = activeSource(), res = null;
    ["preset", "mix", "draw", "paste"].forEach(function (p) { $("panel-" + p).hidden = (p !== src); });
    $("reroll").hidden = !(src === "preset" && $("preset").value === "random");
    if (src === "draw") drawCanvas();   // the canvas has no size while its panel is hidden, so repaint when it appears
    if (src === "preset") { var p = BL.PRESETS[$("preset").value]; $("preset-note").textContent = p.note; res = { u: p.make(state.seed), label: p.label }; }
    else if (src === "mix") res = mixInput();
    else if (src === "draw") res = state.drawing ? { u: state.drawing.snapOn ? state.drawing.snap.u : state.drawing.raw, label: "Your drawing" + (state.drawing.snapOn ? " (snapped)" : " (raw)") } : { error: "Nothing drawn yet." };
    else res = state.pasted ? { u: state.pasted, label: "Pasted numbers" } : { error: "No numbers applied yet." };
    var warn = $("input-warning"); warn.hidden = true; warn.textContent = "";
    if (res && res.u) {
      state.input = { u: res.u, label: res.label };
      var m = BL.maxAbs(res.u), mean = res.u.reduce(function (a, b) { return a + b; }, 0) / NX;
      $("input-summary").textContent = res.label + ": max |u| = " + BL.fmt(m, 2) + ", mean = " + BL.fmt(mean, 2) + ". " + NX + " points.";
      if (m > BL.getTrainMaxAbs()) { warn.hidden = false; warn.textContent = "Heads up: max |u| = " + BL.fmt(m, 2) + " is above the largest value seen in training (" + BL.fmt(BL.getTrainMaxAbs(), 2) + "). The server will flag this and the result may be wrong."; }
    } else { state.input = null; $("input-summary").textContent = (res && res.error) || ""; }
    drawInputPlot();
    $("run").disabled = state.busy || !state.input;
  }
  function drawInputPlot() {
    var c = $("input-plot"), x = BL.grid(NX), tm = BL.getTrainMaxAbs();
    if (!state.input) { var s0 = setup(c); s0.ctx.fillStyle = cssv("--muted"); s0.ctx.textAlign = "center"; s0.ctx.fillText("No input yet", s0.w / 2, s0.h / 2); c.setAttribute("aria-label", "No initial condition chosen yet"); return; }
    var yl = Math.max(1.2, 1.15 * BL.maxAbs(state.input.u)), hl = yl >= tm ? [{ y: tm, label: "training max" }, { y: -tm, label: "" }] : [];
    plotLines(c, { xmin: 0, xmax: 1, ymin: -yl, ymax: yl, xlabel: "x", ylabel: "u(x, 0)", hlines: hl, series: [{ x: x, y: state.input.u, color: cssv("--fno") }] });
    c.setAttribute("aria-label", "Initial condition, " + state.input.label + ". Maximum absolute value " + BL.fmt(BL.maxAbs(state.input.u), 2) + ".");
  }

  // ---------- drawing ----------
  var DRAW_Y = 3.5, stroke = null;
  function drawCanvas() {
    var c = $("draw-canvas"), s = setup(c), ctx = s.ctx, ml = 30, mt = 6, pw = s.w - ml - 6, ph = s.h - mt - 6, tm = BL.getTrainMaxAbs();
    var X = function (x) { return ml + x * pw; }, Y = function (y) { return mt + (1 - (y + DRAW_Y) / (2 * DRAW_Y)) * ph; };
    ctx.strokeStyle = cssv("--line"); ctx.fillStyle = cssv("--muted"); ctx.textAlign = "right"; ctx.textBaseline = "middle";
    [-3, -2, -1, 0, 1, 2, 3].forEach(function (v) { ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke(); ctx.fillText(String(v), ml - 4, Y(v)); });
    ctx.save(); ctx.setLineDash([5, 4]); ctx.strokeStyle = cssv("--muted"); [tm, -tm].forEach(function (v) { ctx.beginPath(); ctx.moveTo(ml, Y(v)); ctx.lineTo(ml + pw, Y(v)); ctx.stroke(); }); ctx.restore();
    ctx.strokeRect(ml, mt, pw, ph);
    var d = state.drawing, pts = stroke || (d && d.points);
    if (!pts) { ctx.textAlign = "center"; ctx.fillText("Draw here: one stroke, left to right", ml + pw / 2, mt + ph / 2); }
    else { ctx.strokeStyle = cssv("--solver"); ctx.lineWidth = 2; ctx.beginPath(); pts.forEach(function (p, i) { if (i) ctx.lineTo(X(p.x), Y(p.y)); else ctx.moveTo(X(p.x), Y(p.y)); }); ctx.stroke(); }
    if (d && !stroke && d.snapOn) { var g = BL.grid(NX); ctx.strokeStyle = cssv("--fno"); ctx.lineWidth = 2; ctx.beginPath(); for (var i = 0; i < NX; i++) { if (i) ctx.lineTo(X(g[i]), Y(d.snap.u[i])); else ctx.moveTo(X(g[i]), Y(d.snap.u[i])); } ctx.stroke(); }
    return { ml: ml, mt: mt, pw: pw, ph: ph };
  }
  function layoutOf(c) { var r = c.getBoundingClientRect(), w = Math.max(220, Math.round(r.width)), h = Math.max(140, Math.round(r.height)); return { ml: 30, mt: 6, pw: w - 36, ph: h - 12 }; }
  function pointerPos(ev) {
    var c = $("draw-canvas"), r = c.getBoundingClientRect(), L = layoutOf(c);
    var x = (ev.clientX - r.left - L.ml) / L.pw, y = (1 - (ev.clientY - r.top - L.mt) / L.ph) * 2 * DRAW_Y - DRAW_Y;
    return { x: Math.min(Math.max(x, 0), 1), y: Math.min(Math.max(y, -DRAW_Y), DRAW_Y) };
  }
  function finishStroke() {
    if (!stroke) return; var pts = stroke; stroke = null;
    var raw = BL.pointsToGrid(pts);
    if (!raw) { $("draw-msg").textContent = "That stroke was too short. Draw a longer stroke from left to right."; state.drawing = null; refreshInput(); drawCanvas(); return; }
    state.drawing = { points: pts, raw: raw, snap: BL.snapDrawing(raw), snapOn: $("snap").checked };
    updateDrawMsg(); refreshInput(); drawCanvas();
  }
  function updateDrawMsg() {
    var d = state.drawing; if (!d) { $("draw-msg").textContent = ""; return; }
    var pct = (100 * d.snap.change).toFixed(0), msg;
    if (d.snapOn) msg = "Snapped: smoothed to at most " + BL.SNAP_KMAX + " waves (changed your drawing by " + pct + "%)." + (d.snap.clamped ? " Values beyond the training range were clipped." : "");
    else msg = "Snapping is off: your raw drawing is sent as is.";
    if (d.snap.outside) msg += d.snapOn ? " WARNING: your drawing has more detail or height than the model's training family covers, so the model is being asked about something close to, but not exactly, what you drew." : " WARNING: this drawing is outside the family the model was trained on.";
    $("draw-msg").textContent = msg;
  }
  (function wireDraw() {
    var c = $("draw-canvas");
    c.addEventListener("pointerdown", function (ev) { ev.preventDefault(); try { c.setPointerCapture(ev.pointerId); } catch (e) { /* capture is a nicety, drawing works without it */ } stroke = [pointerPos(ev)]; drawCanvas(); });
    c.addEventListener("pointermove", function (ev) { if (!stroke) return; stroke.push(pointerPos(ev)); drawCanvas(); });
    c.addEventListener("pointerup", finishStroke); c.addEventListener("pointercancel", finishStroke);
    $("snap").addEventListener("change", function () { if (state.drawing) { state.drawing.snapOn = $("snap").checked; updateDrawMsg(); refreshInput(); drawCanvas(); } });
    $("clear-draw").addEventListener("click", function () { state.drawing = null; stroke = null; $("draw-msg").textContent = ""; refreshInput(); drawCanvas(); });
  })();

  // ---------- paste ----------
  $("apply-paste").addEventListener("click", function () {
    var r = BL.parseNumbers($("paste").value);
    if (r.error) { state.pasted = null; $("paste-msg").textContent = r.error; } else { state.pasted = r.u; $("paste-msg").textContent = "Accepted 256 numbers."; }
    refreshInput();
  });

  // ---------- running ----------
  function setStatus(text, cls) { var s = $("status"); s.textContent = text; s.className = "status" + (cls ? " " + cls : ""); }
  function setBusy(b) { state.busy = b; $("run").disabled = b || !state.input; $("run").setAttribute("aria-busy", b ? "true" : "false"); }
  function errMsg(status, data) {
    var d = data && data.detail;
    if (status === 422 && Array.isArray(d) && d.length) return "The server rejected the input: " + d.map(function (e) { return e.msg; }).join("; ");
    if (status === 503) return "The model is not available on the server right now (" + (d && d.error ? d.error : "service unavailable") + ").";
    return "The server returned an error (HTTP " + status + ")." + (typeof d === "string" ? " " + d : "");
  }
  function clearResults() {
    state.result = null; $("results").hidden = true; $("ood-banner").textContent = ""; $("metrics").textContent = "";
    $("provenance").textContent = ""; $("profile-summary").textContent = ""; $("profile-table").textContent = "";
  }
  function validate(d) {
    function mat(m) { return Array.isArray(m) && m.length === NT && m.every(function (r) { return Array.isArray(r) && r.length === NX; }); }
    return d && d.fno && mat(d.fno.u) && d.solver && mat(d.solver.u) && Array.isArray(d.t) && d.t.length === NT && Array.isArray(d.x) && d.x.length === NX &&
      d.ood && typeof d.ood.is_ood === "boolean" && Array.isArray(d.solver.rel_l2_fno_vs_solver_per_t) && d.solver.rel_l2_fno_vs_solver_per_t.length === NT;
  }
  async function run() {
    if (!state.input) { setStatus("Choose, draw or paste an initial condition first.", "err"); return; }
    clearResults(); setBusy(true); setStatus("Running the FNO and the numerical solver on the server… (about a second)", "busy");
    var ctrl = new AbortController(), to = setTimeout(function () { ctrl.abort(); }, 120000), t0 = performance.now(), sent = new Float64Array(state.input.u);
    try {
      var r = await fetch("/predict", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ u0: Array.from(sent), include_solver: true }), signal: ctrl.signal });
      var txt = await r.text(), data = null; try { data = JSON.parse(txt); } catch (e) { data = null; }
      if (!r.ok) throw new Error(errMsg(r.status, data));
      if (!validate(data)) throw new Error("The server's reply did not have the expected shape, so nothing is shown.");
      var fno = BL.flatten(data.fno.u), solver = BL.flatten(data.solver.u);
      if (!fno || !solver || !Array.prototype.every.call(fno, isFinite) || !Array.prototype.every.call(solver, isFinite)) throw new Error("The server's reply contained invalid numbers, so nothing is shown.");
      var err = new Float64Array(fno.length), emax = 0; for (var i = 0; i < fno.length; i++) { err[i] = Math.abs(fno[i] - solver[i]); if (err[i] > emax) emax = err[i]; }
      state.result = { resp: data, fno: fno, solver: solver, err: err, emax: emax, rtMs: performance.now() - t0, at: new Date(), u0: sent, label: state.input.label };
      setStatus("Done.", ""); renderResults();
    } catch (e) {
      clearResults(); setStatus(e.name === "AbortError" ? "The request took too long and was cancelled." : (e instanceof TypeError ? "Could not reach the server. Check your connection and try again." : e.message), "err");
    } finally { clearTimeout(to); setBusy(false); }
  }

  // ---------- results ----------
  function renderResults() {
    var R = state.result, d = R.resp; if (!R) return;
    $("results").hidden = false;
    var ban = $("ood-banner"); ban.textContent = "";
    if (d.ood.is_ood) {
      var b = el("div", "ood"); b.appendChild(el("strong", "big", "⚠ Out-of-distribution input: do not trust the FNO result"));
      b.appendChild(el("p", null, "The model was never trained on inputs like this, so its answer may be wrong without looking wrong. Compare it with the numerical solver."));
      var ul = el("ul"); d.ood.reasons.forEach(function (r) { ul.appendChild(el("li", null, r)); }); b.appendChild(ul); ban.appendChild(b);
    } else {
      var ok = el("div", "okbox", "No out-of-distribution flag was raised. This does not guarantee the result is accurate: check the solver comparison and the limitations below."); ban.appendChild(ok);
    }
    var rel = d.solver.rel_l2_fno_vs_solver_whole;
    if (rel > 0.06) { var w = el("p", "warn", "The FNO and the solver differ by " + (100 * rel).toFixed(1) + "% (relative L2 over the whole run). On inputs like the training data this is typically 1–3%. Treat the FNO result with caution."); ban.appendChild(w); }
    var mt = $("metrics"); mt.textContent = "";
    mt.appendChild(card("FNO vs solver (whole run)", (100 * rel).toFixed(2) + " %", "relative L2, from the API"));
    mt.appendChild(card("Mass drift (max over time)", "FNO " + BL.fmt(d.fno.mass_drift, 4), "solver " + BL.fmt(d.solver.mass_drift, 2)));
    mt.appendChild(card("FNO inference time", BL.fmt(d.fno.seconds * 1000, 1) + " ms", "on the server"));
    mt.appendChild(card("Numerical solver time", BL.fmt(d.solver.seconds, 2) + " s", "on the server"));
    mt.appendChild(card("Round trip", BL.fmt(R.rtMs, 0) + " ms", "from your browser, includes the solver"));
    $("provenance").textContent = "Received from POST /predict at " + R.at.toLocaleTimeString() + ". Input: " + R.label + ". Model " + d.model.name + ", weights sha256 starts " + d.model.sha256_12 + ". Equation: " + d.equation + ".";
    requestAnimationFrame(function () { drawResultPlots(); var h = $("h-results"); h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: false }); });
  }
  function drawResultPlots() {
    var R = state.result; if (!R || $("results").hidden) return;
    var mx = 0; for (var i = 0; i < R.fno.length; i++) mx = Math.max(mx, Math.abs(R.fno[i]), Math.abs(R.solver[i])); mx = mx || 1;
    drawHeatmap($("hm-fno"), R.fno, -mx, mx, "rdbu", "FNO prediction heatmap");
    drawHeatmap($("hm-solver"), R.solver, -mx, mx, "rdbu", "Numerical solver heatmap");
    drawHeatmap($("hm-error"), R.err, 0, R.emax || 1e-12, "viridis", "Absolute difference between FNO and solver");
    drawProfile();
    var t = R.resp.t;
    plotLines($("err-time-plot"), { xmin: 0, xmax: 2, ymin: 0, ymax: Math.max(...R.resp.solver.rel_l2_fno_vs_solver_per_t) * 1.1 || 1e-6, xlabel: "t", ylabel: "relative L2 over x",
      series: [{ x: t, y: R.resp.solver.rel_l2_fno_vs_solver_per_t, color: cssv("--fno") }] });
    $("err-time-plot").setAttribute("aria-label", "Relative difference between FNO and solver versus time, from " + BL.fmt(R.resp.solver.rel_l2_fno_vs_solver_per_t[0], 3) + " at t=0 to " + BL.fmt(R.resp.solver.rel_l2_fno_vs_solver_per_t[NT - 1], 3) + " at t=2.");
  }
  function drawProfile() {
    var R = state.result; if (!R || $("results").hidden) return;
    var k = state.tIndex, t = R.resp.t[k], x = R.resp.x, f = R.fno.subarray(k * NX, (k + 1) * NX), s = R.solver.subarray(k * NX, (k + 1) * NX);
    var lo = Infinity, hi = -Infinity; for (var i = 0; i < R.fno.length; i++) { lo = Math.min(lo, R.fno[i], R.solver[i]); hi = Math.max(hi, R.fno[i], R.solver[i]); }
    var pad = (hi - lo) * 0.08 || 0.1;
    plotLines($("profile-plot"), { xmin: 0, xmax: 1, ymin: lo - pad, ymax: hi + pad, xlabel: "x", ylabel: "u(x, t)", legend: true,
      series: [{ x: x, y: s, color: cssv("--solver"), width: 3, label: "Solver" }, { x: x, y: f, color: cssv("--fno"), width: 2, dash: [6, 3], label: "FNO" }] });
    var rl = BL.relL2(f, s);
    $("t-readout").textContent = "t = " + t.toFixed(2);
    $("profile-summary").textContent = "At t = " + t.toFixed(2) + ": FNO vs solver relative L2 = " + (100 * rl).toFixed(2) + "%, max |difference| = " + BL.fmt(Math.max.apply(null, Array.prototype.map.call(f, function (v, j) { return Math.abs(v - s[j]); })), 4) + ".";
    $("profile-plot").setAttribute("aria-label", "Solution profile at t = " + t.toFixed(2) + ", FNO and solver. " + $("profile-summary").textContent);
    var tb = $("profile-table"); tb.textContent = ""; var hr = el("tr"); ["x", "FNO", "Solver", "|diff|"].forEach(function (h) { hr.appendChild(el("th", null, h)); }); tb.appendChild(hr);
    for (var j = 0; j < NX; j += 16) { var tr = el("tr"); [x[j], f[j], s[j], Math.abs(f[j] - s[j])].forEach(function (v, n) { tr.appendChild(el("td", null, n === 3 ? BL.fmt(v, 4) : BL.fmt(v, 3))); }); tb.appendChild(tr); }
  }
  $("t-slider").addEventListener("input", function () { state.tIndex = parseInt(this.value, 10); drawProfile(); });

  // ---------- downloads ----------
  function download(name, text, type) { var a = el("a"), url = URL.createObjectURL(new Blob([text], { type: type })); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(url); }, 1000); }
  $("dl-json").addEventListener("click", function () { var R = state.result; if (!R) return; var d = R.resp; download("burgers_result.json", JSON.stringify({ created: R.at.toISOString(), input_label: R.label, u0: Array.from(R.u0), x: d.x, t: d.t, fno_u: d.fno.u, solver_u: d.solver.u, ood: d.ood, model: d.model, equation: d.equation, fno_vs_solver_relL2: d.solver.rel_l2_fno_vs_solver_whole }), "application/json"); });
  $("dl-csv").addEventListener("click", function () { var R = state.result; if (!R) return; download("burgers_result.csv", BL.csvFromResponse(R.resp, R.fno, R.solver), "text/csv"); });

  // ---------- model info + health ----------
  async function loadMeta() {
    try {
      var h = await fetch("/health"), b = $("api-badge"); if (h.ok) { b.textContent = "API: ok"; b.className = "badge ok"; } else { b.textContent = "API: degraded"; b.className = "badge bad"; }
    } catch (e) { $("api-badge").textContent = "API: unreachable"; $("api-badge").className = "badge bad"; }
    var box = $("model-info");
    try {
      var r = await fetch("/model-info"); if (!r.ok) throw new Error("HTTP " + r.status); var m = await r.json(); box.textContent = "";
      var mt = m.metrics, tr = m.runtime.ood_train_range; BL.setTrainMaxAbs(tr.max_abs.max);
      box.appendChild(card("Model", m.name, m.type)); box.appendChild(card("Parameters", m.parameters.toLocaleString(), "weights sha256 " + m.runtime.served_sha256.slice(0, 12) + "…"));
      box.appendChild(card("FNO test error", BL.fmt(mt.fno_test_mean_3_seeds, 4) + " ± " + mt.fno_test_sd_3_seeds.toFixed(4), "mean ± sd of 3 training seeds; relative L2, 1,000 held-out simulations"));
      box.appendChild(card("This deployed model (seed 1)", BL.fmt(mt.deployed_seed1_test, 4), "one of the 3 seeds, not the best-of; validation " + BL.fmt(mt.deployed_seed1_val, 4)));
      box.appendChild(card("U-Net baseline test error", BL.fmt(mt.unet_test_mean_3_seeds, 4), "mean of 3 seeds"));
      box.appendChild(card("Near vs away from shocks", BL.fmt(mt.near_shock_pooled_relL2_test_seed1, 3) + " vs " + BL.fmt(mt.away_from_shock_pooled_relL2_test_seed1, 3), "pooled relative L2, seed 1, test"));
      box.appendChild(card("Mass drift (test)", BL.fmt(mt.mass_drift_pred_test_seed1, 4) + " vs " + BL.fmt(mt.mass_drift_truth_test, 2), "model vs reference data"));
      box.appendChild(card("Training data", m.training.dataset, m.training.train_samples + " train / " + m.training.val + " val / " + m.training.test + " test, " + m.training.epochs + " epochs"));
      refreshInput();
    } catch (e) { box.textContent = ""; box.appendChild(el("p", "warn", "Model information is unavailable (" + e.message + "). Predictions may still work.")); }
  }

  // ---------- init ----------
  function redrawAll() { drawInputPlot(); drawCanvas(); drawResultPlots(); }
  var rt; window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(redrawAll, 120); });
  (function init() {
    var sel = $("preset"); Object.keys(BL.PRESETS).forEach(function (k) { var o = el("option", null, BL.PRESETS[k].label); o.value = k; sel.appendChild(o); });
    buildMixUI();
    var q = new URLSearchParams(location.search);
    if (q.get("theme") === "light" || q.get("theme") === "dark") document.documentElement.setAttribute("data-theme", q.get("theme"));
    if (q.get("preset") && BL.PRESETS[q.get("preset")]) sel.value = q.get("preset");
    if (q.get("seed")) state.seed = parseInt(q.get("seed"), 10) || 1;
    Array.prototype.forEach.call(document.querySelectorAll('input[name="src"]'), function (r) { r.addEventListener("change", refreshInput); });
    sel.addEventListener("change", refreshInput);
    $("reroll").addEventListener("click", function () { state.seed = (state.seed % 1000000) + 1; refreshInput(); });
    $("run").addEventListener("click", run);
    refreshInput(); drawCanvas(); loadMeta();
    if (q.get("run") === "1") run();
  })();
})();
