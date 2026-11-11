/* Builds the dashboard report PDF in the browser, with no library. Loaded only when the user clicks "Download PDF".
   Page 1: the dashboard as seen (canvases + numbers, redrawn in a light print palette). Next pages: the numbers as real text.
   The complete CSV and JSON are attached inside the PDF (open the attachments panel of your PDF viewer). Everything comes from the API response. */
(function () {
  "use strict";
  var enc = new TextEncoder();

  // base-14 PDF fonts cannot show Greek letters, so the text pages use plain ASCII (the picture on page 1 keeps the symbols)
  function asc(s) {
    return String(s).replace(/ν/g, "nu").replace(/π/g, "pi").replace(/∂/g, "d").replace(/²/g, "^2").replace(/[·×]/g, "x").replace(/±/g, "+/-")
      .replace(/[–—−]/g, "-").replace(/[≈]/g, "~").replace(/[“”]/g, '"').replace(/[‘’]/g, "'").replace(/…/g, "...").replace(/[^\x20-\x7e]/g, "?");
  }
  function esc(s) { return asc(s).replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)"); }
  function pad(s, n) { s = String(s); while (s.length < n) s += " "; return s; }
  function padL(s, n) { s = String(s); while (s.length < n) s = " " + s; return s; }
  function wrap(text, n) {
    var words = asc(text).split(/\s+/), out = [], line = "";
    words.forEach(function (w) { if ((line + " " + w).trim().length > n) { out.push(line); line = w; } else line = (line + " " + w).trim(); });
    if (line) out.push(line); return out;
  }
  function pdfDate(d) {
    var z = function (n) { return String(n).padStart(2, "0"); }, tz = -d.getTimezoneOffset(), sg = tz >= 0 ? "+" : "-", a = Math.abs(tz);
    return "D:" + d.getFullYear() + z(d.getMonth() + 1) + z(d.getDate()) + z(d.getHours()) + z(d.getMinutes()) + z(d.getSeconds()) + sg + z(Math.floor(a / 60)) + "'" + z(a % 60) + "'";
  }
  async function deflate(bytes) {
    var cs = new CompressionStream("deflate"), w = cs.writable.getWriter(); w.write(bytes); w.close();
    return new Uint8Array(await new Response(cs.readable).arrayBuffer());
  }

  // ---- a tiny PDF object writer ----
  function Pdf() { this.objs = []; }
  Pdf.prototype.add = function (parts) { this.objs.push(parts); return this.objs.length; };
  Pdf.prototype.reserve = function () { this.objs.push(null); return this.objs.length; };
  Pdf.prototype.set = function (n, parts) { this.objs[n - 1] = parts; };
  Pdf.prototype.blob = function (rootN, infoN) {
    var chunks = [], offsets = [], pos = 0;
    function push(x) { var b = typeof x === "string" ? enc.encode(x) : x; chunks.push(b); pos += b.length; }
    push("%PDF-1.7\n"); push(new Uint8Array([0x25, 0xE2, 0xE3, 0xCF, 0xD3, 0x0A]));      // binary marker comment
    for (var i = 0; i < this.objs.length; i++) {
      offsets.push(pos); push((i + 1) + " 0 obj\n");
      var parts = this.objs[i]; for (var j = 0; j < parts.length; j++) push(parts[j]);
      push("\nendobj\n");
    }
    var xref = pos; push("xref\n0 " + (this.objs.length + 1) + "\n0000000000 65535 f \n");
    offsets.forEach(function (o) { push(String(o).padStart(10, "0") + " 00000 n \n"); });
    push("trailer\n<< /Size " + (this.objs.length + 1) + " /Root " + rootN + " 0 R /Info " + infoN + " 0 R >>\nstartxref\n" + xref + "\n%%EOF\n");
    return new Blob(chunks, { type: "application/pdf" });
  };
  function stream(entries, bytes) { return ["<< " + entries + " /Length " + bytes.length + " >>\nstream\n", bytes, "\nendstream"]; }

  // ---- page 1 picture: the dashboard as the user sees it ----
  var $ = function (id) { return document.getElementById(id); };
  async function renderComposite(B) {
    var root = document.documentElement, prev = root.getAttribute("data-theme");
    root.setAttribute("data-theme", "light"); B.redrawAll();            // print palette: dark text on white, whatever theme is active
    try {
      var W = 1800, H = 1000, c = document.createElement("canvas"); c.width = W; c.height = H;
      var g = c.getContext("2d"); g.fillStyle = "#ffffff"; g.fillRect(0, 0, W, H); g.textBaseline = "top";
      var ink = "#1d2330", muted = "#505a6c";
      g.fillStyle = ink; g.font = "bold 34px Arial, sans-serif"; g.fillText("Burgers Wave Predictor", 20, 12);
      var kind = ($("strip").className.match(/good|warn|bad/) || ["idle"])[0], col = { good: "#1f6f3a", warn: "#7a3200", bad: "#8a1f12", idle: ink }[kind];
      g.fillStyle = col; g.font = "bold 22px Arial, sans-serif"; g.fillText($("strip-text").textContent, 20, 58);
      function box(x, y, w, h, caption, canvasId, note) {
        g.strokeStyle = "#d9d5ca"; g.lineWidth = 2; g.strokeRect(x, y, w, h);
        g.fillStyle = ink; g.font = "bold 22px Arial, sans-serif"; g.fillText(caption, x + 10, y + 8);
        if (!canvasId) return { x: x + 10, y: y + 42, w: w - 20, h: h - 50 };
        var cv = $(canvasId), ar = cv.width / cv.height, bx = x + 8, by = y + 38, bw = w - 16, bh = h - 46 - (note ? 26 : 0), dw = bw, dh = bw / ar;
        if (dh > bh) { dh = bh; dw = bh * ar; }
        g.drawImage(cv, bx + (bw - dw) / 2, by, dw, dh);
        if (note) { g.fillStyle = muted; g.font = "18px Arial, sans-serif"; g.fillText(note, x + 10, y + h - 28); }
        return null;
      }
      var inputNote = $("input-note").textContent;
      box(20, 100, 372, 470, "1 · Starting wave", "input-plot", inputNote.length > 52 ? inputNote.slice(0, 50) + "…" : inputNote);
      box(410, 100, 440, 470, "2 · AI forecast", "hm-fno");
      box(868, 100, 440, 470, "Classic solver (reference)", "hm-solver");
      box(1326, 100, 440, 470, "Where they differ", "hm-error");
      box(20, 590, 580, 390, "Snapshot at one moment (" + $("t-readout").textContent + ")", "profile-plot");
      box(618, 590, 560, 390, "Gap over time", "err-time-plot");
      var nb = box(1196, 590, 570, 390, "3 · Key numbers", null);
      g.fillStyle = ink; g.font = "22px Arial, sans-serif";
      var kids = Array.prototype.slice.call($("nums").children), y = nb.y;
      for (var i = 0; i + 1 < kids.length; i += 2) {
        g.fillStyle = muted; g.fillText(kids[i].textContent, nb.x, y); g.fillStyle = ink; g.font = "bold 22px Arial, sans-serif";
        var v = kids[i + 1].textContent; g.fillText(v, nb.x + nb.w - g.measureText(v).width, y); g.font = "22px Arial, sans-serif"; y += 34;
      }
      g.fillStyle = muted; g.font = "17px Arial, sans-serif";
      g.fillText("Generated " + new Date().toLocaleString() + " from the server's /predict response.", nb.x, 590 + 390 - 30);
      var blob = await new Promise(function (res) { c.toBlob(res, "image/jpeg", 0.92); });
      return { bytes: new Uint8Array(await blob.arrayBuffer()), w: W, h: H };
    } finally {
      if (prev === null) root.removeAttribute("data-theme"); else root.setAttribute("data-theme", prev);
      B.redrawAll();
    }
  }

  // ---- text pages: the numbers as real text ----
  function textLines(B) {
    var S = B.state, R = S.result, d = R.resp, L = [], NX = d.x.length;
    L.push("KEY NUMBERS");
    var kids = Array.prototype.slice.call($("nums").children);
    for (var i = 0; i + 1 < kids.length; i += 2) L.push("  " + pad(asc(kids[i].textContent), 24) + asc(kids[i + 1].textContent));
    L.push(""); L.push("WHAT THE PAGE TOLD YOU");
    wrap($("strip-text").textContent, 150).forEach(function (l) { L.push("  " + l); });
    if (d.ood.is_ood) { L.push("  Out-of-distribution input:"); d.ood.reasons.forEach(function (r) { wrap(r, 146).forEach(function (l) { L.push("    - " + l); }); }); }
    if (d.self_check && d.self_check.flagged) { L.push("  The AI's own sanity checks failed:"); d.self_check.reasons.forEach(function (r) { wrap(r, 146).forEach(function (l) { L.push("    - " + l); }); }); }
    if (d.self_check) L.push("  self-check signals: frame-0 error " + d.self_check.signals.t0.toFixed(4) + ", mass score " + d.self_check.signals.mass.toFixed(4) + ", equation residual " + d.self_check.signals.resid.toFixed(4));
    L.push(""); L.push("MODEL AND EQUATION");
    L.push("  model " + d.model.name + ", weights sha256 starts " + d.model.sha256_12 + ", trained at " + d.model.trained_at_nx + " points, this input has " + d.model.input_nx);
    wrap("equation: " + d.equation, 150).forEach(function (l) { L.push("  " + l); });
    L.push(""); L.push("HOW TO USE THIS FILE");
    wrap("The full result (all 201 time steps x " + NX + " points: time t, position x, AI value, solver value, absolute gap) is attached inside this PDF as burgers_result.csv, and "
      + "everything including the warning information as burgers_result.json. Open the attachments panel of your PDF viewer to extract them. "
      + "Position x_i = (i + 0.5)/" + NX + " on a periodic domain [0, 1); time t = 0.01 k for k = 0..200.", 150).forEach(function (l) { L.push("  " + l); });
    L.push(""); L.push("STARTING WAVE: " + asc(R.label) + " (" + NX + " values at x_i, 8 per line)");
    for (var j = 0; j < R.u0.length; j += 8) {
      var row = padL(j, 4) + ":"; for (var k = j; k < Math.min(j + 8, R.u0.length); k++) row += " " + padL(R.u0[k].toFixed(5), 10); L.push(row);
    }
    L.push(""); L.push("SNAPSHOTS (every 8th point of x; AI / classic solver)");
    var T = [0, 50, 100, 150, 200], head = "   x   ";
    T.forEach(function (k) { head += "| t=" + pad(d.t[k].toFixed(2), 4) + "  AI       solver  "; }); L.push(head);
    for (var xi = 0; xi < NX; xi += 8) {
      var r2 = padL(d.x[xi].toFixed(3), 6) + " "; T.forEach(function (k) { r2 += "| " + padL(d.fno.u[k][xi].toFixed(4), 9) + " " + padL(d.solver.u[k][xi].toFixed(4), 9) + "  "; }); L.push(r2);
    }
    return L;
  }

  async function download(B) {
    var S = B.state, R = S.result; if (!R) throw new Error("no result to export yet");
    var d = R.resp, now = new Date(), pdf = new Pdf(), BL = B.BL;
    var img = await renderComposite(B);
    var cat = pdf.reserve(), pages = pdf.reserve(), info = pdf.reserve();
    var f1 = pdf.add(["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]);
    var f2 = pdf.add(["<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>"]);
    var f3 = pdf.add(["<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]);
    var im = pdf.add(stream("/Type /XObject /Subtype /Image /Width " + img.w + " /Height " + img.h + " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode", img.bytes));
    var fonts = "/Font << /F1 " + f1 + " 0 R /F2 " + f2 + " 0 R /F3 " + f3 + " 0 R >>", PW = 842, PH = 595, kids = [];
    function addPage(content, withImage) {
      var cs = pdf.add(stream("", enc.encode(content)));
      var res = "/Resources << " + fonts + (withImage ? " /XObject << /Im0 " + im + " 0 R >>" : "") + " >>";
      var p = pdf.add(["<< /Type /Page /Parent " + pages + " 0 R /MediaBox [0 0 " + PW + " " + PH + "] " + res + " /Contents " + cs + " 0 R >>"]); kids.push(p);
    }
    // page 1: the dashboard picture, with a one-line caption
    var iw = PW - 36, ih = iw * img.h / img.w, iy = PH - 24 - ih;
    addPage("q " + iw.toFixed(2) + " 0 0 " + ih.toFixed(2) + " 18 " + iy.toFixed(2) + " cm /Im0 Do Q\nBT /F3 9 Tf 18 " + (PH - 16) + " Td (" + esc("Burgers Wave Predictor - dashboard report, " + now.toLocaleString()) + ") Tj ET\n"
      + "BT /F1 8 Tf 18 10 Td (" + esc("Numbers are on the next pages as text. The complete CSV and JSON are attached inside this PDF.") + ") Tj ET\n", true);
    // text pages
    var L = textLines(B), per = 60, n = Math.ceil(L.length / per);
    for (var p = 0; p < n; p++) {
      var chunk = L.slice(p * per, (p + 1) * per), c = "BT /F3 11 Tf 36 " + (PH - 30) + " Td (" + esc("Burgers Wave Predictor - numbers (page " + (p + 1) + " of " + n + ")") + ") Tj ET\n";
      c += "BT /F2 7 Tf 8.2 TL 36 " + (PH - 50) + " Td\n"; chunk.forEach(function (l) { c += "(" + esc(l) + ") Tj T*\n"; }); c += "ET\n"; addPage(c, false);
    }
    // attachments: the full CSV and JSON, deflated, as embedded files
    var csv = enc.encode(BL.csvFromResponse(d, R.fno, R.solver));
    var json = enc.encode(JSON.stringify({ created: R.at.toISOString(), input_label: R.label, u0: Array.from(R.u0), x: d.x, t: d.t, fno_u: d.fno.u, solver_u: d.solver.u, ood: d.ood, self_check: d.self_check || null,
      verdict: d.verdict || null, model: d.model, equation: d.equation, fno_vs_solver_relL2: d.solver.rel_l2_fno_vs_solver_whole }));
    var files = [["burgers_result.csv", "text#2Fcsv", csv, "Full result table: t, x, u_fno, u_solver, abs_error"], ["burgers_result.json", "application#2Fjson", json, "Full result with warnings and model information"]], names = [];
    for (var q = 0; q < files.length; q++) {
      var z = await deflate(files[q][2]);
      var ef = pdf.add(stream("/Type /EmbeddedFile /Subtype /" + files[q][1] + " /Params << /Size " + files[q][2].length + " >> /Filter /FlateDecode", z));
      var fs = pdf.add(["<< /Type /Filespec /F (" + esc(files[q][0]) + ") /UF (" + esc(files[q][0]) + ") /Desc (" + esc(files[q][3]) + ") /EF << /F " + ef + " 0 R /UF " + ef + " 0 R >> >>"]);
      names.push("(" + esc(files[q][0]) + ") " + fs + " 0 R");
    }
    pdf.set(pages, ["<< /Type /Pages /Kids [" + kids.map(function (k) { return k + " 0 R"; }).join(" ") + "] /Count " + kids.length + " >>"]);
    pdf.set(info, ["<< /Title (" + esc("Burgers Wave Predictor - dashboard report") + ") /Producer (" + esc("burgers-fno-service, in-browser PDF writer") + ") /CreationDate (" + pdfDate(now) + ") >>"]);
    pdf.set(cat, ["<< /Type /Catalog /Pages " + pages + " 0 R /Names << /EmbeddedFiles << /Names [" + names.join(" ") + "] >> >> >>"]);
    var blob = pdf.blob(cat, info), a = document.createElement("a"), url = URL.createObjectURL(blob);
    window.__lastPdf = { bytes: blob.size, pages: kids.length, attachments: files.map(function (f) { return f[0]; }) };
    a.href = url; a.download = "burgers_dashboard_report.pdf"; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
    return blob;
  }

  window.BurgersPdf = { download: download };
})();
