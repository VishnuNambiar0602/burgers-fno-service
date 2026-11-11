/* Pure helpers for the Burgers FNO page. No DOM access here, so they can be unit-tested in Node.
   Everything that gets DISPLAYED as a result comes from the API; this file only builds inputs
   (presets, drawing -> grid) and formats API output (colour maps, CSV). */
(function (root) {
  "use strict";
  var NX = 256, NT = 201, DT = 0.01, NU = 0.01;
  var TRAIN_MAX_ABS = 3.35;        // overwritten at runtime from /model-info
  var SNAP_KMAX = 8;               // drawn inputs are low-passed to wavenumber <= 8 (chosen with scripts/snap_eval.py; training median 99.9%-energy cutoff is 4, p99 is 32)

  function grid(nx) { var x = new Float64Array(nx); for (var i = 0; i < nx; i++) x[i] = (i + 0.5) / nx; return x; }

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function fromFn(f) { var u = new Float64Array(NX), x = grid(NX); for (var i = 0; i < NX; i++) u[i] = f(x[i]); return u; }

  // sum of sines: comps = [{amp, k, phase}], plus a constant mean
  function sineMix(comps, mean) {
    var TWO_PI = 2 * Math.PI;
    return fromFn(function (x) {
      var s = mean || 0;
      for (var j = 0; j < comps.length; j++) s += comps[j].amp * Math.sin(TWO_PI * comps[j].k * x + comps[j].phase);
      return s;
    });
  }

  var PRESETS = {
    sine:   { label: "Single sine", note: "One gentle wave. In testing the AI and the solver differed by about 2.7%.", make: function () { return sineMix([{ amp: 1, k: 1, phase: 0 }], 0); } },
    two:    { label: "Two sines", note: "Two waves of different speeds, which makes sharper fronts. In testing the AI and solver differed by about 11%, and no warning is raised for this wave.", make: function () { return sineMix([{ amp: 0.8, k: 2, phase: 0.5 }, { amp: 0.5, k: 5, phase: 1.2 }], 0); } },
    bump:   { label: "Gaussian bump", note: "One smooth bump. In testing the AI and solver differed by about 1.7%.", make: function () { return fromFn(function (x) { return Math.exp(-Math.pow((x - 0.5) / 0.08, 2)); }); } },
    square: { label: "Square wave", note: "Sudden jumps. In testing the AI and solver differed by about 11%, and no warning is raised for this wave.", make: function () { return fromFn(function (x) { return (x > 0.25 && x < 0.75) ? 1 : -1; }); } },
    random: { label: "Random smooth mix (re-roll with the button)", note: "A random mix of 1 to 3 waves; click again for a new one. In testing (40 trials) the typical difference was about 5% and up to about 25%; no warning is raised.", make: function (seed) { return randomMix(seed); } },
    tall:   { label: "Too tall (out-of-distribution demo)", note: "Five times as tall as a basic wave: higher than anything in training. Raises a warning; in testing the AI was off by about 110%.", make: function () { return sineMix([{ amp: 5, k: 1, phase: 0 }], 0); } },
    noise:  { label: "Random noise (out-of-distribution demo)", note: "Random jitter. Raises a warning; in testing the AI was off by about 70%.", make: function (seed) { var r = mulberry32(seed || 7); return fromFn(function () { var a = r() || 1e-9, b = r(); return 0.3 * Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b); }); } },
    wiggly: { label: "Very wiggly sine, 40 waves (out-of-distribution demo)", note: "40 tight oscillations, far finer than the training waves. Raises a warning; in testing the AI was off by about 150%.", make: function () { return sineMix([{ amp: 1, k: 40, phase: 0 }], 0); } }
  };

  function randomMix(seed) {
    var r = mulberry32(seed == null ? 1 : seed), n = 1 + Math.floor(r() * 3), comps = [], tot = 0;
    for (var i = 0; i < n; i++) { var a = 0.3 + 0.9 * r(); tot += a; comps.push({ amp: a, k: 1 + Math.floor(r() * 8), phase: 2 * Math.PI * r() }); }
    var s = tot > 2.4 ? 2.4 / tot : 1;                       // keep max|u| <= 2.4 + mean, inside the training range
    for (var j = 0; j < comps.length; j++) comps[j].amp *= s;
    return sineMix(comps, (r() - 0.5) * 0.8);
  }

  // Interpolate drawn points [{x in [0,1], y}] onto the 256-point periodic grid (linear; the gap across the seam is bridged linearly).
  function pointsToGrid(points) {
    if (points.length < 2) return null;
    var p = points.slice().sort(function (a, b) { return a.x - b.x; }), q = [];
    for (var i = 0; i < p.length; i++) if (!q.length || p[i].x - q[q.length - 1].x > 1e-6) q.push(p[i]);
    if (q.length < 2) return null;
    var ext = [{ x: q[q.length - 1].x - 1, y: q[q.length - 1].y }].concat(q, [{ x: q[0].x + 1, y: q[0].y }]);
    var u = new Float64Array(NX), x = grid(NX), j = 0;
    for (var k = 0; k < NX; k++) {
      while (j < ext.length - 2 && ext[j + 1].x < x[k]) j++;
      var a = ext[j], b = ext[j + 1], t = (b.x - a.x) > 0 ? (x[k] - a.x) / (b.x - a.x) : 0;
      u[k] = a.y + Math.min(Math.max(t, 0), 1) * (b.y - a.y);
    }
    return u;
  }

  // Low-pass: keep the mean and Fourier modes 1..kmax (naive DFT, N is small).
  function lowpass(u, kmax) {
    var n = u.length, out = new Float64Array(n), mean = 0, i, k;
    for (i = 0; i < n; i++) mean += u[i];
    mean /= n; for (i = 0; i < n; i++) out[i] = mean;
    for (k = 1; k <= kmax; k++) {
      var a = 0, b = 0;
      for (i = 0; i < n; i++) { var ang = 2 * Math.PI * k * (i + 0.5) / n; a += u[i] * Math.cos(ang); b += u[i] * Math.sin(ang); }
      a *= 2 / n; b *= 2 / n;
      for (i = 0; i < n; i++) { var an = 2 * Math.PI * k * (i + 0.5) / n; out[i] += a * Math.cos(an) + b * Math.sin(an); }
    }
    return out;
  }

  function norm(u) { var s = 0; for (var i = 0; i < u.length; i++) s += u[i] * u[i]; return Math.sqrt(s); }
  function relL2(a, b) { var s = 0, t = 0; for (var i = 0; i < a.length; i++) { var d = a[i] - b[i]; s += d * d; t += b[i] * b[i]; } return t > 0 ? Math.sqrt(s / t) : NaN; }
  function maxAbs(u) { var m = 0; for (var i = 0; i < u.length; i++) m = Math.max(m, Math.abs(u[i])); return m; }

  // Snap a drawing to the supported family: low-pass to SNAP_KMAX, then report how much it changed and whether it was clamped.
  function snapDrawing(raw, maxAbsLimit) {
    var lim = maxAbsLimit || TRAIN_MAX_ABS, snapped = lowpass(raw, SNAP_KMAX), clamped = false;
    for (var i = 0; i < snapped.length; i++) {
      if (snapped[i] > lim) { snapped[i] = lim; clamped = true; } else if (snapped[i] < -lim) { snapped[i] = -lim; clamped = true; }
    }
    var ref = norm(raw), change = ref > 0 ? relL2(snapped, raw) : 0;
    return { u: snapped, change: change, clamped: clamped, outside: change > 0.25 || clamped };
  }

  // ---- colour maps (ColorBrewer RdBu reversed, and viridis) ----
  var RDBU = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"];
  var VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"];
  function hex(h) { return [parseInt(h.substr(1, 2), 16), parseInt(h.substr(3, 2), 16), parseInt(h.substr(5, 2), 16)]; }
  var CM = { rdbu: RDBU.map(hex), viridis: VIRIDIS.map(hex) };
  function colorAt(name, t) {
    var s = CM[name], x = Math.min(Math.max(t, 0), 1) * (s.length - 1), i = Math.min(Math.floor(x), s.length - 2), f = x - i;
    return [s[i][0] + f * (s[i + 1][0] - s[i][0]), s[i][1] + f * (s[i + 1][1] - s[i][1]), s[i][2] + f * (s[i + 1][2] - s[i][2])];
  }
  // flat array u[nt*nx] (row k = time k) -> RGBA image nx wide, nt tall, time increasing upward
  function heatmapRGBA(u, nt, nx, vmin, vmax, cmap) {
    var img = new Uint8ClampedArray(nt * nx * 4), span = (vmax - vmin) || 1;
    for (var k = 0; k < nt; k++) for (var i = 0; i < nx; i++) {
      var c = colorAt(cmap, (u[k * nx + i] - vmin) / span), o = ((nt - 1 - k) * nx + i) * 4;
      img[o] = c[0]; img[o + 1] = c[1]; img[o + 2] = c[2]; img[o + 3] = 255;
    }
    return img;
  }

  function flatten(rows) {          // array of nt arrays of nx numbers -> Float64Array, or null if ragged
    var nt = rows.length, nx = rows[0].length, out = new Float64Array(nt * nx);
    for (var k = 0; k < nt; k++) { if (rows[k].length !== nx) return null; for (var i = 0; i < nx; i++) out[k * nx + i] = rows[k][i]; }
    return out;
  }

  function csvFromResponse(resp, fno, solver) {
    var nt = resp.t.length, nx = resp.x.length, lines = ["t,x,u_fno,u_solver,abs_error"];
    for (var k = 0; k < nt; k++) for (var i = 0; i < nx; i++) {
      var f = fno[k * nx + i], s = solver[k * nx + i];
      lines.push(resp.t[k] + "," + resp.x[i] + "," + f + "," + s + "," + Number(Math.abs(f - s).toPrecision(7)));
    }
    return lines.join("\n");
  }

  function parseNumbers(text) {
    var parts = text.split(/[\s,;\[\]]+/).filter(function (s) { return s.length; }), out = [];
    for (var i = 0; i < parts.length; i++) { var v = Number(parts[i]); if (!isFinite(v)) return { error: "'" + parts[i] + "' is not a number" }; out.push(v); }
    if (out.length !== NX) return { error: "Expected exactly " + NX + " numbers, got " + out.length };
    return { u: Float64Array.from(out) };
  }

  function fmt(x, d) { if (!isFinite(x)) return "n/a"; var a = Math.abs(x); return (a !== 0 && (a < 1e-3 || a >= 1e4)) ? x.toExponential(d == null ? 2 : d) : x.toFixed(d == null ? 3 : d); }

  var BL = { NX: NX, NT: NT, DT: DT, NU: NU, SNAP_KMAX: SNAP_KMAX, PRESETS: PRESETS, grid: grid, sineMix: sineMix, randomMix: randomMix,
    pointsToGrid: pointsToGrid, lowpass: lowpass, snapDrawing: snapDrawing, relL2: relL2, maxAbs: maxAbs, colorAt: colorAt,
    heatmapRGBA: heatmapRGBA, flatten: flatten, csvFromResponse: csvFromResponse, parseNumbers: parseNumbers, fmt: fmt,
    setTrainMaxAbs: function (v) { if (isFinite(v) && v > 0) TRAIN_MAX_ABS = v; }, getTrainMaxAbs: function () { return TRAIN_MAX_ABS; } };
  if (typeof module !== "undefined" && module.exports) module.exports = BL; else root.BL = BL;
})(typeof window !== "undefined" ? window : this);
