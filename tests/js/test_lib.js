// Node unit tests for app/static/lib.js (no test framework needed). Exit code 1 on failure.
const BL = require("../../app/static/lib.js");
let failed = 0;
function ok(c, m) { if (!c) { failed++; console.log("FAIL:", m); } else console.log("ok  :", m); }
const near = (a, b, t) => Math.abs(a - b) <= t;

for (const [k, p] of Object.entries(BL.PRESETS)) {
  const u = p.make(3);
  ok(u.length === 256 && Array.from(u).every(Number.isFinite), `preset ${k}: 256 finite values`);
}
let worst = 0; for (let s = 1; s <= 200; s++) worst = Math.max(worst, BL.maxAbs(BL.randomMix(s)));
ok(worst < 3.35, `random mix stays inside training range over 200 seeds (max |u| = ${worst.toFixed(3)})`);
ok(BL.maxAbs(BL.PRESETS.tall.make()) > 3.35, "'tall' preset really exceeds the training max");

// drawing -> grid recovers a sine sampled at 60 irregular points
const pts = []; for (let i = 0; i < 60; i++) { const x = (i + 0.37 * (i % 3)) / 62; pts.push({ x, y: Math.sin(2 * Math.PI * x) }); }
const g = BL.pointsToGrid(pts), ref = BL.PRESETS.sine.make();
let e = 0; for (let i = 0; i < 256; i++) if (i / 256 > 0.03 && i / 256 < 0.97) e = Math.max(e, Math.abs(g[i] - ref[i]));
ok(e < 0.08, `drawn sine recovered on the grid away from the seam (max err ${e.toFixed(3)})`);
ok(BL.pointsToGrid([{ x: 0.1, y: 1 }]) === null, "a single point is rejected");

// low-pass
const k3 = BL.sineMix([{ amp: 1, k: 3, phase: 0.4 }], 0.5), lp = BL.lowpass(k3, 16);
ok(near(BL.relL2(lp, k3), 0, 1e-9), "low-pass keeps a k=3 sine + mean exactly");
const k40 = BL.sineMix([{ amp: 1, k: 40, phase: 0 }], 0);
ok(BL.maxAbs(BL.lowpass(k40, 16)) < 1e-9, "low-pass removes a k=40 sine");

// snapping
ok(BL.snapDrawing(BL.PRESETS.sine.make()).outside === false, "smooth sine: not flagged as outside the family");
ok(BL.snapDrawing(BL.PRESETS.noise.make(5)).outside === true, "noise drawing: flagged as outside the family");
ok(BL.snapDrawing(BL.PRESETS.tall.make()).clamped === true, "too-tall drawing: clamped and flagged");

// heatmap orientation: time k=0 must be on the BOTTOM row, k=nt-1 on the top row
const nt = 3, nx = 2, u = new Float64Array([0, 0, 0.5, 0.5, 1, 1]);
const img = BL.heatmapRGBA(u, nt, nx, 0, 1, "viridis"), c0 = BL.colorAt("viridis", 0), c1 = BL.colorAt("viridis", 1);
ok(img[((nt - 1) * nx) * 4] === Math.round(c0[0]) && img[0] === Math.round(c1[0]), "heatmap: t=0 at the bottom, last t at the top");
ok(img.length === nt * nx * 4 && img[3] === 255, "heatmap: RGBA size and opaque alpha");

// CSV and parsing
const resp = { t: [0, 0.01], x: [0.1, 0.2] }, csv = BL.csvFromResponse(resp, [1, 2, 3, 4], [1, 2, 3, 5]).split("\n");
ok(csv.length === 5 && csv[0] === "t,x,u_fno,u_solver,abs_error" && csv[4] === "0.01,0.2,4,5,1", "CSV: header, row count and an example row");
ok(BL.parseNumbers("1,2,3").error !== undefined, "paste: wrong count rejected");
ok(BL.parseNumbers(new Array(256).fill("0.5").join(" ")).u.length === 256, "paste: 256 numbers accepted");
ok(BL.parseNumbers(new Array(255).fill("0.5").concat(["abc"]).join(" ")).error.indexOf("abc") >= 0, "paste: non-number named in the error");
ok(BL.flatten([[1, 2], [3]]) === null, "ragged API array rejected");
ok(BL.fmt(0.000012) === "1.20e-5" && BL.fmt(1.23456) === "1.235", "number formatting");
process.exit(failed ? 1 : 0);
