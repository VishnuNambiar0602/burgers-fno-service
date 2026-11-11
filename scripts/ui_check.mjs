// Drives the dashboard in headless Edge/Chrome through the DevTools protocol (no packages needed; Node >= 22 has WebSocket).
//   node scripts/ui_check.mjs http://127.0.0.1:8000 [outdir] [scenario]     scenario: all (default) | error
// Prints one JSON object of checks; exit code 1 if any check failed.
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const base = process.argv[2] || "http://127.0.0.1:8000";
const outdir = process.argv[3] || "reports";
const scenario = process.argv[4] || "all";
const candidates = ["C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe"];
const exe = candidates.find(p => fs.existsSync(p));
if (!exe) { console.log(JSON.stringify({ skipped: "no Edge/Chrome found" })); process.exit(0); }
const port = 9300 + Math.floor(Math.random() * 500), profile = path.resolve("build", "edge_profile_" + port);
fs.mkdirSync(profile, { recursive: true }); fs.mkdirSync(outdir, { recursive: true });
const proc = spawn(exe, ["--headless=new", "--disable-gpu", "--no-first-run", `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, "about:blank"], { stdio: "ignore" });
const sleep = ms => new Promise(r => setTimeout(r, ms));
const results = {}; let failed = 0;
const check = (name, ok, info) => { results[name] = { ok: !!ok, ...(info !== undefined ? { info } : {}) }; if (!ok) failed++; };

async function connect() {
  for (let i = 0; i < 60; i++) {
    try { const l = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); const pg = l.find(t => t.type === "page"); if (pg) return pg.webSocketDebuggerUrl; } catch (e) { /* not up yet */ }
    await sleep(500);
  }
  throw new Error("browser did not start");
}
const ws = new WebSocket(await connect());
await new Promise(r => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", m => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } });
const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async expr => { const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }); if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails)); return r.result.result.value; };
async function shot(file, beyond = false) { const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: beyond }); fs.writeFileSync(path.join(outdir, file), Buffer.from(r.result.data, "base64")); }
async function open(url, w, h, mobile, scheme, dpr) {
  await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: dpr || (mobile ? 2 : 1), mobile });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme || "light" }] });
  await send("Page.navigate", { url }); await sleep(800);
  for (let i = 0; i < 40; i++) { if (await ev("document.readyState") === "complete") break; await sleep(250); }
  await sleep(500);
}
const styled = () => ev("(function(){var r=getComputedStyle(document.querySelector('#run')),d=getComputedStyle(document.querySelector('.dash'));return {runBg:r.backgroundColor,runRadius:r.borderTopLeftRadius,dash:d.display,bodyBg:getComputedStyle(document.body).backgroundColor}})()");
const styledOk = s => s.dash === "grid" && s.runRadius !== "0px" && s.runBg !== "rgb(239, 239, 239)" && s.runBg !== "rgba(0, 0, 0, 0)";
const stripState = () => ev("document.getElementById('strip').className + '|' + document.getElementById('strip-text').textContent");
async function waitDone() {
  for (let i = 0; i < 120; i++) { const s = await stripState(); if (!s.startsWith("strip busy")) return s; await sleep(500); }
  return "timeout";
}
const key = async k => { const code = { Tab: 9, Escape: 27 }[k]; await send("Input.dispatchKeyEvent", { type: "keyDown", key: k, code: k, windowsVirtualKeyCode: code }); await send("Input.dispatchKeyEvent", { type: "keyUp", key: k, code: k, windowsVirtualKeyCode: code }); };
const nonBlank = cid => ev(`(function(){var c=document.getElementById('${cid}');var d=c.getContext('2d').getImageData(0,0,c.width,c.height).data,seen={},n=0;for(var i=0;i<d.length;i+=4*37){var k=d[i]+','+d[i+1]+','+d[i+2];if(!seen[k]){seen[k]=1;n++}}return n})()`);
// page must fit the viewport: no page scroll in either direction, and no card whose content is cut off
const fit = () => ev(`(function(){var de=document.documentElement,bad=[];document.querySelectorAll('.card').forEach(function(c){if(c.scrollHeight>c.clientHeight+1||c.scrollWidth>c.clientWidth+1)bad.push(c.querySelector('h2').textContent.trim().slice(0,24)+' '+c.scrollWidth+'x'+c.scrollHeight+' in '+c.clientWidth+'x'+c.clientHeight)});return {scrollH:de.scrollHeight,innerH:window.innerHeight,scrollW:de.scrollWidth,clientW:de.clientWidth,clipped:bad}})()`);
const fitOk = f => f.scrollH <= f.innerH + 1 && f.scrollW <= f.clientW + 1 && f.clipped.length === 0;

try {
  await send("Page.enable"); await send("Runtime.enable");
  if (scenario === "matrix") {
    // realistic browser windows: viewport = screen minus toolbars, divided by the Windows scaling factor
    for (const [w, h, d, name] of [[1920, 950, 1, "1080p_100pct"], [1536, 730, 1.25, "1080p_125pct"], [1280, 600, 1.5, "1080p_150pct"], [1366, 650, 1, "laptop_768"], [1280, 560, 1, "short_window"], [1024, 640, 1, "narrow_1024"], [1100, 520, 1, "very_short"], [900, 600, 1, "min_desktop_width"], [1280, 500, 1.5, "tiny_window"]]) {
      await open(base + "/?preset=bump&run=1", w, h, false, "light", d); const s = await waitDone(); await sleep(500);
      const f = await fit(); check(`matrix_${name}_${w}x${h}@${d}`, /^strip (good|warn)\|/.test(s), { fits: fitOk(f), ...f });
      await shot(`ui_matrix_${name}.png`);
    }
  } else if (scenario === "offline") {
    // the page opened WITHOUT the API behind it (file:// or an editor preview): it must still be styled and must say what is wrong
    await open(base, 1366, 768, false, "light"); await sleep(1500);
    const st = await styled(); check("offline_page_is_styled", styledOk(st), st);
    const s = await stripState(); check("offline_strip_explains_how_to_connect", /^strip bad\|/.test(s) && /127\.0\.0\.1:8000/.test(s), s);
    check("offline_predict_disabled", await ev("document.getElementById('run').disabled===true"));
    check("offline_presets_still_render", await ev("document.querySelectorAll('button.pre').length===8"));
    await shot("ui_offline_notice.png");
  } else if (scenario === "error") {
    await open(base + "/?preset=sine&run=1", 1366, 768, false);
    const s = await waitDone();
    check("error_strip_shows_problem", /^strip bad\|/.test(s) && /not available|server/i.test(s), s);
    check("error_no_numbers_shown", await ev("document.getElementById('nums').textContent.indexOf('Press Predict')===0 && document.getElementById('dl-csv').disabled===true"));
    check("error_maps_are_placeholders_not_data", (await nonBlank("hm-fno")) < 30, await nonBlank("hm-fno"));   // text only; a real map has 200+ colours
    await shot("ui_error_state.png");
  } else {
    await open(base + "/", 1366, 768, false, "light");
    const st0 = await styled(); check("stylesheet_applied", styledOk(st0), st0);
    // ---- 1. one screen, no scrolling, at four common desktop sizes, before and after a run
    for (const [w, h] of [[1920, 1080], [1440, 900], [1366, 768], [1280, 720]]) {
      await open(base + "/", w, h, false, "light");
      const f0 = await fit(); check(`fits_${w}x${h}_before_run`, fitOk(f0), f0);
      await ev("document.getElementById('run').click()"); const s = await waitDone(); await sleep(400);
      check(`run_done_${w}x${h}`, /^strip (good|warn)\|/.test(s), s);
      const f1 = await fit(); check(`fits_${w}x${h}_after_run`, fitOk(f1), f1);
      if (w === 1366) { await shot("ui_dashboard_1366x768.png"); }
      if (w === 1920) { await shot("ui_dashboard_1920x1080.png"); }
    }
    // ---- 2. a normal flow at 1366x768: presets, run, maps, numbers, slider, downloads
    await open(base + "/", 1366, 768, false, "light");
    check("presets_visible", await ev("document.querySelectorAll('button.pre').length===8"));
    await ev("document.querySelector('button.pre[data-key=bump]').click()"); await sleep(200);
    check("preset_selected_state", await ev("document.querySelector('button.pre[data-key=bump]').getAttribute('aria-pressed')==='true' && document.querySelectorAll('button.pre[aria-pressed=true]').length===1"));
    check("input_note_updates", await ev("document.getElementById('input-note').textContent.indexOf('Bump')===0"), await ev("document.getElementById('input-note').textContent"));
    check("preset_tooltips_state_measured_error", await ev("[].every.call(document.querySelectorAll('button.pre'),function(b){return /testing|training/.test(b.title)})"));
    check("before_run_no_numbers_and_downloads_disabled", await ev("document.getElementById('dl-csv').disabled && document.getElementById('t-slider').disabled && document.getElementById('nums').textContent.indexOf('Press Predict')===0"));
    check("before_run_strip_is_idle", /^strip idle\|/.test(await stripState()));
    await ev("document.getElementById('run').click()");
    const mid = await stripState(); check("busy_state_shown_while_running", /^strip busy\|/.test(mid), mid);
    const s1 = await waitDone(); check("run_done_good_strip", /^strip good\|The AI matches the classic solver within/.test(s1), s1); await sleep(300);
    const blank = await Promise.all(["hm-fno", "hm-solver", "hm-error", "profile-plot", "err-time-plot", "input-plot"].map(nonBlank));
    check("all_plots_have_content", blank.every(n => n > 8), blank);
    const numsTxt = await ev("document.getElementById('nums').textContent"); check("numbers_from_api_shown", /Difference/.test(numsTxt) && /AI took/.test(numsTxt) && /Solver took/.test(numsTxt) && /Mass drift/.test(numsTxt), numsTxt);
    const rt = await ev("(function(){var t=document.getElementById('nums').textContent;var m=t.match(/Solver took([0-9.]+) s/),r=t.match(/Round trip([0-9.]+) ms/);return {solver:m?parseFloat(m[1]):null,rt:r?parseFloat(r[1]):null}})()");
    check("round_trip_at_least_solver_time", rt.solver !== null && rt.rt >= rt.solver * 1000 * 0.9, rt);
    check("slider_enabled_and_moves_profile", await ev("(function(){var s=document.getElementById('t-slider');if(s.disabled)return false;s.value=40;s.dispatchEvent(new Event('input'));return document.getElementById('t-readout').textContent==='t = 0.40'})()"));
    await ev("window.__blobs=[];var o=URL.createObjectURL;URL.createObjectURL=function(b){window.__blobs.push(b);return o.call(URL,b)};document.getElementById('dl-csv').click();document.getElementById('dl-json').click();0"); await sleep(800);
    const csv = await ev("window.__blobs[0] ? window.__blobs[0].text().then(function(t){var l=t.split(String.fromCharCode(10));return {n:l.length,head:l[0]}}) : null");
    check("download_csv_51457_lines", csv && csv.head === "t,x,u_fno,u_solver,abs_error" && csv.n === 201 * 256 + 1, csv);
    const js = await ev("window.__blobs[1] ? window.__blobs[1].text().then(function(t){var d=JSON.parse(t);return {u0:d.u0.length,fno:d.fno_u.length,solver:d.solver_u.length,ood:typeof d.ood.is_ood}}) : null");
    check("download_json_complete", js && js.u0 === 256 && js.fno === 201 && js.solver === 201 && js.ood === "boolean", js);
    await shot("ui_dashboard_result.png");
    // honest behaviour: a harder wave makes the AI and solver disagree and the strip says so (no out-of-distribution flag here)
    await ev("document.querySelector('button.pre[data-key=two]').click()"); await sleep(200); await ev("document.getElementById('run').click()");
    const sTwo = await waitDone(); check("harder_wave_shows_disagreement_warning", /^strip warn\|The AI and the solver disagree by/.test(sTwo), sTwo);
    check("disagreement_has_why_button", await ev("!document.getElementById('strip-why').hidden"));
    // ---- 3. explanations exist only when asked for
    const txt0 = await ev("document.body.innerText");
    check("explanations_hidden_until_clicked", txt0.indexOf("What each control does") < 0 && txt0.indexOf("Why ν/π and not just ν") < 0 && txt0.indexOf("Mass drift: the total amount") < 0 && txt0.indexOf("When not to trust") >= 0 /* only the button label */);
    const topics = await ev("[].map.call(document.querySelectorAll('[data-help]'),function(b){return b.getAttribute('data-help')})");
    check("nine_help_topics_available", topics.length >= 9, topics);
    let dialogOk = true; const dialogInfo = {};
    for (let i = 0; i < topics.length; i++) {
      await ev(`document.querySelectorAll('[data-help]')[${i}].focus(); document.querySelectorAll('[data-help]')[${i}].click()`); await sleep(250);
      const st = await ev("(function(){var d=document.getElementById('dlg');return {open:d.open,title:document.getElementById('dlg-title').textContent,len:document.getElementById('dlg-body').textContent.length}})()");
      await key("Escape"); await sleep(200);
      const after = await ev(`(function(){return {open:document.getElementById('dlg').open,focusBack:document.activeElement===document.querySelectorAll('[data-help]')[${i}]}})()`);
      const good = st.open && st.title.length > 3 && st.len > 80 && !after.open && after.focusBack; dialogInfo[topics[i]] = { ...st, ...after }; if (!good) dialogOk = false;
    }
    check("every_help_dialog_opens_closes_with_escape_and_returns_focus", dialogOk, dialogInfo);
    await ev("document.querySelector('[data-help=guide]').click()"); await sleep(300); await shot("ui_dialog_guide.png");
    await key("Escape"); await sleep(200);
    await ev("document.getElementById('btn-model').click()"); await sleep(300);
    check("model_dialog_shows_numbers_from_api", await ev("document.getElementById('dlg-model').open && /Test error of the AI/.test(document.getElementById('model-body').textContent) && /0\\.02/.test(document.getElementById('model-body').textContent)"));
    await key("Escape"); await sleep(200);
    // ---- 4. out-of-distribution: red strip, Why? dialog with reasons
    await open(base + "/", 1366, 768, false, "light");
    await ev("document.querySelector('button.pre[data-key=tall]').click()"); await sleep(200);
    check("tall_preset_shows_early_warning", await ev("!document.getElementById('input-warn').hidden"));
    await ev("document.getElementById('run').click()"); const s2 = await waitDone(); await sleep(300);
    check("ood_strip_is_red_and_plain", /^strip bad\|Unfamiliar starting wave/.test(s2), s2);
    check("ood_why_button_visible", await ev("!document.getElementById('strip-why').hidden"));
    await ev("document.getElementById('strip-why').click()"); await sleep(300);
    check("ood_why_dialog_lists_reason", await ev("document.getElementById('dlg-why').open && /amplitude/.test(document.getElementById('why-body').textContent)"), await ev("document.getElementById('why-body').textContent.slice(0,160)"));
    await key("Escape"); await sleep(200);
    const f2 = await fit(); check("fits_1366x768_with_ood_result", fitOk(f2), f2);
    await shot("ui_dashboard_ood.png");
    // ---- 5. keyboard: first Tab is the skip link
    await open(base + "/", 1366, 768, false, "light");
    await key("Tab"); check("keyboard_first_tab_is_skip_link", await ev("document.activeElement.className==='skip'"), await ev("document.activeElement.tagName+'.'+document.activeElement.className"));
    // ---- 6. drawing on the input plot with simulated mouse events
    const box = await ev("(function(){var r=document.getElementById('input-plot').getBoundingClientRect();return {x:r.left,y:r.top,w:r.width,h:r.height}})()");
    const pt = (t, x, y) => send("Input.dispatchMouseEvent", { type: t, x, y, button: "left", buttons: t === "mouseReleased" ? 0 : 1, clickCount: 1, pointerType: "mouse" });
    await pt("mousePressed", box.x + box.w * 0.1, box.y + box.h / 2);
    for (let i = 0; i <= 50; i++) { const fx = 0.1 + 0.8 * i / 50; await pt("mouseMoved", box.x + box.w * fx, box.y + box.h * (0.5 - 0.25 * Math.sin(2 * Math.PI * fx) + 0.015 * ((i * 7) % 5 - 2))); }
    await pt("mouseReleased", box.x + box.w * 0.9, box.y + box.h / 2); await sleep(400);
    const note = await ev("document.getElementById('input-note').textContent"); check("drawing_replaces_the_wave", /^Your drawing, smoothed/.test(note), note);
    check("drawing_deselects_presets", await ev("document.querySelectorAll('button.pre[aria-pressed=true]').length===0"));
    check("drawing_shows_caution", await ev("!document.getElementById('input-warn').hidden"));
    await ev("document.getElementById('run').click()"); const s3 = await waitDone(); check("drawing_run_done", /^strip (good|warn|bad)\|/.test(s3), s3);
    await ev("document.activeElement && document.activeElement.blur()");
    await shot("ui_dashboard_drawing.png");
    // ---- 7. advanced dialog: validation and applying a sine mix
    await ev("document.getElementById('advanced').click()"); await sleep(300);
    check("advanced_dialog_opens", await ev("document.getElementById('dlg-advanced').open"));
    await ev("document.getElementById('paste').value='1,2,3'; document.getElementById('apply-paste').click()");
    check("paste_error_shown_in_dialog", await ev("document.getElementById('paste-msg').textContent.indexOf('Expected exactly 256')===0"));
    await ev("document.getElementById('apply-mix').click()"); await sleep(300);
    check("sine_mix_applies_and_closes_dialog", await ev("!document.getElementById('dlg-advanced').open && /^Your sine mix/.test(document.getElementById('input-note').textContent)"), await ev("document.getElementById('input-note').textContent"));
    // ---- 8. phone-sized screen: scrolling is allowed there, but nothing may overflow sideways
    await open(base + "/?preset=tall&run=1", 375, 800, true, "dark");
    const s4 = await waitDone(); check("mobile_run_done", /^strip (good|warn|bad)\|/.test(s4), s4);
    const fm = await fit(); check("mobile_no_horizontal_overflow", fm.scrollW <= fm.clientW + 1, fm);
    await shot("ui_mobile_dark.png", true);
  }
} catch (e) { check("script_exception", false, String(e)); }
console.log(JSON.stringify(results, null, 1));
try { ws.close(); } catch (e) { /* ignore */ }
proc.kill(); await sleep(500);
try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* ignore */ }
process.exit(failed ? 1 : 0);
