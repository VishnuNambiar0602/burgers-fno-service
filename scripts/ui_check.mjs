// Drives the page in headless Edge/Chrome through the DevTools protocol (no packages needed; Node >= 22 has WebSocket).
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
async function shot(file) { const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true }); fs.writeFileSync(path.join(outdir, file), Buffer.from(r.result.data, "base64")); }
async function open(url, w, h, mobile, scheme) {
  await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: mobile ? 2 : 1, mobile });
  await send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme || "light" }] });
  await send("Page.navigate", { url }); await sleep(800);
  for (let i = 0; i < 40; i++) { if (await ev("document.readyState") === "complete") break; await sleep(250); }
}
async function waitStatus() {
  for (let i = 0; i < 120; i++) { const s = await ev("document.getElementById('status').className + '|' + document.getElementById('status').textContent"); if (!s.startsWith("status busy") && !/^status\|$/.test(s) ) return s; await sleep(500); }
  return "timeout";
}
const nonBlank = id => ev(`(function(){var c=document.getElementById('${id}');var d=c.getContext('2d').getImageData(0,0,c.width,c.height).data,seen={},n=0;for(var i=0;i<d.length;i+=4*37){var k=d[i]+','+d[i+1]+','+d[i+2];if(!seen[k]){seen[k]=1;n++}}return n})()`);
const overflow = () => ev(`(function(){var w=document.documentElement.clientWidth,bad=[];document.querySelectorAll('body *').forEach(function(e){var r=e.getBoundingClientRect();if(r.width>0&&r.right>w+1){var p=e.parentElement,clipped=false;while(p&&p!==document.body){var o=getComputedStyle(p).overflowX;if(o==='auto'||o==='scroll'||o==='hidden'){clipped=true;break}p=p.parentElement}if(!clipped)bad.push(e.tagName+'#'+e.id+'.'+e.className)}});return {scrollW:document.documentElement.scrollWidth,clientW:w,bad:bad.slice(0,6)}})()`);

try {
  await send("Page.enable"); await send("Runtime.enable");
  if (scenario === "error") {
    await open(base + "/?preset=sine&run=1", 1000, 900, false);
    const s = await waitStatus();
    check("error_state_message", /err\|/.test(s) && /not available|server/i.test(s), s);
    check("error_state_no_results_shown", await ev("document.getElementById('results').hidden === true"));
    await shot("ui_error_state.png");
  } else {
    // 1. desktop, light, preset run
    await open(base + "/?preset=sine&run=1", 1280, 900, false, "light");
    const s = await waitStatus(); check("desktop_run_done", /Done/.test(s), s);
    check("desktop_results_visible", await ev("!document.getElementById('results').hidden"));
    check("desktop_ok_banner_no_ood", await ev("!!document.querySelector('#ood-banner .okbox') && !document.querySelector('#ood-banner .ood')"));
    const blank = await Promise.all(["hm-fno", "hm-solver", "hm-error", "profile-plot", "err-time-plot", "input-plot"].map(nonBlank));
    check("desktop_canvases_have_content", blank.every(n => n > 8), blank);
    const rt = await ev("(function(){var c=[].slice.call(document.querySelectorAll('#metrics .card'));function v(k){var x=c.find(function(e){return e.querySelector('.k').textContent.indexOf(k)===0});return x?x.querySelector('.v').textContent:''}return {rt:v('Round trip'),solver:v('Numerical solver')}})()");
    check("desktop_round_trip_exceeds_solver_time", parseFloat(rt.rt) >= parseFloat(rt.solver) * 1000 * 0.9, rt);
    check("desktop_title_and_equation", await ev("document.title.length>0 && document.body.textContent.indexOf('(ν/π)')>=0 && document.body.textContent.indexOf('ν = 0.01 (fixed)')>=0 && document.body.textContent.indexOf('256 grid points')>=0"));
    const o1 = await overflow(); check("desktop_no_horizontal_overflow", o1.scrollW <= o1.clientW + 1 && o1.bad.length === 0, o1);
    await shot("ui_desktop_light.png");
    // 2. keyboard: first Tab lands on the skip link
    await send("Page.navigate", { url: base + "/" }); await sleep(1500);
    await send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 }); await send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 });
    check("keyboard_first_tab_is_skip_link", await ev("document.activeElement.className==='skip'"), await ev("document.activeElement.tagName+'.'+document.activeElement.className"));
    // 3. mobile emulation, dark, OOD preset
    await open(base + "/?preset=tall&run=1", 375, 800, true, "dark");
    const s2 = await waitStatus(); check("mobile_ood_run_done", /Done/.test(s2), s2);
    check("mobile_ood_banner_shown", await ev("!!document.querySelector('#ood-banner .ood') && document.querySelector('#ood-banner').textContent.indexOf('Out-of-distribution')>=0"));
    check("mobile_ood_banner_has_reason", await ev("document.querySelectorAll('#ood-banner .ood li').length>=1"));
    const o2 = await overflow(); check("mobile_no_horizontal_overflow", o2.scrollW <= o2.clientW + 1 && o2.bad.length === 0, o2);
    await shot("ui_mobile_ood_dark.png");
    // 4. drawing with simulated mouse events, then run
    await open(base + "/", 1000, 900, false, "light");
    await ev("document.querySelector('input[name=src][value=draw]').click()"); await sleep(300);
    await ev("document.getElementById('draw-canvas').scrollIntoView({block:'center'})"); await sleep(300);
    const box = await ev("(function(){var r=document.getElementById('draw-canvas').getBoundingClientRect();return {x:r.left,y:r.top,w:r.width,h:r.height}})()");
    await ev("window.__log={};['pointerdown','pointermove','pointerup','mousedown','mouseup'].forEach(function(t){document.getElementById('draw-canvas').addEventListener(t,function(){window.__log[t]=(window.__log[t]||0)+1})})");
    const pt = (t, x, y) => send("Input.dispatchMouseEvent", { type: t, x, y, button: "left", buttons: t === "mouseReleased" ? 0 : 1, clickCount: 1, pointerType: "mouse" });
    await pt("mousePressed", box.x + box.w * 0.05, box.y + box.h / 2);
    for (let i = 0; i <= 60; i++) { const fx = 0.05 + 0.9 * i / 60; await pt("mouseMoved", box.x + box.w * fx, box.y + box.h * (0.5 - 0.3 * Math.sin(2 * Math.PI * fx) + 0.02 * ((i * 7) % 5 - 2))); }
    await pt("mouseReleased", box.x + box.w * 0.95, box.y + box.h / 2); await sleep(400);
    const dm = await ev("document.getElementById('draw-msg').textContent"); check("draw_snapped_message", /Snapped/.test(dm), { msg: dm, events: await ev("window.__log"), box });
    check("draw_input_updated", await ev("document.getElementById('input-summary').textContent.indexOf('Your drawing')===0"), await ev("document.getElementById('input-summary').textContent"));
    await ev("document.getElementById('run').click()"); const s3 = await waitStatus(); check("draw_run_done", /Done/.test(s3), s3);
    await shot("ui_drawing_run.png");
    // 5. paste validation error state
    await ev("document.querySelector('input[name=src][value=paste]').click()"); await ev("document.getElementById('paste').value='1,2,3'; document.getElementById('apply-paste').click()");
    check("paste_error_shown", await ev("document.getElementById('paste-msg').textContent.indexOf('Expected exactly 256')===0"));
    check("run_disabled_without_valid_input", await ev("document.getElementById('run').disabled===true"));
  }
} catch (e) { check("script_exception", false, String(e)); }
console.log(JSON.stringify(results, null, 1));
try { ws.close(); } catch (e) { /* ignore */ }
proc.kill(); await sleep(500);
try { fs.rmSync(profile, { recursive: true, force: true }); } catch (e) { /* ignore */ }
process.exit(failed ? 1 : 0);
