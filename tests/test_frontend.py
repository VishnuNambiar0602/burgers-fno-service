"""Static checks of the served page, Node unit tests of lib.js, and real-browser checks (skipped if Node/Edge are missing)."""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from html.parser import HTMLParser

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.main import app  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC = os.path.join(ROOT, "app", "static")
read = lambda n: open(os.path.join(STATIC, n), encoding="utf-8").read()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.labels_for, self.wrapped, self.attrs, self._in_label, self.h1 = set(), set(), set(), [], 0, 0

    def handle_starttag(self, tag, a):
        a = dict(a)
        self.attrs.append((tag, a))
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "h1":
            self.h1 += 1
        if tag == "label":
            self._in_label += 1
            if "for" in a:
                self.labels_for.add(a["for"])
        if tag in ("input", "select", "textarea") and self._in_label:
            self.wrapped.add(a.get("id", id(a)))

    def handle_endtag(self, tag):
        if tag == "label":
            self._in_label -= 1


def parsed():
    p = Collect()
    p.feed(read("index.html"))
    return p


def test_root_serves_html_with_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert client.head("/").status_code == 200
    assert "content-security-policy" not in client.get("/docs").headers      # Swagger UI needs its own CDN assets


@pytest.mark.parametrize("name,ctype", [("app.js", "javascript"), ("lib.js", "javascript"), ("style.css", "text/css")])
def test_static_assets_served(client, name, ctype):
    r = client.get(f"/static/{name}")
    assert r.status_code == 200 and ctype in r.headers["content-type"] and len(r.content) > 500


def test_page_states_correct_equation_and_fixed_settings():
    html = read("index.html")
    assert "(ν/π)" in html and "ν = 0.01 (fixed)" in html and "256 grid points" in html
    assert "Why ν/π and not just ν?" in html                                   # explained in the equation dialog
    for must in ["Sharp fronts (shocks)", "does not copy its own input", "does not conserve mass", "One physics setting only", "CC BY 4.0",
                 "The warnings are not a guarantee"]:
        assert must in html, must


def _visible_text(html):
    html = re.sub(r"<template.*?</template>", "", html, flags=re.S)
    html = re.sub(r"<dialog.*?</dialog>", "", html, flags=re.S)
    html = re.sub(r"<(script|style).*?</>", "", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def test_explanations_live_only_in_dialogs_and_are_opt_in():
    vis = _visible_text(read("index.html"))
    assert len(vis.split()) < 220, len(vis.split())                            # the always-visible page stays short
    for hidden_text in ["What each control does", "Gibbs", "relative L2", "Why ν/π", "nonlinear transport", "Mass drift: the total"]:
        assert hidden_text not in vis, hidden_text
    topics = re.findall(r'data-help="([a-z]+)"', read("index.html"))
    assert len(set(topics)) >= 9
    for t in set(topics):
        assert f'<template id="help-{t}">' in read("index.html"), t


def test_every_id_used_by_the_script_exists_in_the_page():
    used = set(re.findall(r'\$\("([A-Za-z0-9_-]+)"\)', read("app.js")))
    missing = used - parsed().ids
    assert not missing, missing


def test_no_external_hosts_and_no_inline_code_or_styles():
    for n in ("index.html", "app.js", "lib.js", "style.css"):
        assert not re.search(r"https?://(?!127\.0\.0\.1)", read(n)), n      # nothing is loaded from third parties (the local address appears only in help text)
    p = parsed()
    assert not any("style" in a or any(k.startswith("on") for k in a) for _, a in p.attrs)
    assert not any(t == "script" and "src" not in a for t, a in p.attrs)    # CSP: no inline scripts


def test_accessibility_basics():
    p = parsed()
    html = read("index.html")
    assert '<html lang="en">' in html and 'name="viewport"' in html and p.h1 == 1 and 'class="skip"' in html
    for tag, a in p.attrs:
        if tag in ("input", "select", "textarea"):
            assert a.get("aria-label") or a.get("id") in p.labels_for or a.get("id") in p.wrapped or id(a) in p.wrapped, (tag, a)
        if tag == "canvas":
            assert a.get("role") == "img" and a.get("aria-label"), a
        if tag == "section":
            assert a.get("aria-labelledby"), a
    assert 'role="status"' in html and 'aria-live="polite"' in html
    assert html.count("<dialog") >= 4 and all(a.get("aria-labelledby") for tag, a in p.attrs if tag == "dialog")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_lib_js_unit_tests():
    r = subprocess.run(["node", "tests/js/test_lib.js"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert r.returncode == 0, r.stdout[-1500:]
    r = subprocess.run(["node", "--check", "app/static/app.js"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
    assert r.returncode == 0, r.stderr


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _serve(env_extra=None):
    port = _free_port()
    env = {**os.environ, **(env_extra or {})}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api", timeout=2)
            break
        except Exception:
            time.sleep(0.5)
    return proc, port


def _browser(port, scenario, url=None):
    out = os.path.join(ROOT, "build", "ui_test_out")
    os.makedirs(out, exist_ok=True)
    r = subprocess.run(["node", "scripts/ui_check.mjs", url or f"http://127.0.0.1:{port}", out, scenario], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=400)
    try:
        res = json.loads(r.stdout)
    except Exception:
        pytest.fail(r.stdout[-800:] + r.stderr[-800:])
    return r.returncode, res


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_end_to_end():
    proc, port = _serve()
    try:
        code, res = _browser(port, "all")
        if "skipped" in res:
            pytest.skip(res["skipped"])
        bad = {k: v for k, v in res.items() if not v["ok"]}
        assert code == 0 and not bad, bad
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_shows_error_and_no_result_when_model_is_missing():
    proc, port = _serve({"MODEL_PATH": os.path.join(ROOT, "model", "does_not_exist.pt")})
    try:
        code, res = _browser(port, "error")
        if "skipped" in res:
            pytest.skip(res["skipped"])
        bad = {k: v for k, v in res.items() if not v["ok"]}
        assert code == 0 and not bad, bad
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_html_uses_relative_asset_links_so_it_also_works_from_disk():
    html = read("index.html")
    assert 'href="style.css"' in html and 'src="lib.js"' in html and 'src="app.js"' in html
    assert "/static/" not in html.split("<body>")[0] and 'src="/static/' not in html
    for n in ("style.css", "lib.js", "app.js"):
        assert os.path.exists(os.path.join(STATIC, n))          # the files sit next to index.html


def test_root_serves_absolute_versioned_links_and_never_a_stale_copy(client):
    r = client.get("/")
    assert re.search(r'href="/static/style\.css\?v=\d+"', r.text) and re.search(r'src="/static/app\.js\?v=\d+"', r.text)
    assert r.headers["cache-control"] == "no-cache"
    assert client.get("/static/style.css").headers["cache-control"] == "no-cache"
    s = client.get("/static/index.html")
    assert s.status_code == 200 and 'href="style.css"' in s.text                 # the raw page also works at /static/


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_page_opened_from_disk_is_styled_and_explains_itself():
    url = "file:///" + os.path.join(STATIC, "index.html").replace("\\", "/")
    code, res = _browser(0, "offline", url)
    if "skipped" in res:
        pytest.skip(res["skipped"])
    bad = {k: v for k, v in res.items() if not v["ok"]}
    assert code == 0 and not bad, bad


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_static_files_without_the_api_are_styled_and_explain_themselves():
    port = _free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", STATIC], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.5)
        code, res = _browser(port, "offline", f"http://127.0.0.1:{port}/index.html")
        if "skipped" in res:
            pytest.skip(res["skipped"])
        bad = {k: v for k, v in res.items() if not v["ok"]}
        assert code == 0 and not bad, bad
    finally:
        srv.terminate()
        srv.wait(timeout=10)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_one_screen_in_realistic_laptop_windows():
    proc, port = _serve()
    try:
        code, res = _browser(port, "matrix")
        if "skipped" in res:
            pytest.skip(res["skipped"])
        bad = {k: v for k, v in res.items() if not (v["ok"] and v["info"]["fits"])}
        assert code == 0 and not bad, bad
    finally:
        proc.terminate()
        proc.wait(timeout=10)
