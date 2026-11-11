"""Static checks of the Agent-mode toggle and the PDF button, plus real-browser checks. Written 2026-11-11; NOT yet run (no test runs until the user says)."""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC = os.path.join(ROOT, "app", "static")
read = lambda n: open(os.path.join(STATIC, n), encoding="utf-8").read()


def test_buttons_exist_and_machine_code_is_not_in_the_default_page():
    html = read("index.html")
    assert 'id="agent-toggle"' in html and 'aria-pressed="false"' in html and 'id="dl-pdf"' in html
    assert re.search(r'id="dl-pdf"[^>]*disabled', html)                                   # disabled until there is a result
    for n in ("index.html", "app.js", "lib.js"):                                           # lazily loaded: the page itself never contains machine content
        txt = read(n)
        for needle in ("agent-manifest", "agent-state", "/mcp"):
            assert needle not in txt, (n, needle)
    assert "agent.js" not in html and "pdf.js" not in html                                 # not referenced by script tags
    assert 'loadScript("agent.js")' in read("app.js") and 'loadScript("pdf.js")' in read("app.js")


def test_agent_and_pdf_scripts_exist_and_are_csp_friendly():
    for n in ("agent.js", "pdf.js"):
        txt = read(n)
        assert len(txt) > 3000 and not re.search(r"https?://", txt), n
        assert "eval(" not in txt and "innerHTML" not in txt, n
    assert "burgers:state" in read("agent.js") and "/mcp" in read("agent.js") and "EmbeddedFile" in read("pdf.js") and "DCTDecode" in read("pdf.js")


def test_machine_view_is_removed_when_the_mode_is_switched_off():
    js = read("agent.js")
    assert "root.remove()" in js and 'removeEventListener("burgers:state"' in js


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_real_browser_agent_mode_and_pdf():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(80):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        out = os.path.join(ROOT, "build", "ui_test_out")
        os.makedirs(out, exist_ok=True)
        r = subprocess.run(["node", "scripts/ui_check.mjs", f"http://127.0.0.1:{port}", out, "features"], cwd=ROOT, capture_output=True, text=True, timeout=400)
        res = json.loads(r.stdout)
        if "skipped" in res:
            pytest.skip(res["skipped"])
        bad = {k: v for k, v in res.items() if not v["ok"]}
        assert r.returncode == 0 and not bad, bad
    finally:
        proc.terminate()
        proc.wait(timeout=10)
