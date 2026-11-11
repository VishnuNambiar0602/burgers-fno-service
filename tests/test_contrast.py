"""WCAG contrast of the page's colour tokens, light and dark themes (text >= 4.5:1, plotted lines >= 3:1)."""
import os
import re

STYLE = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "style.css"), encoding="utf-8").read()


def tokens(block_start):
    i = STYLE.index(block_start)
    body = STYLE[STYLE.index("{", i) + 1:STYLE.index("}", i)]
    return dict(re.findall(r"--([a-z0-9-]+):(#[0-9a-fA-F]{6})", body))


def lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted([lum(a), lum(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


THEMES = {"light": tokens(":root{"), "dark": tokens(':root[data-theme="dark"]')}
TEXT_PAIRS = [("ink", "bg"), ("ink", "card"), ("muted", "bg"), ("muted", "card"), ("brand", "bg"), ("brand-ink", "brand"),
              ("warn-ink", "warn-bg"), ("bad-ink", "bad-bg"), ("ink", "ok-bg"), ("ink", "idle-bg")]


def test_text_contrast_meets_aa_in_both_themes():
    for name, t in THEMES.items():
        for fg, bg in TEXT_PAIRS:
            if fg in t and bg in t:
                assert ratio(t[fg], t[bg]) >= 4.5, (name, fg, bg, round(ratio(t[fg], t[bg]), 2))


def test_plot_line_colours_are_visible_on_the_card():
    for name, t in THEMES.items():
        for line in ("fno", "solver"):
            assert ratio(t[line], t["card"]) >= 3.0, (name, line, round(ratio(t[line], t["card"]), 2))
