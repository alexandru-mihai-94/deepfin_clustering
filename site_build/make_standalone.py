"""
Bundle the exhibit in docs/ into one self-contained HTML file plus an images/ folder, for dropping into
another website or web framework.

    python3 site_build/make_standalone.py [output folder]     (default: ../deepfin_exhibit_standalone)

It writes two versions that share images/: deepfin_exhibit.html (the site's look, light or dark with the
visitor's system setting) and deepfin_exhibit_white.html (always white, with the OIST talk colours: near-black
text and one red for rules and selected buttons, no logos).

The four pages become hash routes of a single page (#home, #species, #stages, #classifier, plus
#species/map and #stages/map). To keep out of the way of a host page:
  - every page's CSS is scoped to its page wrapper, and the colour variables live on #deepfin-exhibit,
    not :root
  - element ids get a dfx- prefix, the page scripts run inside one closure (each page in its own function),
    and the only global is window.DFX (DFX.go('species') navigates)
  - the offline cache, manifest and home-screen icons are left out (they belong to the published site)
Only one page is in the DOM at a time; going to a page (or "Play again") rebuilds it from its <template>.
Every rewrite below is checked, so a change to a page that this script does not expect stops it with an error.
"""
import json, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO.parent / "deepfin_exhibit_standalone"
PAGES = {"home": "index.html", "species": "spot_the_species.html", "stages": "stages.html", "classifier": "how_ai_sees.html"}
IMAGE_DIRS = ("ai_eye", "spot_species", "stages")
# the white version: OIST talk palette; percula keeps an orange (OIST frame rust, darkened to 4.5:1 for text)
WHITE = {"--bg": "#fff", "--fg": "#1a1a1a", "--muted": "#52514e", "--card": "#fff", "--line": "#c9c8c2",
         "--perc": "#b4570e", "--ocel": "#0d6b9a", "--accent": "#c80019", "--ok": "#15803d", "--bad": "#aa001e"}
WHITE_CSS = """
#deepfin-exhibit { --red:#c80019; }
#deepfin-exhibit :is(h1, h2, h3) { font-family:Georgia, 'Times New Roman', serif; }
#deepfin-exhibit h1 { padding-bottom:10px; border-bottom:2px solid var(--red); }
#deepfin-exhibit :is(.card, .act) { border-radius:0; border-top:2px solid var(--red); }
#deepfin-exhibit .act:is(:hover, :active) { border-color:var(--red); }
#deepfin-exhibit .act span.go { color:var(--red); }
#deepfin-exhibit :is(button, .btn, .top a, .site a) { border-radius:4px; }
#deepfin-exhibit :is(button.go, .btn.go, a[aria-current="page"], :is(.seg, .objs) button[aria-pressed="true"]) { background:var(--red); border-color:var(--red); color:#fff; }
#deepfin-exhibit .pick button[aria-pressed="true"] { box-shadow:0 0 0 4px var(--red); }
#deepfin-exhibit :is(.dots i.on, .bar div) { background:var(--red); }
"""
LINKS = {"index.html": "#home", "spot_the_species.html#map": "#species/map", "spot_the_species.html": "#species",
         "stages.html#map": "#stages/map", "stages.html": "#stages", "how_ai_sees.html": "#classifier"}


def sub(text, old, new, count=1):
    """Replace exactly `count` occurrences (0 = any number, at least one)."""
    n = text.count(old)
    assert n and (count == 0 or n == count), f"expected {count or 'some'} x {old[:70]!r}, found {n}"
    return text.replace(old, new)


# ---- CSS: scope every rule to the page wrapper; collect the colour tokens ----------------------------
def blocks(css):
    """Yield (prelude, body) for each top-level rule, matching nested braces."""
    i = 0
    while True:
        j = css.find("{", i)
        if j < 0:
            return
        depth, k = 1, j + 1
        while depth:
            depth += {"{": 1, "}": -1}.get(css[k], 0); k += 1
        yield css[i:j].strip(), css[j + 1:k - 1]
        i = k


def scope_css(css, page, tokens):
    scope = f".dfx-{page}"
    out = []
    for pre, body in blocks(re.sub(r"/\*.*?\*/", "", css, flags=re.S)):
        if pre.startswith("@media") and ":root" in body:  # dark-mode colour tokens
            tokens.setdefault(pre, {}).update(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))
        elif pre.startswith("@media"):
            out.append(f"{pre} {{\n{scope_css(body, page, tokens)}\n}}")
        elif pre.startswith("@keyframes"):
            out.append(f"{pre.replace('pop', 'dfx-pop')} {{{body}}}")
        elif pre == ":root":
            tokens.setdefault("", {}).update(re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", body))
        elif pre == "html":
            continue  # touch-action moves onto #deepfin-exhibit
        else:
            sels = []
            for s in (x.strip() for x in pre.split(",")):
                s = re.sub(r"#([A-Za-z][\w-]*)", r"#dfx-\1", s)  # id selectors follow the id prefix
                sels.append(scope if s == "body" else f"{scope} {s}")
            out.append(f"{', '.join(sels)} {{{body.replace('animation:pop', 'animation:dfx-pop')}}}")
    return "\n".join(out)


# ---- markup and scripts --------------------------------------------------------------------------------
def page_parts(name, file):
    html = (DOCS / file).read_text()
    css = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
    markup = re.search(r"<main>.*</main>", html, re.S).group(0)
    scripts = [s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
               if "serviceWorker" not in s and "exhibit iPad (home-screen app only)" not in s[:80]]
    js = "\n".join(scripts)

    for old, new in sorted(LINKS.items(), key=lambda kv: -len(kv[0])):  # longest first: stages.html#map before stages.html
        markup = markup.replace(f'href="{old}"', f'href="{new}"')
    markup = markup.replace('href="#map"', f'href="#{name}/map"')
    for target, route in (("'stages.html'", "stages"), ("'spot_the_species.html'", "species")):
        markup = markup.replace(f'onclick="location.href = {target}"', f"onclick=\"DFX.go('{route}')\"")
    markup = markup.replace('onclick="location.href = location.pathname"', f"onclick=\"DFX.go('{name}')\"")
    assert "location." not in markup and ".html" not in markup.replace("https://", "").split("http")[0], f"{file}: unhandled link"

    if js:
        js = sub(js, "$ = id => document.getElementById(id)", "$ = id => document.getElementById('dfx-' + id)")
        js = js.replace("scrollTo(0, 0)", "root.scrollIntoView({ block: 'start' })")
    if name in ("species", "stages"):
        js = sub(js, "addEventListener('hashchange', () => location.reload());  // nav links to #map on this same page\n", "")
        js = sub(js, "if (location.hash === '#map')", "if (sub === 'map')")
    if name == "species":
        js = sub(js, "document.querySelectorAll('[data-pick]')", "root.querySelectorAll('[data-pick]')")
    if name == "classifier":
        js = sub(js, "document.querySelectorAll('.screen')", "root.querySelectorAll('.screen')")
        js = sub(js, "document.addEventListener('click'", "root.addEventListener('click'")
        js = sub(js, js[js.index("// exhibit iPad (home-screen app only)"):js.index("function paint(")], "")
    assert "location." not in js and "document.addEventListener" not in js, f"{file}: unhandled navigation in script"

    both = markup + "\x00" + js  # ids (also those built in script templates) and image paths, in one pass
    both = re.sub(r'(?<![\w-])id="', 'id="dfx-', both)
    both = re.sub(rf"(?<![\w./#-])({'|'.join(IMAGE_DIRS)})/", r"images/\1/", both)
    markup, js = both.split("\x00")
    return css, markup, js


tokens, styles, templates, inits = {}, [], [], []
for name, file in PAGES.items():
    css, markup, js = page_parts(name, file)
    styles.append(f"/* ---- {name} ({file}) ---- */\n" + scope_css(css, name, tokens))
    templates.append(f'<template id="dfx-tpl-{name}">\n{markup}\n</template>')
    inits.append(f"  {name}(root, sub) {{\n{js}\n  }},")

data = {}
for var, f in (("SPOT", "spot_species/data.js"), ("AIE", "ai_eye/data.js")):
    text = (DOCS / f).read_text()
    data[var] = re.search(rf"window\.{var} = (.*);\s*$", text, re.S).group(1)
    json.loads(data[var])  # must be plain JSON
data = {k: re.sub(rf"(?<![\w./#-])({'|'.join(IMAGE_DIRS)})/", r"images/\1/", v) for k, v in data.items()}

base = tokens.pop("")
assert set(WHITE) <= set(base), f"WHITE names colours the pages do not have: {set(WHITE) - set(base)}"
css_vars = lambda t: "".join(f" {k}:{v};" for k, v in t.items())
dark = "\n".join(f"{q} {{ #deepfin-exhibit {{{css_vars(t)} }} }}" for q, t in tokens.items())
VERSIONS = {  # file: (colour variables, extra CSS)
    "deepfin_exhibit.html": (css_vars(base), "body { margin:0; background:#f7f9fb; }\n"
                             "@media (prefers-color-scheme: dark) { body { background:#0f1720; } }\n" + dark),
    "deepfin_exhibit_white.html": (css_vars({**base, **WHITE}), "body { margin:0; background:#fff; }" + WHITE_CSS),
}


def page(colours, extra):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DEEPFIN: How an AI Sorts Clownfish</title>
<!-- Generated by deepfin_clustering/site_build/make_standalone.py from the pages in docs/. To embed: copy the
     <style>, the #deepfin-exhibit element, the four <template>s and the <script>; keep images/ next to the page. -->
<style>
#deepfin-exhibit {{{colours} touch-action:manipulation; }}
{chr(10).join(styles)}
{extra}
</style>
</head>
<body>
<div id="deepfin-exhibit"></div>
{chr(10).join(templates)}
<script>
(() => {{
const SPOT = {data["SPOT"]};  // species maps (from site_build/spot_species_data.py)
const AIE = {data["AIE"]};  // classifier page data (from site_build/make_exhibit_data.py)

const PAGES = {{
{chr(10).join(inits)}
}};

// router: one page in the DOM at a time, rebuilt from its template on every visit
const app = document.getElementById('deepfin-exhibit');
function show() {{
  const [name, sub] = location.hash.slice(1).split('/');
  const page = PAGES[name] ? name : 'home';
  const root = document.createElement('div');
  root.className = 'dfx-page dfx-' + page;
  root.append(document.getElementById('dfx-tpl-' + page).content.cloneNode(true));
  app.replaceChildren(root);
  if (location.hash) root.scrollIntoView({{ block: 'start' }});
  PAGES[page](root, sub);
}}
function go(name, sub) {{
  const hash = '#' + name + (sub ? '/' + sub : '');
  if (location.hash === hash) show(); else location.hash = hash;
}}
// a link to the page you are on restarts it, as reloading a separate page did
app.addEventListener('click', e => {{
  const a = e.target.closest('a[href^="#"]');
  if (a && a.getAttribute('href') === location.hash) {{ e.preventDefault(); show(); }}
}});
addEventListener('hashchange', show);
// exhibit iPad (home-screen app only): back to the home page after 2 minutes without a touch
if (matchMedia('(display-mode: standalone)').matches) {{
  let idle; const wake = () => {{ clearTimeout(idle); idle = setTimeout(() => go('home'), 120000); }};
  ['pointerdown', 'keydown'].forEach(ev => addEventListener(ev, wake)); wake();
}}
window.DFX = {{ go }};
show();
}})();
</script>
</body>
</html>
"""

OUT.mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "images", ignore_errors=True)
for d in IMAGE_DIRS:
    shutil.copytree(DOCS / d, OUT / "images" / d, ignore=shutil.ignore_patterns("data.js", ".*"))
for file, (colours, extra) in VERSIONS.items():
    html = page(colours, extra)
    (OUT / file).write_text(html)
    print(f"wrote {OUT / file} ({len(html) / 1e3:.0f} kB)")
n_img = sum(1 for _ in (OUT / "images").rglob("*.*"))
size = sum(p.stat().st_size for p in (OUT / "images").rglob("*.*"))
print(f"and images/ ({n_img} files, {size / 1e6:.1f} MB)")
