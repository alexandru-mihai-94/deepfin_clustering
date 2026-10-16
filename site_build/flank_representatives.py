"""
Pick one representative photograph per individual and flank for the left/right flank
analysis, and write ../flank_umap/representatives.csv (processed + original filename).

Candidates are the images left in docs/percula_ocellaris/processed_rembg_v3 (bad frames
already deleted). Within each fish+flank, the representative is the medoid: the duplicate
closest to the group's mean appearance (cut-out flattened on white, all in the canonical
head-left pose, compared at 256x120 RGB). Also writes representatives_sheet.jpg to check them.

    site_build/venv/bin/python site_build/flank_representatives.py [--force]

The website, the saved features and the model scores were all built from the existing representatives.csv,
so the script will not overwrite it without --force. Ties (a fish flank with two photos is always a tie: both are
equally far from their mean) go to the lower duplicate number; the first version of this file left them to
rounding, which picked differently for 7 of the 118 flanks on another machine.
"""
import csv, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent  # deepfin_clustering/
DATA = REPO.parent  # the analysis folder next to the repo: flank_umap/, rembg_FishImages/, inat_wild/ (not in git)
IMG = REPO / "docs" / "percula_ocellaris"
OUT = DATA / "flank_umap"
NAME = re.compile(r"^(ocellaris|percula)_sample_(\d+)_([LR])_duplicate(\d+)\.png$")


def on_white(path):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    return Image.alpha_composite(bg, im).convert("RGB")  # transparent pixels still hold the old backdrop


groups = defaultdict(list)
for f in sorted((IMG / "processed_rembg_v3").glob("*.png")):
    m = NAME.match(f.name)
    assert m, f"unexpected filename {f.name}"
    groups[(m[1], int(m[2]), m[3])].append(f)

rows = []
for (sp, n, side), files in sorted(groups.items()):
    X = np.stack([np.asarray(on_white(f).resize((256, 120)), np.float32).ravel() / 255 for f in files])
    d = np.round(((X - X.mean(0)) ** 2).sum(1), 3)  # round so exact ties stay ties on any machine
    pick = files[int(np.argmin(d))]  # first minimum = lowest duplicate number
    orig = IMG / "originals_all" / (pick.stem + ".jpg")
    assert orig.exists(), orig
    rows.append(dict(species=sp, individual=f"{sp}_sample_{n:02d}", side=side, n_candidates=len(files),
                     processed_file=pick.name, original_file=orig.name))

OUT.mkdir(exist_ok=True)
if (OUT / "representatives.csv").exists() and "--force" not in sys.argv:
    raise SystemExit(f"{OUT / 'representatives.csv'} exists and the site is built from it; pass --force to remake it")
with open(OUT / "representatives.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)

# contact sheet: processed | original for every representative
W, H, C = 220, 100, 6
sheet = Image.new("RGB", (C * 2 * W, ((len(rows) + C - 1) // C) * (H + 14)), "white")
d = ImageDraw.Draw(sheet)
for k, r in enumerate(rows):
    x, y = (k % C) * 2 * W, (k // C) * (H + 14)
    for j, im in enumerate((on_white(IMG / "processed_rembg_v3" / r["processed_file"]),
                            Image.open(IMG / "originals_all" / r["original_file"]).convert("RGB"))):
        im.thumbnail((W - 4, H)); sheet.paste(im, (x + j * W + 2, y + 14))
    d.text((x + 3, y + 1), f"{r['individual']} {r['side']}  ({r['n_candidates']} cand.)",
           fill="#c2410c" if r["species"] == "percula" else "#1d4ed8")
sheet.save(OUT / "representatives_sheet.jpg", quality=85)

ind = {r["individual"] for r in rows}
both = sum(1 for i in ind if {r["side"] for r in rows if r["individual"] == i} == {"L", "R"})
print(f"{len(rows)} representatives, {len(ind)} individuals ({both} with both flanks); "
      f"percula {sum(r['species'] == 'percula' for r in rows)}, ocellaris {sum(r['species'] == 'ocellaris' for r in rows)}")
print("wrote", OUT / "representatives.csv", "and", OUT / "representatives_sheet.jpg")
