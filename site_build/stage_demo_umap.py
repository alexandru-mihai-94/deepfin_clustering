"""
UMAP maps for docs/stages.html: A. ocellaris developmental stages, embedded with the 3 networks
used on the species page (src/deepfin/models.py). For the 144 thumbnail fish listed in the page:
crop each 4K frame to the fish, pad to a square with black, extract features, UMAP 2D
(n_neighbors=15, min_dist=0.3, seed 24), and write the coordinates into the page's
`const UMAP = {...};` line, one map per network. min_dist 0.3 rather than the paper's 0.01:
these features pack each stage into tight clumps at 0.01 (fish stack on top of each other on the
page), while 0.3 shows the same order as a continuous trajectory with near-identical
same-stage neighbour scores (73/78/66% vs 75/77/68%).
Images in rembg_FishImages already have their background removed with rembg.

    site_build/venv/bin/python -u site_build/stage_demo_umap.py
"""
import json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import umap

REPO = Path(__file__).resolve().parent.parent  # deepfin_clustering/
DATA = REPO.parent  # the analysis folder next to the repo: flank_umap/, rembg_FishImages/, inat_wild/ (not in git)
sys.path.insert(0, str(REPO / "src"))
from deepfin.models import extract

IMG_DIR = DATA / "rembg_FishImages"
PAGE = REPO / "docs" / "stages.html"
MODELS = ["MobileNetV2", "VGG16", "DenseNet121"]  # same as docs/spot_the_species.html

page = PAGE.read_text()
blocks = dict(re.findall(r'(\d): "([^"]+)"', page))
ids = [i for s in sorted(blocks) for i in blocks[s].split()]
stage = np.array([int(s) for s in sorted(blocks) for _ in blocks[s].split()])
assert len(ids) == 144, len(ids)


def crop_to_fish(f):
    im = Image.open(f).convert("RGB")
    im.thumbnail((1600, 1600))
    x0, y0, x1, y1 = im.convert("L").point(lambda v: 255 if v > 25 else 0).getbbox()
    p = int(0.04 * (x1 - x0))
    return im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))


imgs = None
maps = {}
for m in MODELS:
    fp = REPO / "site_build" / "cache" / f"stages144_{m}.npz"
    if fp.exists():
        X = np.load(fp)["features"]
    else:
        imgs = imgs or [crop_to_fish(next(IMG_DIR.glob(f"{i}_*rembg.jpg"))) for i in ids]
        X = extract(imgs, m, bg=(0, 0, 0), tag="stages")
        np.savez_compressed(fp, features=X, ids=np.array(ids), stage=stage)
    xy = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=24).fit_transform(X)
    nbr = NearestNeighbors(n_neighbors=6).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    same = float(np.mean(stage[nbr] == stage[:, None]))  # share of map neighbours at the same stage
    within1 = float(np.mean(np.abs(stage[nbr] - stage[:, None]) <= 1))
    print(f"{m}: {X.shape[1]} features; 5 nearest map neighbours: same stage {same:.0%}, within one stage {within1:.0%}")
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))  # 0..1 for CSS positioning
    maps[m] = {"same_stage": round(same, 3), "xy": {i: [round(float(a), 4), round(float(b), 4)] for i, (a, b) in zip(ids, xy)}}

page, n = re.subn(r"^const UMAP = .*;$", lambda _: f"const UMAP = {json.dumps(maps, separators=(',', ':'))};", page, flags=re.M)
assert n == 1, "const UMAP line not found in stages.html"
PAGE.write_text(page)
print("updated", PAGE)
