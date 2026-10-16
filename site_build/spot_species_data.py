"""
Data for docs/spot_the_species.html, from the Taiwan left/right flank set.

- One image per fish: the left-flank representative if there is one, else the right
  (from ../flank_umap/representatives.csv; processed_rembg_v3 cut-outs on white).
- Features of the top 3 models of the flank sweep, reused from ../flank_umap/features/.
- Per model: UMAP n_neighbors=5 (min_dist/seed in PARAMS), then k-means k=2 on the map,
  clusters named by best match to species (that is DEEPFIN's call for each fish).
  DenseNet121's perfect split is seed-dependent: 8/30 seeds at min_dist 0.01, 2/30 at 0.1.
- Quiz: the 5 Percula + 5 Ocellaris that look most like the other species: cosine distance
  to the nearest fish of the other species, ranked within each model, ranks summed over models.
  QUIZ_EXCLUDE drops fish whose cut-out has visible artefacts.
- Two wild iNaturalist examples: the unmodified original photos, placed by hand in
  docs/spot_species as wild_<species>_raw.jpg (credits from ../inat_wild/credits.csv).

Writes docs/spot_species/ (images + data.js) and ../flank_umap/spot_species_dataset.csv.

    site_build/venv/bin/python site_build/spot_species_data.py
"""
import csv, json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import umap

REPO = Path(__file__).resolve().parent.parent  # deepfin_clustering/
DATA = REPO.parent  # the analysis folder next to the repo: flank_umap/, rembg_FishImages/, inat_wild/ (not in git)
SRC = REPO / "docs" / "percula_ocellaris" / "processed_rembg_v3"
FLANK = DATA / "flank_umap"
WEB = REPO / "docs" / "spot_species"
# top 3 on processed images in flank_umap/scores.csv: (n_neighbors, min_dist, seed)
PARAMS = {"MobileNetV2": (5, 0.1, 24), "VGG16": (5, 0.1, 24), "DenseNet121": (5, 0.01, 4)}
MODELS = list(PARAMS)
QUIZ_EXCLUDE = {"percula_sample_21"}  # flat grey cut-out artefact in front of the tail
WILD = {"Ocellaris": "Ocellaris_254417945.jpg", "Percula": "Percula_282414654.jpg"}
NAME = {"percula": "Percula", "ocellaris": "Ocellaris"}

reps = list(csv.DictReader(open(FLANK / "representatives.csv")))
by_fish = {}
for r in reps:
    if r["individual"] not in by_fish or r["side"] == "L":
        by_fish[r["individual"]] = r
rows = sorted(by_fish.values(), key=lambda r: r["individual"])
ids = [r["individual"] for r in rows]
sp = np.array([NAME[r["species"]] for r in rows])
print(f"{len(rows)} fish: {np.sum(sp == 'Percula')} Percula, {np.sum(sp == 'Ocellaris')} Ocellaris; "
      f"{sum(r['side'] == 'R' for r in rows)} use the right flank")

out = {"fish": dict(zip(ids, sp.tolist())), "models": {}}
cross_rank = np.zeros(len(rows))
for m in MODELS:
    f = np.load(FLANK / "features" / f"processed_{m}.npz")
    pos = {name: i for i, name in enumerate(f["files"])}
    X = f["features"][[pos[r["processed_file"]] for r in rows]]
    nb, md, seed = PARAMS[m]
    xy = umap.UMAP(n_neighbors=nb, min_dist=md, random_state=seed).fit_transform(X)
    lab = KMeans(2, random_state=24, n_init=10).fit_predict(xy)
    ct = np.array([[np.sum((sp == s) & (lab == c)) for c in (0, 1)] for s in ("Percula", "Ocellaris")])
    ri, ci = linear_sum_assignment(-ct)
    group = {int(c): ("Percula", "Ocellaris")[r] for r, c in zip(ri, ci)}
    acc = ct[ri, ci].sum() / len(rows)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    out["models"][m] = {"accuracy": round(float(acc), 3), "umap": {"n_neighbors": nb, "min_dist": md, "seed": seed},
                        "xy": {i: [round(float(a), 4), round(float(b), 4)] for i, (a, b) in zip(ids, xy)},
                        "group": {i: group[int(l)] for i, l in zip(ids, lab)}}
    D = pairwise_distances(X, metric="cosine")
    cross = np.array([D[i, sp != sp[i]].min() for i in range(len(rows))])
    cross_rank += rankdata(cross)
    wrong = [i for i, l in zip(ids, lab) if out["models"][m]["group"][i] != out["fish"][i]]
    print(f"{m}: species accuracy {acc:.1%} (k-means on UMAP n={nb} md={md} seed={seed}); misplaced: {wrong}")

ok = np.array([i not in QUIZ_EXCLUDE for i in ids])
quiz = [ids[i] for s in ("Percula", "Ocellaris") for i in np.argsort(np.where((sp == s) & ok, cross_rank, np.inf))[:5]]
out["quiz"] = quiz

WEB.mkdir(exist_ok=True)
for f in WEB.glob("*_sample_*.jpg"):
    f.unlink()  # drop images of fish no longer in the quiz (the wild_*_raw.jpg photos stay)
credits = {r["file"]: r for r in csv.DictReader(open(DATA / "inat_wild" / "credits.csv"))}
out["wild"] = []
for s, f in WILD.items():
    assert (WEB / f"wild_{s.lower()}_raw.jpg").exists(), f"missing wild_{s.lower()}_raw.jpg"
    c = credits[f]
    out["wild"].append({"species": s, "img": f"wild_{s.lower()}_raw.jpg", "place": c["place"],
                        "credit": c["attribution"].replace("(c) ", "").split(",")[0],
                        "license": c["license"].upper().replace("CC-", "CC "), "url": c["observation"]})
for r in rows:
    p = Image.open(SRC / r["processed_file"]).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", p.size, "white"), p).convert("RGB")
    if r["individual"] in quiz:
        big = im.copy(); big.thumbnail((900, 900)); big.save(WEB / f"{r['individual']}.jpg", quality=85)
    im.thumbnail((180, 180)); im.save(WEB / f"{r['individual']}_t.jpg", quality=80)
(WEB / "data.js").write_text("// generated by site_build/spot_species_data.py\nwindow.SPOT = "
                             + json.dumps(out, separators=(",", ":")) + ";\n")

with open(FLANK / "spot_species_dataset.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["individual", "species", "side", "processed_file", "original_file", "quiz", "lookalike_rank_sum",
                *[f"{m}_{k}" for m in MODELS for k in ("x", "y", "group")]])
    for i, r in enumerate(rows):
        w.writerow([r["individual"], sp[i], r["side"], r["processed_file"], r["original_file"], int(ids[i] in quiz),
                    int(cross_rank[i]),
                    *[v for m in MODELS for v in (*out["models"][m]["xy"][ids[i]], out["models"][m]["group"][ids[i]])]])

print("quiz fish:", quiz)
for m in MODELS:
    print(f"  {m}: DEEPFIN right on {sum(out['models'][m]['group'][q] == out['fish'][q] for q in quiz)}/10 quiz fish")
print("wrote", WEB, "and", FLANK / "spot_species_dataset.csv")
