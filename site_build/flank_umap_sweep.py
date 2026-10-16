"""
Model x UMAP sweep for the left/right flank dataset (A. ocellaris / A. percula, Taiwan).

Inputs : ../flank_umap/representatives.csv (from flank_representatives.py)
Image sets:
  processed  docs/percula_ocellaris/processed_rembg_v3 cut-outs, flattened on white
  original   docs/percula_ocellaris/originals_all full photographs
Every image is padded to a square (no stretching) and resized to the model's input size.

1. Features: global-average-pooled ImageNet features from each model, with that model's
   own preprocess_input. "EfficientNetV2L_deepfin" repeats the repo's current choice
   (vgg16.preprocess_input on EfficientNetV2L) for comparison. Cached per set+model.
2. UMAP sweep over n_neighbors x min_dist, scored on 3 seeds:
     species_kmeans  k-means k=2 on the 2-D map vs species (best label matching)
     species_5nn     leave-one-out 5-NN species accuracy in the 2-D map
     flank_match     for fish with both flanks: is the nearest neighbour the same fish's other flank
   Feature-space baselines (no UMAP, cosine) are reported per model too.
3. A thumbnail figure for every set x model x n_neighbors x min_dist (seed 24), an overview
   grid per set x model, and scores.csv ranked by species_kmeans.

    site_build/venv/bin/python -u site_build/flank_umap_sweep.py
"""
import csv, os, sys, time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

REPO = Path(__file__).resolve().parent.parent  # deepfin_clustering/
DATA = REPO.parent  # the analysis folder next to the repo: flank_umap/, rembg_FishImages/, inat_wild/ (not in git)
sys.path.insert(0, str(REPO / "src"))
from deepfin.models import MODELS, extract

IMG = REPO / "docs" / "percula_ocellaris"
OUT = DATA / "flank_umap"
NEIGHBORS = (5, 10, 15, 30, 50)
MIN_DIST = (0.0, 0.01, 0.1, 0.5)
SEEDS = (24, 0, 1)  # figures use the first
COLOR = {"percula": "#c2410c", "ocellaris": "#1d4ed8"}

rows = list(csv.DictReader(open(OUT / "representatives.csv")))
species = np.array([r["species"] for r in rows])
fish = np.array([r["individual"] for r in rows])
n = len(rows)
# index of the other flank of the same fish, or -1
other = np.array([next((j for j in range(n) if j != i and fish[j] == fish[i]), -1) for i in range(n)])


def load_sets():
    sets = {"processed": [], "original": []}
    for r in rows:
        p = Image.open(IMG / "processed_rembg_v3" / r["processed_file"]).convert("RGBA")
        sets["processed"].append(Image.alpha_composite(Image.new("RGBA", p.size, "white"), p).convert("RGB"))
        o = Image.open(IMG / "originals_all" / r["original_file"])
        o.draft("RGB", (1200, 1200))  # fast JPEG decode at reduced size (16 MP originals)
        o = o.convert("RGB"); o.thumbnail((1200, 1200))
        sets["original"].append(o)
    return sets


def scores(Z, metric="euclidean"):
    """species_kmeans (2-D maps only), species_5nn, flank_match."""
    nn = NearestNeighbors(n_neighbors=6, metric=metric).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]  # drop self
    knn = np.mean([np.mean(species[idx[i]] == species[i]) > 0.5 for i in range(n)])
    has = other >= 0
    flank = np.mean(idx[has, 0] == other[has])
    km = None
    if Z.shape[1] == 2:
        lab = KMeans(2, random_state=24, n_init=10).fit_predict(Z)
        ct = np.array([[np.sum((species == s) & (lab == c)) for c in (0, 1)] for s in ("percula", "ocellaris")])
        r, c = linear_sum_assignment(-ct)
        km = ct[r, c].sum() / n
    return km, knn, flank


def thumb(im, size=84):
    t = im.copy(); t.thumbnail((size, size))
    return np.asarray(t)


def figure(xy, thumbs, title, path):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    for i in np.where(other > np.arange(n))[0]:  # thin line joining the two flanks of a fish
        ax.plot(*xy[[i, other[i]]].T, color="#999", lw=0.6, zorder=0)
    ax.scatter(xy[:, 0], xy[:, 1], s=0)
    for t, (a, b), s in zip(thumbs, xy, species):
        ax.add_artist(AnnotationBbox(OffsetImage(t, zoom=1.0), (a, b), frameon=True, pad=0.05,
                                     bboxprops=dict(edgecolor=COLOR[s], lw=1.2)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    print(f"{n} images, {len(set(fish))} individuals, {np.sum(other >= 0) // 2} with both flanks", flush=True)
    sets = load_sets()
    thumbs = {k: [thumb(im) for im in v] for k, v in sets.items()}
    results, base = [], []
    for tag, imgs in sets.items():
        for name in MODELS:
            fp = OUT / "features" / f"{tag}_{name}.npz"
            if fp.exists():
                X = np.load(fp)["features"]
            else:
                X = extract(imgs, name, tag=tag)
                np.savez_compressed(fp, features=X, files=np.array([r[f"{tag}_file"] for r in rows]))
            _, knn, flank = scores(X, metric="cosine")
            base.append([tag, name, X.shape[1], knn, flank])
            print(f"[feature space] {tag}/{name}: species 5-NN {knn:.0%}, flank match {flank:.0%}", flush=True)

            figdir = OUT / "figures" / tag / name; figdir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(len(NEIGHBORS), len(MIN_DIST), figsize=(4 * len(MIN_DIST), 3.4 * len(NEIGHBORS)), dpi=90)
            t0 = time.time()
            for a, nb in enumerate(NEIGHBORS):
                for b, md in enumerate(MIN_DIST):
                    per_seed = []
                    for seed in SEEDS:
                        xy = umap.UMAP(n_neighbors=nb, min_dist=md, random_state=seed).fit_transform(X)
                        per_seed.append(scores(xy))
                        if seed == SEEDS[0]:
                            xy0 = xy
                    m = np.array(per_seed, dtype=float)
                    results.append([tag, name, nb, md, *m.mean(0), *m.std(0)])
                    lbl = (f"species k-means {m[:, 0].mean():.0%}  5-NN {m[:, 1].mean():.0%}  "
                           f"flank match {m[:, 2].mean():.0%}  (mean of {len(SEEDS)} seeds)")
                    figure(xy0, thumbs[tag], f"{tag} / {name} / n_neighbors={nb} min_dist={md}\n{lbl}",
                           figdir / f"umap_n{nb:02d}_md{md}.png")
                    ax = axes[a, b]
                    for s in COLOR:
                        ax.scatter(*xy0[species == s].T, s=8, c=COLOR[s])
                    ax.set_title(f"n={nb} md={md}\nkm {m[:, 0].mean():.0%} 5nn {m[:, 1].mean():.0%} flank {m[:, 2].mean():.0%}", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"{tag} / {name}  (orange = percula, blue = ocellaris; seed {SEEDS[0]} shown)")
            fig.tight_layout(); fig.savefig(OUT / f"overview_{tag}_{name}.png"); plt.close(fig)
            print(f"[umap sweep] {tag}/{name}: {len(NEIGHBORS) * len(MIN_DIST)} settings done ({time.time() - t0:.0f}s)", flush=True)

    results.sort(key=lambda r: (-r[4], -r[5], -r[6]))
    with open(OUT / "scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_set", "model", "n_neighbors", "min_dist", "species_kmeans", "species_5nn", "flank_match",
                    "species_kmeans_sd", "species_5nn_sd", "flank_match_sd", "figure"])
        for r in results:
            w.writerow([*r[:4], *(round(v, 3) for v in r[4:]), f"figures/{r[0]}/{r[1]}/umap_n{r[2]:02d}_md{r[3]}.png"])
    with open(OUT / "feature_space_baseline.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_set", "model", "n_features", "species_5nn_cosine", "flank_match_cosine"])
        w.writerows([[a, b, c, round(d, 3), round(e, 3)] for a, b, c, d, e in base])
    print("\nTop 10 by species k-means accuracy (mean of 3 seeds):", flush=True)
    for r in results[:10]:
        print(f"  {r[0]:9s} {r[1]:24s} n={r[2]:2d} md={r[3]:<4}  k-means {r[4]:.0%} (sd {r[7]:.0%})  5-NN {r[5]:.0%}  flank {r[6]:.0%}")
    print("wrote", OUT / "scores.csv", OUT / "feature_space_baseline.csv", "and figures/", flush=True)


if __name__ == "__main__":
    main()
