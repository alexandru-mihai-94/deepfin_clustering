"""
Sort a folder of images into groups in four choices, without writing any code:

  1. a folder of images
  2. a neural network that turns each image into a fingerprint of numbers (see deepfin.models)
  3. a way to squeeze the fingerprints into a few dimensions: PCA (fast, linear) or UMAP (keeps neighbours together);
     the map shows the first two
  4. a way to find groups in those dimensions: HDBSCAN (finds the number of groups itself) or k-means (you set it)

Use it from the point-and-click app (`deepfin-gui`), the command line (`deepfin --help`), or Python:

    from deepfin.explore import run
    run("my_images", model="MobileNetV2", reduce="umap", cluster="kmeans", k=3)

or step by step, which is what the app does to redo steps 3-4 without recomputing the fingerprints:

    paths = find_images("my_images"); paths, thumbs, X = fingerprint(paths, "VGG16")
    emb, variance = embed(X, "pca", dims=5); labels = group(emb, "kmeans", k=2)

Results go to an output folder (`save`): map.png and map.pdf (the map with thumbnails, coloured by group),
clusters.csv (one row per image: file, group and its coordinates in every dimension), features.csv (the raw
fingerprints), pca_variance.csv (PCA only), settings.json (the choices used) and sorted/ with a copy of every image
in one folder per group (HDBSCAN puts images that fit no group in sorted/noise).
"""
from __future__ import annotations

import argparse, csv, io, json, os, shutil, time
from pathlib import Path

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# the three networks of the exhibit first: they separated Percula/Ocellaris best (deepfin_clustering/docs)
MODELS = ["MobileNetV2", "VGG16", "DenseNet121", "ResNet50", "EfficientNetB0", "InceptionV3", "EfficientNetV2L"]
NOISE_GREY = "#9aa0a6"


def find_images(folder) -> list[Path]:
    """Image files directly inside `folder` (not in sub-folders), sorted by name."""
    return sorted(p for p in Path(folder).expanduser().iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES
                  and not p.name.startswith("."))


def check(n, reduce="umap", dims=2, cluster="hdbscan", k=2, min_cluster_size=5):
    """Stop before the slow steps on choices that cannot work for n images."""
    if reduce not in ("umap", "pca") or cluster not in ("hdbscan", "kmeans"):
        raise ValueError("reduce must be 'umap' or 'pca', and cluster 'hdbscan' or 'kmeans'")
    if n < 3:
        raise ValueError(f"Found {n} images; need at least 3 (jpg, png, tif, bmp or webp).")
    if not 2 <= dims <= n - 1:
        raise ValueError(f"The number of dimensions must be between 2 and {n - 1} (one less than the number of images).")
    if cluster == "kmeans" and not 2 <= k <= n:
        raise ValueError(f"The number of groups must be between 2 and the number of images ({n}).")
    if cluster == "hdbscan" and not 2 <= min_cluster_size <= n:
        raise ValueError(f"The smallest group size must be between 2 and the number of images ({n}).")


def load_images(paths, remove_background=False, log=print):
    from PIL import Image
    if remove_background:
        try:
            from rembg import remove
        except ImportError:
            raise RuntimeError("Background removal needs the rembg package (pip install rembg), or untick it.")
    images = []
    for i, p in enumerate(paths, 1):
        if remove_background:
            im = Image.open(io.BytesIO(remove(p.read_bytes()))).convert("RGBA")
        else:
            im = Image.open(p)
            im.load()
            im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
        if im.mode == "RGBA":  # transparent parts become white
            im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im)
        images.append(im.convert("RGB"))
        if remove_background and (i % 10 == 0 or i == len(paths)):
            log(f"Removed backgrounds: {i}/{len(paths)}")
    return images


def fingerprint(paths, model="MobileNetV2", remove_background=False, log=print):
    """Steps 1-2: (paths, 96-pixel thumbnails as arrays, fingerprints: one row of numbers per image)."""
    import contextlib, sys
    import numpy as np
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    try:  # python.org's macOS Python has no certificates for the one-time model download unless certifi is used
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    log(f"1/4  Loading {len(paths)} images from {paths[0].parent}")
    images = load_images(paths, remove_background, log)

    log(f"2/4  Fingerprinting each image with {model} (the first time, the network is downloaded once)")
    from .models import extract
    real_stdout = sys.stdout

    def relay(line):  # extract() prints its progress; hand each line to `log`, which may itself print
        with contextlib.redirect_stdout(real_stdout):
            log(line)

    with contextlib.redirect_stdout(_Lines(relay)):
        X = extract(images, model, tag="images")
    thumbs = []
    for im in images:
        im.thumbnail((96, 96))
        thumbs.append(np.asarray(im))
    return paths, thumbs, X


def embed(X, reduce="umap", dims=2, n_neighbors=15, min_dist=0.1, seed=24, log=print):
    """Step 3: (coordinates, n images x dims; share of the variance of the first 10 PCs, or None for UMAP)."""
    n = len(X)
    log(f"3/4  Squeezing the fingerprints into {dims} dimensions with {reduce.upper()}")
    if reduce == "umap":
        import umap
        nb = min(n_neighbors, n - 1)
        if nb != n_neighbors:
            log(f"     (only {n} images: using {nb} neighbours instead of {n_neighbors})")
        return umap.UMAP(n_components=dims, n_neighbors=nb, min_dist=min_dist, random_state=seed).fit_transform(X), None
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(max(dims, 10), n, X.shape[1]), random_state=seed).fit(X)
    variance = pca.explained_variance_ratio_[:10]
    log("     variance explained: " + ", ".join(f"PC{i} {v:.1%}" for i, v in enumerate(variance, 1))
        + f"; together {variance.sum():.1%}")
    return pca.transform(X)[:, :dims], variance


def group(emb, cluster="hdbscan", min_cluster_size=5, min_samples=None, k=2, seed=24, log=print):
    """Step 4: a group number per image (-1: HDBSCAN found no group for it), using all dimensions of `emb`."""
    log(f"4/4  Finding groups with {'HDBSCAN' if cluster == 'hdbscan' else f'k-means (k = {k})'}")
    if cluster == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples or None, copy=True).fit_predict(emb)
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb)


def summary(labels) -> str:
    found = sorted(set(int(l) for l in labels) - {-1})
    noise = int((labels < 0).sum())
    return (f"{len(found)} groups (" + ", ".join(f"group {c}: {int((labels == c).sum())}" for c in found) + ")"
            + (f", {noise} images in no group" if noise else ""))


def settings_for(folder, images, model, remove_background, reduce, dims, n_neighbors, min_dist, seed,
                 cluster, min_cluster_size, min_samples, k) -> dict:
    """The choices that matter for a run, as saved in settings.json."""
    s = dict(folder=str(folder), images=images, model=model, remove_background=remove_background, reduce=reduce,
             dims=dims, seed=seed, cluster=cluster)
    s.update(dict(n_neighbors=n_neighbors, min_dist=min_dist) if reduce == "umap" else {})
    s.update(dict(min_cluster_size=min_cluster_size, min_samples=min_samples) if cluster == "hdbscan" else dict(k=k))
    return s


def title_for(s) -> str:
    dims = f"{s['dims']}-D" + (", first 2 shown" if s["dims"] > 2 else "")
    groups = "HDBSCAN" if s["cluster"] == "hdbscan" else f"k-means (k = {s['k']})"
    return f"{s['model']} · {s['reduce'].upper()} {dims} · {groups}"


def draw_map(fig, emb, labels, thumbs, title, variance=None, max_thumbs=200):
    """Draw the first two dimensions on matplotlib Figure `fig`, a thumbnail per image (up to max_thumbs) framed in
    its group's colour; for PCA, the axes show each component's share of the variance and a bar chart below shows
    the first 10 components."""
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    fig.clear()
    if variance is None:
        ax = fig.add_subplot()
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    else:
        ax, bars = fig.subplots(2, 1, height_ratios=[4, 1])
        ax.set_xlabel(f"PC1 ({variance[0]:.1%} of the variance)"); ax.set_ylabel(f"PC2 ({variance[1]:.1%})")
        b = bars.bar([f"PC{i}" for i in range(1, len(variance) + 1)], variance * 100, color=NOISE_GREY)
        bars.bar_label(b, fmt="%.0f%%", fontsize=8)
        bars.set_ylabel("% of variance")
        bars.set_title(f"Variance explained by the first {len(variance)} components (together {variance.sum():.0%})",
                       fontsize=10)
        bars.spines[["top", "right"]].set_visible(False)

    found = sorted(set(int(l) for l in labels) - {-1})
    cmap = colormaps["tab10" if len(found) <= 10 else "tab20"]
    color = {c: cmap(i % cmap.N) for i, c in enumerate(found)}
    color[-1] = NOISE_GREY
    for c in found + ([-1] if (labels < 0).any() else []):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=18, color=color[c],
                   label=f"{'no group' if c < 0 else f'group {c}'} ({int(m.sum())})")
    show = np.random.default_rng(0).permutation(len(emb))[:max_thumbs]
    zoom = min(0.6, 0.5 * fig.get_figwidth() / np.sqrt(len(show)))  # ponytail: fewer images, bigger thumbnails
    for i in show:
        ax.add_artist(AnnotationBbox(OffsetImage(thumbs[i], zoom=zoom), emb[i, :2], frameon=True, pad=0.05,
                                     bboxprops=dict(edgecolor=color[int(labels[i])], lw=2)))
    ax.margins(0.08)  # room for the thumbnails at the edges
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)  # beside the map, not over it


def save(out, paths, X, emb, labels, settings, fig=None, features=True, coords=True, copies=True, variance=None):
    """Write the chosen results to `out`; settings.json always. Returns the output folder."""
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    if fig is not None:
        fig.savefig(out / "map.png", dpi=200)
        fig.savefig(out / "map.pdf")
    if coords:
        axis = "pc" if settings["reduce"] == "pca" else "umap"
        with open(out / "clusters.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "cluster"] + [f"{axis}_{d}" for d in range(1, emb.shape[1] + 1)])
            w.writerows([p.name, "noise" if l < 0 else int(l)] + [round(float(v), 5) for v in row]
                        for p, l, row in zip(paths, labels, emb))
        if variance is not None:
            with open(out / "pca_variance.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["component", "explained_variance", "cumulative"])
                w.writerows([f"PC{i}", round(float(v), 5), round(float(c), 5)]
                            for i, (v, c) in enumerate(zip(variance, variance.cumsum()), 1))
    if features:
        with open(out / "features.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file"] + [f"f{j}" for j in range(1, X.shape[1] + 1)])
            w.writerows([p.name] + [f"{v:.6g}" for v in row] for p, row in zip(paths, X))
    if copies:
        sorted_dir = out / "sorted"
        shutil.rmtree(sorted_dir, ignore_errors=True)
        for p, l in zip(paths, labels):
            d = sorted_dir / ("noise" if l < 0 else f"cluster_{l}")
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / p.name)
    return out


def run(folder, model="MobileNetV2", reduce="umap", n_neighbors=15, min_dist=0.1, cluster="hdbscan",
        min_cluster_size=5, min_samples=None, k=2, seed=24, remove_background=False, out=None,
        sort_copies=True, log=print, dims=2) -> Path:
    """Run the four steps on `folder` and save every result; returns the output folder."""
    from matplotlib.figure import Figure
    folder = Path(folder).expanduser()
    paths = find_images(folder)
    check(len(paths), reduce, dims, cluster, k, min_cluster_size)
    t0 = time.time()
    paths, thumbs, X = fingerprint(paths, model, remove_background, log)
    emb, variance = embed(X, reduce, dims, n_neighbors, min_dist, seed, log)
    labels = group(emb, cluster, min_cluster_size, min_samples, k, seed, log)
    s = settings_for(folder, len(paths), model, remove_background, reduce, dims, n_neighbors, min_dist, seed,
                     cluster, min_cluster_size, min_samples, k)
    fig = Figure(figsize=(12, 11 if variance is not None else 9), layout="constrained")
    draw_map(fig, emb, labels, thumbs, title_for(s), variance)
    out = save(out or folder.parent / f"{folder.name}_deepfin", paths, X, emb, labels, s, fig,
               copies=sort_copies, variance=variance)
    log(f"Done in {time.time() - t0:.0f} s: {summary(labels)}")
    log(f"Results: {out}")
    return out


class _Lines(io.TextIOBase):
    """A text stream that hands each complete line to a callback."""
    def __init__(self, callback):
        self.callback, self.buf = callback, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.callback("     " + line.strip())
        return len(s)


def main(argv=None):
    a = argparse.ArgumentParser(prog="deepfin", description="Sort a folder of images into groups (see deepfin-gui for the point-and-click app).")
    a.add_argument("folder", help="folder with the images")
    a.add_argument("--model", default="MobileNetV2", choices=MODELS, help="network for the fingerprints (default MobileNetV2)")
    a.add_argument("--remove-background", action="store_true", help="cut each subject out with rembg first (slower)")
    a.add_argument("--reduce", default="umap", choices=["umap", "pca"], help="how to squeeze the fingerprints (default umap)")
    a.add_argument("--dims", type=int, default=2, help="number of dimensions to squeeze into; groups use all, the map shows the first 2 (default 2)")
    a.add_argument("--n-neighbors", type=int, default=15, help="UMAP: neighbours per image (default 15)")
    a.add_argument("--min-dist", type=float, default=0.1, help="UMAP: how tightly points may pack, 0 to 1 (default 0.1)")
    a.add_argument("--cluster", default="hdbscan", choices=["hdbscan", "kmeans"], help="how to find groups (default hdbscan)")
    a.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN: smallest group (default 5)")
    a.add_argument("--min-samples", type=int, default=None, help="HDBSCAN: higher = more images left in no group (default = min cluster size)")
    a.add_argument("-k", "--clusters", dest="k", type=int, default=2, help="k-means: number of groups (default 2)")
    a.add_argument("--seed", type=int, default=24, help="random seed, for repeatable maps (default 24)")
    a.add_argument("--out", help="output folder (default: <folder>_deepfin next to the images)")
    a.add_argument("--no-copies", action="store_true", help="do not copy the images into one folder per group")
    o = a.parse_args(argv)
    try:
        run(o.folder, o.model, o.reduce, o.n_neighbors, o.min_dist, o.cluster, o.min_cluster_size, o.min_samples,
            o.k, o.seed, o.remove_background, o.out, not o.no_copies, dims=o.dims)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        a.exit(1, f"deepfin: {e}\n")


if __name__ == "__main__":
    main()
