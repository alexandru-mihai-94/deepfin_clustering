"""
Sort a folder of images into groups in four choices, without writing any code:

  1. a folder of images
  2. a neural network that turns each image into a fingerprint of numbers (see deepfin.models)
  3. a way to squeeze the fingerprints onto a flat 2-D map: PCA (fast, linear) or UMAP (keeps neighbours together)
  4. a way to find groups on the map: HDBSCAN (finds the number of groups itself) or k-means (you set the number)

Use it from the point-and-click app (`deepfin-gui`), the command line (`deepfin --help`), or Python:

    from deepfin.explore import run
    run("my_images", model="MobileNetV2", reduce="umap", cluster="kmeans", k=3)

Results go to an output folder: clusters.csv (one row per image: file, cluster, map x and y), map.png (the map with
thumbnails, coloured by cluster), settings.json (the choices used) and, optionally, sorted/ with a copy of every
image in one folder per cluster (HDBSCAN puts images that fit no group in sorted/noise).
"""
from __future__ import annotations

import argparse, csv, io, json, os, shutil, time
from pathlib import Path

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
# the three networks of the exhibit first: they separated Percula/Ocellaris best (deepfin_clustering/docs)
MODELS = ["MobileNetV2", "VGG16", "DenseNet121", "ResNet50", "EfficientNetB0", "InceptionV3", "EfficientNetV2L"]


def find_images(folder) -> list[Path]:
    """Image files directly inside `folder` (not in sub-folders), sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES
                  and not p.name.startswith("."))


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


def run(folder, model="MobileNetV2", reduce="umap", n_neighbors=15, min_dist=0.1, cluster="hdbscan",
        min_cluster_size=5, min_samples=None, k=2, seed=24, remove_background=False, out=None,
        sort_copies=True, log=print) -> Path:
    """Run the four steps on `folder` and write the results; returns the output folder."""
    folder = Path(folder).expanduser()
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    if reduce not in ("umap", "pca") or cluster not in ("hdbscan", "kmeans"):
        raise ValueError("reduce must be 'umap' or 'pca', and cluster 'hdbscan' or 'kmeans'")
    paths = find_images(folder)
    n = len(paths)
    if n < 3:
        raise ValueError(f"Found {n} images in {folder}; need at least 3 (jpg, png, tif, bmp or webp).")
    if cluster == "kmeans" and not 2 <= k <= n:
        raise ValueError(f"The number of clusters must be between 2 and the number of images ({n}).")
    if cluster == "hdbscan" and not 2 <= min_cluster_size <= n:
        raise ValueError(f"The minimum cluster size must be between 2 and the number of images ({n}).")
    out = Path(out).expanduser() if out else folder.parent / f"{folder.name}_deepfin"
    t0 = time.time()

    try:  # python.org's macOS Python has no certificates for the one-time model download unless certifi is used
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    import numpy as np

    log(f"1/4  Loading {n} images from {folder}")
    images = load_images(paths, remove_background, log)

    log(f"2/4  Fingerprinting each image with {model} (the first time, the network is downloaded once)")
    from .models import extract
    import contextlib, sys
    real_stdout = sys.stdout

    def relay(line):  # extract() prints its progress; hand each line to `log`, which may itself print
        with contextlib.redirect_stdout(real_stdout):
            log(line)

    with contextlib.redirect_stdout(_Lines(relay)):
        X = extract(images, model, tag="images")

    log(f"3/4  Making a 2-D map with {reduce.upper()}")
    if reduce == "umap":
        import umap
        nb = min(n_neighbors, n - 1)
        if nb != n_neighbors:
            log(f"     (only {n} images: using {nb} neighbours instead of {n_neighbors})")
        xy = umap.UMAP(n_neighbors=nb, min_dist=min_dist, random_state=seed).fit_transform(X)
    else:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(X)

    log(f"4/4  Finding groups with {'HDBSCAN' if cluster == 'hdbscan' else f'k-means (k = {k})'}")
    if cluster == "hdbscan":
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples or None).fit_predict(xy)
    else:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(xy)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "clusters.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "cluster", "map_x", "map_y"])
        w.writerows([p.name, "noise" if l < 0 else int(l), round(float(a), 4), round(float(b), 4)]
                    for p, l, (a, b) in zip(paths, labels, xy))
    settings = dict(folder=str(folder), images=n, model=model, reduce=reduce, cluster=cluster, seed=seed,
                    remove_background=remove_background)
    settings.update(dict(n_neighbors=n_neighbors, min_dist=min_dist) if reduce == "umap" else {})
    settings.update(dict(min_cluster_size=min_cluster_size, min_samples=min_samples) if cluster == "hdbscan" else dict(k=k))
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    plot_map(xy, labels, images, out / "map.png", f"{model} · {reduce.upper()} · {cluster}")
    if sort_copies:
        sorted_dir = out / "sorted"
        shutil.rmtree(sorted_dir, ignore_errors=True)
        for p, l in zip(paths, labels):
            d = sorted_dir / ("noise" if l < 0 else f"cluster_{l}")
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / p.name)

    found = sorted(set(int(l) for l in labels) - {-1})
    sizes = ", ".join(f"cluster {c}: {int((labels == c).sum())}" for c in found)
    noise = int((labels < 0).sum())
    log(f"Done in {time.time() - t0:.0f} s: {len(found)} clusters ({sizes})" + (f", {noise} images in no group" if noise else ""))
    log(f"Results: {out}")
    return out


def plot_map(xy, labels, images, path, title, max_thumbs=200):
    """The 2-D map with a thumbnail per image (up to max_thumbs), framed in its cluster's colour."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    clusters = sorted(set(int(l) for l in labels))
    cmap = plt.get_cmap("tab10" if len(clusters) <= 10 else "tab20")
    color = {c: ("#9aa0a6" if c < 0 else cmap(i % cmap.N)) for i, c in enumerate(c for c in clusters if c >= 0)}
    color[-1] = "#9aa0a6"
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    for c in clusters:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=18, color=color[c], label="no group" if c < 0 else f"cluster {c}")
    show = np.random.default_rng(0).permutation(len(images))[:max_thumbs]
    for i in show:
        t = images[i].copy()
        t.thumbnail((72, 72))
        ax.add_artist(AnnotationBbox(OffsetImage(np.asarray(t), zoom=1), xy[i], frameon=True, pad=0.05,
                                     bboxprops=dict(edgecolor=color[int(labels[i])], lw=2)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)  # beside the map, not over it
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    a.add_argument("--reduce", default="umap", choices=["umap", "pca"], help="how to make the 2-D map (default umap)")
    a.add_argument("--n-neighbors", type=int, default=15, help="UMAP: neighbours per image (default 15)")
    a.add_argument("--min-dist", type=float, default=0.1, help="UMAP: how tightly points may pack, 0 to 1 (default 0.1)")
    a.add_argument("--cluster", default="hdbscan", choices=["hdbscan", "kmeans"], help="how to find groups (default hdbscan)")
    a.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN: smallest group (default 5)")
    a.add_argument("--min-samples", type=int, default=None, help="HDBSCAN: higher = more images left in no group (default = min cluster size)")
    a.add_argument("-k", "--clusters", dest="k", type=int, default=2, help="k-means: number of groups (default 2)")
    a.add_argument("--seed", type=int, default=24, help="random seed, for repeatable maps (default 24)")
    a.add_argument("--out", help="output folder (default: <folder>_deepfin next to the images)")
    a.add_argument("--no-copies", action="store_true", help="do not copy the images into one folder per cluster")
    o = a.parse_args(argv)
    try:
        run(o.folder, o.model, o.reduce, o.n_neighbors, o.min_dist, o.cluster, o.min_cluster_size, o.min_samples,
            o.k, o.seed, o.remove_background, o.out, not o.no_copies)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        a.exit(1, f"deepfin: {e}\n")


if __name__ == "__main__":
    main()
