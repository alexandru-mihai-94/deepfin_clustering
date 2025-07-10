
from pathlib import Path
from .preprocessing import segment_images
from .feature_extract import embed_cnn
from .clustering import run_clustering
from .viz import plot_umap

def run_demo(img_dir: str | Path, out_dir: str | Path = "results"):
    """One-command demo used in README & CI."""
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs      = segment_images(img_dir)          # ndarray | list[ndarray]
    features  = embed_cnn(imgs, model="efficientnet_b0")
    labels, xy = run_clustering(features)        # labels, 2-D coords
    plot_umap(xy, labels, out_dir/"umap_demo.png")
    print(" Demo finished ", out_dir/"umap_demo.png")
