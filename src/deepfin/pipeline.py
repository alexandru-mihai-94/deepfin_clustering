
from pathlib import Path
from .preprocessing import rembg_images
from .preprocessing import load_images
from .feature_extract import CNN_embed
from .clustering import run_clustering
from .viz import plot_umap
from .viz import plot_umap_thumbnails

def run_demo(img_dir: str | Path, out_dir: str | Path = "results"):
    """One-command demo used in README & CI."""
    img_dir, out_dir = Path(img_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # imgs      = load_images(img_dir)          # ndarray | list[ndarray]
    thumbs   = rembg_images(img_dir)
    print("Image background removal complete")
    features  = CNN_embed(thumbs, model="EfficientNetV2L")
    print("Feature extraction complete")
    labels = []
    xy = []
    for i in range(5,30,5):
        labels_t, xy_t = run_clustering(features,umap_n_neighbors=i,umap_min_dist = 0.01)
        plot_umap(xy_t, labels_t, out_dir/f"umap_demo_n{i}.png")
        # labels, 2-D coords
        labels.append(labels_t)
        xy.append(xy_t)

    plot_umap_thumbnails(xy[1], labels[1], out_dir/"umap_demo_thumbnails.png", 
                            images=thumbs, 
                            max_thumbnails=60, 
                            thumb_zoom=0.1,
                            title = "Acipenseridae vs Tinca Tinca")

    print(" Demo finished ", out_dir/"umap_demo.png")
