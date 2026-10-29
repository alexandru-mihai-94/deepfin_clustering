"""Quick check of deepfin.explore on 8 sample images: PCA + k-means (MobileNetV2 is downloaded the first time)."""
import csv, shutil, tempfile
from pathlib import Path

from deepfin.explore import run


def test_explore_writes_results():
    tmp = Path(tempfile.mkdtemp())
    try:
        images = tmp / "images"
        images.mkdir()
        for p in sorted((Path(__file__).parent.parent / "sample_images").glob("*.jpg"))[:8]:
            shutil.copy(p, images / p.name)
        out = run(images, model="MobileNetV2", reduce="pca", cluster="kmeans", k=2, out=tmp / "out", log=lambda s: None)
        rows = list(csv.DictReader(open(out / "clusters.csv")))
        assert len(rows) == 8 and {r["cluster"] for r in rows} <= {"0", "1"}
        assert (out / "map.png").exists() and len(list((out / "sorted").rglob("*.jpg"))) == 8
    finally:
        shutil.rmtree(tmp)
