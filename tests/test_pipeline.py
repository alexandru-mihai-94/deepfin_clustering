from deepfin.pipeline import run_demo
import pathlib, tempfile, shutil

def test_demo_runs():
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        run_demo("sample_images/", out_dir=tmp)
        assert (tmp / "umap_demo.png").exists()
    finally:
        shutil.rmtree(tmp)
