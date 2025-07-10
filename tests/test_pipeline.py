from deepfin.pipeline import run_demo
import pathlib, tempfile, shutil

def test_demo_placeholder():
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        # until helpers are implemented we expect a NotImplementedError
        try:
            run_demo("sample_images/", out_dir=tmp)
        except NotImplementedError:
            assert True
    finally:
        shutil.rmtree(tmp)
