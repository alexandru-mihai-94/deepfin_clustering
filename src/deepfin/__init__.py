"""DEEPFIN – unsupervised fish-image clustering."""
__version__ = "0.1.0"

def run_demo(img_dir, out_dir="results"):
    """One-command demo used in README & CI."""
    from .pipeline import run_demo as _run_demo
    return _run_demo(img_dir, out_dir)    run_demo(img_dir, out_dir)
