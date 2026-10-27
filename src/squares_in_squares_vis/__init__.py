"""squares-in-squares-vis: interactive viewer for "Squares in Squares" (Kaggle) submission files."""
from .viewer import PackingViewer, load_submission, score_n, side_length, main  # noqa: F401

__version__ = "0.1.0"
__all__ = ["PackingViewer", "load_submission", "score_n", "side_length", "main"]
