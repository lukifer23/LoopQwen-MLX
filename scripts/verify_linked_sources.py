"""Reconstruct corrected HybridQA sources, both prompt views and every required CSV return."""

from loopqwen.io import read
from scripts.linked_rollin_material import build
from scripts.linked_source_overlap import run as overlap
from scripts.linked_table_archive import audit as original
from scripts.linked_table_material import run as bind_callbacks
from scripts.linked_table_sources import DEST
from scripts.linked_text_repair import audit as corrected


if __name__ == "__main__":
    original(read(DEST / "enrollment.json"))
    corrected()
    overlap(text_disjoint=True)
    bind_callbacks(text_disjoint=True)
    build(text_disjoint=True)
