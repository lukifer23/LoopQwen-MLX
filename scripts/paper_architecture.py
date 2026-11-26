"""Draw the two qualified recurrence placements and the external observation path."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def make(finish):
    fig, ax = plt.subplots(figsize=(7.15, 2.95))
    ax.set(xlim=(-0.1, 10.1), ylim=(-0.15, 4.2))
    ax.axis("off")

    def box(x, y, w, label, shared=False):
        edge = "#126457" if shared else "#697471"
        patch = FancyBboxPatch(
            (x, y),
            w,
            0.6,
            boxstyle="round,pad=0.04,rounding_size=.06",
            edgecolor=edge,
            facecolor="#e0f0eb" if shared else "#f4f5f2",
            linewidth=1,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.30, label, ha="center", va="center", fontsize=8)
        return x, x + w

    def arrow(x, y, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color="#425752",
            )
        )

    ax.text(0, 4.03, "Induction  ·  Base initialization", weight="bold", fontsize=10)
    row = [
        (0, 1.78, "Encode facts\nand question"),
        (2.25, 1.5, "Layer 15\npass 1"),
        (4.20, 1.5, "Layer 15\npass 2"),
        (6.15, 1.5, "Layer 15\npass 3"),
        (8.12, 1.78, "Vocabulary-logit\ncorrection"),
    ]
    boxes = [box(x, 2.46, w, label, 0 < i < 4) for i, (x, w, label) in enumerate(row)]
    for left, right in zip(boxes, boxes[1:]):
        arrow(left[1], 2.76, right[0], 2.76)
    ax.text(
        4.95,
        3.69,
        "Fixed premise representations",
        va="center",
        ha="center",
        fontsize=8,
    )
    ax.plot([3.0, 6.9], [3.45, 3.45], lw=0.8, color="#425752")
    for x in (3, 4.95, 6.9):
        arrow(x, 3.46, x, 3.07)
    ax.text(
        4.95,
        2.26,
        "The question state changes at each pass.",
        va="top",
        ha="center",
        fontsize=8,
    )

    ax.text(
        0,
        1.71,
        "Tool calling  ·  Instruction-sibling initialization",
        weight="bold",
        fontsize=10,
    )
    row = [
        (0, 1.78, "Request and\ntool schemas"),
        (2.25, 1.5, "Frozen prefix\nlayers 0–22"),
        (4.20, 1.5, "Layer 23\nKV state 1"),
        (6.15, 1.5, "Layer 23\nKV state 2"),
        (8.12, 1.78, "Generate a call\nor final answer"),
    ]
    boxes = [box(x, 0.81, w, label, i in (2, 3)) for i, (x, w, label) in enumerate(row)]
    for left, right in zip(boxes, boxes[1:]):
        arrow(left[1], 1.11, right[0], 1.11)
    ax.plot([9.01, 9.01, 0.89], [0.76, 0.29, 0.29], lw=0.8, color="#425752")
    arrow(0.89, 0.28, 0.89, 0.76)
    ax.text(
        4.95,
        0.14,
        "A valid call executes. Its actual observation enters the next model turn.",
        ha="center",
        va="top",
        fontsize=8,
    )
    finish(fig, "recurrent-paths")
