import itertools
from pathlib import Path

from loopqwen.csv_tools import tools
from loopqwen.research.linked_rollin import exact_sign_flip
from loopqwen.research.linked_rollin_data import reached, score
from scripts.linked_table_material import cases, render


def test_article_text_and_table_link_must_be_observed_in_order(tmp_path):
    _, source, table, pages = next(cases())
    view = render(source, table, pages, tmp_path)["file"]
    expected = view["events"]
    w = {
        "files": view["files"],
        "gold": source["answer"],
        "events": expected,
        "page": source["link"],
    }
    reader = tools([tmp_path / Path(name) for name in view["files"]], keyed=True)[0]
    alt = next(k for k in pages if k != source["link"])
    args = {**expected[-1]["call"]["arguments"], "key": alt}
    wrong = {
        "call": {"name": reader.name, "arguments": args},
        "observation": reader.callback(**args),
    }
    assert wrong["observation"]["value"] == pages[alt]

    def grade(events, answer):
        return score(
            None, w, 0, {"status": "answered", "answer": answer, "turns": events}
        )

    assert reached(w, [[expected[1]], [expected[0]]]) == 1
    assert grade([expected[0], wrong], w["gold"])["correct_answer"]
    assert not grade([expected[0], wrong], w["gold"])["success"]
    batch = {
        "calls": [e["call"] for e in expected],
        "observations": [e["observation"] for e in expected],
    }
    assert reached(w, [expected]) == 1
    assert not grade([batch], w["gold"])["success"]
    assert grade([wrong, *expected], w["gold"])["success"]
    assert not grade(expected, "An unrelated place")["success"]


def test_exact_source_sign_flip_includes_zero_differences_and_ties():
    delta = [0, 2, 0, -1, 2, -1]
    brute = sum(
        abs(sum(a * b for a, b in zip(delta, signs, strict=True))) >= abs(sum(delta))
        for signs in itertools.product((-1, 1), repeat=len(delta))
    ) / 2 ** len(delta)
    assert exact_sign_flip(delta) == brute
    assert exact_sign_flip([1] * 24) == 2 / 2**24
