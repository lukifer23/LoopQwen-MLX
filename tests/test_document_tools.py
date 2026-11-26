import pytest

from loopqwen.document_tools import chunks, documents


def test_document_search_indexes_exact_snapshot_offsets_and_scopes_queries(tmp_path):
    a, b = (tmp_path / n for n in ("one.md", "two.txt"))
    text = "Résumé of the project.\r\n\r\n" + "fox jumps. " * 140 + "Acacia grew here."
    a.write_bytes(b"\xef\xbb\xbf" + text.encode())
    b.write_text("Acacia blossoms elsewhere.")
    with documents([a, b]) as (tool,):
        a.write_text("new material")
        hits = tool.callback('Acacia "OR" grew', document=a.name)["passages"]
        assert hits and all(
            h["document"] == a.name and h["text"] == text[h["start"] : h["stop"]]
            for h in hits
        )
        assert "Acacia" in hits[0]["text"]
        assert not tool.callback("absentword")["passages"]
        assert list(chunks("")) == []
        with pytest.raises(ValueError):
            tool.callback("Acacia", document="../one.md")
    with pytest.raises(ValueError), documents([a, a]):
        pass


def test_reading_grounding_requires_annotated_location_and_completed_answer():
    from loopqwen.research.document_search import grade

    item = {"document": "a.txt", "refs": ["the acacia"], "answer_ranges": [[10, 16]]}

    def result(start, stop):
        return {
            "answer": "Acacia.",
            "status": "answered",
            "turns": [
                {
                    "observation": {
                        "passages": [
                            {
                                "document": "a.txt",
                                "start": start,
                                "stop": stop,
                                "text": "acacia",
                            }
                        ]
                    }
                }
            ],
        }

    assert grade(item, result(10, 16))["grounded"]
    assert grade(item, result(30, 36))["correct"]
    assert not grade(item, result(30, 36))["grounded"]
    assert not grade(item, {**result(10, 16), "status": "token_limit"})["grounded"]
