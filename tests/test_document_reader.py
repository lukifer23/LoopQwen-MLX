import itertools

from mlx_lm.utils import load_tokenizer

from loopqwen.document_reader import DocumentReader
from loopqwen.model_loader import local_snapshot
from loopqwen.tool_session import prompt


def test_document_question_cannot_retrieve_itself_or_assistant_text():
    tok = load_tokenizer(local_snapshot())
    reader = DocumentReader(tok, None)
    source = ("ordinary prose in a source line\n" * 90) + "the last row"
    query = "Which service handles Artemis shipments?"
    message = prompt(tok, [{"role": "user", "content": source + "\n\n" + query}])
    ids = tok.encode(message, add_special_tokens=False)
    document, terms = reader.split(ids)
    assert query == tok.decode(terms)
    assert query not in tok.decode(document)
    assert (
        reader.split(ids + tok.encode("Possible answer", add_special_tokens=False))
        is None
    )
    assert reader.split(tok.encode(query, add_special_tokens=False)) is None
    for a, b in reader.spans(document):
        assert 0 <= a < b <= len(document) and b - a <= 192


def test_long_physical_lines_preserve_every_original_token():
    tok = load_tokenizer(local_snapshot())
    reader = DocumentReader(tok, None)
    ids = tok.encode("request " * 490 + "\nthird line", add_special_tokens=False)
    spans = reader.spans(ids)
    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    assert all(b - a <= 192 for a, b in spans)
    assert all(c <= b for (_, b), (c, _) in itertools.pairwise(spans))
