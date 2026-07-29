from rec_researcher.core.settings import Settings
from rec_researcher.retrieval.chunker import PassageChunker


def _chunker(*, size: int, overlap: int) -> PassageChunker:
    return PassageChunker(
        Settings(_env_file=None, chunk_size=size, chunk_overlap=overlap)
    )


def test_empty_text_produces_no_chunks() -> None:
    chunker = _chunker(size=20, overlap=5)

    assert chunker.chunk("source", "") == []
    assert chunker.chunk("source", "  \n\n ") == []


def test_short_text_produces_one_safe_chunk() -> None:
    passages = _chunker(size=100, overlap=20).chunk("source", "  short text  ")

    assert len(passages) == 1
    assert passages[0].text == "short text"
    assert passages[0].start_offset == 2
    assert passages[0].end_offset == 12


def test_chunks_prefer_paragraphs_and_preserve_offsets() -> None:
    text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    passages = _chunker(size=34, overlap=0).chunk("source", text)

    assert [passage.text for passage in passages] == [
        "first paragraph\n\nsecond paragraph",
        "third paragraph",
    ]
    for index, passage in enumerate(passages):
        assert passage.position == index
        assert passage.source_id == "source"
        assert text[passage.start_offset : passage.end_offset] == passage.text


def test_overlap_is_preserved_for_hard_splits() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    passages = _chunker(size=10, overlap=3).chunk("source", text)

    assert [passage.text for passage in passages] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]
    assert passages[1].start_offset == passages[0].end_offset - 3
