from rec_researcher.core.models import PassageRecord
from rec_researcher.retrieval.bm25 import BM25Retriever, mixed_tokenize


def _passage(identifier: str, text: str) -> PassageRecord:
    return PassageRecord(id=identifier, source_id="source", text=text)


def test_bm25_ranks_mixed_language_match_first() -> None:
    retriever = BM25Retriever(
        [
            _passage("relevant", "向量检索 vector retrieval system"),
            _passage("other-1", "烹饪食谱 cooking recipe"),
            _passage("other-2", "天气预报 weather report"),
        ]
    )

    results = retriever.search("向量检索 vector", limit=2)

    assert results[0].passage_id == "relevant"
    assert results[0].rank == 1
    assert results[0].score > results[1].score
    assert mixed_tokenize("中文 API") == ["中文", "api"]


def test_bm25_empty_inputs_are_safe() -> None:
    assert BM25Retriever([]).search("query") == []
    assert BM25Retriever([_passage("p", "text")]).search("") == []
    assert BM25Retriever([_passage("p", "text")]).search("text", limit=0) == []
