"""In-memory BM25 retrieval with lightweight mixed-language tokenization."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi

from rec_researcher.core.models import PassageRecord

_PARTS = re.compile(r"[A-Za-z0-9]+(?:['_-][A-Za-z0-9]+)*|[\u3400-\u9fff]+")
_CHINESE = re.compile(r"^[\u3400-\u9fff]+$")


class BM25Result(BaseModel):
    """One traceable lexical retrieval hit."""

    model_config = ConfigDict(extra="forbid")

    passage_id: str
    rank: int
    score: float


def mixed_tokenize(text: str) -> list[str]:
    """Tokenize English as words and Chinese runs as character bigrams."""

    tokens: list[str] = []
    for part in _PARTS.findall(text.casefold()):
        if not _CHINESE.fullmatch(part):
            tokens.append(part)
        elif len(part) == 1:
            tokens.append(part)
        else:
            tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
    return tokens


class BM25Retriever:
    """Rank a fixed passage corpus using BM25Okapi."""

    def __init__(self, passages: Sequence[PassageRecord]) -> None:
        """Build the lexical index, accepting an empty corpus."""

        self._passages = list(passages)
        corpus = [mixed_tokenize(passage.text) for passage in passages]
        self._index = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, *, limit: int = 10) -> list[BM25Result]:
        """Return ranked passage identifiers for a non-empty query."""

        query_tokens = mixed_tokenize(query)
        if self._index is None or not query_tokens or limit <= 0:
            return []
        scores = self._index.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [
            BM25Result(
                passage_id=self._passages[index].id,
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(order[:limit], start=1)
        ]
