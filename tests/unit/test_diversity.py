from rec_researcher.retrieval.diversity import (
    DiversityCandidate,
    maximal_marginal_relevance,
)


def test_mmr_avoids_highly_similar_text() -> None:
    candidates = [
        DiversityCandidate(
            passage_id="a",
            source_id="s1",
            text="vector search retrieval ranking",
            relevance_score=1.0,
        ),
        DiversityCandidate(
            passage_id="duplicate",
            source_id="s1",
            text="vector search retrieval ranking system",
            relevance_score=0.99,
        ),
        DiversityCandidate(
            passage_id="diverse",
            source_id="s2",
            text="collaborative filtering user preferences",
            relevance_score=0.8,
        ),
    ]

    results = maximal_marginal_relevance(candidates, top_k=2, lambda_=0.5)

    assert [result.passage_id for result in results] == ["a", "diverse"]


def test_mmr_empty_and_oversized_top_k_are_safe() -> None:
    assert maximal_marginal_relevance([], top_k=10) == []
    candidate = DiversityCandidate(
        passage_id="a", source_id="s", text="text", relevance_score=1.0
    )
    assert len(maximal_marginal_relevance([candidate], top_k=10)) == 1
