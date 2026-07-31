from rec_researcher.core.models import (
    EvidenceRecord,
    ReportClaim,
    SourceRecord,
)
from rec_researcher.evaluation.metrics import evidence_support_metrics, precision_at_k
from rec_researcher.evidence.claim_verifier import ClaimEvidenceVerifier, ClaimExtractor
from rec_researcher.planning.query_rewrite import QueryRewriter
from rec_researcher.retrieval.source_quality import score_source


def _source(identifier: str, url: str, title: str = "retrieval method") -> SourceRecord:
    return SourceRecord(
        id=identifier,
        title=title,
        url=url,
        snippet="recommendation retrieval ranking",
        provider="test",
    )


def test_query_rewrite_is_bounded_deduplicated_and_deterministic() -> None:
    rewriter = QueryRewriter(max_queries=4)
    first = rewriter.rewrite("YouTubeDNN 负采样")
    assert first == rewriter.rewrite("YouTubeDNN 负采样")
    assert len(first) == 4
    assert first[0].query_type == "original"
    assert len({item.query.casefold() for item in first}) == len(first)
    assert first[1].query == "YouTubeDNN original paper PDF arXiv DOI"


def test_source_quality_prefers_primary_source_over_blog() -> None:
    paper = score_source(_source("paper", "https://arxiv.org/abs/1234"))
    blog = score_source(_source("blog", "https://medium.com/post"))
    assert paper.final_score > blog.final_score
    assert paper.authority > blog.authority


def test_claim_extraction_and_all_verification_statuses() -> None:
    claims = ClaimExtractor().extract(
        "## 方法\n模型使用 sampled softmax 训练。[S1]\n效果提升 12%。\n"
    )
    assert len(claims) == 2
    source = _source("src", "https://arxiv.org/abs/1")
    evidence = EvidenceRecord(
        evidence_id="E1",
        source_id="src",
        passage_id="P1",
        claim_hint="sampled softmax",
        excerpt="模型使用 sampled softmax 训练",
    )
    results = ClaimEvidenceVerifier().verify(claims, [source], [evidence])
    assert results[0].status == "supported"
    assert results[1].status == "missing_citation"
    metrics = evidence_support_metrics(results)
    assert metrics["evidence_support_rate"] == 0.5
    assert metrics["missing_citation_rate"] == 0.5


def test_invalid_citation_and_precision() -> None:
    result = ClaimEvidenceVerifier().verify(
        [ReportClaim(claim_id="C1", text="事实陈述 12", citation_ids=["S9"])],
        [_source("src", "https://example.com")],
        [],
    )
    assert result[0].status == "invalid_citation"
    assert (
        precision_at_k(["https://a.test", "https://x.test"], {"https://a.test": 1}, k=5)
        == 0.5
    )
