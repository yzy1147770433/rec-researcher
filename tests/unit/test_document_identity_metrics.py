from rec_researcher.core.models import GoldDocument, SourceRecord
from rec_researcher.evaluation.metrics import document_identity_metrics


def source(title: str, url: str) -> SourceRecord:
    return SourceRecord(
        id=url,
        title=title,
        url=url,
        snippet="paper",
        provider="test",
    )


def test_arxiv_pdf_and_abstract_are_same_document() -> None:
    gold = GoldDocument(
        document_id="arxiv:2004.12832",
        title=(
            "ColBERT Efficient and Effective Passage Search via Contextualized "
            "Late Interaction over BERT"
        ),
        arxiv_id="2004.12832",
        accepted_urls=["https://arxiv.org/abs/2004.12832"],
    )
    metrics = document_identity_metrics(
        [
            source(
                "ColBERT paper",
                "https://arxiv.org/pdf/2004.12832.pdf",
            )
        ],
        [gold],
    )
    assert metrics["document_recall_at_3"] == 1.0
    assert metrics["document_mrr"] == 1.0


def test_long_title_alias_matches_but_short_model_name_does_not() -> None:
    long_gold = GoldDocument(
        document_id="paper:colbert",
        title=(
            "ColBERT Efficient and Effective Passage Search via Contextualized "
            "Late Interaction over BERT"
        ),
    )
    short_gold = GoldDocument(document_id="paper:deepfm", title="DeepFM")
    retrieved = [
        source(
            "ColBERT: Efficient and Effective Passage Search via Contextualized "
            "Late Interaction over BERT",
            "https://example.org/paper.pdf",
        ),
        source("A blog introduction to DeepFM", "https://example.org/blog"),
    ]
    assert (
        document_identity_metrics(retrieved, [long_gold])["document_recall_at_3"] == 1.0
    )
    assert (
        document_identity_metrics(retrieved, [short_gold])["document_recall_at_3"]
        == 0.0
    )
