from rec_researcher.core.models import EvidenceRecord, SourceRecord
from rec_researcher.domain.recommender import (
    RecommendationDomainAnalyzer,
    ReproductionDifficulty,
)


def source(snippet: str, *, url: str = "https://example.com/paper") -> SourceRecord:
    return SourceRecord(
        id="source-1",
        title="Recommendation paper",
        url=url,
        snippet=snippet,
        provider="fixture",
    )


def evidence(text: str) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="evidence-1",
        source_id="source-1",
        passage_id="passage-1",
        claim_hint=text,
        excerpt=text,
    )


def test_identifies_generative_recommendation() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("A generative recommender using autoregressive generation.")], []
    )

    assert "生成式推荐" in profile.task_type
    assert "Autoregressive Generation" in profile.core_techniques


def test_identifies_two_tower_model() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("A two-tower retrieval model evaluated on MovieLens.")], []
    )

    assert "召回" in profile.task_type
    assert "双塔" in profile.model_family


def test_extracts_github_url_and_public_code_status() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("Code: https://github.com/example/recommender")], []
    )

    assert profile.code_urls == ["https://github.com/example/recommender"]
    assert profile.has_public_code is True


def test_gpu_memory_remains_unknown_without_explicit_evidence() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("The model was trained efficiently on a GPU.")], []
    )

    assert profile.estimated_gpu_memory_gb is None
    assert profile.hardware_requirements_stated is False


def test_reproduction_difficulty_has_explainable_score() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("A proprietary dataset is used and code is not publicly available.")],
        [],
    )

    assert profile.reproduction_score == 4
    assert profile.reproduction_difficulty == ReproductionDifficulty.HIGH
    assert profile.reproduction_score_reasons == [
        "+2：未识别到公开代码",
        "+2：未识别到公开数据集",
    ]


def test_conflicting_sources_preserve_uncertainty() -> None:
    profile = RecommendationDomainAnalyzer().analyze(
        [source("Code is not publicly available.")],
        [evidence("Implementation: https://github.com/example/recommender")],
    )

    assert profile.has_public_code is True
    assert profile.code_urls
    assert profile.uncertainty
    assert "冲突" in profile.uncertainty[0]
