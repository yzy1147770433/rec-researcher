import pytest

from rec_researcher.core.exceptions import ConfigurationError
from rec_researcher.core.settings import Settings
from rec_researcher.providers.factory import ProviderFactory
from rec_researcher.providers.mock import (
    MockLanguageModel,
    MockPassageReranker,
    MockSearchProvider,
    MockTextEmbedder,
)
from rec_researcher.retrieval.pipeline import RetrievalPipeline
from rec_researcher.retrieval.vector_store import InMemoryVectorIndex


def test_mock_factory_creates_deterministic_offline_composition() -> None:
    factory = ProviderFactory(
        Settings(_env_file=None), mode="mock", retrieval_mode="hybrid"
    )

    assert isinstance(factory.create_language_model(), MockLanguageModel)
    assert isinstance(factory.create_search_provider(), MockSearchProvider)
    assert isinstance(factory.create_text_embedder(), MockTextEmbedder)
    assert isinstance(factory.create_passage_reranker(), MockPassageReranker)
    assert isinstance(factory.create_vector_index(), InMemoryVectorIndex)
    assert isinstance(factory.create_retrieval_pipeline(), RetrievalPipeline)


def test_real_snippet_mode_does_not_require_siliconflow() -> None:
    settings = Settings(
        _env_file=None,
        mode="real",
        llm_base_url="https://llm.invalid/v1",
        llm_api_key="llm-secret",
        llm_model="llm-model",
        tavily_api_key="tavily-secret",
        siliconflow_api_key=None,
        embedding_model=None,
        reranker_model=None,
    )
    factory = ProviderFactory(settings, retrieval_mode="snippet")

    factory.validate_configuration()
    assert factory.create_retrieval_pipeline() is None


def test_real_hybrid_mode_requires_siliconflow_without_exposing_secret() -> None:
    secret = "factory-test-complete-secret"
    settings = Settings(
        _env_file=None,
        mode="real",
        siliconflow_api_key=secret,
        embedding_model=None,
        reranker_model=None,
    )
    factory = ProviderFactory(settings, retrieval_mode="hybrid")

    with pytest.raises(ConfigurationError) as caught:
        factory.create_retrieval_pipeline()

    assert "embedding_model" in str(caught.value)
    assert secret not in str(caught.value)
    assert secret not in str(settings.safe_summary())


def test_none_providers_return_none_and_are_rejected_for_hybrid() -> None:
    factory = ProviderFactory(
        Settings(_env_file=None),
        mode="mock",
        retrieval_mode="hybrid",
        embedding_provider="none",
        reranker_provider="none",
        vector_store="none",
    )

    assert factory.create_text_embedder() is None
    assert factory.create_passage_reranker() is None
    assert factory.create_vector_index() is None
    with pytest.raises(ConfigurationError, match="Hybrid retrieval requires"):
        factory.create_retrieval_pipeline()
