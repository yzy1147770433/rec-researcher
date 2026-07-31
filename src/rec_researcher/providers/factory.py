"""Central construction of provider and retrieval dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rec_researcher.core.exceptions import ConfigurationError
from rec_researcher.core.settings import (
    EmbeddingProvider,
    RerankerProvider,
    RetrievalMode,
    Settings,
    VectorStore,
    _is_configured,
)
from rec_researcher.providers.base import (
    LanguageModel,
    PassageReranker,
    SearchProvider,
    TextEmbedder,
    VectorIndex,
)
from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel
from rec_researcher.providers.mock import (
    MockLanguageModel,
    MockPassageReranker,
    MockSearchProvider,
    MockTextEmbedder,
)
from rec_researcher.providers.siliconflow import (
    SiliconFlowEmbedder,
    SiliconFlowReranker,
)
from rec_researcher.providers.tavily import TavilySearchProvider
from rec_researcher.retrieval.pipeline import RetrievalPipeline
from rec_researcher.retrieval.vector_store import InMemoryVectorIndex, MilvusLiteIndex

ProviderMode = Literal["mock", "real"]


@dataclass(frozen=True, slots=True)
class ProviderSelection:
    """Explicit provider choices for one runtime composition."""

    mode: ProviderMode
    retrieval_mode: RetrievalMode
    embedding_provider: EmbeddingProvider
    reranker_provider: RerankerProvider
    vector_store: VectorStore


class ProviderFactory:
    """Create configured providers without silently changing implementations."""

    def __init__(
        self,
        settings: Settings,
        *,
        mode: ProviderMode | None = None,
        retrieval_mode: RetrievalMode | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker_provider: RerankerProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Capture settings and explicit provider selections."""

        selected_mode = mode or settings.mode
        default_embedding: EmbeddingProvider = (
            "mock" if selected_mode == "mock" else settings.embedding_provider
        )
        default_reranker: RerankerProvider = (
            "mock" if selected_mode == "mock" else settings.reranker_provider
        )
        default_vector_store: VectorStore = (
            "memory" if selected_mode == "mock" else settings.vector_store
        )
        self.settings = settings
        self.selection = ProviderSelection(
            mode=selected_mode,
            retrieval_mode=retrieval_mode or settings.retrieval_mode,
            embedding_provider=embedding_provider or default_embedding,
            reranker_provider=reranker_provider or default_reranker,
            vector_store=vector_store or default_vector_store,
        )

    def validate_configuration(self) -> None:
        """Validate the selected composition without constructing resources."""

        if self.selection.mode == "real":
            self._require(
                llm_base_url=self.settings.llm_base_url,
                llm_api_key=self.settings.llm_api_key,
                llm_model=self.settings.llm_model,
                tavily_api_key=self.settings.tavily_api_key,
            )

    def create_language_model(self) -> LanguageModel:
        """Create the language model selected by runtime mode."""

        if self.selection.mode == "mock":
            return MockLanguageModel()
        self._require(
            llm_base_url=self.settings.llm_base_url,
            llm_api_key=self.settings.llm_api_key,
            llm_model=self.settings.llm_model,
        )
        return OpenAICompatibleLanguageModel(self.settings)

    def create_search_provider(self) -> SearchProvider:
        """Create the search provider selected by runtime mode."""

        if self.selection.mode == "mock":
            return MockSearchProvider()
        self._require(tavily_api_key=self.settings.tavily_api_key)
        return TavilySearchProvider(self.settings)

    def create_text_embedder(self) -> TextEmbedder | None:
        """Create the explicitly selected embedder, or return ``None``."""

        provider = self.selection.embedding_provider
        if provider == "none":
            return None
        if provider == "mock":
            return MockTextEmbedder()
        self._require(
            siliconflow_api_key=self.settings.siliconflow_api_key,
            embedding_model=self.settings.embedding_model,
        )
        return SiliconFlowEmbedder(self.settings)

    def create_passage_reranker(self) -> PassageReranker | None:
        """Create the explicitly selected reranker, or return ``None``."""

        provider = self.selection.reranker_provider
        if provider == "none":
            return None
        if provider == "mock":
            return MockPassageReranker()
        self._require(
            siliconflow_api_key=self.settings.siliconflow_api_key,
            reranker_model=self.settings.reranker_model,
        )
        return SiliconFlowReranker(self.settings)

    def create_vector_index(self) -> VectorIndex | None:
        """Create the selected vector index, or return ``None``."""

        provider = self.selection.vector_store
        if provider == "none":
            return None
        if provider == "memory":
            return InMemoryVectorIndex()
        return MilvusLiteIndex(
            self.settings.milvus_uri,
            collection_name=self.settings.milvus_collection,
        )

    def create_retrieval_pipeline(
        self, *, run_namespace: str | None = None
    ) -> RetrievalPipeline | None:
        """Create hybrid retrieval, while snippet mode returns no pipeline."""

        del run_namespace  # Per-run/task scoping occurs when retrieval is invoked.

        if self.selection.retrieval_mode == "snippet":
            return None
        self._validate_hybrid_configuration()
        embedder = self.create_text_embedder()
        reranker = self.create_passage_reranker()
        vector_index = self.create_vector_index()
        assert embedder is not None
        assert reranker is not None
        assert vector_index is not None
        return RetrievalPipeline(
            embedder=embedder,
            vector_index=vector_index,
            reranker=reranker,
            retrieval_top_k=self.settings.retrieval_top_k,
            rerank_top_k=self.settings.rerank_top_k,
            mmr_top_k=self.settings.mmr_top_k,
            rrf_k=self.settings.rrf_k,
            mmr_lambda=self.settings.mmr_lambda,
        )

    def _validate_hybrid_configuration(self) -> None:
        disabled = [
            name
            for name, value in (
                ("embedding_provider", self.selection.embedding_provider),
                ("reranker_provider", self.selection.reranker_provider),
                ("vector_store", self.selection.vector_store),
            )
            if value == "none"
        ]
        if disabled:
            raise ConfigurationError(
                "Hybrid retrieval requires configured providers: " + ", ".join(disabled)
            )
        required: dict[str, object] = {}
        if self.selection.embedding_provider == "siliconflow":
            required.update(
                siliconflow_api_key=self.settings.siliconflow_api_key,
                embedding_model=self.settings.embedding_model,
            )
        if self.selection.reranker_provider == "siliconflow":
            required.update(
                siliconflow_api_key=self.settings.siliconflow_api_key,
                reranker_model=self.settings.reranker_model,
            )
        self._require(**required)

    @staticmethod
    def _require(**values: object) -> None:
        missing = [name for name, value in values.items() if not _is_configured(value)]
        if missing:
            raise ConfigurationError(
                "Missing real-mode configuration: " + ", ".join(missing)
            )
