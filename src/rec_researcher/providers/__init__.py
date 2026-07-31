"""Provider interfaces and implementations."""

from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel
from rec_researcher.providers.mock import (
    MockLanguageModel,
    MockPassageReranker,
    MockSearchProvider,
    MockTextEmbedder,
    MockWebFetcher,
)
from rec_researcher.providers.siliconflow import (
    SiliconFlowEmbedder,
    SiliconFlowReranker,
    SiliconFlowRerankResult,
)
from rec_researcher.providers.tavily import TavilySearchProvider

__all__ = [
    "MockLanguageModel",
    "MockPassageReranker",
    "MockSearchProvider",
    "MockTextEmbedder",
    "MockWebFetcher",
    "OpenAICompatibleLanguageModel",
    "ProviderFactory",
    "ProviderSelection",
    "SiliconFlowEmbedder",
    "SiliconFlowReranker",
    "SiliconFlowRerankResult",
    "TavilySearchProvider",
]


def __getattr__(name: str) -> object:
    """Load factory exports lazily to avoid retrieval/provider import cycles."""

    if name in {"ProviderFactory", "ProviderSelection"}:
        from rec_researcher.providers.factory import (
            ProviderFactory,
            ProviderSelection,
        )

        return {
            "ProviderFactory": ProviderFactory,
            "ProviderSelection": ProviderSelection,
        }[name]
    raise AttributeError(name)
