"""Opt-in real-network coverage for the complete hybrid retrieval workflow."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rec_researcher.core.settings import Settings
from rec_researcher.planning.planner import ResearchPlanner
from rec_researcher.providers.factory import ProviderFactory
from rec_researcher.reporting.writer import RealReportWriter
from rec_researcher.retrieval.chunker import PassageChunker
from rec_researcher.retrieval.fetcher import AsyncWebFetcher
from rec_researcher.retrieval.pipeline import RetrievalPipeline
from rec_researcher.workflow.orchestrator import ResearchOrchestrator

pytestmark = [pytest.mark.asyncio, pytest.mark.network, pytest.mark.network_e2e]

_QUESTION = "YouTubeDNN 双塔召回模型的核心原理是什么？"
_REQUIRED_ENV = (
    "REC_LLM_API_KEY",
    "REC_TAVILY_API_KEY",
    "REC_SILICONFLOW_API_KEY",
    "REC_LLM_BASE_URL",
    "REC_LLM_MODEL",
    "REC_EMBEDDING_MODEL",
    "REC_RERANKER_MODEL",
)


def _real_settings(tmp_path: Path) -> Settings:
    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.skip("missing real E2E configuration: " + ", ".join(missing))
    return Settings(
        _env_file=None,
        mode="real",
        output_dir=tmp_path / "results",
        max_tasks=3,
        max_sources_per_query=2,
        max_total_sources=5,
        max_concurrency=2,
        fetch_concurrency=2,
        retrieval_mode="hybrid",
        embedding_provider="siliconflow",
        reranker_provider="siliconflow",
        vector_store="milvus",
        milvus_uri=str(tmp_path / "milvus-lite.db"),
        retrieval_top_k=10,
        rerank_top_k=5,
        mmr_top_k=5,
        llm_max_tokens=1200,
        request_timeout_seconds=45,
        max_retries=1,
    )


async def test_complete_real_hybrid_research_chain(tmp_path: Path) -> None:
    settings = _real_settings(tmp_path)
    factory = ProviderFactory(settings, mode="real", retrieval_mode="hybrid")
    factory.validate_configuration()
    llm = factory.create_language_model()
    search = factory.create_search_provider()
    embedder = factory.create_text_embedder()
    reranker = factory.create_passage_reranker()
    vector_index = factory.create_vector_index()
    assert embedder is not None
    assert reranker is not None
    assert vector_index is not None

    fetcher = AsyncWebFetcher(settings)
    orchestrator = ResearchOrchestrator(
        output_dir=settings.output_dir,
        planner=ResearchPlanner(llm, max_tasks=3),
        search_provider=search,
        writer=RealReportWriter(llm),
        mode="real",
        max_tasks=3,
        max_sources=5,
        sources_per_query=2,
        max_concurrency=2,
        retrieval_concurrency=2,
        fetch_concurrency=2,
        timeout=180,
        retrieval_mode="hybrid",
        web_fetcher=fetcher,
        passage_chunker=PassageChunker(settings),
        retrieval_pipeline=RetrievalPipeline(
            embedder=embedder,
            vector_index=vector_index,
            reranker=reranker,
            retrieval_top_k=settings.retrieval_top_k,
            rerank_top_k=settings.rerank_top_k,
            mmr_top_k=settings.mmr_top_k,
            rrf_k=settings.rrf_k,
            mmr_lambda=settings.mmr_lambda,
        ),
        final_passage_limit=5,
    )

    try:
        run = await orchestrator.run(_QUESTION)
    finally:
        await fetcher.aclose()
        for provider in (llm, search, embedder, reranker):
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()
        close_index = getattr(vector_index, "close", None)
        if close_index is not None:
            close_index()

    urls = [str(source.url) for source in run.output.sources]
    assert any(url.startswith(("http://", "https://")) for url in urls)
    assert all("example.com" not in url for url in urls)
    stats = run.output.statistics
    assert stats.fetch_successes > 0 or any(
        "fetch fallback" in warning for warning in run.budget.warnings
    )
    assert stats.bm25_candidate_count > 0
    assert stats.embedding_calls > 0
    assert stats.fused_candidate_count > 0
    assert stats.reranker_calls > 0 or any(
        "reranker failed" in warning for warning in run.budget.warnings
    )
    assert 0 < stats.final_passage_count <= 5
    assert run.output.validation.valid is True

    run_dir = settings.output_dir / run.run_id
    artifact_names = (
        "report.md",
        "sources.json",
        "evidence.json",
        "run.json",
        "validation.json",
    )
    assert all((run_dir / name).is_file() for name in artifact_names)
    serialized = "\n".join(
        (run_dir / name).read_text(encoding="utf-8") for name in artifact_names
    )
    serialized += "\n" + "\n".join(run.budget.warnings)
    for env_name in (
        "REC_LLM_API_KEY",
        "REC_TAVILY_API_KEY",
        "REC_SILICONFLOW_API_KEY",
    ):
        secret = os.environ[env_name]
        assert secret not in serialized
