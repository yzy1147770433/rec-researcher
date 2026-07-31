# RecResearcher v0.2.0

v0.2.0 connects the complete hybrid retrieval path to the command line while
keeping mock runs and the default test suite deterministic and offline.

## 主要功能

- Bounded asynchronous planning, searching, fetching, and retrieval with
  per-source failure isolation.
- Hybrid retrieval with page extraction, chunking, deduplication, BM25, dense
  retrieval, weighted RRF, reranking, and MMR.
- Real-provider adapters for an OpenAI-compatible LLM, Tavily, SiliconFlow,
  and Milvus Lite, all isolated behind typed interfaces.
- Source-linked passages, evidence, reports, and deterministic citation
  validation.
- Versioned benchmarks with named ablations, null-aware metrics, resumable
  execution, and comparison artifacts.

## 安装

Python 3.11 or 3.12 and uv are supported:

```bash
git clone https://github.com/yzy1147770433/rec-researcher.git
cd rec-researcher
uv sync --all-groups
uv run rec-researcher doctor
```

## Mock 演示

This run needs no API key or network connection and exercises the complete
hybrid pipeline with deterministic fakes:

```bash
uv run rec-researcher run \
  "How does generative retrieval differ from two-tower retrieval?" \
  --mode mock --retrieval-mode hybrid
```

Artifacts are written below `outputs/<run-id>/`. The shorter snippet path is
available with `--retrieval-mode snippet`.

## Real hybrid 演示

Copy `.env.example` to the ignored `.env` file and set the OpenAI-compatible
LLM, Tavily, and SiliconFlow values. Never commit real credentials.

```dotenv
REC_LLM_BASE_URL=https://your-provider.invalid/v1
REC_LLM_API_KEY=replace-with-your-key
REC_LLM_MODEL=replace-with-your-model
REC_TAVILY_API_KEY=replace-with-your-key
REC_SILICONFLOW_API_KEY=replace-with-your-key
REC_EMBEDDING_MODEL=replace-with-your-embedding-model
REC_RERANKER_MODEL=replace-with-your-reranker-model
```

```bash
uv run rec-researcher doctor --real
uv run rec-researcher run \
  "Representative work on generative retrieval for recommender systems" \
  --mode real --retrieval-mode hybrid \
  --embedding-provider siliconflow \
  --reranker-provider siliconflow \
  --vector-store milvus
```

Real mode makes external requests and may incur provider charges. Milvus Lite
uses `./data/rec_researcher.db` by default.

## 测试

The default suite excludes `network`, `network_e2e`, and `integration` markers:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
uv build
```

Milvus Lite is tested separately because it binds a local socket:

```bash
uv run pytest -q tests/unit/test_vector_store.py -m integration
```

Real network tests require explicit credentials and marker selection; see
[`real-e2e.md`](real-e2e.md).

## 已知限制

- PDF structure, tables, equations, and appendices have limited extraction
  support; binary documents commonly fall back to snippets.
- Citation validation checks structure and source mappings, not factual truth,
  freshness, or source independence.
- Real runs are non-deterministic and depend on provider uptime, quotas, search
  indexes, and mutable pages.
- The current labeled benchmark is small and should not be treated as a broad
  quality claim.
- Rule-based recommendation-domain extraction can miss or misassociate
  entities and requires manual review for publication-quality research.
