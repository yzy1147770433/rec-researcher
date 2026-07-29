# RecResearcher architecture

## Design

RecResearcher uses an asyncio-oriented layered architecture. Domain data flows
inward through typed interfaces, while concrete external services remain at the
edge. Pydantic v2 models define serializable boundaries. Provider protocols make
offline deterministic fakes and real adapters interchangeable only through
explicit configuration.

## Layers

### `core`

Owns provider-independent data models, configuration, shared exceptions, run
state, and serialization rules. No provider implementation may be imported here.

### `providers`

Defines protocols for language models, search, embeddings, reranking, and vector
indexes. Concrete adapters translate external errors into project exceptions.
Each external adapter must have a deterministic offline fake.

### `planning`

Turns a research question into bounded, ordered inquiry tasks. Planning consumes
the language-model protocol rather than a concrete SDK.

### `workflow`

Coordinates the run with asyncio, applies concurrency and work budgets, isolates
per-source failures, and accumulates run statistics. Runtime mode selection is
explicit: real mode cannot substitute mock providers.

### `retrieval`

Fetches documents, extracts text, creates source-linked chunks, performs sparse
and dense recall, fuses rankings, reranks candidates, and applies diversity
selection. Empty candidate collections are normal results.

### `evidence`

Builds evidence records from passages and validates that claims and citations
resolve to retained source IDs, passage IDs, and URLs.

### `reporting`

Produces the Markdown report and reproducibility suggestions. It rejects major
claims without valid numbered citations.

### `domain`

Contains recommender-system-specific analysis, including dataset, objective,
baseline, offline metric, online metric, bias, and reproducibility concepts. It
depends on generic core contracts, not provider implementations.

### `evaluation`

Measures planning coverage, retrieval quality, citation validity, report quality,
failure isolation, determinism, and budget compliance using offline fixtures by
default.

## Dependency direction

`core` is the innermost layer. Provider interfaces may use core models. Planning,
retrieval, evidence, reporting, and domain use core types and provider protocols.
Workflow composes those capabilities; evaluation observes them through public
interfaces. Concrete provider SDKs do not leak into models or orchestration.

The project does not use LangChain or LangGraph. Orchestration is implemented
directly with asyncio and small typed components.
