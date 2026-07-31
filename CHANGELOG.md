# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-31

### Added

- Complete hybrid retrieval from the CLI: page extraction, chunking, BM25,
  dense retrieval, weighted RRF, reranking, and MMR.
- Deterministic mock fetch, embedding, reranking, and vector implementations
  for offline hybrid runs and tests.
- Versioned recommendation-system benchmarks, null-aware retrieval metrics,
  named ablations, resumable runs, and Markdown/JSON comparison artifacts.
- Opt-in real-provider and full hybrid end-to-end tests, plus an independent
  Milvus Lite integration job in CI.

### Changed

- Real hybrid mode now wires Tavily, page fetching, SiliconFlow embeddings and
  reranking, and Milvus Lite through provider interfaces.
- Research orchestration now records hybrid-stage statistics, bounded
  concurrency, source-linked passages, and explicit degradation warnings.
- CI covers Python 3.11 and 3.12, caches uv dependencies, runs offline checks,
  and builds distributions.

### Fixed

- Isolated failed sources and provider-stage failures so one failure does not
  terminate the complete research run.
- Preserved source identifiers and URLs through chunking, retrieval, evidence,
  report generation, and citation validation.
- Safely handled empty search results, passage corpora, and reranker inputs.

### Known limitations

- PDF structure, tables, equations, and appendices have limited extraction
  support and may fall back to search snippets.
- Citation validation verifies structure and source mappings, not factual
  correctness, freshness, or source independence.
- Real runs depend on external providers, mutable pages, credentials, quotas,
  and network conditions, so they are non-deterministic and opt-in.
- The labeled benchmark is small and intended for regression analysis rather
  than broad performance claims; domain extraction still needs manual review.

## [0.1.1] - 2026-07-29

### Added

- GitHub Actions checks for linting, offline tests, and package builds.
- MIT license and package project metadata.

### Changed

- Replaced the README with an English version.
- Classified Milvus Lite tests as integration tests because they bind a local
  socket.

## [0.1.0] - 2026-07-29

### Added

- Async research workflow with bounded planning, scheduling, and source
  failure isolation.
- Deterministic offline providers and opt-in real providers.
- Source-linked evidence, citation validation, recommendation-system analysis,
  and Markdown/JSON artifacts.
- Independently tested lexical, vector, fusion, reranking, and diversity
  retrieval components.
- Offline benchmark runner and command-line interface.

[Unreleased]: https://github.com/yzy1147770433/rec-researcher/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yzy1147770433/rec-researcher/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/yzy1147770433/rec-researcher/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yzy1147770433/rec-researcher/releases/tag/v0.1.0
