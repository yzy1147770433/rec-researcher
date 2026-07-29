# RecResearcher MVP requirements

## Scope

RecResearcher accepts one recommender-system research question and produces a
traceable research package. The MVP establishes the local domain, provider
boundaries, orchestration contract, and reporting contract. Real network search
is outside this foundation milestone.

## Input

The required input is a non-empty natural-language question about recommender
systems. Runtime configuration selects either `mock` or `real` mode and supplies
budgets and provider settings. Mock fixtures may be selected explicitly for
deterministic development and evaluation.

## Output

One run returns:

- decomposed research subtasks and the result of each subtask;
- discovered sources, including stable source IDs, titles, URLs, snippets, and
  provider names;
- passages and evidence records that retain their source relationships;
- a Markdown report;
- reproducibility suggestions, such as configurations, datasets, seeds, and
  evaluation steps to record;
- run statistics covering work counts, failures, and elapsed time.

Every major conclusion in the Markdown report must cite one or more source
numbers. Citation validation must be able to map each number back to a retained
source ID and URL. Unsupported major conclusions are report-validation errors.

## Runtime guarantees

- Mock mode is completely offline: it must neither issue network requests nor
  require API credentials. Its provider behavior must be deterministic.
- Real mode must never silently fall back to mock data or mock providers. Missing
  real-mode configuration is an explicit configuration error; provider failures
  remain visible in results and statistics.
- Failure to fetch, parse, retrieve, or rank one source must not fail the entire
  research run. The workflow records the failure and continues with other
  sources whenever useful work remains.
- Empty search results and empty reranking inputs are valid boundary cases and
  return empty collections safely.
- Secrets, authorization headers, and complete credential values must never be
  logged or included in run output.

## Acceptance boundary

Default tests run without internet access or API keys. Network tests are
explicitly marked and opt-in. This milestone does not implement real search,
fetching, model calls, or vector-database operations.
