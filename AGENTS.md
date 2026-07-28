# RecResearcher Repository Instructions

## Repository boundary

Work only inside the current rec-researcher repository.

Do not read, search, inspect, import, copy, or reference:
- ~/projects/deepresearch
- github.com/Leochang7/deepresearch
- any source code, tests, prompts, schemas, comments, configuration structure,
  documentation wording, class names, or function names from that project.

This project is an independent implementation based only on public technical
ideas and the requirements stored in this repository.

## Technical requirements

- Python 3.11 or later.
- Use the src/ package layout.
- Use asyncio for orchestration.
- Use Pydantic v2 for domain and configuration models.
- External APIs must be isolated behind Protocol interfaces.
- All public functions and methods must have type annotations.
- Do not log API keys, Authorization headers, or complete secret values.
- Do not hardcode provider credentials.
- Network tests must be opt-in.
- Default tests must work without API keys or internet access.
- Every external provider needs a deterministic mock or fake implementation.
- Empty search results and empty rerank documents must be handled safely.
- One failed source must not terminate an entire research run.
- Generated claims must preserve source identifiers and URLs.
- Do not add LangChain or LangGraph.

## Development workflow

For every task:
1. Inspect the existing repository.
2. State the files that will be changed.
3. Implement the smallest coherent feature.
4. Add or update tests.
5. Run Ruff and Pytest.
6. Report commands run and remaining limitations.

Do not create Git commits.
Do not rewrite unrelated files.
Do not claim success if tests fail.
