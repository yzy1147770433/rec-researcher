"""Provider-independent domain objects for a research run."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, computed_field


class DomainModel(BaseModel):
    """Common validation policy for domain records."""

    model_config = ConfigDict(extra="forbid")


class WorkState(StrEnum):
    """Lifecycle state shared by planned and completed work."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class InquiryTask(DomainModel):
    """One bounded research question produced by planning."""

    id: str
    question: str = Field(min_length=1)
    state: WorkState = WorkState.PENDING
    priority: int = Field(default=0, ge=0)
    search_queries: list[str] = Field(default_factory=list)
    parent_id: str | None = None


class SourceRecord(DomainModel):
    """A provider result retained for citation and retrieval."""

    id: str
    title: str
    url: HttpUrl
    snippet: str
    provider: str
    score: float | None = None


class PassageRecord(DomainModel):
    """A source-linked text segment used for retrieval."""

    id: str
    source_id: str
    text: str
    position: int = Field(default=0, ge=0)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    content_origin: Literal["fetched", "search_snippet_fallback"] = "fetched"

    @computed_field
    @property
    def passage_id(self) -> str:
        """Expose the stable passage identifier under its domain name."""

        return self.id

    @computed_field
    @property
    def chunk_index(self) -> int:
        """Expose the chunk index while retaining ``position`` compatibility."""

        return self.position


class EvidenceRecord(DomainModel):
    """Evidence that binds a finding to both its passage and source."""

    evidence_id: str
    source_id: str
    passage_id: str
    claim_hint: str
    excerpt: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CitationValidation(DomainModel):
    """Deterministic result of checking one generated report."""

    valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    citation_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    cited_source_count: int = Field(default=0, ge=0)
    available_source_count: int = Field(default=0, ge=0)
    uncited_long_paragraphs: list[str] = Field(default_factory=list)


class TaskResult(DomainModel):
    """Collected outcome for one inquiry task."""

    task_id: str
    state: WorkState
    source_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RunStatistics(DomainModel):
    """Counters and timings for a complete research run."""

    planned_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    sources_found: int = Field(default=0, ge=0)
    sources_failed: int = Field(default=0, ge=0)
    passages_created: int = Field(default=0, ge=0)
    evidence_items: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    fetch_attempts: int = Field(default=0, ge=0)
    fetch_successes: int = Field(default=0, ge=0)
    fetch_failures: int = Field(default=0, ge=0)
    fallback_passages: int = Field(default=0, ge=0)
    raw_passage_count: int = Field(default=0, ge=0)
    deduplicated_passage_count: int = Field(default=0, ge=0)


class ResearchOutput(DomainModel):
    """Serializable output contract for one research question."""

    question: str = Field(min_length=1)
    tasks: list[InquiryTask] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    passages: list[PassageRecord] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    markdown_report: str = ""
    validation: CitationValidation = Field(default_factory=CitationValidation)
    reproduction_suggestions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    statistics: RunStatistics = Field(default_factory=RunStatistics)


class RunBudgetRecord(DomainModel):
    """Persisted snapshot of bounded work performed during a run."""

    start_time: datetime
    elapsed_seconds: float = Field(ge=0.0)
    llm_calls: int = Field(default=0, ge=0)
    search_calls: int = Field(default=0, ge=0)
    embedding_calls: int = Field(default=0, ge=0)
    reranker_calls: int = Field(default=0, ge=0)
    fetched_pages: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    passage_count: int = Field(default=0, ge=0)
    fetch_attempts: int = Field(default=0, ge=0)
    fetch_successes: int = Field(default=0, ge=0)
    fetch_failures: int = Field(default=0, ge=0)
    fallback_passages: int = Field(default=0, ge=0)
    raw_passage_count: int = Field(default=0, ge=0)
    deduplicated_passage_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)


class ResearchRun(DomainModel):
    """Persisted metadata and structured result for one workflow run."""

    run_id: str
    mode: str
    status: WorkState = WorkState.COMPLETED
    output: ResearchOutput
    budget: RunBudgetRecord
