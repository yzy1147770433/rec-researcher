"""Bounded mock and LLM-backed research planning."""

from __future__ import annotations

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from rec_researcher.core.exceptions import LanguageModelResponseError
from rec_researcher.core.models import InquiryTask
from rec_researcher.providers.base import LanguageModel
from rec_researcher.providers.llm_http import OpenAICompatibleLanguageModel

_NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _PlannedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: _NonEmptyText
    queries: list[_NonEmptyText] = Field(min_length=1)


class _Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[_PlannedTask] = Field(min_length=3, max_length=5)


class ResearchPlanner:
    """Split a question into three to five bounded research tasks."""

    _ASPECTS = (
        ("technical-principles", "技术原理"),
        ("representative-work", "代表工作"),
        ("experiments-metrics", "实验与指标"),
        ("code-datasets", "开源代码与数据集"),
        ("reproduction", "复现建议"),
    )

    def __init__(self, language_model: LanguageModel | None = None) -> None:
        """Use deterministic planning unless a real language model is supplied."""

        self.language_model = language_model

    async def create_tasks(self, question: str) -> list[InquiryTask]:
        """Create a validated plan, attempting exactly one JSON format repair."""

        normalized = question.strip()
        if not normalized:
            raise ValueError("research question must not be empty")
        if self.language_model is None:
            return self._mock_tasks(normalized)

        prompt = (
            "Create a research plan for the question below. Return JSON only with "
            'this exact shape: {"tasks":[{"objective":"...","queries":["..."]}]}. '
            "Return 3 to 5 tasks. Every objective and query must be non-empty.\n\n"
            f"Question: {normalized}"
        )
        raw = await self.language_model.generate(prompt)
        try:
            plan = self._parse_plan(raw)
        except LanguageModelResponseError as first_error:
            repair_prompt = (
                "Repair the following response into valid JSON only. Preserve its "
                "meaning, use the required schema, and return 3 to 5 tasks.\n"
                'Schema: {"tasks":[{"objective":"...","queries":["..."]}]}\n'
                f"Invalid response:\n{raw}"
            )
            repaired = await self.language_model.generate(repair_prompt)
            try:
                plan = self._parse_plan(repaired)
            except LanguageModelResponseError as repair_error:
                raise LanguageModelResponseError(
                    "LLM research plan remained invalid after one format repair: "
                    f"{repair_error}"
                ) from first_error

        return [
            InquiryTask(
                id=f"task-{index}",
                question=task.objective,
                priority=index,
                search_queries=task.queries,
            )
            for index, task in enumerate(plan.tasks, start=1)
        ]

    @classmethod
    def _parse_plan(cls, raw: str) -> _Plan:
        try:
            payload = OpenAICompatibleLanguageModel.parse_json(raw)
            return _Plan.model_validate(payload)
        except ValidationError as exc:
            details = json.dumps(exc.errors(include_url=False), ensure_ascii=False)
            raise LanguageModelResponseError(
                f"LLM research plan does not match the required schema: {details}"
            ) from exc

    @classmethod
    def _mock_tasks(cls, question: str) -> list[InquiryTask]:
        return [
            InquiryTask(
                id=f"task-{index}",
                question=f"围绕“{question}”研究{label}",
                priority=index,
                search_queries=[f"{question} {label}"],
            )
            for index, (_, label) in enumerate(cls._ASPECTS, start=1)
        ]
