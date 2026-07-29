"""Bounded deterministic research planning."""

from rec_researcher.core.models import InquiryTask


class ResearchPlanner:
    """Split a question into a stable set of recommender research tasks."""

    _ASPECTS = (
        ("technical-principles", "技术原理"),
        ("representative-work", "代表工作"),
        ("experiments-metrics", "实验与指标"),
        ("code-datasets", "开源代码与数据集"),
        ("reproduction", "复现建议"),
    )

    async def create_tasks(self, question: str) -> list[InquiryTask]:
        """Create five ordered tasks, rejecting an empty question."""

        normalized = question.strip()
        if not normalized:
            raise ValueError("research question must not be empty")
        return [
            InquiryTask(
                id=f"task-{index}",
                question=f"围绕“{normalized}”研究{label}",
                priority=index,
                search_queries=[f"{normalized} {label}"],
            )
            for index, (_, label) in enumerate(self._ASPECTS, start=1)
        ]
