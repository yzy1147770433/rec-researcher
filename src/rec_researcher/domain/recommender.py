"""Evidence-grounded analysis specialized for recommender-system papers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum

from pydantic import Field

from rec_researcher.core.models import DomainModel, EvidenceRecord, SourceRecord


class ReproductionDifficulty(StrEnum):
    """Coarse, explainable reproduction difficulty level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendationPaperProfile(DomainModel):
    """Structured, evidence-bounded profile of a recommendation paper."""

    title: str
    task_type: list[str] = Field(default_factory=list)
    model_family: list[str] = Field(default_factory=list)
    core_techniques: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    code_urls: list[str] = Field(default_factory=list)
    has_public_code: bool | None = None
    hardware_requirements_stated: bool = False
    estimated_gpu_memory_gb: float | None = Field(default=None, gt=0)
    reproduction_difficulty: ReproductionDifficulty
    reproduction_score: int = Field(ge=-2)
    reproduction_score_reasons: list[str] = Field(default_factory=list)
    reproduction_steps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    suitability_reason: str
    uncertainty: list[str] = Field(default_factory=list)


class RecommendationDomainAnalyzer:
    """Extract recommendation-specific signals with deterministic rules."""

    _TASKS = {
        "召回": (r"召回", r"retrieval", r"candidate generation"),
        "排序": (r"排序", r"ranking", r"ranker"),
        "多任务学习": (r"多任务", r"multi[- ]task"),
        "序列推荐": (r"序列推荐", r"sequential recommendation"),
        "图推荐": (r"图推荐", r"graph recommendation", r"graph-based recommender"),
        "生成式推荐": (
            r"生成式推荐",
            r"generative recommendation",
            r"generative recommender",
        ),
    }
    _TECHNIQUES = {
        "ItemCF": (r"\bitemcf\b", r"item[- ]based collaborative filtering"),
        "双塔": (r"双塔", r"two[- ]tower", r"dual[- ]encoder"),
        "DeepFM": (r"\bdeepfm\b",),
        "DIN": (r"\bdin\b", r"deep interest network"),
        "MMOE": (r"\bmmoe\b", r"multi-gate mixture-of-experts"),
        "GNN": (r"\bgnn\b", r"graph neural network"),
        "Transformer": (r"\btransformer\b",),
        "Semantic ID": (r"semantic ids?", r"语义\s*id"),
        "Autoregressive Generation": (
            r"autoregressive generation",
            r"自回归生成",
        ),
    }
    _DATASETS = (
        "MovieLens",
        "Amazon Reviews",
        "Yelp",
        "Gowalla",
        "Last.fm",
        "MIND",
        "Criteo",
        "Ali-CCP",
        "KuaiRec",
        "Taobao",
    )
    _METRICS = {
        "Recall": r"\brecall(?:@\d+)?\b",
        "NDCG": r"\bndcg(?:@\d+)?\b",
        "Hit Rate": r"\bhit[ -]?rate(?:@\d+)?\b|\bhr@\d+\b",
        "MRR": r"\bmrr\b",
        "AUC": r"\bauc\b",
        "LogLoss": r"\blog[ -]?loss\b",
    }
    _GITHUB_URL = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.I)
    _GPU_MEMORY = (
        re.compile(r"(?:gpu|显存)[^\n.;]{0,40}?(\d+(?:\.\d+)?)\s*(?:gb|gib)", re.I),
        re.compile(
            r"(\d+(?:\.\d+)?)\s*(?:gb|gib)[^\n.;]{0,30}?(?:gpu memory|显存)",
            re.I,
        ),
    )

    def analyze(
        self,
        sources: Sequence[SourceRecord],
        evidence: Sequence[EvidenceRecord],
        *,
        title: str | None = None,
    ) -> RecommendationPaperProfile:
        """Analyze source metadata and evidence without filling evidential gaps."""

        text = self._combined_text(sources, evidence)
        task_types = self._matches(text, self._TASKS)
        techniques = self._matches(text, self._TECHNIQUES)
        datasets = [
            dataset
            for dataset in self._DATASETS
            if re.search(re.escape(dataset), text, re.I)
        ]
        metrics = [
            metric
            for metric, pattern in self._METRICS.items()
            if re.search(pattern, text, re.I)
        ]
        code_urls = sorted(set(self._GITHUB_URL.findall(text)))
        explicit_no_code = bool(
            re.search(
                r"no (?:public(?:ly available)? )?code"
                r"|code (?:is )?not public(?:ly available)?"
                r"|代码(?:未|不)公开",
                text,
                re.I,
            )
        )
        has_public_code = True if code_urls else (False if explicit_no_code else None)
        uncertainty: list[str] = []
        if code_urls and explicit_no_code:
            uncertainty.append("来源对代码是否公开存在冲突；已保留发现的 GitHub URL。")

        memory_values = {
            float(match.group(1))
            for pattern in self._GPU_MEMORY
            for match in pattern.finditer(text)
        }
        if len(memory_values) == 1:
            gpu_memory = memory_values.pop()
        else:
            gpu_memory = None
            if len(memory_values) > 1:
                uncertainty.append("来源给出了相互冲突的 GPU 显存数值。")
        hardware_stated = bool(
            memory_values
            or gpu_memory is not None
            or re.search(
                r"multi[- ]gpu|multi[- ]node|distributed|多卡|分布式|单卡",
                text,
                re.I,
            )
        )

        score, reasons = self._difficulty(text, has_public_code, datasets)
        difficulty = self._difficulty_level(score)
        risks = self._risks(has_public_code, datasets, uncertainty)
        selected_title = title or (sources[0].title if sources else "Unknown paper")
        suitability = self._suitability(task_types, techniques, datasets, metrics)
        return RecommendationPaperProfile(
            title=selected_title,
            task_type=task_types,
            model_family=techniques,
            core_techniques=techniques,
            datasets=datasets,
            metrics=metrics,
            code_urls=code_urls,
            has_public_code=has_public_code,
            hardware_requirements_stated=hardware_stated,
            estimated_gpu_memory_gb=gpu_memory,
            reproduction_difficulty=difficulty,
            reproduction_score=score,
            reproduction_score_reasons=reasons,
            reproduction_steps=self._steps(has_public_code, datasets),
            risks=risks,
            suitability_reason=suitability,
            uncertainty=uncertainty,
        )

    @staticmethod
    def _combined_text(
        sources: Sequence[SourceRecord], evidence: Sequence[EvidenceRecord]
    ) -> str:
        source_text = "\n".join(
            f"{source.title}\n{source.snippet}\n{source.url}" for source in sources
        )
        evidence_text = "\n".join(
            f"{item.claim_hint}\n{item.excerpt}" for item in evidence
        )
        return f"{source_text}\n{evidence_text}"

    @staticmethod
    def _matches(text: str, patterns: dict[str, tuple[str, ...]]) -> list[str]:
        return [
            label
            for label, variants in patterns.items()
            if any(re.search(pattern, text, re.I) for pattern in variants)
        ]

    @staticmethod
    def _difficulty(
        text: str, has_public_code: bool | None, datasets: list[str]
    ) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []

        def add(points: int, reason: str) -> None:
            nonlocal score
            score += points
            reasons.append(f"{points:+d}：{reason}")

        if has_public_code is not True:
            add(2, "未识别到公开代码")
        if not datasets:
            add(2, "未识别到公开数据集")
        llm_training = (
            r"(?:large language model|\bllm\b|大语言模型)"
            r".{0,30}(?:train|训练|fine[- ]tun)"
        )
        if re.search(llm_training, text, re.I):
            add(2, "需要大语言模型训练")
        if re.search(r"multi[- ]gpu|multi[- ]node|distributed|多卡|分布式", text, re.I):
            add(2, "需要多卡或分布式运行")
        if re.search(r"single[- ]gpu|single gpu|单卡", text, re.I):
            add(-1, "来源明确说明单卡可运行")
        complete_artifacts = (
            r"(?:complete|full|完整).{0,20}(?:config|配置).{0,40}checkpoint"
            r"|checkpoint.{0,40}(?:complete|full|完整).{0,20}(?:config|配置)"
        )
        if re.search(complete_artifacts, text, re.I):
            add(-1, "提供完整配置和 checkpoint")
        return score, reasons

    @staticmethod
    def _difficulty_level(score: int) -> ReproductionDifficulty:
        if score <= 0:
            return ReproductionDifficulty.LOW
        if score <= 3:
            return ReproductionDifficulty.MEDIUM
        return ReproductionDifficulty.HIGH

    @staticmethod
    def _steps(has_public_code: bool | None, datasets: list[str]) -> list[str]:
        steps = ["核对论文、代码与数据版本并固定实验配置。"]
        steps.append(
            "运行公开代码的最小样例并保存环境锁文件。"
            if has_public_code
            else "先实现最小基线，明确记录论文中缺失的实现细节。"
        )
        steps.append(
            f"在 {datasets[0]} 上复现主指标并与论文口径对齐。"
            if datasets
            else "获得合法可用的数据后，再对齐划分、负采样与评估口径。"
        )
        return steps

    @staticmethod
    def _risks(
        has_public_code: bool | None,
        datasets: list[str],
        uncertainty: list[str],
    ) -> list[str]:
        risks = list(uncertainty)
        if has_public_code is not True:
            risks.append("公开代码不可确认，论文细节可能不足以重建实现。")
        if not datasets:
            risks.append("公开数据不可确认，实验对比可能不可复核。")
        return risks

    @staticmethod
    def _suitability(
        tasks: list[str], techniques: list[str], datasets: list[str], metrics: list[str]
    ) -> str:
        signals = len(tasks) + len(techniques) + len(datasets) + len(metrics)
        if signals == 0:
            return "当前证据未提供足够的推荐系统专属信息，需补充来源。"
        return (
            f"证据覆盖 {len(tasks)} 类推荐任务、{len(techniques)} 类模型技术、"
            f"{len(datasets)} 个常见数据集和 {len(metrics)} 项常见指标。"
        )
