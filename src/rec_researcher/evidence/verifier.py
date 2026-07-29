"""Local, deterministic report citation verification."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rec_researcher.core.models import CitationValidation, SourceRecord
from rec_researcher.reporting.citation import CitationRegistry

_CITATION_RE = re.compile(r"\[(S\d+)\]")
_REFERENCE_RE = re.compile(r"(?m)^\s*-\s*\[(S\d+)\]\s+.*?—\s*(\S+)\s*$")
_SECTION_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")


class CitationVerifier:
    """Validate citations without making network or language-model calls."""

    def __init__(self, *, long_paragraph_length: int = 120) -> None:
        """Configure the threshold used to flag uncited factual prose."""

        if long_paragraph_length < 1:
            raise ValueError("long_paragraph_length must be at least 1")
        self.long_paragraph_length = long_paragraph_length

    def verify(
        self,
        report: str,
        sources: Sequence[SourceRecord],
        registry: CitationRegistry | None = None,
    ) -> CitationValidation:
        """Return all citation errors and report quality statistics."""

        citation_registry = registry or CitationRegistry(sources)
        known = set(citation_registry.labels)
        body, references = self._split_references(report)
        body_citations = _CITATION_RE.findall(body)
        all_citations = _CITATION_RE.findall(report)
        reference_entries = _REFERENCE_RE.findall(references)
        reference_labels = [label for label, _ in reference_entries]
        errors: list[str] = []

        if "S0" in all_citations:
            errors.append("citation [S0] is not allowed")
        unknown = sorted(set(all_citations) - known, key=self._label_number)
        if unknown:
            errors.append(
                "report contains citations absent from source registry: "
                + ", ".join(f"[{item}]" for item in unknown)
            )
        unknown_references = sorted(
            set(reference_labels) - known, key=self._label_number
        )
        if unknown_references:
            errors.append(
                "References contain entries absent from SourceRecord registry: "
                + ", ".join(f"[{item}]" for item in unknown_references)
            )
        missing_references = sorted(
            set(body_citations) - set(reference_labels), key=self._label_number
        )
        if missing_references:
            errors.append(
                "body citations missing from References: "
                + ", ".join(f"[{item}]" for item in missing_references)
            )
        if len(reference_labels) != len(set(reference_labels)):
            errors.append("References contain duplicate citation labels")

        numbered = sorted(
            {self._label_number(item) for item in all_citations if item != "S0"}
        )
        if numbered and numbered != list(range(1, numbered[-1] + 1)):
            errors.append("citation numbering contains a gap")
        if reference_labels != [f"S{i}" for i in range(1, len(reference_labels) + 1)]:
            errors.append("References must use consecutive labels starting at [S1]")

        source_urls = {
            f"S{index}": str(source.url)
            for index, source in enumerate(citation_registry.sources, start=1)
        }
        for label, url in reference_entries:
            if not url.startswith(("http://", "https://")):
                errors.append(f"reference [{label}] URL must use http or https")
            elif source_urls.get(label) != url:
                errors.append(
                    f"reference [{label}] URL does not match its SourceRecord"
                )

        cited_sections = self._cited_major_sections(body)
        if not cited_sections:
            errors.append("at least one major report section must contain a citation")

        paragraphs = self._prose_paragraphs(body)
        factual = [item for item in paragraphs if len(item) >= 40]
        cited_factual = [item for item in factual if _CITATION_RE.search(item)]
        coverage = len(cited_factual) / len(factual) if factual else 0.0
        uncited_long = [
            item
            for item in factual
            if len(item) >= self.long_paragraph_length and not _CITATION_RE.search(item)
        ]
        cited_known = set(body_citations) & known
        diversity = len(cited_known) / len(known) if known else 0.0
        warnings = (
            [f"{len(uncited_long)} long factual paragraph(s) have no citation"]
            if uncited_long
            else []
        )
        return CitationValidation(
            valid=not errors,
            errors=errors,
            warnings=warnings,
            citation_coverage=coverage,
            source_diversity=diversity,
            cited_source_count=len(cited_known),
            available_source_count=len(known),
            uncited_long_paragraphs=uncited_long,
        )

    @staticmethod
    def _split_references(report: str) -> tuple[str, str]:
        match = re.search(r"(?m)^##\s+References\s*$", report)
        if match is None:
            return report, ""
        return report[: match.start()], report[match.end() :]

    @staticmethod
    def _label_number(label: str) -> int:
        return int(label[1:])

    @staticmethod
    def _prose_paragraphs(body: str) -> list[str]:
        return [
            " ".join(block.split())
            for block in re.split(r"\n\s*\n", body)
            if block.strip()
            and not block.lstrip().startswith("#")
            and not block.lstrip().startswith("-")
        ]

    @staticmethod
    def _cited_major_sections(body: str) -> list[str]:
        matches = list(_SECTION_RE.finditer(body))
        cited: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            title = match.group(1).strip().casefold()
            eligible = title not in {"references", "局限性", "limitations"}
            if eligible and _CITATION_RE.search(body[match.end() : end]):
                cited.append(title)
        return cited
