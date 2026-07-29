"""Stable source citation allocation and reference rendering."""

from __future__ import annotations

from collections.abc import Sequence

from rec_researcher.core.models import SourceRecord


class CitationRegistry:
    """Assign one stable citation label to each unique source URL."""

    def __init__(self, sources: Sequence[SourceRecord] = ()) -> None:
        """Register sources in deterministic input order."""

        self._sources: list[SourceRecord] = []
        self._labels_by_url: dict[str, str] = {}
        self._labels_by_source_id: dict[str, str] = {}
        for source in sources:
            self.register(source)

    def register(self, source: SourceRecord) -> str:
        """Return the existing or newly allocated label for a source."""

        url = str(source.url)
        label = self._labels_by_url.get(url)
        if label is None:
            label = f"S{len(self._sources) + 1}"
            self._labels_by_url[url] = label
            self._sources.append(source)
        self._labels_by_source_id[source.id] = label
        return label

    def label_for_source(self, source_id: str) -> str:
        """Resolve an original source identifier to its bracket-free label."""

        try:
            return self._labels_by_source_id[source_id]
        except KeyError as exc:
            raise KeyError(f"source is not registered: {source_id}") from exc

    @property
    def labels(self) -> tuple[str, ...]:
        """Return allocated labels in reference order."""

        return tuple(f"S{index}" for index in range(1, len(self._sources) + 1))

    @property
    def sources(self) -> tuple[SourceRecord, ...]:
        """Return the canonical source behind each unique URL."""

        return tuple(self._sources)

    def references_markdown(self) -> str:
        """Render the canonical Markdown References section."""

        lines = ["## References", ""]
        lines.extend(
            f"- [S{index}] {source.title} — {source.url}"
            for index, source in enumerate(self._sources, start=1)
        )
        if not self._sources:
            lines.append("- 无引用来源。")
        return "\n".join(lines)
