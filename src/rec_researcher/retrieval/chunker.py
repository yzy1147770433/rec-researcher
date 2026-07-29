"""Paragraph-aware source text chunking."""

from __future__ import annotations

from rec_researcher.core.models import PassageRecord
from rec_researcher.core.settings import Settings


class PassageChunker:
    """Split text into bounded, overlapping, source-linked passages."""

    def __init__(self, settings: Settings) -> None:
        """Read chunk size and overlap from application settings."""

        self.chunk_size = settings.chunk_size
        self.overlap = min(settings.chunk_overlap, settings.chunk_size - 1)

    def chunk(self, source_id: str, text: str) -> list[PassageRecord]:
        """Create non-empty chunks, preferring paragraph boundaries."""

        if not text or not text.strip():
            return []
        passages: list[PassageRecord] = []
        start = 0
        position = 0
        text_length = len(text)
        while start < text_length:
            limit = min(start + self.chunk_size, text_length)
            end = self._preferred_end(text, start, limit)
            trimmed_start, trimmed_end = self._trim_bounds(text, start, end)
            if trimmed_start < trimmed_end:
                passages.append(
                    PassageRecord(
                        id=f"{source_id}:{position}",
                        source_id=source_id,
                        text=text[trimmed_start:trimmed_end],
                        position=position,
                        start_offset=trimmed_start,
                        end_offset=trimmed_end,
                    )
                )
                position += 1
            if end >= text_length:
                break
            next_start = end - self.overlap
            start = next_start if next_start > start else end
        return passages

    @staticmethod
    def _preferred_end(text: str, start: int, limit: int) -> int:
        if limit == len(text):
            return limit
        boundary = text.rfind("\n\n", start + 1, limit + 1)
        return boundary + 2 if boundary >= start else limit

    @staticmethod
    def _trim_bounds(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end
