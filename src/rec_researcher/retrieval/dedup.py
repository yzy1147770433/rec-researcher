"""Lightweight URL and passage deduplication."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from rec_researcher.core.models import PassageRecord, SourceRecord

_WHITESPACE = re.compile(r"\s+")


def normalize_url(url: str) -> str:
    """Return a stable HTTP URL without a fragment or redundant trailing slash."""

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    port = parts.port
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    if port and not default_port:
        hostname = f"{hostname}:{port}"
    path = parts.path.rstrip("/") or "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, hostname, path, query, ""))


def deduplicate_sources(sources: Iterable[SourceRecord]) -> list[SourceRecord]:
    """Keep the first source for each normalized URL."""

    seen: set[str] = set()
    unique: list[SourceRecord] = []
    for source in sources:
        key = normalize_url(str(source.url))
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def text_fingerprint(text: str) -> str:
    """Hash case-folded, whitespace-normalized text using SHA-256."""

    normalized = _WHITESPACE.sub(" ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_passages(passages: Iterable[PassageRecord]) -> list[PassageRecord]:
    """Keep the first passage for each normalized-text fingerprint."""

    seen: set[str] = set()
    unique: list[PassageRecord] = []
    for passage in passages:
        fingerprint = text_fingerprint(passage.text)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(passage)
    return unique
