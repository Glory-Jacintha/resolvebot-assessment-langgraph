"""Policy search (the RAG part).

* policies.md is split into documents at each "# " title line, then into
  paragraphs at each "### §x.y" heading.
* manifest.json (matched by doc_id) decides which documents may be used:
    - status "current" and effective_date <= APP_TODAY  -> in effect
    - status "current" and effective_date >  APP_TODAY  -> future change (flagged)
    - anything else (e.g. superseded)                   -> never used
* Ranking is BM25 over the paragraphs. Pure Python, no model download.
* No policy facts are written in this code; everything comes from the files.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import get_settings
from rules import fmt_date, parse_date

_DOC_ID_RE = re.compile(r"\*\*Document ID:\*\*\s*([A-Za-z0-9_\-]+)")
_GROUP_RE = re.compile(r"^##\s+(§\d+)\s+(.*)$")
_SECTION_RE = re.compile(r"^###\s+(§[\d.]+)\s+(.*)$")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were",
    "be", "it", "this", "that", "i", "my", "me", "we", "you", "your", "do", "does", "can",
    "how", "what", "when", "if", "with", "at", "by", "from", "as", "have", "has", "will",
    "there", "any", "get", "about", "so", "no", "not", "than", "then", "into", "up",
}


@dataclass
class Chunk:
    doc_id: str
    title: str
    section: str
    heading: str
    group: str
    text: str
    tokens: list[str]


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_date: str
    chunks: list[Chunk]


# --------------------------------------------------------------- loading
def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]  # shipp -> ship
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9₹]+", text.lower())
    return [_stem(w) for w in words if w not in _STOPWORDS]


def _split_documents(markdown: str) -> list[tuple[str, list[str]]]:
    docs: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    title = ""
    for line in markdown.splitlines():
        if line.startswith("# "):
            if current is not None:
                docs.append((title, current))
            title, current = line[2:].strip(), []
        elif current is not None:
            current.append(line)
    if current is not None:
        docs.append((title, current))
    return docs


def _build_chunks(doc_id: str, title: str, lines: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    group = ""
    section = heading = ""
    body: list[str] = []

    def flush() -> None:
        if not section:
            return
        text = "\n".join(body).strip()
        if text:
            searchable = f"{title} {group} {heading} {text}"
            chunks.append(Chunk(doc_id, title, section, heading, group, text, tokenize(searchable)))

    for line in lines:
        g = _GROUP_RE.match(line)
        s = _SECTION_RE.match(line)
        if g:
            flush()
            section, heading, body = "", "", []
            group = g.group(2).strip()
        elif s:
            flush()
            section, heading, body = s.group(1), s.group(2).strip(), []
        elif section:
            body.append(line)
    flush()
    return chunks


_cache: dict[tuple, list[Document]] = {}


def load_documents() -> list[Document]:
    """Parse the files (cached until a file changes)."""
    settings = get_settings()
    policies, manifest = Path(settings.policies_path), Path(settings.manifest_path)
    key = (str(policies), policies.stat().st_mtime, str(manifest), manifest.stat().st_mtime)
    if key in _cache:
        return _cache[key]

    meta = {m["doc_id"]: m for m in json.loads(manifest.read_text(encoding="utf-8"))}
    documents: list[Document] = []
    for title, lines in _split_documents(policies.read_text(encoding="utf-8")):
        match = next((m for m in map(_DOC_ID_RE.search, lines) if m), None)
        doc_id = match.group(1) if match else None
        if doc_id is None or doc_id not in meta:
            continue  # unknown documents are never used
        entry = meta[doc_id]
        documents.append(
            Document(
                doc_id=doc_id,
                title=title,
                doc_type=str(entry.get("doc_type", "policy")),
                status=str(entry.get("status", "")).strip().lower(),
                effective_date=str(entry.get("effective_date", "")),
                chunks=_build_chunks(doc_id, title, lines),
            )
        )
    _cache.clear()
    _cache[key] = documents
    return documents


# ---------------------------------------------------------------- search
def _usable_documents() -> list[tuple[Document, bool]]:
    """(document, is_future) for every document that may be shown."""
    today = get_settings().app_today
    usable = []
    for doc in load_documents():
        if doc.status != "current":
            continue  # superseded or unknown status: never used
        start = parse_date(doc.effective_date)
        if start is None:
            continue
        usable.append((doc, start > today))
    return usable


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Best matching paragraphs, as dicts ready to hand to the LLM."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return []

    candidates = [(doc, future, chunk) for doc, future in _usable_documents() for chunk in doc.chunks]
    if not candidates:
        return []

    n = len(candidates)
    avg_len = sum(len(c.tokens) for _, _, c in candidates) / n or 1.0
    doc_freq: Counter[str] = Counter()
    for _, _, chunk in candidates:
        doc_freq.update(set(chunk.tokens))

    k1, b = 1.5, 0.75
    scored = []
    for doc, future, chunk in candidates:
        counts = Counter(chunk.tokens)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(chunk.tokens) / avg_len))
        if score > 0:
            scored.append((score, doc, future, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:k]

    # If a policy and the FAQ could disagree, the policy wins: drop FAQ paragraphs
    # whenever at least one real policy paragraph was found.
    if any(doc.doc_type != "faq" for _, doc, _, _ in top):
        top = [item for item in top if item[1].doc_type != "faq"]

    results = []
    for score, doc, future, chunk in top:
        item: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "doc_type": doc.doc_type,
            "section": chunk.section,
            "heading": chunk.heading,
            "text": chunk.text,
            "effective_date": doc.effective_date,
            "in_effect": not future,
        }
        if future:
            start = fmt_date(parse_date(doc.effective_date))
            item["note"] = (
                f"NOT IN EFFECT YET. This is a future change that starts on {start}. "
                "Mention it only as an upcoming change with that date."
            )
        results.append(item)
    return results
