"""Text preprocessing pipelines for contextual embeddings and lexical feature extraction."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar

from ticket_classification.models.entities import TicketRecord

STOP_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "done",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "please",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "would",
    "you",
    "your",
    "yours",
    "hello",
    "hi",
    "dear",
    "regards",
    "thanks",
    "thank",
    "sincerely",
}


class TextCleaningHandler(ABC):
    """Abstract handler in the cleaning Chain of Responsibility."""

    def __init__(self) -> None:
        self._next: TextCleaningHandler | None = None

    def set_next(self, handler: TextCleaningHandler) -> TextCleaningHandler:
        self._next = handler
        return handler

    def handle(self, record: TicketRecord) -> TicketRecord:
        record = self._process(record)
        if self._next is not None:
            return self._next.handle(record)
        return record

    @abstractmethod
    def _process(self, record: TicketRecord) -> TicketRecord:
        """Apply single cleaning transformation to record."""


class EmailHeaderAndFooterHandler(TextCleaningHandler):
    """Removes email routing headers, forwarding traces, attachments, and disclaimers."""

    HEADER_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        re.compile(r"(?im)^(from|sent|received|to|cc|bcc)\s*:.*$"),
        re.compile(r"(?im)^subject\s*:\s*"),
        re.compile(r"\[cid:[^\]]+\]", re.IGNORECASE),
        re.compile(
            r"the information in this email is confidential.*?(?:recipient\.|$)",
            re.IGNORECASE | re.DOTALL,
        ),
        re.compile(r"<[^>]+>"),  # HTML tags
    ]

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual or record.raw_text or ""
        for pat in self.HEADER_PATTERNS:
            text = pat.sub(" ", text)
        record.text_contextual = re.sub(r"\s+", " ", text).strip()
        return record


class TypedPIIMaskHandler(TextCleaningHandler):
    """Replaces sensitive PII and system IDs with semantic placeholders."""

    RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    RE_EMAIL = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
    RE_TICKET_ID = re.compile(
        r"\b(?:INC|RITM|TKT|REQ|ERR|CLM)[-_]?\d+\b", re.IGNORECASE
    )
    RE_AMOUNT = re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?")
    RE_PHONE = re.compile(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
    )

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual or record.raw_text or ""
        text = self.RE_URL.sub("<URL>", text)
        text = self.RE_EMAIL.sub("<EMAIL>", text)
        text = self.RE_TICKET_ID.sub("<TICKET_ID>", text)
        text = self.RE_AMOUNT.sub("<AMOUNT>", text)
        text = self.RE_PHONE.sub("<PHONE>", text)
        record.text_contextual = re.sub(r"\s+", " ", text).strip()
        return record


class RedactionMaskNormalizerHandler(TextCleaningHandler):
    """Normalizes CFPB / financial claim redaction masks (XXXX, XX/XX/XXXX) into <REDACTED>."""

    RE_XXXX = re.compile(r"\b(?:X{2,}|x{2,})(?:/(?:X{2,}|x{2,}))*\b")
    RE_BRACKET_REDACT = re.compile(
        r"\[(?:REDACTED|MASKED|CONFIDENTIAL)\]", re.IGNORECASE
    )

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual or record.raw_text or ""
        text = self.RE_XXXX.sub("<REDACTED>", text)
        text = self.RE_BRACKET_REDACT.sub("<REDACTED>", text)
        record.text_contextual = re.sub(r"\s+", " ", text).strip()
        return record


class TemplateSignatureHandler(TextCleaningHandler):
    """Computes a structural template signature for near-duplicate grouping."""

    RE_NUMBERS = re.compile(r"\b\d+\b")
    RE_QUOTED = re.compile(r'"[^"]+"|\'[^\']+\'')

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = (record.text_contextual or record.raw_text or "").lower()
        sig = self.RE_QUOTED.sub("<STR>", text)
        sig = self.RE_NUMBERS.sub("<NUM>", sig)
        sig = re.sub(r"[^a-z0-9_<>]+", " ", sig)
        record.template_signature = re.sub(r"\s+", " ", sig).strip()[:180]
        return record


class DualTrackBuilderHandler(TextCleaningHandler):
    """Produces contextual text for dense embeddings and normalized tokens for term frequency analysis."""

    RE_WORDS = re.compile(r"\b[a-z]{2,}\b")

    def __init__(self, min_token_len: int = 2) -> None:
        super().__init__()
        self.min_token_len = min_token_len

    SILENT_E_ENDINGS: ClassVar[tuple[str, ...]] = (
        "rg",
        "chang",
        "iv",
        "lv",
        "rv",
        "vv",
        "dat",
        "put",
        "lut",
        "mut",
        "rut",
        "anc",
        "enc",
        "inc",
        "orc",
        "urc",
        "ic",
        "as",
        "is",
        "os",
        "us",
        "iz",
        "az",
        "oz",
        "uz",
        "lin",
        "pir",
        "uir",
        "tir",
        "vir",
        "tur",
        "cur",
        "sur",
    )

    @classmethod
    def _restore_stem(cls, stem: str) -> str:
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] in "bdgmnprt":
            return stem[:-1]
        if stem.endswith(cls.SILENT_E_ENDINGS) and stem not in {"put", "cut", "out"}:
            return stem + "e"
        return stem

    @classmethod
    def _light_lemmatize(cls, word: str) -> str:
        if len(word) > 4:
            if word.endswith(("ies", "ied")):
                return word[:-3] + "y"
            if word.endswith("ing") and len(word) > 5:
                return cls._restore_stem(word[:-3])
            if word.endswith("ed") and len(word) > 4:
                return cls._restore_stem(word[:-2])
            if word.endswith("es") and word[:-2].endswith(("sh", "ch", "ss", "x", "z")):
                return word[:-2]
            if word.endswith("s") and not word.endswith(("ss", "us", "is")):
                return word[:-1]
        return word

    def _process(self, record: TicketRecord) -> TicketRecord:
        contextual = record.text_contextual or record.raw_text or ""
        if not contextual.strip():
            contextual = "empty_ticket"
        record.text_contextual = contextual

        lowered = re.sub(r"<[A-Z_]+>", " ", contextual).lower()
        tokens = [
            self._light_lemmatize(w)
            for w in self.RE_WORDS.findall(lowered)
            if len(w) >= self.min_token_len and w not in STOP_WORDS
        ]
        record.text_lexical = " ".join(tokens) if tokens else "empty"
        return record


def build_default_cleaning_chain() -> TextCleaningHandler:
    """Builds the default sequential handler pipeline for ticket and claim normalization."""
    h1 = EmailHeaderAndFooterHandler()
    h2 = TypedPIIMaskHandler()
    h3 = RedactionMaskNormalizerHandler()
    h4 = TemplateSignatureHandler()
    h5 = DualTrackBuilderHandler()
    h1.set_next(h2).set_next(h3).set_next(h4).set_next(h5)
    return h1
