"""Text preprocessing pipelines for contextual embeddings and lexical feature extraction."""

from __future__ import annotations

import html
import re
import unicodedata
from abc import ABC, abstractmethod
from itertools import pairwise
from typing import ClassVar

from ticket_classification.config.pipeline import CleaningPolicy, LanguagePolicy
from ticket_classification.domain.cleaning.languages import (
    detect_language,
    lexical_tokens,
)
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
        re.compile(
            r"</?(?:html|body|head|div|span|p|br|a|b|i|strong|em|table|tr|td|img|ul|ol|li)(?:\s[^>]*)?/?>",
            re.IGNORECASE,
        ),
    ]

    def __init__(self, policy: CleaningPolicy | None = None) -> None:
        super().__init__()
        self.policy = policy or CleaningPolicy()

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual
        for index, pattern in enumerate(self.HEADER_PATTERNS):
            if index < 3 and not self.policy.strip_email_headers:
                continue
            if index == 3 and not self.policy.strip_disclaimers:
                continue
            text = pattern.sub(" ", text)
        for pattern in self.policy.boilerplate_patterns:
            text = re.sub(pattern, " ", text)
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

    def __init__(self, policy: CleaningPolicy | None = None) -> None:
        super().__init__()
        self.policy = policy or CleaningPolicy()

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual
        if not self.policy.mask_pii:
            return record
        text = self.RE_URL.sub("<URL>", text)
        text = self.RE_EMAIL.sub("<EMAIL>", text)
        text = self.RE_TICKET_ID.sub("<TICKET_ID>", text)
        text = self.RE_AMOUNT.sub("<AMOUNT>", text)
        text = self.RE_PHONE.sub("<PHONE>", text)
        for token, pattern in self.policy.pii_patterns.items():
            text = re.sub(pattern, f"<{token.upper()}>", text)
        record.text_contextual = re.sub(r"\s+", " ", text).strip()
        return record


class RedactionMaskNormalizerHandler(TextCleaningHandler):
    """Normalizes CFPB / financial claim redaction masks (XXXX, XX/XX/XXXX) into <REDACTED>."""

    RE_XXXX = re.compile(r"\b(?:X{2,}|x{2,})(?:/(?:X{2,}|x{2,}))*\b")
    RE_BRACKET_REDACT = re.compile(
        r"\[(?:REDACTED|MASKED|CONFIDENTIAL)\]", re.IGNORECASE
    )

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual
        text = self.RE_XXXX.sub("<REDACTED>", text)
        text = self.RE_BRACKET_REDACT.sub("<REDACTED>", text)
        record.text_contextual = re.sub(r"\s+", " ", text).strip()
        return record


class TemplateSignatureHandler(TextCleaningHandler):
    """Computes a structural template signature for near-duplicate grouping."""

    RE_NUMBERS = re.compile(r"\b\d+\b")
    RE_QUOTED = re.compile(r'"[^"]+"|\'[^\']+\'')

    def __init__(self, mask_numbers: bool = False) -> None:
        super().__init__()
        self.mask_numbers = mask_numbers

    def _process(self, record: TicketRecord) -> TicketRecord:
        text = record.text_contextual.casefold()
        if self.mask_numbers:
            text = self.RE_NUMBERS.sub("<NUM>", text)
        record.template_signature = re.sub(r"\s+", " ", text).strip()
        return record


class DualTrackBuilderHandler(TextCleaningHandler):
    """Produces contextual text for dense embeddings and normalized tokens for term frequency analysis."""

    def __init__(
        self, min_token_len: int = 2, language_policy: LanguagePolicy | None = None
    ) -> None:
        super().__init__()
        self.min_token_len = min_token_len
        self.language_policy = language_policy or LanguagePolicy()

    def _process(self, record: TicketRecord) -> TicketRecord:
        contextual = record.text_contextual
        record.text_contextual = contextual

        detected = detect_language(
            re.sub(r"<[A-Z_]+>", " ", contextual),
            self.language_policy,
            record.language_hint,
        )
        record.languages = detected.languages
        record.language_confidence = detected.confidence
        record.language_status = detected.status

        tokens = lexical_tokens(
            contextual,
            self.language_policy,
            STOP_WORDS,
            self.min_token_len,
            record.languages,
        )
        record.text_lexical = " ".join(tokens) if tokens else "empty"
        return record


class UnicodeLanguageHandler(TextCleaningHandler):
    def __init__(self, policy: LanguagePolicy) -> None:
        super().__init__()
        self.policy = policy

    def _process(self, record: TicketRecord) -> TicketRecord:
        record.text_contextual = unicodedata.normalize(
            self.policy.normalization, html.unescape(record.raw_text)
        )
        return record


def build_default_cleaning_chain(
    cleaning: CleaningPolicy | None = None, language: LanguagePolicy | None = None
) -> TextCleaningHandler:
    """Compose domain cleaning with a separately configured language policy."""
    cleaning = cleaning or CleaningPolicy()
    language = language or LanguagePolicy()
    handlers = [
        UnicodeLanguageHandler(language),
        EmailHeaderAndFooterHandler(cleaning),
        TypedPIIMaskHandler(cleaning),
    ]
    if cleaning.normalize_redactions:
        handlers.append(RedactionMaskNormalizerHandler())
    handlers.extend(
        [
            TemplateSignatureHandler(cleaning.template_mask_numbers),
            DualTrackBuilderHandler(cleaning.min_token_length, language),
        ]
    )
    for current, following in pairwise(handlers):
        current.set_next(following)
    return handlers[0]
