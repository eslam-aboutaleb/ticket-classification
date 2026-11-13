"""Language detection and registered lexical transforms, independent of domain."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException
from langdetect.detector_factory import PROFILES_DIRECTORY

from ticket_classification.config.pipeline import LanguagePolicy

_FACTORY = DetectorFactory()
_FACTORY.load_profile(PROFILES_DIRECTORY)
_FACTORY.seed = 0


@dataclass
class LanguageResult:
    languages: list[str]
    confidence: float
    status: str
    hint: str | None = None


def _detect_langdetect(
    text: str, policy: LanguagePolicy, hint: str | None = None
) -> LanguageResult:
    """Detect script-mixed text; short or ambiguous input is explicitly undetermined."""
    letters = "".join(c for c in text if c.isalpha())
    if len(letters) < policy.min_detection_chars:
        return LanguageResult([], 0.0, "undetermined", hint)
    arabic = " ".join(re.findall(r"[\u0600-\u06ff]+", text))
    latin = " ".join(re.findall(r"[A-Za-z\u00c0-\u024f]+", text))
    # Isolated codes such as HTTP/API are too short to identify a language reliably.
    # Only split scripts when both segments have enough letters for detection.
    inputs = (
        [arabic, latin]
        if all(
            sum(c.isalpha() for c in segment) >= policy.min_detection_chars
            for segment in (arabic, latin)
        )
        else [text]
    )
    found: list[str] = []
    confidences = []
    for segment in inputs:
        detector = _FACTORY.create()
        try:
            detector.append(segment)
            result = detector.get_probabilities()[0]
        except (LangDetectException, IndexError):
            continue
        if result.prob >= policy.min_detection_confidence:
            found.append(result.lang)
            confidences.append(float(result.prob))
    languages = sorted(set(found))
    status = (
        "supported"
        if languages and all(x in policy.supported for x in languages)
        else "unsupported"
        if languages
        else "undetermined"
    )
    return LanguageResult(languages, min(confidences, default=0.0), status, hint)


DETECTORS: dict[str, Callable] = {"langdetect": _detect_langdetect}


def register_language_detector(name: str, detector: Callable) -> None:
    """Register a detector returning LanguageResult; register again before model reload."""
    DETECTORS[name] = detector


def detect_language(
    text: str, policy: LanguagePolicy, hint: str | None = None
) -> LanguageResult:
    if policy.detector not in DETECTORS:
        raise ValueError(f"Unknown language detector: {policy.detector}")
    return DETECTORS[policy.detector](text, policy, hint)


def _unicode_tokens(text: str) -> list[str]:
    # Explicit character fallback for scripts without obligatory whitespace.
    text = re.sub(r"([\u3400-\u9fff\u3040-\u30ff])", r" \1 ", text)
    return re.findall(r"[^\W_]+(?:[._/+:-][^\W_]+)*", text, flags=re.UNICODE)


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"unicode": _unicode_tokens}


def register_tokenizer(name: str, tokenizer: Callable[[str], list[str]]) -> None:
    TOKENIZERS[name] = tokenizer


LexicalPlugin = Callable[[list[str]], list[str]]
LEXICAL_PLUGINS: dict[str, LexicalPlugin] = {
    "unicode": lambda words: words,
    "english": lambda words: words,
    "arabic": lambda words: words,
}


def register_lexical_plugin(name: str, transform: LexicalPlugin) -> None:
    """Register a tokenizer postprocessor at application startup; persist its name."""
    LEXICAL_PLUGINS[name] = transform


def lexical_tokens(
    text: str,
    policy: LanguagePolicy,
    english_stopwords: set[str],
    min_length: int = 2,
    languages: list[str] | None = None,
) -> list[str]:
    text = unicodedata.normalize(
        policy.normalization, re.sub(r"<[A-Z_]+>", " ", text)
    ).casefold()
    if policy.arabic_remove_diacritics:
        text = re.sub(r"[\u064b-\u065f\u0670]", "", text)
    if policy.arabic_fold_alef:
        text = re.sub("[أإآٱ]", "ا", text)
    if policy.tokenizer not in TOKENIZERS:
        raise ValueError(f"Unknown tokenizer: {policy.tokenizer}")
    words = TOKENIZERS[policy.tokenizer](text)
    preserved = {x.casefold() for x in policy.preserve_tokens}
    languages = languages or []
    stopwords = (
        english_stopwords - {"no", "nor", "not", "never"}
        if "en" in languages
        else set()
    )
    for language, values in policy.extra_stopwords.items():
        if language not in languages:
            continue
        stopwords = stopwords | {w.casefold() for w in values}
    words = [
        w
        for w in words
        if w in preserved
        or (
            (len(w) >= min_length or re.search(r"[\u3400-\u9fff\u3040-\u30ff]", w))
            and w not in stopwords
        )
    ]
    # Apply each language transform only to tokens in that detected language's script.
    for language, plugin in policy.lexical_plugins.items():
        if plugin not in LEXICAL_PLUGINS:
            raise ValueError(f"Unknown lexical plugin: {plugin}")
        if language not in languages:
            continue
        pattern = (
            r"[\u0600-\u06ff]"
            if language == "ar"
            else r"^[a-z]+$"
            if language == "en"
            else r"[^\W\d_]"
        )
        words = [
            part
            for word in words
            for part in (
                LEXICAL_PLUGINS[plugin]([word])
                if word not in preserved and re.search(pattern, word)
                else [word]
            )
        ]
    return words
